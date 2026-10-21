"""Batch evaluator for canonical thesis experiments.

Reads configs/evaluation_manifest.json, evaluates every listed model with the
common metric implementation in evaluate_model_outputs.py, and writes per-run
JSON files plus CSV/Markdown registries.

The manifest is explicit on purpose: obsolete/debug runs under results/ are
not silently mixed with the canonical experiment set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from evaluate_model_outputs import (
    compute_loudness_metrics,
    compute_mrstft,
    compute_time_metrics,
    load_mono_wav,
    match_lengths,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "evaluation_manifest.json"
DEFAULT_OUTDIR = ROOT / "results" / "evaluation"
DEFAULT_CSV = DEFAULT_OUTDIR / "experiment_registry.csv"
DEFAULT_MD = ROOT / "docs" / "experiment_registry.md"

ESR_TOLERANCE = 1e-5


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def best_epoch(values):
    if not values:
        return None
    return min(range(len(values)), key=lambda i: values[i]) + 1


def load_training_metadata(run_dir: Path) -> dict:
    metadata = {
        "final_epoch": None,
        "best_epoch": None,
        "best_val": None,
        "reference_esr": None,
        "reference_esr_source": None,
    }

    stats_path = run_dir / "training_stats.json"
    if stats_path.exists():
        stats = load_json(stats_path)
        vals = stats.get("validation_losses", [])
        metadata["final_epoch"] = stats.get("current_epoch")
        metadata["best_epoch"] = best_epoch(vals)
        metadata["best_val"] = stats.get(
            "best_val_loss",
            min(vals) if vals else None,
        )

        ref = stats.get("test_lossESR_best")
        if ref is not None:
            metadata["reference_esr"] = float(ref)
            metadata["reference_esr_source"] = (
                "training_stats.json:test_lossESR_best"
            )

    if metadata["reference_esr"] is None:
        metrics_path = run_dir / "test_metrics.json"
        if metrics_path.exists():
            old = load_json(metrics_path)
            ref = old.get("test_ESR")
            if ref is not None:
                metadata["reference_esr"] = float(ref)
                metadata["reference_esr_source"] = (
                    "test_metrics.json:test_ESR"
                )

    return metadata


def evaluate_one(run: dict, outdir: Path, length_policy: str) -> dict:
    run_id = run["id"]
    run_dir = ROOT / run["run_dir"]
    target_path = ROOT / run["target"]
    prediction_path = run_dir / run.get("prediction", "test_out_bestv.wav")
    checkpoint_path = run_dir / run.get("checkpoint", "model_best.json")

    missing = [
        str(path.relative_to(ROOT))
        for path in (target_path, prediction_path)
        if not path.exists()
    ]
    if missing:
        return {
            "id": run_id,
            "status": "missing_files",
            "missing_files": missing,
            "effect": run.get("effect"),
            "setting": run.get("setting"),
            "architecture": run.get("architecture"),
            "training_mode": run.get("training_mode"),
            "role": run.get("role"),
        }

    target_sr, target, target_dtype = load_mono_wav(target_path)
    pred_sr, prediction, pred_dtype = load_mono_wav(prediction_path)

    if target_sr != pred_sr:
        raise ValueError(
            f"{run_id}: sample-rate mismatch "
            f"(target={target_sr}, prediction={pred_sr})"
        )

    original_target_samples = len(target)
    original_prediction_samples = len(prediction)

    target, prediction, evaluated_samples = match_lengths(
        target,
        prediction,
        length_policy,
    )

    time_metrics = compute_time_metrics(target, prediction)
    stft_metrics = compute_mrstft(
        target,
        prediction,
        fft_sizes=[512, 2048, 8192],
        hop_sizes=[128, 512, 2048],
    )
    loudness_metrics = compute_loudness_metrics(
        target,
        prediction,
        target_sr,
    )

    training = load_training_metadata(run_dir)
    ref_esr = training["reference_esr"]

    if ref_esr is None:
        esr_abs_diff = None
        esr_matches_reference = None
    else:
        esr_abs_diff = abs(time_metrics["esr"] - ref_esr)
        esr_matches_reference = esr_abs_diff <= ESR_TOLERANCE

    result = {
        "schema_version": 1,
        "status": "ok",
        "id": run_id,
        "effect": run.get("effect"),
        "setting": run.get("setting"),
        "architecture": run.get("architecture"),
        "training_mode": run.get("training_mode"),
        "pretraining_source": run.get("pretraining_source"),
        "role": run.get("role"),
        "run_dir": run["run_dir"],
        "target": run["target"],
        "prediction": str(
            prediction_path.relative_to(ROOT)
        ).replace("\\", "/"),
        "checkpoint": (
            str(checkpoint_path.relative_to(ROOT)).replace("\\", "/")
            if checkpoint_path.exists()
            else None
        ),
        "checkpoint_size_bytes": (
            checkpoint_path.stat().st_size
            if checkpoint_path.exists()
            else None
        ),
        "audio": {
            "sample_rate_hz": target_sr,
            "evaluated_samples": evaluated_samples,
            "audio_seconds": evaluated_samples / target_sr,
            "target_original_samples": original_target_samples,
            "prediction_original_samples": original_prediction_samples,
            "target_dtype": target_dtype,
            "prediction_dtype": pred_dtype,
            "length_policy": length_policy,
            "amplitude_normalization": False,
            "resampling": False,
        },
        "training": {
            "final_epoch": training["final_epoch"],
            "best_epoch": training["best_epoch"],
            "best_val": training["best_val"],
        },
        "metrics": {
            **time_metrics,
            **stft_metrics,
            "loudness": loudness_metrics,
        },
        "validation": {
            "reference_esr": ref_esr,
            "reference_esr_source": training["reference_esr_source"],
            "esr_abs_diff": esr_abs_diff,
            "esr_matches_reference": esr_matches_reference,
            "tolerance": ESR_TOLERANCE,
        },
        "metric_configuration": {
            "mrstft_fft_sizes": [512, 2048, 8192],
            "mrstft_hop_sizes": [128, 512, 2048],
            "mrstft_window": "Hann",
            "mrstft_center": True,
            "mrstft_combination": (
                "mean_over_resolutions("
                "spectral_convergence + log_magnitude_L1)"
            ),
        },
    }

    outdir.mkdir(parents=True, exist_ok=True)
    with open(
        outdir / f"{run_id}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def fmt(value, decimals=6):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{decimals}f}"
    return str(value)


def registry_row(result: dict) -> dict:
    loudness = result.get("metrics", {}).get("loudness", {})
    training = result.get("training", {})
    validation = result.get("validation", {})

    return {
        "id": result.get("id"),
        "role": result.get("role"),
        "effect": result.get("effect"),
        "setting": result.get("setting"),
        "architecture": result.get("architecture"),
        "training_mode": result.get("training_mode"),
        "pretraining_source": result.get("pretraining_source"),
        "final_epoch": training.get("final_epoch"),
        "best_epoch": training.get("best_epoch"),
        "best_val": training.get("best_val"),
        "esr": result.get("metrics", {}).get("esr"),
        "mae": result.get("metrics", {}).get("mae"),
        "mse": result.get("metrics", {}).get("mse"),
        "rmse": result.get("metrics", {}).get("rmse"),
        "mrstft": result.get("metrics", {}).get("mrstft"),
        "mrstft_sc": result.get("metrics", {}).get(
            "mrstft_spectral_convergence"
        ),
        "mrstft_logmag": result.get("metrics", {}).get(
            "mrstft_log_magnitude_l1"
        ),
        "target_lufs": loudness.get("target_lufs"),
        "prediction_lufs": loudness.get("prediction_lufs"),
        "lufs_error": loudness.get("lufs_absolute_error"),
        "checkpoint_size_bytes": result.get("checkpoint_size_bytes"),
        "reference_esr": validation.get("reference_esr"),
        "esr_abs_diff": validation.get("esr_abs_diff"),
        "esr_validation_ok": validation.get("esr_matches_reference"),
        "status": result.get("status"),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "role",
        "effect",
        "setting",
        "architecture",
        "training_mode",
        "pretraining_source",
        "final_epoch",
        "best_epoch",
        "best_val",
        "esr",
        "mae",
        "mse",
        "rmse",
        "mrstft",
        "mrstft_sc",
        "mrstft_logmag",
        "target_lufs",
        "prediction_lufs",
        "lufs_error",
        "checkpoint_size_bytes",
        "reference_esr",
        "esr_abs_diff",
        "esr_validation_ok",
        "status",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(items: list[dict]) -> str:
    lines = [
        "| Effect | Setting | Architecture | Training | ESR | MAE | MR-STFT | LUFS err (dB) | Best val | Best epoch |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for r in items:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.get("effect", "")),
                    str(r.get("setting", "")),
                    str(r.get("architecture", "")),
                    str(r.get("training_mode", "")),
                    fmt(r.get("esr")),
                    fmt(r.get("mae")),
                    fmt(r.get("mrstft")),
                    fmt(r.get("lufs_error"), 3),
                    fmt(r.get("best_val")),
                    fmt(r.get("best_epoch"), 0),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    core = [
        r for r in rows
        if r.get("role") == "core" and r.get("status") == "ok"
    ]
    transfer = [
        r for r in rows
        if r.get("role") == "transfer_learning"
        and r.get("status") == "ok"
    ]
    missing = [r for r in rows if r.get("status") != "ok"]

    validated = [
        r for r in rows
        if r.get("status") == "ok"
        and r.get("reference_esr") is not None
    ]
    ok_count = sum(
        bool(r.get("esr_validation_ok"))
        for r in validated
    )

    text = [
        "# Experiment registry",
        "",
        "Generated by scripts/evaluate_all_existing_models.py.",
        "",
        (
            "All fidelity metrics are recalculated with the common "
            "external evaluator rather than copied from "
            "architecture-specific training losses."
        ),
        "",
        "## Core architecture experiments",
        "",
        (
            markdown_table(core)
            if core
            else "_No core experiments evaluated yet._"
        ),
        "",
        "## Complementary transfer-learning experiments",
        "",
        (
            markdown_table(transfer)
            if transfer
            else "_No transfer-learning experiments evaluated yet._"
        ),
        "",
        "## ESR reproducibility check",
        "",
        (
            "When an older CoreAudioML ESR is available, the pipeline "
            "compares it against the independently recomputed ESR with "
            "tolerance 1e-5."
        ),
        "",
        (
            f"Validated against previous ESR values: "
            f"**{ok_count}/{len(validated)}** within tolerance."
            if validated
            else "No previous ESR values were available for validation."
        ),
        "",
    ]

    if missing:
        text.extend(["## Missing/incomplete runs", ""])
        for r in missing:
            text.append(
                f"- {r.get('id')}: {r.get('status')}"
            )
        text.append("")

    text.extend(
        [
            "## Notes",
            "",
            (
                "- Lower is better for ESR, MAE, MSE, RMSE, "
                "MR-STFT and LUFS error."
            ),
            (
                "- best_val remains useful for checkpoint selection "
                "and convergence, but is not the common cross-"
                "architecture fidelity metric."
            ),
            (
                "- Computational benchmarks (parameters, inference "
                "time, RTF and peak inference memory) will be added "
                "in the dedicated efficiency benchmark stage."
            ),
            "",
        ]
    )

    path.write_text("\n".join(text), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all canonical thesis runs and build "
            "a metric registry."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_MD,
    )
    parser.add_argument(
        "--length-policy",
        choices=("strict", "trim"),
        default="strict",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path

    manifest = load_json(manifest_path)
    runs = manifest["runs"]

    print("=" * 88)
    print("BATCH THESIS EVALUATION")
    print("=" * 88)
    print(f"Manifest: {manifest_path}")
    print(f"Runs:     {len(runs)}")
    print()

    results = []

    for i, run in enumerate(runs, 1):
        print(f"[{i:02d}/{len(runs):02d}] {run['id']}")

        try:
            result = evaluate_one(
                run,
                args.outdir,
                args.length_policy,
            )
        except Exception as exc:
            result = {
                "id": run["id"],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "effect": run.get("effect"),
                "setting": run.get("setting"),
                "architecture": run.get("architecture"),
                "training_mode": run.get("training_mode"),
                "role": run.get("role"),
            }

        results.append(result)

        if result["status"] == "ok":
            metrics = result["metrics"]
            lufs = metrics["loudness"].get(
                "lufs_absolute_error"
            )
            ref = result["validation"].get(
                "reference_esr"
            )
            ref_ok = result["validation"].get(
                "esr_matches_reference"
            )

            print(
                f"    ESR={metrics['esr']:.6f}  "
                f"MAE={metrics['mae']:.6f}  "
                f"MR-STFT={metrics['mrstft']:.6f}  "
                f"LUFSerr={fmt(lufs, 3) or 'N/A'}"
            )

            if ref is not None:
                print(
                    f"    ESR check: reference={ref:.9f}  "
                    f"diff="
                    f"{result['validation']['esr_abs_diff']:.3e}  "
                    f"{'OK' if ref_ok else 'MISMATCH'}"
                )
        else:
            detail = result.get(
                "error",
                result.get("missing_files"),
            )
            print(
                f"    {result['status']}: {detail}"
            )

    rows = [registry_row(r) for r in results]

    write_csv(rows, args.csv)
    write_markdown(rows, args.markdown)

    ok = sum(r["status"] == "ok" for r in results)
    failed = len(results) - ok

    print()
    print("=" * 88)
    print(
        f"DONE: {ok} evaluated, "
        f"{failed} missing/error"
    )
    print(f"Per-run JSON: {args.outdir}")
    print(f"CSV registry: {args.csv}")
    print(f"Markdown:     {args.markdown}")
    print("=" * 88)


if __name__ == "__main__":
    main()
