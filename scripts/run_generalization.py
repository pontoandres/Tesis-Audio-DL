#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "generalization"
PROCESSED = ROOT / "data" / "generalization" / "processed"
WRIGHT_EVAL = ROOT / "external" / "Automated-GuitarAmpModelling" / "evaluate_model.py"
COMMON_EVAL = ROOT / "scripts" / "evaluate_model_outputs.py"

WRIGHT_RUNS = [
    ("starcaster", "CNN", ROOT / "results" / "starcaster50_cnn100_causal128_finealign_v1-" / "model_best.json"),
    ("starcaster", "LSTM", ROOT / "results" / "starcaster50_lstm32_scratch_v1-" / "model_best.json"),
    ("dynacomp", "CNN", ROOT / "results" / "mxrdynacomp50_cnn100_causal128_v1-" / "model_best.json"),
    ("dynacomp", "LSTM", ROOT / "results" / "mxrdynacomp50_lstm32_scratch_v1-" / "model_best.json"),
]

TCN_RUNS = [
    ("starcaster", "TCN", ROOT / "results" / "starcaster50_tcn300c_scratch_v1-" / "model_best.pt"),
    ("dynacomp", "TCN", ROOT / "results" / "mxrdynacomp50_tcn300c_scratch_v1-" / "model_best.pt"),
]


def run_cmd(cmd: list[str]) -> None:
    print("\n> " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def process_wright(effect: str, arch: str, model: Path, cuda: int, chunk: int) -> Path:
    data_location = PROCESSED / effect / "50"
    out = RESULTS / f"{effect}50_{arch.lower()}_prediction.wav"
    internal_metrics = RESULTS / "wright_internal" / f"{effect}50_{arch.lower()}.json"
    internal_metrics.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([
        sys.executable, str(WRIGHT_EVAL),
        "--model", str(model),
        "--data_location", str(data_location),
        "--file_name", "generalization50",
        "--output", str(out),
        "--metrics", str(internal_metrics),
        "--test_chunk", str(chunk),
        "--cuda", str(cuda),
    ])
    return out


def read_mono(path: Path) -> tuple[int, np.ndarray]:
    fs, x = wavfile.read(path)
    if x.ndim == 2:
        if x.shape[1] != 1:
            x = np.mean(x.astype(np.float64), axis=1)
        else:
            x = x[:, 0]
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    else:
        x = x.astype(np.float32)
    return int(fs), np.ascontiguousarray(x)


def process_tcn(effect: str, model_path: Path, device_name: str, chunk: int) -> Path:
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from train_tcn import CausalTCN300, infer_full  # type: ignore

    device = torch.device(device_name)
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["model_config"]
    model = CausalTCN300(
        channels=int(cfg["channels"]),
        kernel_size=int(cfg["kernel_size"]),
        dilations=tuple(int(x) for x in cfg["dilations"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    input_path = PROCESSED / effect / "50" / "test" / "generalization50-input.wav"
    fs, x = read_mono(input_path)
    pred = infer_full(model, x, device, chunk).numpy()
    out = RESULTS / f"{effect}50_tcn_prediction.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(out, fs, pred.astype(np.float32))
    print(f"Saved TCN prediction: {out}")
    return out


def common_evaluate(effect: str, arch: str, prediction: Path) -> Path:
    target = PROCESSED / effect / "50" / "test" / "generalization50-target.wav"
    out = RESULTS / "evaluation" / f"{effect}50_{arch.lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = "model_best.pt" if arch == "TCN" else "model_best.json"
    run_cmd([
        sys.executable, str(COMMON_EVAL),
        "--target", str(target),
        "--prediction", str(prediction),
        "--output", str(out),
        "--effect", effect,
        "--setting", "50",
        "--architecture", arch,
        "--run-name", f"generalization_{effect}50_{arch.lower()}",
        "--checkpoint", checkpoint,
    ])
    return out


def build_summary(paths: list[Path]) -> None:
    rows = []
    for path in paths:
        d = json.loads(path.read_text(encoding="utf-8"))
        exp = d["experiment"]
        m = d["metrics"]
        loud = m["loudness"]
        rows.append({
            "effect": exp["effect"],
            "setting": exp["setting"],
            "architecture": exp["architecture"],
            "esr": m["esr"],
            "mae": m["mae"],
            "mrstft": m["mrstft"],
            "lufs_error": loud["lufs_absolute_error"],
            "target_lufs": loud["target_lufs"],
            "prediction_lufs": loud["prediction_lufs"],
            "seconds": d["audio"]["audio_seconds"],
        })

    csv_path = RESULTS / "generalization_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = RESULTS / "generalization_summary.md"
    lines = [
        "# Generalization evaluation summary",
        "",
        "Lower is better for ESR, MAE, MR-STFT and LUFS error.",
        "",
        "| Effect | Architecture | ESR | MAE | MR-STFT | LUFS error (dB) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        le = "N/A" if r["lufs_error"] is None else f'{r["lufs_error"]:.6f}'
        lines.append(f'| {r["effect"]} | {r["architecture"]} | {r["esr"]:.6f} | {r["mae"]:.6f} | {r["mrstft"]:.6f} | {le} |')
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "="*72)
    print("GENERALIZATION SUMMARY")
    print("="*72)
    for r in rows:
        print(f"{r['effect']:10s} {r['architecture']:4s}  ESR={r['esr']:.6f}  MAE={r['mae']:.6f}  MR-STFT={r['mrstft']:.6f}  LUFSerr={r['lufs_error']}")
    print(f"\nCSV: {csv_path}")
    print(f"MD : {md_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run all 50% generalization inference and common thesis metrics.")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--chunk", type=int, default=20000)
    args = p.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = "cuda"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_flag = 1 if device == "cuda" else 0

    required_audio = [
        PROCESSED / e / "50" / "test" / name
        for e in ("starcaster", "dynacomp")
        for name in ("generalization50-input.wav", "generalization50-target.wav")
    ]
    missing_audio = [p for p in required_audio if not p.exists()]
    if missing_audio:
        raise FileNotFoundError("Run scripts\\prepare_generalization.py first. Missing:\n" + "\n".join(str(p) for p in missing_audio))

    model_files = [x[2] for x in WRIGHT_RUNS] + [x[2] for x in TCN_RUNS]
    missing_models = [p for p in model_files if not p.exists()]
    if missing_models:
        print("\nMissing model checkpoints:")
        for pth in missing_models:
            print(f"  {pth}")
        print("\nNOTE: TCN .pt files are gitignored. If these are the missing files, copy the 50% model_best.pt checkpoints from the machine where the TCNs were trained, or run this script on that machine after copying data\\generalization there.")
        raise FileNotFoundError("Required model checkpoint(s) missing.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    predictions: dict[tuple[str,str],Path] = {}
    for effect, arch, model in WRIGHT_RUNS:
        predictions[(effect, arch)] = process_wright(effect, arch, model, cuda_flag, args.chunk)
    for effect, arch, model in TCN_RUNS:
        predictions[(effect, arch)] = process_tcn(effect, model, device, args.chunk)

    evals = []
    for effect in ("starcaster", "dynacomp"):
        for arch in ("CNN", "LSTM", "TCN"):
            evals.append(common_evaluate(effect, arch, predictions[(effect, arch)]))
    build_summary(evals)
    print("\nDONE. No model was trained or modified.")


if __name__ == "__main__":
    main()
