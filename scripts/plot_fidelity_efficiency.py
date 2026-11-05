#!/usr/bin/env python
"""
Generate thesis figures that combine fidelity and computational efficiency.

Inputs
------
results/evaluation/experiment_registry.csv
    Common fidelity metrics for the canonical thesis experiments.

results/efficiency/benchmark_efficiency.csv
    Common CPU/GPU efficiency benchmark.

The core fidelity-vs-efficiency figures use only the 24 scratch/core runs.
For each pedal and architecture, fidelity is summarized across the four
settings (25/50/75/100) with the mean as the point and one standard deviation
as a vertical error bar. Computational cost is architecture-level and therefore
comes from the common efficiency benchmark.

Lower is better for both axes in the trade-off plots. The lower-left region is
therefore preferable, but no single scalar "overall score" is constructed:
ESR, MAE, MR-STFT and LUFS error remain separate fidelity dimensions.

Outputs are written to:
docs/figures/fidelity_efficiency/
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVAL_CSV = ROOT / "results" / "evaluation" / "experiment_registry.csv"
EFF_CSV = ROOT / "results" / "efficiency" / "benchmark_efficiency.csv"
OUT_DIR = ROOT / "docs" / "figures" / "fidelity_efficiency"

ARCH_ORDER = ("CNN", "LSTM", "TCN")
EFFECT_ORDER = ("starcaster", "dynacomp")
DEVICE_ORDER = ("cpu", "cuda")

METRICS = {
    "esr": "ESR",
    "mae": "MAE",
    "mrstft": "MR-STFT",
    "lufs_error": "LUFS absolute error (dB)",
}


def normalize_architecture(value: str) -> str:
    s = value.strip().lower()
    if "tcn" in s:
        return "TCN"
    if "lstm" in s:
        return "LSTM"
    if "cnn" in s or "gatedconvnet" in s:
        return "CNN"
    raise ValueError(f"Unknown architecture label: {value!r}")


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required input not found: {path}\n"
            "Run the corresponding evaluation/benchmark stage first."
        )

    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_fidelity() -> dict:
    rows = read_csv(EVAL_CSV)

    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}

    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("role") != "core":
            continue
        if row.get("training_mode") != "scratch":
            continue

        effect = row["effect"].strip().lower()
        arch = normalize_architecture(row["architecture"])

        if effect not in EFFECT_ORDER or arch not in ARCH_ORDER:
            continue

        key = (effect, arch)
        if key not in grouped:
            grouped[key] = {metric: [] for metric in METRICS}

        for metric in METRICS:
            value = row.get(metric, "")
            if value not in ("", None):
                grouped[key][metric].append(float(value))

    summary = {}

    for effect in EFFECT_ORDER:
        for arch in ARCH_ORDER:
            key = (effect, arch)

            if key not in grouped:
                raise RuntimeError(
                    f"Missing fidelity rows for {effect} / {arch}"
                )

            summary[key] = {}

            for metric in METRICS:
                values = grouped[key][metric]

                if len(values) != 4:
                    raise RuntimeError(
                        f"Expected 4 core settings for {effect} / {arch} / "
                        f"{metric}, found {len(values)}"
                    )

                summary[key][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": values,
                }

    return summary


def load_efficiency() -> dict:
    rows = read_csv(EFF_CSV)
    summary = {}

    for row in rows:
        arch = normalize_architecture(row["architecture"])
        device = row["device"].strip().lower()

        if arch not in ARCH_ORDER or device not in DEVICE_ORDER:
            continue

        key = (device, arch)

        if key in summary:
            raise RuntimeError(
                f"Duplicate efficiency row for {device} / {arch}"
            )

        peak = row.get("cuda_peak_allocated_bytes", "")

        summary[key] = {
            "parameters": int(row["trainable_parameters"]),
            "state_size_mb": float(row["state_size_mb"]),
            "median_inference_seconds": float(
                row["median_inference_seconds"]
            ),
            "p95_inference_seconds": float(
                row["p95_inference_seconds"]
            ),
            "rtf": float(row["median_rtf"]),
            "realtime_x": float(row["median_realtime_x"]),
            "cuda_peak_mb": (
                None
                if peak in ("", None)
                else float(peak) / (1024 ** 2)
            ),
        }

    for device in DEVICE_ORDER:
        for arch in ARCH_ORDER:
            if (device, arch) not in summary:
                raise RuntimeError(
                    f"Missing efficiency result for {device} / {arch}"
                )

    return summary


def effect_label(effect: str) -> str:
    if effect == "starcaster":
        return "Starcaster Distortion"
    if effect == "dynacomp":
        return "MXR Dyna Comp"
    return effect


def device_label(device: str) -> str:
    return "GPU" if device == "cuda" else "CPU"


def metric_filename(metric: str) -> str:
    if metric == "lufs_error":
        return "lufs"
    return metric


def plot_tradeoff(
    fidelity: dict,
    efficiency: dict,
    effect: str,
    device: str,
    metric: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.5))

    for arch in ARCH_ORDER:
        x = efficiency[(device, arch)]["rtf"]
        y = fidelity[(effect, arch)][metric]["mean"]
        y_std = fidelity[(effect, arch)][metric]["std"]

        ax.errorbar(
            x,
            y,
            yerr=y_std,
            fmt="o",
            markersize=8,
            capsize=4,
            linewidth=1.2,
        )

        ax.annotate(
            arch,
            (x, y),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_xlabel(
        f"Real-time factor (RTF) on {device_label(device)}"
    )
    ax.set_ylabel(f"Mean {METRICS[metric]} across pedal settings")
    ax.set_title(
        f"{effect_label(effect)}: fidelity vs {device_label(device)} cost"
    )
    ax.grid(True, alpha=0.25)

    ax.text(
        0.02,
        0.02,
        "Lower-left is better\n(error ↓, RTF ↓)",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rtf_by_device(
    efficiency: dict,
    path: Path,
) -> None:
    x = np.arange(len(ARCH_ORDER))
    width = 0.36

    cpu = [efficiency[("cpu", arch)]["rtf"] for arch in ARCH_ORDER]
    gpu = [efficiency[("cuda", arch)]["rtf"] for arch in ARCH_ORDER]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    ax.bar(x - width / 2, cpu, width, label="CPU")
    ax.bar(x + width / 2, gpu, width, label="GPU")

    ax.set_xticks(x, ARCH_ORDER)
    ax.set_ylabel("Median real-time factor (RTF)")
    ax.set_title("Inference cost by architecture and device")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_parameters(
    efficiency: dict,
    path: Path,
) -> None:
    values = [
        efficiency[("cpu", arch)]["parameters"]
        for arch in ARCH_ORDER
    ]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    bars = ax.bar(ARCH_ORDER, values)

    ax.set_ylabel("Trainable parameters")
    ax.set_title("Trainable parameter count by architecture")
    ax.grid(True, axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_state_size(
    efficiency: dict,
    path: Path,
) -> None:
    values = [
        efficiency[("cpu", arch)]["state_size_mb"]
        for arch in ARCH_ORDER
    ]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    bars = ax.bar(ARCH_ORDER, values)

    ax.set_ylabel("Standardized model state size (MB)")
    ax.set_title("Model-state size by architecture")
    ax.grid(True, axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gpu_memory(
    efficiency: dict,
    path: Path,
) -> None:
    values = [
        efficiency[("cuda", arch)]["cuda_peak_mb"]
        for arch in ARCH_ORDER
    ]

    if any(value is None for value in values):
        print("Skipping GPU memory plot: CUDA peak data incomplete.")
        return

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    bars = ax.bar(ARCH_ORDER, values)

    ax.set_ylabel("Peak allocated CUDA memory (MB)")
    ax.set_title("Peak GPU memory during inference")
    ax.grid(True, axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(
    fidelity: dict,
    efficiency: dict,
    path: Path,
) -> None:
    fieldnames = [
        "effect",
        "architecture",
        "esr_mean",
        "esr_std",
        "mae_mean",
        "mae_std",
        "mrstft_mean",
        "mrstft_std",
        "lufs_error_mean",
        "lufs_error_std",
        "parameters",
        "state_size_mb",
        "cpu_rtf",
        "cpu_realtime_x",
        "gpu_rtf",
        "gpu_realtime_x",
        "gpu_peak_mb",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for effect in EFFECT_ORDER:
            for arch in ARCH_ORDER:
                writer.writerow(
                    {
                        "effect": effect,
                        "architecture": arch,
                        "esr_mean": fidelity[(effect, arch)]["esr"]["mean"],
                        "esr_std": fidelity[(effect, arch)]["esr"]["std"],
                        "mae_mean": fidelity[(effect, arch)]["mae"]["mean"],
                        "mae_std": fidelity[(effect, arch)]["mae"]["std"],
                        "mrstft_mean": fidelity[(effect, arch)]["mrstft"]["mean"],
                        "mrstft_std": fidelity[(effect, arch)]["mrstft"]["std"],
                        "lufs_error_mean": fidelity[(effect, arch)]["lufs_error"]["mean"],
                        "lufs_error_std": fidelity[(effect, arch)]["lufs_error"]["std"],
                        "parameters": efficiency[("cpu", arch)]["parameters"],
                        "state_size_mb": efficiency[("cpu", arch)]["state_size_mb"],
                        "cpu_rtf": efficiency[("cpu", arch)]["rtf"],
                        "cpu_realtime_x": efficiency[("cpu", arch)]["realtime_x"],
                        "gpu_rtf": efficiency[("cuda", arch)]["rtf"],
                        "gpu_realtime_x": efficiency[("cuda", arch)]["realtime_x"],
                        "gpu_peak_mb": efficiency[("cuda", arch)]["cuda_peak_mb"],
                    }
                )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fidelity = load_fidelity()
    efficiency = load_efficiency()

    generated = []

    # Main trade-off plots: ESR and MR-STFT on CPU and GPU.
    for effect in EFFECT_ORDER:
        for device in DEVICE_ORDER:
            for metric in ("esr", "mrstft"):
                path = OUT_DIR / (
                    f"{effect}_{metric_filename(metric)}_vs_"
                    f"{device}_rtf.png"
                )
                plot_tradeoff(
                    fidelity,
                    efficiency,
                    effect,
                    device,
                    metric,
                    path,
                )
                generated.append(path)

    # Complementary GPU plots for amplitude/loudness metrics.
    for effect in EFFECT_ORDER:
        for metric in ("mae", "lufs_error"):
            path = OUT_DIR / (
                f"{effect}_{metric_filename(metric)}_vs_gpu_rtf.png"
            )
            plot_tradeoff(
                fidelity,
                efficiency,
                effect,
                "cuda",
                metric,
                path,
            )
            generated.append(path)

    extra_plots = [
        (
            plot_rtf_by_device,
            OUT_DIR / "architecture_rtf_cpu_gpu.png",
        ),
        (
            plot_parameters,
            OUT_DIR / "architecture_parameters.png",
        ),
        (
            plot_state_size,
            OUT_DIR / "architecture_state_size.png",
        ),
        (
            plot_gpu_memory,
            OUT_DIR / "architecture_gpu_peak_memory.png",
        ),
    ]

    for function, path in extra_plots:
        function(efficiency, path)
        if path.exists():
            generated.append(path)

    summary_csv = OUT_DIR / "fidelity_efficiency_summary.csv"
    write_summary_csv(fidelity, efficiency, summary_csv)

    print("=" * 78)
    print("FIDELITY vs EFFICIENCY FIGURES")
    print("=" * 78)
    print(f"Evaluation input : {EVAL_CSV}")
    print(f"Efficiency input : {EFF_CSV}")
    print(f"Output directory : {OUT_DIR}")
    print()
    print(f"Generated {len(generated)} figures:")
    for path in generated:
        print(f"  {path.relative_to(ROOT)}")
    print()
    print(f"Summary CSV: {summary_csv.relative_to(ROOT)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
