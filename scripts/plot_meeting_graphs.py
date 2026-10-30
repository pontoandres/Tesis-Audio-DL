import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "results" / "evaluation"
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_metric(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def parse_run_name(name):
    if name.startswith("starcaster"):
        effect = "Starcaster"
        rest = name[len("starcaster"):]
    elif name.startswith("dynacomp"):
        effect = "DynaComp"
        rest = name[len("dynacomp"):]
    else:
        effect = "Unknown"
        rest = name

    m = re.match(r"(\d+)", rest)
    setting = int(m.group(1)) if m else None

    if "_cnn_" in name:
        architecture = "CNN"
    elif "_lstm32_" in name:
        architecture = "LSTM"
    else:
        architecture = "Other"

    if "_ft_" in name:
        mode = "FT"
    else:
        mode = "Scratch"

    return effect, setting, architecture, mode


def load_rows():
    rows = []
    for path in sorted(EVAL_DIR.glob("*.json")):
        if path.name == "experiment_registry.json":
            continue

        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)

        run_name = path.stem
        effect, setting, architecture, mode = parse_run_name(run_name)

        metrics = d.get("metrics", {})
        loudness = metrics.get("loudness", {})

        esr = metrics.get("esr")
        mrstft = metrics.get("mrstft")
        lufs_error = loudness.get("lufs_absolute_error")

        if esr is None:
            print(f"WARNING: ESR missing for {run_name}")

        rows.append({
            "run_name": run_name,
            "effect": effect,
            "setting": setting,
            "architecture": architecture,
            "mode": mode,
            "esr": esr,
            "mrstft": mrstft,
            "lufs_error": lufs_error,
        })
    return rows


def find_value(rows, effect, setting, architecture, mode, metric):
    for r in rows:
        if (
            r["effect"] == effect
            and r["setting"] == setting
            and r["architecture"] == architecture
            and r["mode"] == mode
        ):
            return r[metric]
    return None


def plot_scratch_metric(rows, effect, metric, ylabel, filename):
    settings = [25, 50, 75, 100]

    cnn_vals = [find_value(rows, effect, s, "CNN", "Scratch", metric) for s in settings]
    lstm_vals = [find_value(rows, effect, s, "LSTM", "Scratch", metric) for s in settings]

    x = np.arange(len(settings))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, cnn_vals, width, label="CNN scratch")
    plt.bar(x + width / 2, lstm_vals, width, label="LSTM scratch")

    plt.xticks(x, [str(s) for s in settings])
    plt.xlabel("Setting (%)")
    plt.ylabel(ylabel)
    plt.title(f"{effect} - {ylabel} (Scratch)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=200)
    plt.close()


def plot_ft_50_esr(rows):
    labels = [
        "Star CNN\nscratch",
        "Star CNN\nFT",
        "Star LSTM\nscratch",
        "Star LSTM\nFT",
        "Dyna CNN\nscratch",
        "Dyna CNN\nFT",
        "Dyna LSTM\nscratch",
        "Dyna LSTM\nFT",
    ]

    values = [
        find_value(rows, "Starcaster", 50, "CNN", "Scratch", "esr"),
        find_value(rows, "Starcaster", 50, "CNN", "FT", "esr"),
        find_value(rows, "Starcaster", 50, "LSTM", "Scratch", "esr"),
        find_value(rows, "Starcaster", 50, "LSTM", "FT", "esr"),
        find_value(rows, "DynaComp", 50, "CNN", "Scratch", "esr"),
        find_value(rows, "DynaComp", 50, "CNN", "FT", "esr"),
        find_value(rows, "DynaComp", 50, "LSTM", "Scratch", "esr"),
        find_value(rows, "DynaComp", 50, "LSTM", "FT", "esr"),
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.ylabel("ESR")
    plt.title("Fine-tuning experiments at 50% setting (ESR)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ft_50_esr_comparison.png", dpi=200)
    plt.close()


def main():
    rows = load_rows()

    plot_scratch_metric(
        rows,
        effect="Starcaster",
        metric="esr",
        ylabel="ESR",
        filename="starcaster_scratch_esr.png",
    )

    plot_scratch_metric(
        rows,
        effect="DynaComp",
        metric="esr",
        ylabel="ESR",
        filename="dynacomp_scratch_esr.png",
    )

    plot_scratch_metric(
    rows,
    effect="Starcaster",
    metric="mrstft",
    ylabel="MR-STFT",
    filename="starcaster_scratch_mrstft.png",
)

    plot_scratch_metric(
        rows,
        effect="DynaComp",
        metric="mrstft",
        ylabel="MR-STFT",
        filename="dynacomp_scratch_mrstft.png",
    )

    plot_scratch_metric(
        rows,
        effect="Starcaster",
        metric="lufs_error",
        ylabel="LUFS absolute error (dB)",
        filename="starcaster_scratch_lufs.png",
    )

    plot_scratch_metric(
        rows,
        effect="DynaComp",
        metric="lufs_error",
        ylabel="LUFS absolute error (dB)",
        filename="dynacomp_scratch_lufs.png",
    )

    plot_ft_50_esr(rows)

    print("Saved figures in:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()