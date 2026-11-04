import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "results" / "evaluation"
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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

    match = re.match(r"(\d+)", rest)
    setting = int(match.group(1)) if match else None

    if "_cnn_" in name:
        architecture = "CNN"
    elif "_lstm32_" in name:
        architecture = "LSTM"
    elif "_tcn300c_" in name:
        architecture = "TCN"
    else:
        architecture = "Other"

    mode = "FT" if "_ft_" in name else "Scratch"
    return effect, setting, architecture, mode


def load_rows():
    rows = []

    for path in sorted(EVAL_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        run_name = path.stem
        effect, setting, architecture, mode = parse_run_name(run_name)

        metrics = data.get("metrics", {})
        loudness = metrics.get("loudness", {})

        rows.append(
            {
                "run_name": run_name,
                "effect": effect,
                "setting": setting,
                "architecture": architecture,
                "mode": mode,
                "esr": metrics.get("esr"),
                "mae": metrics.get("mae"),
                "mrstft": metrics.get("mrstft"),
                "lufs_error": loudness.get("lufs_absolute_error"),
            }
        )

    return rows


def find_value(rows, effect, setting, architecture, mode, metric):
    for row in rows:
        if (
            row["effect"] == effect
            and row["setting"] == setting
            and row["architecture"] == architecture
            and row["mode"] == mode
        ):
            return row[metric]
    return None


def plot_scratch_metric(rows, effect, metric, ylabel, filename):
    settings = [25, 50, 75, 100]
    architectures = ["CNN", "LSTM", "TCN"]

    plt.figure(figsize=(8, 5))

    for architecture in architectures:
        values = [
            find_value(
                rows,
                effect,
                setting,
                architecture,
                "Scratch",
                metric,
            )
            for setting in settings
        ]

        missing = [
            setting
            for setting, value in zip(settings, values)
            if value is None
        ]
        if missing:
            raise ValueError(
                f"Missing {metric} values for {effect} {architecture} "
                f"at settings {missing}"
            )

        plt.plot(
            settings,
            values,
            marker="o",
            linewidth=2,
            label=architecture,
        )

    plt.xticks(settings)
    plt.xlabel("Setting (%)")
    plt.ylabel(ylabel)
    plt.title(f"{effect} - {ylabel} (Scratch)")
    plt.grid(alpha=0.25)
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

    if any(value is None for value in values):
        raise ValueError("Missing ESR values for the 50% transfer-learning plot.")

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.ylabel("ESR")
    plt.title("Fine-tuning experiments at 50% setting (ESR)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ft_50_esr_comparison.png", dpi=200)
    plt.close()


def main():
    rows = load_rows()

    metrics = [
        ("esr", "ESR", "esr"),
        ("mae", "MAE", "mae"),
        ("mrstft", "MR-STFT", "mrstft"),
        ("lufs_error", "LUFS absolute error (dB)", "lufs"),
    ]

    for effect, prefix in [
        ("Starcaster", "starcaster"),
        ("DynaComp", "dynacomp"),
    ]:
        for metric, ylabel, suffix in metrics:
            plot_scratch_metric(
                rows,
                effect=effect,
                metric=metric,
                ylabel=ylabel,
                filename=f"{prefix}_scratch_{suffix}.png",
            )

    plot_ft_50_esr(rows)

    print("Saved figures in:")
    print(OUT_DIR)
    print("Generated scratch plots for CNN, LSTM and TCN:")
    print("  ESR, MAE, MR-STFT and LUFS error for both effects.")


if __name__ == "__main__":
    main()
