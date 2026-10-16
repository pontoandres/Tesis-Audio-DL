from pathlib import Path
import json
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTDIR = ROOT / "docs" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    "Starcaster50 CNN scratch": RESULTS / "starcaster50_cnn100_causal128_finealign_v1-" / "training_stats.json",
    "Starcaster50 CNN FT BigMuff": RESULTS / "starcaster50_cnn_ft_bigMuff_v1-" / "training_stats.json",
    "Starcaster50 LSTM scratch": RESULTS / "starcaster50_lstm32_scratch_v1-" / "training_stats.json",
    "Starcaster50 LSTM FT WrightMuff1": RESULTS / "starcaster50_lstm32_ft_wrightMuff1_v1-" / "training_stats.json",
    "DynaComp50 CNN scratch": RESULTS / "mxrdynacomp50_cnn100_causal128_v1-" / "training_stats.json",
    "DynaComp50 CNN FT BigMuff": RESULTS / "mxrdynacomp50_cnn_ft_bigMuff_v1-" / "training_stats.json",
    "DynaComp50 LSTM scratch": RESULTS / "mxrdynacomp50_lstm32_scratch_v1-" / "training_stats.json",
    "DynaComp50 LSTM FT WrightMuff1": RESULTS / "mxrdynacomp50_lstm32_ft_wrightMuff1_v1-" / "training_stats.json",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def best_epoch(validation_losses):
    if not validation_losses:
        return None
    return min(range(len(validation_losses)), key=lambda i: validation_losses[i]) + 1


def best_val(validation_losses):
    if not validation_losses:
        return None
    return min(validation_losses)


def extract_metrics(name, path):
    data = load_json(path)
    val_losses = data.get("validation_losses", [])

    test_loss_best = data.get("test_loss_best")
    test_esr_best = data.get("test_lossESR_best")

    # Some older runs (notably Starcaster50 CNN scratch) were evaluated
    # separately after the integrated test stage hit a GPU-memory error.
    # In that case, recover the best-checkpoint test metrics from
    # test_metrics.json in the same result folder.
    if test_loss_best is None or test_esr_best is None:
        fallback = path.parent / "test_metrics.json"
        if fallback.exists():
            fallback_data = load_json(fallback)
            if test_loss_best is None:
                test_loss_best = fallback_data.get("test_loss")
            if test_esr_best is None:
                test_esr_best = fallback_data.get("test_ESR")

    return {
        "name": name,
        "path": str(path),
        "validation_losses": val_losses,
        "epoch_final": data.get("current_epoch"),
        "epoch_best": best_epoch(val_losses),
        "best_val": data.get("best_val_loss", best_val(val_losses)),
        "test_loss_best": test_loss_best,
        "test_esr_best": test_esr_best,
    }


all_data = {
    name: extract_metrics(name, path)
    for name, path in EXPERIMENTS.items()
}


def plot_validation_group(title, names, filename):
    plt.figure(figsize=(10, 6))

    for name in names:
        vals = all_data[name]["validation_losses"]
        epochs = list(range(1, len(vals) + 1))
        plt.plot(epochs, vals, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTDIR / filename, dpi=200)
    plt.close()


def plot_metric_bar(title, metric_key, names, filename, ylabel):
    labels = []
    values = []

    for name in names:
        value = all_data[name][metric_key]
        if value is not None:
            labels.append(name)
            values.append(value)

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTDIR / filename, dpi=200)
    plt.close()


starcaster_names = [
    "Starcaster50 CNN scratch",
    "Starcaster50 CNN FT BigMuff",
    "Starcaster50 LSTM scratch",
    "Starcaster50 LSTM FT WrightMuff1",
]

dynacomp_names = [
    "DynaComp50 CNN scratch",
    "DynaComp50 CNN FT BigMuff",
    "DynaComp50 LSTM scratch",
    "DynaComp50 LSTM FT WrightMuff1",
]

summary_names = starcaster_names + dynacomp_names

plot_validation_group(
    "Starcaster 50 - Validation loss",
    starcaster_names,
    "starcaster50_validation_curves.png",
)

plot_validation_group(
    "DynaComp 50 - Validation loss",
    dynacomp_names,
    "dynacomp50_validation_curves.png",
)

plot_metric_bar(
    "Best validation loss - all experiments",
    "best_val",
    summary_names,
    "summary_best_val.png",
    "Best validation loss",
)

plot_metric_bar(
    "Test loss (best-validation checkpoint) - all experiments",
    "test_loss_best",
    summary_names,
    "summary_test_loss.png",
    "Test loss",
)

plot_metric_bar(
    "Test ESR (best-validation checkpoint) - all experiments",
    "test_esr_best",
    summary_names,
    "summary_test_esr.png",
    "Test ESR",
)

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(
    f"{'Experiment':42} "
    f"{'FinalEp':>8} "
    f"{'BestEp':>8} "
    f"{'BestVal':>12} "
    f"{'TestLoss':>12} "
    f"{'TestESR':>12}"
)

for name in summary_names:
    d = all_data[name]

    best_val_text = (
        f"{d['best_val']:.6f}" if d["best_val"] is not None else "N/A"
    )
    test_loss_text = (
        f"{d['test_loss_best']:.6f}"
        if d["test_loss_best"] is not None
        else "N/A"
    )
    test_esr_text = (
        f"{d['test_esr_best']:.6f}"
        if d["test_esr_best"] is not None
        else "N/A"
    )

    print(
        f"{name:42} "
        f"{str(d['epoch_final']):>8} "
        f"{str(d['epoch_best']):>8} "
        f"{best_val_text:>12} "
        f"{test_loss_text:>12} "
        f"{test_esr_text:>12}"
    )

print()
print(f"Saved figures in: {OUTDIR}")
