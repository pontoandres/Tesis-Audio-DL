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
WRIGHT_ROOT = ROOT / "external" / "Automated-GuitarAmpModelling"
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


def ensure_wright_imports() -> None:
    if not WRIGHT_ROOT.is_dir():
        raise FileNotFoundError(
            f"Wright submodule not found: {WRIGHT_ROOT}\n"
            "Run: git submodule update --init --recursive"
        )
    if str(WRIGHT_ROOT) not in sys.path:
        sys.path.insert(0, str(WRIGHT_ROOT))


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


def cnn_receptive_field(model: torch.nn.Module) -> int:
    blocks = len(model.blocks) - 1
    dilation_sum = sum(
        model.dilation_growth ** layer
        for layer in range(model.layers)
    )
    return 1 + blocks * (model.kernel_size - 1) * dilation_sum


def reset_runtime_state(model: torch.nn.Module) -> None:
    reset = getattr(model, "reset_hidden", None)
    if callable(reset):
        reset()


def infer_wright_stream(
    model: torch.nn.Module,
    architecture: str,
    signal: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    """
    Stream Wright/CoreAudioML models safely.

    GatedConvNet does not accept a chunk argument in process_data(), so using
    that method on a long file processes the whole signal at once. Here CNN
    inference is explicitly chunked with causal left context, while LSTM keeps
    recurrent state between consecutive blocks.
    """
    model = model.to(device)
    model.eval()
    reset_runtime_state(model)

    if architecture == "CNN":
        context = cnn_receptive_field(model) - 1
    elif architecture == "LSTM":
        context = 0
    else:
        raise ValueError(f"Unsupported Wright architecture: {architecture}")

    outputs: list[np.ndarray] = []
    total_samples = len(signal)

    # GatedConvNet creates torch.empty tensors without an explicit device.
    # This context makes those allocations follow the selected device.
    with torch.device(device), torch.inference_mode():
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            expected = end - start

            if architecture == "CNN":
                left = start - context
                if left >= 0:
                    x_np = signal[left:end]
                else:
                    x_np = np.concatenate(
                        [
                            np.zeros(-left, dtype=np.float32),
                            signal[:end],
                        ]
                    )

                x = (
                    torch.from_numpy(
                        np.ascontiguousarray(x_np, dtype=np.float32)
                    )
                    .view(-1, 1, 1)
                    .to(device)
                )
                full_output = model(x)
                output = full_output[-expected:]

            else:
                x_np = signal[start:end]
                x = (
                    torch.from_numpy(
                        np.ascontiguousarray(x_np, dtype=np.float32)
                    )
                    .view(-1, 1, 1)
                    .to(device)
                )
                output = model(x)
                detach = getattr(model, "detach_hidden", None)
                if callable(detach):
                    detach()

            actual = int(output.shape[0])
            if actual != expected:
                raise RuntimeError(
                    f"{architecture} output length mismatch in "
                    f"{start}:{end}: {actual} != {expected}"
                )

            outputs.append(
                output.detach().cpu().numpy()[:, 0, 0].astype(np.float32)
            )

            del x, output
            if architecture == "CNN":
                del full_output

    reset_runtime_state(model)
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    result = np.concatenate(outputs)
    if len(result) != total_samples:
        raise RuntimeError(
            f"{architecture} produced {len(result)} samples, "
            f"expected {total_samples}"
        )
    return result


def process_wright(
    effect: str,
    arch: str,
    model_path: Path,
    device_name: str,
    chunk: int,
) -> Path:
    ensure_wright_imports()
    import CoreAudioML.networks as networks  # type: ignore

    payload = json.loads(model_path.read_text(encoding="utf-8"))
    network = networks.load_model(payload)

    input_path = (
        PROCESSED
        / effect
        / "50"
        / "test"
        / "generalization50-input.wav"
    )
    fs, x = read_mono(input_path)
    device = torch.device(device_name)

    print("\n" + "=" * 72)
    print(f"{effect.upper()} 50% - {arch} GENERALIZATION INFERENCE")
    print("=" * 72)
    print(f"Input       : {input_path}")
    print(f"Model       : {model_path}")
    print(f"Device      : {device}")
    print(f"Chunk size  : {chunk}")
    if arch == "CNN":
        print(f"Causal ctx  : {cnn_receptive_field(network) - 1} samples")

    pred = infer_wright_stream(network, arch, x, device, chunk)

    out = RESULTS / f"{effect}50_{arch.lower()}_prediction.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(out, fs, pred.astype(np.float32))
    print(f"Saved       : {out}")
    return out


def process_tcn(
    effect: str,
    model_path: Path,
    device_name: str,
    chunk: int,
) -> Path:
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

    input_path = (
        PROCESSED
        / effect
        / "50"
        / "test"
        / "generalization50-input.wav"
    )
    fs, x = read_mono(input_path)
    pred = infer_full(model, x, device, chunk).numpy()

    out = RESULTS / f"{effect}50_tcn_prediction.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(out, fs, pred.astype(np.float32))
    print(f"Saved TCN prediction: {out}")

    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def common_evaluate(effect: str, arch: str, prediction: Path) -> Path:
    target = (
        PROCESSED
        / effect
        / "50"
        / "test"
        / "generalization50-target.wav"
    )
    out = RESULTS / "evaluation" / f"{effect}50_{arch.lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = "model_best.pt" if arch == "TCN" else "model_best.json"
    run_cmd([
        sys.executable,
        str(COMMON_EVAL),
        "--target",
        str(target),
        "--prediction",
        str(prediction),
        "--output",
        str(out),
        "--effect",
        effect,
        "--setting",
        "50",
        "--architecture",
        arch,
        "--run-name",
        f"generalization_{effect}50_{arch.lower()}",
        "--checkpoint",
        checkpoint,
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
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = RESULTS / "generalization_summary.md"
    lines = [
        "# Generalization evaluation summary",
        "",
        "Lower is better for ESR, MAE, MR-STFT and LUFS error.",
        "",
        "| Effect | Architecture | ESR | MAE | MR-STFT | LUFS error (dB) |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for row in rows:
        lufs_error = (
            "N/A"
            if row["lufs_error"] is None
            else f'{row["lufs_error"]:.6f}'
        )
        lines.append(
            f'| {row["effect"]} | {row["architecture"]} | '
            f'{row["esr"]:.6f} | {row["mae"]:.6f} | '
            f'{row["mrstft"]:.6f} | {lufs_error} |'
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 72)
    print("GENERALIZATION SUMMARY")
    print("=" * 72)

    for row in rows:
        print(
            f"{row['effect']:10s} {row['architecture']:4s}  "
            f"ESR={row['esr']:.6f}  "
            f"MAE={row['mae']:.6f}  "
            f"MR-STFT={row['mrstft']:.6f}  "
            f"LUFSerr={row['lufs_error']}"
        )

    print(f"\nCSV: {csv_path}")
    print(f"MD : {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run all 50% generalization inference and common thesis metrics."
        )
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument("--chunk", type=int, default=20000)
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = "cuda"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    required_audio = [
        PROCESSED / effect / "50" / "test" / name
        for effect in ("starcaster", "dynacomp")
        for name in (
            "generalization50-input.wav",
            "generalization50-target.wav",
        )
    ]
    missing_audio = [path for path in required_audio if not path.exists()]
    if missing_audio:
        raise FileNotFoundError(
            "Run scripts\\prepare_generalization.py first. Missing:\n"
            + "\n".join(str(path) for path in missing_audio)
        )

    model_files = [run[2] for run in WRIGHT_RUNS] + [
        run[2] for run in TCN_RUNS
    ]
    missing_models = [path for path in model_files if not path.exists()]
    if missing_models:
        print("\nMissing model checkpoints:")
        for path in missing_models:
            print(f"  {path}")
        raise FileNotFoundError("Required model checkpoint(s) missing.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    predictions: dict[tuple[str, str], Path] = {}

    for effect, arch, model_path in WRIGHT_RUNS:
        predictions[(effect, arch)] = process_wright(
            effect,
            arch,
            model_path,
            device,
            args.chunk,
        )

    for effect, arch, model_path in TCN_RUNS:
        predictions[(effect, arch)] = process_tcn(
            effect,
            model_path,
            device,
            args.chunk,
        )

    evaluations = []
    for effect in ("starcaster", "dynacomp"):
        for arch in ("CNN", "LSTM", "TCN"):
            evaluations.append(
                common_evaluate(
                    effect,
                    arch,
                    predictions[(effect, arch)],
                )
            )

    build_summary(evaluations)
    print("\nDONE. No model was trained or modified.")


if __name__ == "__main__":
    main()
