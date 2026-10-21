"""Common output-level evaluator for thesis audio-model experiments.

The goal of this script is to evaluate *all* architectures with the same
post-hoc metrics, independently of the loss used to train each model.

It compares one target WAV against one predicted WAV and writes a JSON file
with reproducible metric/configuration metadata.

Core metrics:
- ESR (time-domain Error-to-Signal Ratio)
- MAE
- MSE / RMSE (secondary, included because they are cheap and useful)
- Multi-resolution STFT error
- Integrated loudness error (LUFS, optional dependency: pyloudnorm)

Important:
- No amplitude normalization is performed.
- No resampling is performed.
- Stereo is rejected instead of being silently downmixed.
- Length mismatch is rejected by default.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None


EPS = 1e-8
ESR_EPS = 1e-5


def pcm_to_float(audio: np.ndarray) -> np.ndarray:
    """Convert common WAV PCM representations to floating point without normalization."""
    if audio.dtype == np.int16:
        return audio.astype(np.float64) / 32768.0
    if audio.dtype == np.int32:
        return audio.astype(np.float64) / 2147483648.0
    if audio.dtype == np.uint8:
        return (audio.astype(np.float64) - 128.0) / 128.0
    if np.issubdtype(audio.dtype, np.floating):
        return audio.astype(np.float64, copy=False)

    raise TypeError(f"Unsupported WAV dtype: {audio.dtype}")


def load_mono_wav(path: Path) -> tuple[int, np.ndarray, str]:
    if not path.exists():
        raise FileNotFoundError(path)

    sample_rate, raw = wavfile.read(path)
    dtype_name = str(raw.dtype)

    if raw.ndim == 2:
        if raw.shape[1] == 1:
            raw = raw[:, 0]
        else:
            raise ValueError(
                f"{path} has {raw.shape[1]} channels. "
                "This evaluator does not silently downmix stereo files."
            )
    elif raw.ndim != 1:
        raise ValueError(f"Unexpected WAV shape for {path}: {raw.shape}")

    audio = pcm_to_float(raw)

    if not np.all(np.isfinite(audio)):
        raise ValueError(f"{path} contains NaN or infinite samples.")

    return int(sample_rate), audio, dtype_name


def match_lengths(
    target: np.ndarray,
    prediction: np.ndarray,
    policy: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    if len(target) == len(prediction):
        return target, prediction, len(target)

    if policy == "strict":
        raise ValueError(
            "Target/prediction length mismatch: "
            f"{len(target)} vs {len(prediction)} samples. "
            "Use --length-policy trim only if the mismatch is expected and documented."
        )

    n = min(len(target), len(prediction))
    print(
        f"WARNING: trimming both signals to {n} samples "
        f"(target={len(target)}, prediction={len(prediction)}).",
        file=sys.stderr,
    )
    return target[:n], prediction[:n], n


def compute_time_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    error = target - prediction

    mse = float(np.mean(np.square(error)))
    mae = float(np.mean(np.abs(error)))
    rmse = float(math.sqrt(mse))
    target_energy = float(np.mean(np.square(target)))
    esr = float(mse / (target_energy + ESR_EPS))

    return {
        "esr": esr,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "target_mean_square": target_energy,
    }


def stft_magnitude(
    audio: torch.Tensor,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    window = torch.hann_window(
        n_fft,
        dtype=audio.dtype,
        device=audio.device,
    )
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    return spec.abs().clamp_min(EPS)


def compute_mrstft(
    target: np.ndarray,
    prediction: np.ndarray,
    fft_sizes: list[int],
    hop_sizes: list[int],
) -> dict:
    if len(fft_sizes) != len(hop_sizes):
        raise ValueError("--fft-sizes and --hop-sizes must contain the same number of values.")

    target_t = torch.from_numpy(target).to(torch.float32)
    pred_t = torch.from_numpy(prediction).to(torch.float32)

    per_resolution = []
    sc_values = []
    logmag_values = []
    total_values = []

    with torch.inference_mode():
        for n_fft, hop in zip(fft_sizes, hop_sizes):
            if n_fft <= 0 or hop <= 0:
                raise ValueError("FFT and hop sizes must be positive.")

            target_mag = stft_magnitude(target_t, n_fft, hop)
            pred_mag = stft_magnitude(pred_t, n_fft, hop)

            # Spectral convergence:
            # || |Y| - |Yhat| ||_F / (|| |Y| ||_F + eps)
            sc = torch.linalg.vector_norm(target_mag - pred_mag) / (
                torch.linalg.vector_norm(target_mag) + EPS
            )

            # Mean absolute log-magnitude difference.
            logmag = torch.mean(
                torch.abs(torch.log(target_mag) - torch.log(pred_mag))
            )

            # Equal weighting of the two terms, followed by equal averaging
            # across STFT resolutions.
            total = sc + logmag

            sc_f = float(sc.cpu())
            logmag_f = float(logmag.cpu())
            total_f = float(total.cpu())

            sc_values.append(sc_f)
            logmag_values.append(logmag_f)
            total_values.append(total_f)

            per_resolution.append(
                {
                    "n_fft": int(n_fft),
                    "hop_length": int(hop),
                    "spectral_convergence": sc_f,
                    "log_magnitude_l1": logmag_f,
                    "combined": total_f,
                }
            )

    return {
        "mrstft": float(np.mean(total_values)),
        "mrstft_spectral_convergence": float(np.mean(sc_values)),
        "mrstft_log_magnitude_l1": float(np.mean(logmag_values)),
        "mrstft_resolutions": per_resolution,
    }


def safe_float_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def compute_loudness_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    sample_rate: int,
) -> dict:
    if pyln is None:
        return {
            "available": False,
            "reason": (
                "pyloudnorm is not installed. "
                "Install with: python -m pip install pyloudnorm"
            ),
            "target_lufs": None,
            "prediction_lufs": None,
            "lufs_absolute_error": None,
        }

    meter = pyln.Meter(sample_rate)
    target_lufs_raw = meter.integrated_loudness(target.astype(np.float64))
    pred_lufs_raw = meter.integrated_loudness(prediction.astype(np.float64))

    target_lufs = safe_float_or_none(target_lufs_raw)
    pred_lufs = safe_float_or_none(pred_lufs_raw)

    if target_lufs is None or pred_lufs is None:
        error = None
    else:
        error = abs(target_lufs - pred_lufs)

    return {
        "available": True,
        "standard": "ITU-R BS.1770 (via pyloudnorm)",
        "target_lufs": target_lufs,
        "prediction_lufs": pred_lufs,
        "lufs_absolute_error": error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate target/prediction WAV pairs with a common thesis metric protocol."
    )
    parser.add_argument("--target", required=True, type=Path, help="Ground-truth pedal output WAV.")
    parser.add_argument("--prediction", required=True, type=Path, help="Model-generated WAV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON. Default: evaluation_metrics.json next to prediction.",
    )
    parser.add_argument(
        "--length-policy",
        choices=("strict", "trim"),
        default="strict",
        help="How to handle unequal sample counts. Default: strict.",
    )
    parser.add_argument(
        "--fft-sizes",
        nargs="+",
        type=int,
        default=[512, 2048, 8192],
        help="FFT sizes for MR-STFT. Default: 512 2048 8192.",
    )
    parser.add_argument(
        "--hop-sizes",
        nargs="+",
        type=int,
        default=[128, 512, 2048],
        help="Hop sizes for MR-STFT. Default: 128 512 2048.",
    )

    # Optional experiment metadata for the JSON registry.
    parser.add_argument("--effect", default=None)
    parser.add_argument("--setting", type=int, default=None)
    parser.add_argument("--architecture", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target_sr, target, target_dtype = load_mono_wav(args.target)
    pred_sr, prediction, pred_dtype = load_mono_wav(args.prediction)

    if target_sr != pred_sr:
        raise ValueError(
            f"Sample-rate mismatch: target={target_sr} Hz, prediction={pred_sr} Hz. "
            "This evaluator does not resample."
        )

    original_target_samples = len(target)
    original_prediction_samples = len(prediction)

    target, prediction, evaluated_samples = match_lengths(
        target,
        prediction,
        args.length_policy,
    )

    time_metrics = compute_time_metrics(target, prediction)
    stft_metrics = compute_mrstft(
        target,
        prediction,
        args.fft_sizes,
        args.hop_sizes,
    )
    loudness_metrics = compute_loudness_metrics(
        target,
        prediction,
        target_sr,
    )

    output_path = args.output
    if output_path is None:
        output_path = args.prediction.parent / "evaluation_metrics.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "effect": args.effect,
            "setting": args.setting,
            "architecture": args.architecture,
            "run_name": args.run_name,
            "checkpoint": args.checkpoint,
        },
        "files": {
            "target": str(args.target),
            "prediction": str(args.prediction),
        },
        "audio": {
            "sample_rate_hz": target_sr,
            "evaluated_samples": evaluated_samples,
            "audio_seconds": evaluated_samples / target_sr,
            "target_original_samples": original_target_samples,
            "prediction_original_samples": original_prediction_samples,
            "target_dtype": target_dtype,
            "prediction_dtype": pred_dtype,
            "length_policy": args.length_policy,
            "amplitude_normalization": False,
            "resampling": False,
        },
        "metrics": {
            **time_metrics,
            **stft_metrics,
            "loudness": loudness_metrics,
        },
        "metric_configuration": {
            "esr_epsilon": ESR_EPS,
            "mrstft_fft_sizes": args.fft_sizes,
            "mrstft_hop_sizes": args.hop_sizes,
            "mrstft_window": "Hann",
            "mrstft_center": True,
            "mrstft_combination": (
                "mean_over_resolutions(spectral_convergence + log_magnitude_L1)"
            ),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("COMMON AUDIO EVALUATION")
    print("=" * 72)
    print(f"Target:      {args.target}")
    print(f"Prediction:  {args.prediction}")
    print(f"Sample rate: {target_sr} Hz")
    print(f"Duration:    {evaluated_samples / target_sr:.3f} s")
    print()
    print(f"ESR:         {time_metrics['esr']:.9f}")
    print(f"MAE:         {time_metrics['mae']:.9f}")
    print(f"MSE:         {time_metrics['mse']:.9f}")
    print(f"RMSE:        {time_metrics['rmse']:.9f}")
    print(f"MR-STFT:     {stft_metrics['mrstft']:.9f}")
    print(
        f"  SC mean:   {stft_metrics['mrstft_spectral_convergence']:.9f}"
    )
    print(
        f"  LogMag:    {stft_metrics['mrstft_log_magnitude_l1']:.9f}"
    )

    if loudness_metrics["available"]:
        print(f"Target LUFS: {loudness_metrics['target_lufs']}")
        print(f"Pred LUFS:   {loudness_metrics['prediction_lufs']}")
        print(f"LUFS error:  {loudness_metrics['lufs_absolute_error']}")
    else:
        print("LUFS:         skipped (pyloudnorm not installed)")

    print()
    print(f"Saved JSON:  {output_path}")


if __name__ == "__main__":
    main()
