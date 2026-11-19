#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "generalization" / "raw"
DEFAULT_OUT = ROOT / "data" / "generalization" / "processed"
TARGET_RESIDUAL = 128
SAMPLE_RATE = 44100


def to_float_mono(x: np.ndarray, name: str) -> np.ndarray:
    if x.ndim == 2:
        if x.shape[1] == 1:
            x = x[:, 0]
        else:
            print(f"[INFO] {name}: {x.shape[1]} channels -> mono by channel mean")
            x = np.mean(x.astype(np.float64), axis=1)
    if x.dtype == np.int16:
        return np.ascontiguousarray(x.astype(np.float64) / 32768.0)
    if x.dtype == np.int32:
        return np.ascontiguousarray(x.astype(np.float64) / 2147483648.0)
    if np.issubdtype(x.dtype, np.floating):
        return np.ascontiguousarray(x.astype(np.float64))
    raise TypeError(f"Unsupported dtype for {name}: {x.dtype}")


def to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)


def load(path: Path) -> tuple[int, np.ndarray]:
    fs, x = wavfile.read(path)
    return int(fs), to_float_mono(x, str(path))


def coarse_lag(x: np.ndarray, y: np.ndarray, fs: int, max_lag_s: float, decim: int) -> int:
    xd = x[::decim].copy()
    yd = y[::decim].copy()
    xd -= np.mean(xd)
    yd -= np.mean(yd)
    sx = np.std(xd)
    sy = np.std(yd)
    if sx < 1e-12 or sy < 1e-12:
        raise ValueError("One signal is effectively silent; cannot estimate lag.")
    xd /= sx
    yd /= sy
    c = correlate(yd, xd, mode="full", method="fft")
    lags = correlation_lags(len(yd), len(xd), mode="full")
    max_lag = int(round(max_lag_s * fs / decim))
    mask = np.abs(lags) <= max_lag
    j = int(np.argmax(np.abs(c[mask])))
    return int(lags[mask][j]) * decim


def crop_to_zero_lag(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        y = y[lag:]
    elif lag < 0:
        x = x[-lag:]
    n = min(len(x), len(y))
    return x[:n], y[:n]


def find_active_window(x: np.ndarray, fs: int, seconds: float = 12.0) -> tuple[int, int]:
    win = min(len(x), int(seconds * fs))
    if len(x) <= win:
        return 0, len(x)
    block = fs
    nblocks = len(x) // block
    energies = np.array([np.mean(x[i*block:(i+1)*block] ** 2) for i in range(nblocks)])
    wb = max(1, int(round(seconds)))
    if len(energies) < wb:
        return 0, win
    score = np.convolve(energies, np.ones(wb), mode="valid")
    start = int(np.argmax(score)) * block
    return start, min(start + win, len(x))


def refine_lag(x: np.ndarray, y: np.ndarray, fs: int, radius: int = 1024) -> int:
    start, end = find_active_window(x, fs)
    a = x[start:end].copy()
    b = y[start:end].copy()
    a -= np.mean(a)
    b -= np.mean(b)
    c = correlate(b, a, mode="full", method="fft")
    lags = correlation_lags(len(b), len(a), mode="full")
    mask = np.abs(lags) <= radius
    j = int(np.argmax(np.abs(c[mask])))
    return int(lags[mask][j])


def align_to_residual(x: np.ndarray, y: np.ndarray, raw_lag: int, residual: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = crop_to_zero_lag(x, y, raw_lag)
    if residual > 0:
        x = x[residual:]
    n = min(len(x), len(y))
    return x[:n], y[:n]


def residual_lags(x: np.ndarray, y: np.ndarray, fs: int, expected: int = 128, radius: int = 64) -> list[dict]:
    positions = [0.05, 0.25, 0.50, 0.75, 0.95]
    win = min(int(8 * fs), len(x))
    results = []
    for frac in positions:
        start = int(frac * max(0, len(x) - win))
        a = x[start:start+win].copy()
        b = y[start:start+win].copy()
        a -= np.mean(a)
        b -= np.mean(b)
        c = correlate(b, a, mode="full", method="fft")
        lags = correlation_lags(len(b), len(a), mode="full")
        mask = (lags >= expected-radius) & (lags <= expected+radius)
        cc = c[mask]
        ll = lags[mask]
        j = int(np.argmax(np.abs(cc)))
        lag = int(ll[j])
        denom = np.sqrt(np.sum(a*a) * np.sum(b*b)) + 1e-15
        results.append({"position_s": start/fs, "lag_samples": lag, "corr": float(cc[j]/denom)})
    return results


def process(effect: str, clean_path: Path, target_path: Path, out_root: Path, max_lag_s: float, decim: int) -> dict:
    fsx, x0 = load(clean_path)
    fsy, y0 = load(target_path)
    if fsx != fsy:
        raise ValueError(f"{effect}: sample rates differ: {fsx} vs {fsy}")
    if fsx != SAMPLE_RATE:
        raise ValueError(f"{effect}: expected {SAMPLE_RATE} Hz, got {fsx} Hz")

    lag0 = coarse_lag(x0, y0, fsx, max_lag_s=max_lag_s, decim=decim)
    xa, ya = crop_to_zero_lag(x0, y0, lag0)
    residual0 = refine_lag(xa, ya, fsx)
    total_lag = lag0 + residual0

    x, y = align_to_residual(x0, y0, total_lag, TARGET_RESIDUAL)
    checks = residual_lags(x, y, fsx)
    median = int(np.median([r["lag_samples"] for r in checks]))

    out_dir = out_root / effect / "50" / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "generalization50"
    input_out = out_dir / f"{prefix}-input.wav"
    target_out = out_dir / f"{prefix}-target.wav"
    wavfile.write(input_out, fsx, to_int16(x))
    wavfile.write(target_out, fsx, to_int16(y))

    print("\n" + "="*72)
    print(f"{effect.upper()} 50% GENERALIZATION")
    print("="*72)
    print(f"Raw coarse lag      : {lag0:+d} samples")
    print(f"Full-rate correction: {residual0:+d} samples")
    print(f"Total raw lag       : {total_lag:+d} samples")
    print(f"Desired residual    : {TARGET_RESIDUAL} samples")
    print(f"Measured median     : {median} samples")
    print(f"Final duration      : {len(x)/fsx:.3f} s")
    print(f"Saved input         : {input_out}")
    print(f"Saved target        : {target_out}")
    print("Residual checks:")
    for r in checks:
        print(f"  {r['position_s']:8.2f}s  lag={r['lag_samples']:4d}  corr={r['corr']:+.4f}")

    return {
        "effect": effect,
        "clean_raw": str(clean_path),
        "target_raw": str(target_path),
        "sample_rate": fsx,
        "coarse_lag_samples": lag0,
        "refine_lag_samples": residual0,
        "raw_lag_samples": total_lag,
        "target_residual_samples": TARGET_RESIDUAL,
        "measured_residual_median_samples": median,
        "residual_checks": checks,
        "processed_input": str(input_out),
        "processed_target": str(target_out),
        "samples": len(x),
        "seconds": len(x)/fsx,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare unseen generalization audio using the thesis alignment convention.")
    p.add_argument("--clean", type=Path, default=DEFAULT_RAW / "generalization_clean.wav")
    p.add_argument("--starcaster", type=Path, default=DEFAULT_RAW / "starcaster50_target_raw.wav")
    p.add_argument("--dynacomp", type=Path, default=DEFAULT_RAW / "dynacomp50_target_raw.wav")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-lag-seconds", type=float, default=10.0)
    p.add_argument("--coarse-decimation", type=int, default=16)
    args = p.parse_args()

    for path in (args.clean, args.starcaster, args.dynacomp):
        if not path.exists():
            raise FileNotFoundError(f"Missing WAV: {path}")

    reports = [
        process("starcaster", args.clean, args.starcaster, args.output_root, args.max_lag_seconds, args.coarse_decimation),
        process("dynacomp", args.clean, args.dynacomp, args.output_root, args.max_lag_seconds, args.coarse_decimation),
    ]
    report_path = args.output_root.parent / "alignment_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"schema_version": 1, "runs": reports}, indent=2), encoding="utf-8")
    print(f"\nAlignment report: {report_path}")
    print("DONE")


if __name__ == "__main__":
    main()
