from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags

ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# DATASETS TO CHECK
# ============================================================

PEDAL = "starcaster"

# Descomenta SOLO la combinación que quieras revisar:

# CONFIGS = ("25",)
# CONFIGS = ("50",)
# CONFIGS = ("25", "50")
CONFIGS = ("75", "100")
# CONFIGS = ("25", "50", "75", "100")

SPLITS = ("train", "val", "test")


PAIRS = []

for config in CONFIGS:
    for split in SPLITS:

        PAIRS.append(
            (
                f"{PEDAL.capitalize()}{config} {split.upper()}",

                ROOT
                / "data"
                / "processed"
                / PEDAL
                / config
                / split
                / f"{PEDAL}{config}-input.wav",

                ROOT
                / "data"
                / "processed"
                / PEDAL
                / config
                / split
                / f"{PEDAL}{config}-target.wav",
            )
        )

WINDOW_S = 4
MAX_LAG_MS = 100
POSITIONS = [0.05, 0.25, 0.50, 0.75, 0.95]


def mono_float(x):
    if x.ndim > 1:
        x = x[:, 0]

    if x.dtype == np.int16:
        return x.astype(np.float64) / 32768.0

    if x.dtype == np.int32:
        return x.astype(np.float64) / 2147483648.0

    return x.astype(np.float64)


def check(name, in_path, target_path):

    fs1, x = wavfile.read(in_path, mmap=True)
    fs2, y = wavfile.read(target_path, mmap=True)

    if fs1 != fs2:
        raise ValueError(
            f"{name}: sample rates differ: {fs1} vs {fs2}"
        )

    n = min(len(x), len(y))

    win = int(WINDOW_S * fs1)
    max_lag = int(MAX_LAG_MS * fs1 / 1000)

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)
    print(f"Sample rate: {fs1}")
    print(f"Duration: {n / fs1:.2f} s")
    print()
    print("position_s   lag_samples   lag_ms    corr")

    for frac in POSITIONS:

        start = int(frac * max(0, n - win))

        a = mono_float(
            np.asarray(x[start:start + win])
        ).copy()

        b = mono_float(
            np.asarray(y[start:start + win])
        ).copy()

        a -= np.mean(a)
        b -= np.mean(b)

        c = correlate(
            b,
            a,
            mode="full",
            method="fft"
        )

        lags = correlation_lags(
            len(b),
            len(a),
            mode="full"
        )

        mask = np.abs(lags) <= max_lag

        c2 = c[mask]
        l2 = lags[mask]

        j = int(np.argmin(c2))

        lag = int(l2[j])

        denom = (
            np.sqrt(
                np.sum(a * a)
                * np.sum(b * b)
            )
            + 1e-15
        )

        corr = float(c2[j] / denom)

        print(
            f"{start/fs1:10.2f}   "
            f"{lag:11d}   "
            f"{1000*lag/fs1:7.3f}   "
            f"{corr:+.4f}"
        )


for pair in PAIRS:
    check(*pair)