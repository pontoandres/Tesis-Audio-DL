from pathlib import Path

import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# DATASETS TO CHECK
# ============================================================

PEDAL = "starcaster"

# Descomenta SOLO la combinación que quieras revisar:
# SETTINGS = ("25",)
# SETTINGS = ("50",)
# SETTINGS = ("25", "50")
SETTINGS = ("75", "100")
# SETTINGS = ("25", "50", "75", "100")

SPLITS = ("train", "val", "test")


def to_float(audio):
    if audio.dtype == np.int16:
        return audio.astype(np.float64) / 32768.0

    if audio.dtype == np.int32:
        return audio.astype(np.float64) / 2147483648.0

    if np.issubdtype(audio.dtype, np.floating):
        return audio.astype(np.float64)

    raise TypeError(f"Unsupported dtype: {audio.dtype}")


def rms(x):
    return np.sqrt(np.mean(x ** 2))


def db(x):
    return 20.0 * np.log10(max(x, 1e-12))


def analyse(setting, split):

    data = (
        ROOT
        / "data"
        / "processed"
        / PEDAL
        / setting
    )

    input_path = (
        data
        / split
        / f"{PEDAL}{setting}-input.wav"
    )

    target_path = (
        data
        / split
        / f"{PEDAL}{setting}-target.wav"
    )

    fs_x, x = wavfile.read(input_path)
    fs_y, y = wavfile.read(target_path)

    if fs_x != fs_y:
        raise ValueError(
            f"{PEDAL} {setting}% {split}: sample-rate mismatch"
        )

    n = min(len(x), len(y))

    x = to_float(x[:n])
    y = to_float(y[:n])

    rms_x = rms(x)
    rms_y = rms(y)

    peak_x = np.max(np.abs(x))
    peak_y = np.max(np.abs(y))

    corr = np.corrcoef(x, y)[0, 1]

    # Mejor aproximación lineal:
    # target ≈ a * input
    scalar = np.dot(x, y) / np.dot(x, x)

    y_linear = scalar * x

    error = y - y_linear

    esr_linear = (
        np.sum(error ** 2)
        / np.sum(y ** 2)
    )

    print()
    print("=" * 60)
    print(f"{PEDAL.capitalize()} {setting}% - {split.upper()}")
    print("=" * 60)

    print(f"Duration: {n / fs_x:.3f} s")
    print(f"Samples:  {n}")
    print()

    print("INPUT")
    print(f"  RMS:       {rms_x:.6f}")
    print(f"  RMS dBFS:  {db(rms_x):.2f} dB")
    print(f"  Peak dBFS: {db(peak_x):.2f} dB")

    print()
    print("TARGET")
    print(f"  RMS:       {rms_y:.6f}")
    print(f"  RMS dBFS:  {db(rms_y):.2f} dB")
    print(f"  Peak dBFS: {db(peak_y):.2f} dB")

    print()
    print("RELATION")
    print(
        f"  Target/Input RMS difference: "
        f"{db(rms_y) - db(rms_x):+.2f} dB"
    )

    print(
        f"  Correlation: {corr:+.4f}"
    )

    print(
        f"  Optimal scalar: {scalar:+.4f}"
    )

    print(
        f"  Linear baseline ESR: "
        f"{esr_linear:.4f}"
    )


def main():

    print(
        f"{PEDAL.capitalize()} settings {SETTINGS} - "
        "Train/Validation/Test comparison"
    )

    for setting in SETTINGS:
        for split in SPLITS:
            analyse(setting, split)


if __name__ == "__main__":
    main()
