import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags


def to_float(audio):
    """Convierte audio PCM o float a float64 aproximadamente entre -1 y 1."""
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(int(info.min)), abs(int(info.max))))
        return audio.astype(np.float64) / scale

    return audio.astype(np.float64)


def to_mono(audio):
    """Convierte stereo/multicanal a mono para hacer la correlación."""
    if audio.ndim == 1:
        return audio

    return np.mean(audio, axis=1)


def find_active_window(audio, sample_rate, window_seconds=10):
    """
    Busca una ventana con bastante energía para evitar correlacionar
    silencios o partes demasiado débiles.
    """
    window_samples = int(window_seconds * sample_rate)

    if len(audio) <= window_samples:
        return 0, len(audio)

    block_size = sample_rate  # bloques de 1 segundo
    num_blocks = len(audio) // block_size

    energies = []

    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        block = audio[start:end]

        energies.append(np.mean(block ** 2))

    energies = np.array(energies)

    window_blocks = max(1, int(round(window_seconds)))

    if len(energies) < window_blocks:
        return 0, min(window_samples, len(audio))

    rolling_energy = np.convolve(
        energies,
        np.ones(window_blocks),
        mode="valid"
    )

    best_block = int(np.argmax(rolling_energy))

    start = best_block * block_size
    end = min(start + window_samples, len(audio))

    return start, end


def estimate_delay(input_audio, target_audio, sample_rate, max_lag_ms=100):
    """
    Devuelve:
        lag_samples
        lag_ms
        normalized_correlation

    Convención:
        lag > 0  -> TARGET está retrasado respecto a INPUT
        lag < 0  -> TARGET está adelantado respecto a INPUT
    """

    # Quitar DC offset
    x = input_audio - np.mean(input_audio)
    y = target_audio - np.mean(target_audio)

    # Evitar problemas si alguna señal está vacía/silenciosa
    x_std = np.std(x)
    y_std = np.std(y)

    if x_std < 1e-12 or y_std < 1e-12:
        raise ValueError("Una de las señales parece estar en silencio.")

    # Normalización
    x = x / x_std
    y = y / y_std

    # correlate(target, input)
    correlation = correlate(
        y,
        x,
        mode="full",
        method="fft"
    )

    lags = correlation_lags(
        len(y),
        len(x),
        mode="full"
    )

    max_lag_samples = int(
        sample_rate * max_lag_ms / 1000.0
    )

    valid = np.abs(lags) <= max_lag_samples

    correlation_valid = correlation[valid]
    lags_valid = lags[valid]

    # abs() permite detectar también una posible inversión de polaridad
    best_index = np.argmax(np.abs(correlation_valid))

    lag_samples = int(lags_valid[best_index])
    peak = correlation_valid[best_index]

    normalization = np.sqrt(
        np.sum(x ** 2) * np.sum(y ** 2)
    )

    normalized_corr = peak / normalization

    lag_ms = (
        lag_samples / sample_rate
    ) * 1000.0

    return lag_samples, lag_ms, normalized_corr


def main():
    parser = argparse.ArgumentParser(
        description="Estimate time offset between clean input and pedal target WAV."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Clean/input WAV"
    )

    parser.add_argument(
        "target",
        type=Path,
        help="Processed/target WAV"
    )

    parser.add_argument(
        "--window",
        type=float,
        default=10.0,
        help="Seconds used for correlation (default: 10)"
    )

    parser.add_argument(
        "--max-lag-ms",
        type=float,
        default=100.0,
        help="Maximum latency to search in milliseconds (default: 100)"
    )

    args = parser.parse_args()

    # -----------------------------
    # Load files
    # -----------------------------

    fs_input, input_raw = wavfile.read(args.input)
    fs_target, target_raw = wavfile.read(args.target)

    print("\n=== FILE INFO ===")
    print(f"Input:  {args.input}")
    print(f"Target: {args.target}")
    print(f"Input sample rate:  {fs_input} Hz")
    print(f"Target sample rate: {fs_target} Hz")

    if fs_input != fs_target:
        raise ValueError(
            "ERROR: Input and target have different sample rates."
        )

    sample_rate = fs_input

    input_audio = to_mono(to_float(input_raw))
    target_audio = to_mono(to_float(target_raw))

    print(f"Input samples:  {len(input_audio)}")
    print(f"Target samples: {len(target_audio)}")

    duration_input = len(input_audio) / sample_rate
    duration_target = len(target_audio) / sample_rate

    print(f"Input duration:  {duration_input:.3f} s")
    print(f"Target duration: {duration_target:.3f} s")

    # Trabajar con longitud común
    common_length = min(
        len(input_audio),
        len(target_audio)
    )

    input_audio = input_audio[:common_length]
    target_audio = target_audio[:common_length]

    # -----------------------------
    # Find useful analysis section
    # -----------------------------

    start, end = find_active_window(
        input_audio,
        sample_rate,
        args.window
    )

    input_window = input_audio[start:end]
    target_window = target_audio[start:end]

    print("\n=== ANALYSIS WINDOW ===")
    print(
        f"{start / sample_rate:.3f} s "
        f"to {end / sample_rate:.3f} s"
    )

    # -----------------------------
    # Estimate latency
    # -----------------------------

    lag_samples, lag_ms, corr = estimate_delay(
        input_window,
        target_window,
        sample_rate,
        args.max_lag_ms
    )

    print("\n==============================")
    print("      ALIGNMENT RESULT")
    print("==============================")

    print(f"Lag: {lag_samples:+d} samples")
    print(f"Lag: {lag_ms:+.4f} ms")
    print(f"Correlation: {corr:+.4f}")

    print()

    if lag_samples > 0:
        print(
            f"TARGET is approximately {lag_samples} samples "
            f"({lag_ms:.4f} ms) LATE relative to INPUT."
        )

        print(
            f"To align them, approximately the first "
            f"{lag_samples} samples of TARGET would need to be removed."
        )

    elif lag_samples < 0:
        delay = abs(lag_samples)

        print(
            f"TARGET is approximately {delay} samples "
            f"({abs(lag_ms):.4f} ms) EARLY relative to INPUT."
        )

        print(
            f"To align them, approximately the first "
            f"{delay} samples of INPUT would need to be removed."
        )

    else:
        print("No measurable sample offset was detected.")

    if corr < 0:
        print(
            "\nNOTE: The negative correlation suggests a possible "
            "polarity inversion between input and target."
        )

    print("==============================\n")


if __name__ == "__main__":
    main()