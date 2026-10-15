from pathlib import Path

import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]

# Reutilizamos exactamente los mismos dry del Starcaster
INPUT_SOURCE = ROOT / "data" / "starcaster"

# Targets capturados con el MXR Dyna Comp
TARGET_SOURCE = ROOT / "data" / "mxrdynacomp"

OUTPUT = ROOT / "data" / "processed" / "mxrdynacomp"


# Coarse lags medidos con check_alignment.py
PAIRS = [
    ("25", "train", -73114),
    ("50", "train", -73114),
    ("75", "train", -73114),
    ("100", "train", -73114),

    ("25", "val", 27),
    ("50", "val", 27),
    ("75", "val", 27),
    ("100", "val", 27),

    ("25", "test", 149),
    ("50", "test", 148),
    ("75", "test", 148),
    ("100", "test", 148),
]


# PROVISIONAL:
# empezamos con 128 para todos.
# check_drift_dynacomp.py nos dirá cuánto corregir cada uno.
CAUSAL_DELAY = {
    ("25", "train"): 158,
    ("25", "val"): 117,
    ("25", "test"): 141,

    ("50", "train"): 158,
    ("50", "val"): 117,
    ("50", "test"): 141,

    ("75", "train"): 158,
    ("75", "val"): 117,
    ("75", "test"): 141,

    ("100", "train"): 158,
    ("100", "val"): 117,
    ("100", "test"): 141,
}


def ensure_mono(audio, name):
    if audio.ndim == 1:
        return audio

    if audio.ndim == 2 and audio.shape[1] == 1:
        return audio[:, 0]

    raise ValueError(
        f"{name} is not mono. Shape: {audio.shape}"
    )


def to_int16(audio):
    """
    Convierte a PCM16 sin normalizar independientemente.
    """

    if audio.dtype == np.int16:
        return audio

    if audio.dtype == np.int32:
        audio_float = (
            audio.astype(np.float64)
            / 2147483648.0
        )

    elif np.issubdtype(audio.dtype, np.floating):
        audio_float = audio.astype(np.float64)

    else:
        raise TypeError(
            f"Unsupported dtype: {audio.dtype}"
        )

    audio_float = np.clip(
        audio_float,
        -1.0,
        1.0
    )

    audio_int16 = np.round(
        audio_float * 32767.0
    )

    audio_int16 = np.clip(
        audio_int16,
        -32768,
        32767
    )

    return audio_int16.astype(np.int16)


def align_pair(input_audio, target_audio, lag):
    """
    Convención de check_alignment.py:

    lag > 0:
        target está atrasado
        -> cortar comienzo del target

    lag < 0:
        target está adelantado
        -> cortar comienzo del input
    """

    if lag > 0:
        target_audio = target_audio[lag:]

    elif lag < 0:
        input_audio = input_audio[abs(lag):]

    length = min(
        len(input_audio),
        len(target_audio)
    )

    return (
        input_audio[:length],
        target_audio[:length]
    )


def add_causal_delay(
    input_audio,
    target_audio,
    delay
):
    """
    Dejamos el TARGET aproximadamente 'delay'
    muestras después del INPUT.

    Para ello adelantamos el INPUT cortando
    'delay' muestras de su comienzo.
    """

    if delay > 0:
        input_audio = input_audio[delay:]

    length = min(
        len(input_audio),
        len(target_audio)
    )

    return (
        input_audio[:length],
        target_audio[:length]
    )


def process(config, split, lag):

    input_path = (
        INPUT_SOURCE
        / split
        / f"starcaster_input_{split}.wav"
    )

    target_path = (
        TARGET_SOURCE
        / split
        / f"MXRDynaComp_{config}_target_44_{split}.wav"
    )

    fs_input, input_audio = wavfile.read(
        input_path
    )

    fs_target, target_audio = wavfile.read(
        target_path
    )

    input_audio = ensure_mono(
        input_audio,
        str(input_path)
    )

    target_audio = ensure_mono(
        target_audio,
        str(target_path)
    )

    if fs_input != fs_target:
        raise ValueError(
            f"Sample rate mismatch in "
            f"{config}% {split}: "
            f"{fs_input} vs {fs_target}"
        )

    print()
    print("=" * 65)
    print(
        f"MXR Dyna Comp {config}% - {split}"
    )
    print("=" * 65)

    print(
        f"Input original dtype:  "
        f"{input_audio.dtype}"
    )

    print(
        f"Target original dtype: "
        f"{target_audio.dtype}"
    )

    print(
        f"Input original samples:  "
        f"{len(input_audio)}"
    )

    print(
        f"Target original samples: "
        f"{len(target_audio)}"
    )

    print(
        f"Coarse lag: "
        f"{lag:+d} samples"
    )

    # 1. Coarse alignment
    input_audio, target_audio = align_pair(
        input_audio,
        target_audio,
        lag
    )

    # 2. Causal margin provisional/final
    delay = CAUSAL_DELAY[
        (config, split)
    ]

    input_audio, target_audio = add_causal_delay(
        input_audio,
        target_audio,
        delay
    )

    print(
        f"Causal target delay: "
        f"{delay} samples "
        f"({1000.0 * delay / fs_input:.3f} ms)"
    )

    # 3. Homogeneizar PCM16
    input_audio = to_int16(
        input_audio
    )

    target_audio = to_int16(
        target_audio
    )

    # 4. Estructura compatible con Wright
    output_dir = (
        OUTPUT
        / config
        / split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    input_output = (
        output_dir
        / f"mxrdynacomp{config}-input.wav"
    )

    target_output = (
        output_dir
        / f"mxrdynacomp{config}-target.wav"
    )

    wavfile.write(
        input_output,
        fs_input,
        input_audio
    )

    wavfile.write(
        target_output,
        fs_target,
        target_audio
    )

    print(
        f"Final samples: "
        f"{len(input_audio)}"
    )

    print(
        f"Duration: "
        f"{len(input_audio) / fs_input:.3f} s"
    )

    print(
        f"Input final dtype:  "
        f"{input_audio.dtype}"
    )

    print(
        f"Target final dtype: "
        f"{target_audio.dtype}"
    )

    print(
        f"Saved: {input_output}"
    )

    print(
        f"Saved: {target_output}"
    )


def main():

    print(
        "Preparing PROVISIONAL MXR Dyna Comp datasets"
    )

    print(
        "Coarse alignment + provisional "
        "128-sample causal delay + PCM16."
    )

    print(
        "Raw original files will NOT be modified."
    )

    for config, split, lag in PAIRS:
        process(
            config,
            split,
            lag
        )

    print()
    print("=" * 65)
    print("DONE")
    print("=" * 65)

    print(
        f"Processed datasets: {OUTPUT}"
    )


if __name__ == "__main__":
    main()