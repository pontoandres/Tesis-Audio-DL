from pathlib import Path

import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "starcaster"
OUTPUT = ROOT / "data" / "processed" / "starcaster"


# Desfases gruesos medidos originalmente con check_alignment.py
PAIRS = [
    ("25", "train", -73142),
    ("50", "train", -73139),

    ("25", "val", 95),
    ("50", "val", 97),

    ("25", "test", 109),
    ("50", "test", -110595),
]


# Experimento causal:
# dejamos 25% intacto por ahora.
# Para 50%, dejamos el target ~128 samples DESPUÉS del input.
CAUSAL_DELAY = {
    "25": 0,
    "50": 128,
}


def to_int16(audio):
    """
    Convierte el audio a PCM16 sin normalizar
    el volumen del archivo.
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
    Convención usada por check_alignment.py:

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

    input_audio = input_audio[:length]
    target_audio = target_audio[:length]

    return input_audio, target_audio


def add_causal_delay(
    input_audio,
    target_audio,
    delay
):
    """
    Hace que el target quede retrasado respecto al input.

    Para lograrlo, avanzamos el INPUT 'delay' muestras.

    Ejemplo:
        delay = 128

        input[t] -> target[t + 128]

    Esto le da a una red causal acceso a muestras
    anteriores del input para producir el target.
    """

    if delay > 0:
        input_audio = input_audio[delay:]

    length = min(
        len(input_audio),
        len(target_audio)
    )

    input_audio = input_audio[:length]
    target_audio = target_audio[:length]

    return input_audio, target_audio


def process(config, split, lag):

    input_path = (
        SOURCE
        / split
        / f"starcaster_input_{split}.wav"
    )

    target_path = (
        SOURCE
        / split
        / f"starcaster_{config}_target_44_{split}.wav"
    )

    fs_input, input_audio = wavfile.read(
        input_path
    )

    fs_target, target_audio = wavfile.read(
        target_path
    )

    if fs_input != fs_target:
        raise ValueError(
            f"Sample rate mismatch in "
            f"{config}% {split}"
        )

    print()
    print("=" * 60)
    print(
        f"Starcaster {config}% - {split}"
    )
    print("=" * 60)

    print(
        f"Input original dtype:  "
        f"{input_audio.dtype}"
    )

    print(
        f"Target original dtype: "
        f"{target_audio.dtype}"
    )

    print(
        f"Coarse lag: "
        f"{lag:+d} samples"
    )

    # 1. Alineación gruesa original
    input_audio, target_audio = align_pair(
        input_audio,
        target_audio,
        lag
    )

    # 2. Delay causal adicional
    delay = CAUSAL_DELAY[config]

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

    # 3. Convertir AMBOS a PCM16
    input_audio = to_int16(
        input_audio
    )

    target_audio = to_int16(
        target_audio
    )

    # 4. Guardar directamente
    # en formato esperado por Wright
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
        / f"starcaster{config}-input.wav"
    )

    target_output = (
        output_dir
        / f"starcaster{config}-target.wav"
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
        "Preparing FINAL Starcaster datasets"
    )

    print(
        "Alignment + causal delay + "
        "PCM16 conversion."
    )

    print(
        "Original files will NOT be modified."
    )

    for config, split, lag in PAIRS:
        process(
            config,
            split,
            lag
        )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        "Final datasets:"
    )

    print(
        OUTPUT
    )


if __name__ == "__main__":
    main()