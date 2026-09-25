"""Prepare aligned Starcaster 50% DRY/WET pairs for Wright's trainer.

Run from the thesis root:
    .venv\\Scripts\\python.exe scripts\\prepare50.py

Source WAVs and existing _16bit copies are never overwritten. The script
checks all three splits before writing any aligned training pairs.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import ndimage, signal

FS = 44100
PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "data" / "starcaster"
REPO = PROJECT / "external" / "Automated-GuitarAmpModelling"
NAME = "starcaster50_aligned_16bit"


def load_mono(path: Path):
    audio, sr = sf.read(path, dtype="float32")
    if sr != FS or audio.ndim != 1:
        raise ValueError(f"Se esperaba mono a 44100 Hz: {path}")
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"NaN o infinito en {path}")
    return audio


def wet_16bit_path(split: str) -> Path:
    folder = ROOT / split
    original = folder / f"starcaster_50_target_44_{split}.wav"
    converted = folder / f"starcaster_50_target_44_{split}_16bit.wav"
    if converted.exists():
        info = sf.info(converted)
        source = sf.info(original)
        if (info.samplerate != FS or info.channels != 1 or
                info.subtype != "PCM_16" or info.frames != source.frames):
            raise ValueError(f"La copia de 16 bits no coincide con el original: {converted}")
        print(f"Ya existe copia PCM16: {converted.relative_to(PROJECT)}")
    else:
        audio = load_mono(original)
        sf.write(converted, audio, FS, subtype="PCM_16")
        print(f"Copia PCM16 creada: {converted.relative_to(PROJECT)}")
    return converted


def coarse_lag(dry: np.ndarray, wet: np.ndarray) -> int:
    """Estimate approximate wet-vs-dry delay from RMS envelopes."""
    hop = 220  # ~5 ms
    def envelope(audio):
        blocks = audio[:len(audio) // hop * hop].reshape(-1, hop)
        rms = np.sqrt(np.mean(blocks * blocks, axis=1))
        return ndimage.gaussian_filter1d(rms.astype("float64"), 3.0)

    a, b = envelope(dry), envelope(wet)
    a -= a.mean()
    b -= b.mean()
    max_seconds = max(5., abs(len(dry) - len(wet)) / FS + 2.)
    max_frames = int(max_seconds * FS / hop)
    cor = signal.correlate(b, a, method="fft")
    lags = signal.correlation_lags(len(b), len(a))
    within = np.abs(lags) <= max_frames
    index = np.argmax(cor[within])
    frames = int(lags[within][index])
    strength = float(cor[within][index] / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"  Envolvente: desfase aproximado WET respecto a DRY: "
          f"{frames * hop / FS * 1000:+.1f} ms; correlacion {strength:.3f}")
    if strength < 0.30:
        raise ValueError("No se reconoce una correspondencia temporal clara en las envolventes.")
    return frames * hop


def fine_lag(dry: np.ndarray, wet: np.ndarray, approx: int) -> tuple[int, float, list[tuple[float, float]]]:
    """Refine to sample precision using several energetic, separated windows.

    Keep the SIGN of the aggregate cross-correlation: a distortion pedal can
    invert polarity. Do not flip the WET signal; it is the physical target.
    """
    margin = int(FS * .15)
    size = 3 * FS
    first = max(0, margin - approx)
    last = min(len(dry) - size, len(wet) - approx - margin - size)
    if first > last:
        raise ValueError("No hay suficiente audio superpuesto para medir el desfase.")

    starts = np.unique(np.linspace(first, last, 11, dtype="int64"))
    total_corr = None
    total_y_sq = None
    total_x_sq = 0.0
    windows = []
    for start in starts:
        start = int(start)
        xx = dry[start:start + size].astype("float64")