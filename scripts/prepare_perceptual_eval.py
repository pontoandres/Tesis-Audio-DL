#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]
GENERALIZATION = ROOT / "data" / "generalization" / "processed"
RESULTS = ROOT / "results" / "generalization"
DEFAULT_OUT = ROOT / "data" / "perceptual" / "form_pack"

EFFECTS = ("starcaster", "dynacomp")
ARCHITECTURES = ("CNN", "LSTM", "TCN")


def read_mono_float(path: Path) -> tuple[int, np.ndarray]:
    fs, x = wavfile.read(path)

    if x.ndim == 2:
        if x.shape[1] == 1:
            x = x[:, 0]
        else:
            x = np.mean(x.astype(np.float64), axis=1)

    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float64) / 2147483648.0
    elif np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float64)
    else:
        raise TypeError(f"Unsupported WAV dtype in {path}: {x.dtype}")

    return int(fs), np.ascontiguousarray(x)


def write_pcm16(path: Path, fs: int, x: np.ndarray) -> dict:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    clipped = int(np.count_nonzero(np.abs(x) > 1.0))
    y = np.clip(x, -1.0, 1.0)
    pcm = np.clip(np.round(y * 32767.0), -32768, 32767).astype(np.int16)
    wavfile.write(path, fs, pcm)
    return {
        "peak_before_pcm16": peak,
        "samples_over_full_scale": clipped,
    }


def rms_db(x: np.ndarray) -> float:
    rms = math.sqrt(float(np.mean(np.square(x))) + 1e-20)
    return 20.0 * math.log10(max(rms, 1e-12))


def choose_start(
    clean: np.ndarray,
    fs: int,
    duration_s: float,
    margin_s: float = 5.0,
) -> tuple[float, dict]:
    """
    Select an active, dynamically varied excerpt using only the dry input.

    This avoids choosing a segment because one model happens to sound better
    or worse. Candidate windows are scored from short-time RMS statistics.
    """
    frame_s = 0.25
    frame = max(1, int(round(frame_s * fs)))
    duration = int(round(duration_s * fs))
    margin = int(round(margin_s * fs))

    if len(clean) < duration + 2 * margin:
        raise ValueError(
            f"Audio too short for {duration_s:.2f}s excerpt with "
            f"{margin_s:.2f}s margins."
        )

    starts = range(margin, len(clean) - margin - duration + 1, fs)
    best = None

    for start in starts:
        segment = clean[start:start + duration]
        nframes = len(segment) // frame
        if nframes < 2:
            continue

        db = np.array(
            [
                rms_db(segment[i * frame:(i + 1) * frame])
                for i in range(nframes)
            ],
            dtype=np.float64,
        )

        active_fraction = float(np.mean(db > -45.0))
        if active_fraction < 0.75:
            continue

        median_db = float(np.median(db))
        dynamic_db = float(np.percentile(db, 90) - np.percentile(db, 10))
        peak = float(np.max(np.abs(segment)))

        # Favor audible activity plus some dynamic variation. Penalize windows
        # that already touch digital full scale.
        score = median_db + 0.50 * dynamic_db
        if peak >= 0.999:
            score -= 20.0

        candidate = {
            "start_sample": int(start),
            "start_seconds": float(start / fs),
            "score": float(score),
            "median_frame_rms_dbfs": median_db,
            "p90_p10_dynamic_db": dynamic_db,
            "active_fraction": active_fraction,
            "dry_peak": peak,
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if best is None:
        raise RuntimeError(
            "Could not find an excerpt with enough active audio. "
            "Use --start to choose a segment manually."
        )

    return float(best["start_seconds"]), best


def source_paths(effect: str) -> dict[str, Path]:
    return {
        "reference": (
            GENERALIZATION
            / effect
            / "50"
            / "test"
            / "generalization50-target.wav"
        ),
        "CNN": RESULTS / f"{effect}50_cnn_prediction.wav",
        "LSTM": RESULTS / f"{effect}50_lstm_prediction.wav",
        "TCN": RESULTS / f"{effect}50_tcn_prediction.wav",
    }


def crop_exact(
    path: Path,
    start_s: float,
    duration_s: float,
    expected_fs: int,
) -> tuple[np.ndarray, dict]:
    fs, x = read_mono_float(path)
    if fs != expected_fs:
        raise ValueError(f"{path}: expected {expected_fs} Hz, got {fs} Hz")

    start = int(round(start_s * fs))
    count = int(round(duration_s * fs))
    end = start + count

    if end > len(x):
        raise ValueError(
            f"{path}: requested {start_s:.3f}-{start_s + duration_s:.3f}s "
            f"but file lasts {len(x) / fs:.3f}s"
        )

    y = np.ascontiguousarray(x[start:end])
    return y, {
        "source": str(path),
        "source_seconds": float(len(x) / fs),
        "clip_start_sample": start,
        "clip_samples": count,
        "clip_rms_dbfs": rms_db(y),
        "clip_peak": float(np.max(np.abs(y))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare anonymized clips for the thesis perceptual evaluation."
        )
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help=(
            "Excerpt start in seconds. If omitted, select automatically "
            "from dry input only."
        ),
    )
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    clean_path = (
        GENERALIZATION
        / "starcaster"
        / "50"
        / "test"
        / "generalization50-input.wav"
    )
    if not clean_path.exists():
        raise FileNotFoundError(
            f"Missing generalization input: {clean_path}\n"
            "Run scripts\\prepare_generalization.py first."
        )

    all_required = [clean_path]
    for effect in EFFECTS:
        all_required.extend(source_paths(effect).values())

    missing = [path for path in all_required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required perceptual source files:\n"
            + "\n".join(str(path) for path in missing)
        )

    fs, clean = read_mono_float(clean_path)

    if args.start is None:
        start_s, selection = choose_start(
            clean,
            fs,
            args.duration,
        )
        selection_mode = "automatic_from_dry_only"
    else:
        start_s = float(args.start)
        selection = {
            "start_seconds": start_s,
            "selection_reason": "manual_start_argument",
        }
        selection_mode = "manual"

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    report = {
        "schema_version": 1,
        "sample_rate_hz": fs,
        "duration_seconds": args.duration,
        "start_seconds": start_s,
        "selection_mode": selection_mode,
        "selection_details": selection,
        "no_loudness_normalization": True,
        "seed": args.seed,
        "effects": {},
    }

    print("\n" + "=" * 72)
    print("PERCEPTUAL EVALUATION CLIP PREPARATION")
    print("=" * 72)
    print(f"Start       : {start_s:.3f} s")
    print(f"Duration    : {args.duration:.3f} s")
    print(f"Selection   : {selection_mode}")
    if selection_mode == "automatic_from_dry_only":
        print(
            f"Dry RMS med.: "
            f"{selection['median_frame_rms_dbfs']:.2f} dBFS"
        )
        print(
            f"Dry dynamics: "
            f"{selection['p90_p10_dynamic_db']:.2f} dB"
        )

    for effect_index, effect in enumerate(EFFECTS):
        paths = source_paths(effect)

        arch_order = list(ARCHITECTURES)
        effect_rng = random.Random(args.seed + 1000 * effect_index)
        effect_rng.shuffle(arch_order)

        mapping = {
            label: architecture
            for label, architecture in zip(("A", "B", "C"), arch_order)
        }

        effect_report = {
            "mapping": mapping,
            "files": {},
        }

        reference, ref_meta = crop_exact(
            paths["reference"],
            start_s,
            args.duration,
            fs,
        )
        ref_out = out_dir / f"{effect}_REFERENCE.wav"
        ref_write = write_pcm16(ref_out, fs, reference)
        effect_report["files"]["REFERENCE"] = {
            **ref_meta,
            **ref_write,
            "output": str(ref_out),
        }

        for label in ("A", "B", "C"):
            architecture = mapping[label]
            clip, meta = crop_exact(
                paths[architecture],
                start_s,
                args.duration,
                fs,
            )
            out = out_dir / f"{effect}_{label}.wav"
            write_meta = write_pcm16(out, fs, clip)
            effect_report["files"][label] = {
                "architecture": architecture,
                **meta,
                **write_meta,
                "output": str(out),
            }

        report["effects"][effect] = effect_report

        print(f"\n{effect.upper()}")
        print(f"  REFERENCE -> physical pedal target")
        for label in ("A", "B", "C"):
            print(f"  {label} -> anonymized model output")
        print(
            "  Internal answer key: "
            + ", ".join(
                f"{label}={mapping[label]}" for label in ("A", "B", "C")
            )
        )

    key_path = out_dir / "answer_key.json"
    key_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    instructions = f"""Perceptual evaluation pack

Selected excerpt
- Start: {start_s:.3f} s
- Duration: {args.duration:.3f} s
- Sample rate: {fs} Hz
- Selection: {selection_mode}

Important
- REFERENCE is the physical pedal target.
- A/B/C are anonymized CNN/LSTM/TCN predictions.
- No independent loudness normalization or gain matching was applied.
- Use the exact same files in the questionnaire.
- Keep answer_key.json private until data collection is finished.

Suggested rating question
Comparando con la referencia, ¿qué tan similar te parece el audio X?

1 = Muy diferente
2 = Bastante diferente
3 = Moderadamente similar
4 = Muy similar
5 = Prácticamente indistinguible
"""
    (out_dir / "README_FORM.txt").write_text(
        instructions,
        encoding="utf-8",
    )

    print(f"\nOutput dir  : {out_dir}")
    print(f"Answer key  : {key_path}")
    print(
        "No model was retrained and no loudness normalization was applied."
    )
    print("DONE")


if __name__ == "__main__":
    main()
