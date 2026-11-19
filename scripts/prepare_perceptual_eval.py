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

    # Normalize PCM before any channel averaging. This preserves the correct
    # amplitude scale if a source happens to be stereo.
    if x.dtype == np.int16:
        y = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32:
        y = x.astype(np.float64) / 2147483648.0
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float64)
    else:
        raise TypeError(f"Unsupported WAV dtype in {path}: {x.dtype}")

    if y.ndim == 2:
        if y.shape[1] == 1:
            y = y[:, 0]
        else:
            y = np.mean(y, axis=1)

    return int(fs), np.ascontiguousarray(y)


def write_pcm16(path: Path, fs: int, x: np.ndarray) -> dict:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    clipped = int(np.count_nonzero(np.abs(x) > 1.0))
    y = np.clip(x, -1.0, 1.0)
    pcm = np.clip(
        np.round(y * 32767.0),
        -32768,
        32767,
    ).astype(np.int16)
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
    Select one active, dynamically varied excerpt using ONLY the dry signal.

    Predictions and pedal targets are deliberately not inspected during
    selection, avoiding cherry-picking a window that favors one model.
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

    starts = range(
        margin,
        len(clean) - margin - duration + 1,
        fs,
    )
    best = None

    for start in starts:
        segment = clean[start:start + duration]
        nframes = len(segment) // frame
        if nframes < 2:
            continue

        frame_db = np.array(
            [
                rms_db(segment[i * frame:(i + 1) * frame])
                for i in range(nframes)
            ],
            dtype=np.float64,
        )

        active_fraction = float(np.mean(frame_db > -45.0))
        if active_fraction < 0.75:
            continue

        median_db = float(np.median(frame_db))
        dynamic_db = float(
            np.percentile(frame_db, 90)
            - np.percentile(frame_db, 10)
        )
        peak = float(np.max(np.abs(segment)))

        # Favor audible activity and dynamic variation. Penalize clipping.
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


def clean_path() -> Path:
    return (
        GENERALIZATION
        / "starcaster"
        / "50"
        / "test"
        / "generalization50-input.wav"
    )


def source_paths(effect: str) -> dict[str, Path]:
    return {
        "PEDAL_REAL": (
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
        raise ValueError(
            f"{path}: expected {expected_fs} Hz, got {fs} Hz"
        )

    start = int(round(start_s * fs))
    count = int(round(duration_s * fs))
    end = start + count

    if end > len(x):
        raise ValueError(
            f"{path}: requested "
            f"{start_s:.3f}-{start_s + duration_s:.3f}s "
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
            "Prepare clean, real-pedal and anonymized model clips "
            "for the thesis perceptual evaluation."
        )
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help=(
            "Excerpt start in seconds. If omitted, choose automatically "
            "using only the dry input."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducible A/B/C architecture mapping.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
    )
    args = parser.parse_args()

    dry_path = clean_path()

    required = [dry_path]
    for effect in EFFECTS:
        required.extend(source_paths(effect).values())

    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required perceptual source files:\n"
            + "\n".join(str(path) for path in missing)
        )

    fs, clean = read_mono_float(dry_path)

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

    report = {
        "schema_version": 2,
        "sample_rate_hz": fs,
        "duration_seconds": args.duration,
        "start_seconds": start_s,
        "selection_mode": selection_mode,
        "selection_details": selection,
        "no_loudness_normalization": True,
        "participant_knows_clean_and_real_pedal": True,
        "architecture_names_hidden_from_participant": True,
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

    clean_clip, clean_meta = crop_exact(
        dry_path,
        start_s,
        args.duration,
        fs,
    )

    for effect_index, effect in enumerate(EFFECTS):
        paths = source_paths(effect)

        architecture_order = list(ARCHITECTURES)
        effect_rng = random.Random(
            args.seed + 1000 * effect_index
        )
        effect_rng.shuffle(architecture_order)

        mapping = {
            label: architecture
            for label, architecture in zip(
                ("A", "B", "C"),
                architecture_order,
            )
        }

        effect_report = {
            "mapping": mapping,
            "files": {},
        }

        # Duplicate the same dry excerpt under each effect name so the final
        # form pack contains exactly five clearly named files per block.
        clean_out = out_dir / f"{effect}_CLEAN.wav"
        clean_write = write_pcm16(
            clean_out,
            fs,
            clean_clip,
        )
        effect_report["files"]["CLEAN"] = {
            **clean_meta,
            **clean_write,
            "output": str(clean_out),
            "participant_label": "Guitarra limpia, sin pedal",
        }

        pedal_clip, pedal_meta = crop_exact(
            paths["PEDAL_REAL"],
            start_s,
            args.duration,
            fs,
        )
        pedal_out = out_dir / f"{effect}_PEDAL_REAL.wav"
        pedal_write = write_pcm16(
            pedal_out,
            fs,
            pedal_clip,
        )
        effect_report["files"]["PEDAL_REAL"] = {
            **pedal_meta,
            **pedal_write,
            "output": str(pedal_out),
            "participant_label": "Pedal físico real",
        }

        for label in ("A", "B", "C"):
            architecture = mapping[label]
            clip, meta = crop_exact(
                paths[architecture],
                start_s,
                args.duration,
                fs,
            )
            out = out_dir / f"{effect}_MODELO_{label}.wav"
            write_meta = write_pcm16(
                out,
                fs,
                clip,
            )
            effect_report["files"][f"MODELO_{label}"] = {
                "architecture": architecture,
                **meta,
                **write_meta,
                "output": str(out),
                "participant_label": f"Emulación Modelo {label}",
            }

        report["effects"][effect] = effect_report

        print(f"\n{effect.upper()}")
        print("  CLEAN       -> participant knows: no pedal")
        print("  PEDAL_REAL  -> participant knows: physical pedal")
        for label in ("A", "B", "C"):
            print(
                f"  MODELO_{label}    -> participant knows: "
                "deep-learning emulation"
            )
        print(
            "  Private mapping: "
            + ", ".join(
                f"{label}={mapping[label]}"
                for label in ("A", "B", "C")
            )
        )

    key_path = out_dir / "answer_key.json"
    key_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    instructions = f"""PAQUETE PARA EVALUACION PERCEPTUAL

Fragmento seleccionado automaticamente
- Inicio: {start_s:.3f} s
- Duracion: {args.duration:.3f} s
- Sample rate: {fs} Hz
- Seleccion: {selection_mode}

ARCHIVOS POR EFECTO
1. <efecto>_CLEAN.wav
   Guitarra limpia, sin pedal.

2. <efecto>_PEDAL_REAL.wav
   Salida real del pedal fisico. Esta es la referencia.

3. <efecto>_MODELO_A.wav
4. <efecto>_MODELO_B.wav
5. <efecto>_MODELO_C.wav
   Tres emulaciones mediante modelos de aprendizaje profundo.

IMPORTANTE
- El participante SI debe saber cual es CLEAN y cual es PEDAL_REAL.
- El participante SI debe saber que A/B/C son emulaciones computacionales.
- No necesita saber cual arquitectura corresponde a A/B/C.
- No se aplico normalizacion independiente de loudness ni gain matching.
- Los cinco archivos usan exactamente el mismo intervalo temporal.
- answer_key.json es privado hasta terminar la recoleccion.

PREGUNTA PRINCIPAL PARA CADA MODELO
Comparando el Modelo X con el pedal real, ¿que tan fiel te parece la
emulacion del efecto?

1 = Muy poco fiel
2 = Poco fiel
3 = Moderadamente fiel
4 = Muy fiel
5 = Practicamente indistinguible del pedal real

MULTIPLE CHOICE / CHECKBOXES
¿Que diferencias notas entre el Modelo X y el pedal real?
Selecciona todas las que apliquen.

Para Starcaster Distortion:
- Nivel o volumen
- Timbre / color del sonido
- Cantidad o caracter de la distorsion
- Ataque / respuesta al tocar
- Sustain
- Ruido o artefactos
- No noto una diferencia clara
- Otro

Para MXR Dyna Comp:
- Nivel o volumen
- Timbre / color del sonido
- Ataque
- Sustain
- Cantidad o caracter de la compresion
- Respuesta dinamica
- Ruido o artefactos
- No noto una diferencia clara
- Otro

TEXT BOX ABIERTO
Si quieres, describe con tus propias palabras cualquier diferencia que
notes entre el Modelo X y el pedal real.

PREGUNTA FINAL OPCIONAL POR EFECTO
De los Modelos A, B y C, ¿cual te parecio mas fiel al pedal real?
- Modelo A
- Modelo B
- Modelo C
- No percibi una diferencia clara

No model was retrained. No manual audio cutting is required.
"""

    (out_dir / "README_FORM.txt").write_text(
        instructions,
        encoding="utf-8",
    )

    print(f"\nOutput dir  : {out_dir}")
    print(f"Answer key  : {key_path}")
    print("Generated   : 10 WAV files, five per effect")
    print(
        "No model was retrained and no loudness normalization was applied."
    )
    print("DONE")


if __name__ == "__main__":
    main()
