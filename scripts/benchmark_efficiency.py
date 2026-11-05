#!/usr/bin/env python
"""
Benchmark computational efficiency of the three thesis architectures.

This benchmark measures architecture cost, not model fidelity. It reconstructs
representative CNN, LSTM and TCN architectures from versioned metadata and uses
the same deterministic synthetic mono input for every architecture.

Why architecture-only reconstruction?
- CNN/LSTM checkpoints are JSON while TCN checkpoints are PyTorch .pt files,
  so comparing raw checkpoint file sizes would mix serialization formats.
- TCN .pt checkpoints are intentionally git-ignored and may not exist on every
  machine.
- Dense inference cost depends on architecture/tensor shapes, not on the
  learned numerical values of the weights.

Reported metrics:
- trainable parameter count;
- standardized float32 model-state size (parameters + persistent buffers);
- inference time (median, mean, p95, min, max);
- real-time factor (RTF = inference_seconds / audio_seconds);
- real-time speed multiplier (1 / RTF);
- CUDA peak allocated/reserved memory when benchmarking on GPU.

Inference is streamed in fixed output blocks. The LSTM preserves recurrent
state between blocks, while CNN/TCN are supplied with the causal left context
required for each block. This avoids unsupported very-long cuDNN sequences and
matches the chunked inference strategy already validated during thesis testing.

The default CPU benchmark uses one PyTorch compute thread to better represent
a real-time audio processing thread and to improve reproducibility.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
WRIGHT_ROOT = ROOT / "external" / "Automated-GuitarAmpModelling"

CNN_MODEL = (
    ROOT
    / "results"
    / "starcaster50_cnn100_causal128_finealign_v1-"
    / "model_best.json"
)
LSTM_MODEL = (
    ROOT
    / "results"
    / "starcaster50_lstm32_scratch_v1-"
    / "model_best.json"
)
TCN_CONFIG = (
    ROOT
    / "results"
    / "starcaster50_tcn300c_scratch_v1-"
    / "model_config.json"
)

DEFAULT_OUTDIR = ROOT / "results" / "efficiency"
DEFAULT_JSON = DEFAULT_OUTDIR / "benchmark_efficiency.json"
DEFAULT_CSV = DEFAULT_OUTDIR / "benchmark_efficiency.csv"
DEFAULT_MD = ROOT / "docs" / "efficiency_benchmark.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CNN, LSTM and TCN thesis architectures."
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "both"),
        default="both",
        help="Device(s) to benchmark.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="Output audio duration per timed inference.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20000,
        help=(
            "Output samples processed per streaming block. "
            "20000 matches the validated LSTM/TCN evaluation chunk size."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Untimed warm-up iterations.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Timed repetitions per architecture/device.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="PyTorch CPU compute threads.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
    )
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_import_paths() -> None:
    if not WRIGHT_ROOT.is_dir():
        raise FileNotFoundError(
            f"Wright submodule not found: {WRIGHT_ROOT}\n"
            "Run: git submodule update --init --recursive"
        )

    for path in (str(SCRIPT_DIR), str(WRIGHT_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def build_models() -> dict[str, torch.nn.Module]:
    ensure_import_paths()

    import CoreAudioML.networks as networks
    from train_tcn import CausalTCN300

    cnn_payload = load_json(CNN_MODEL)
    lstm_payload = load_json(LSTM_MODEL)
    tcn_config = load_json(TCN_CONFIG)

    # Architecture-only reconstruction. Do not load learned state values.
    cnn = networks.load_model(
        {"model_data": copy.deepcopy(cnn_payload["model_data"])}
    )
    lstm = networks.load_model(
        {"model_data": copy.deepcopy(lstm_payload["model_data"])}
    )
    tcn = CausalTCN300(
        channels=int(tcn_config["channels"]),
        kernel_size=int(tcn_config["kernel_size"]),
        dilations=tuple(int(x) for x in tcn_config["dilations"]),
    )

    expected_tcn = tcn_config.get("trainable_parameters")
    actual_tcn = count_trainable_parameters(tcn)
    if expected_tcn is not None and int(expected_tcn) != actual_tcn:
        raise RuntimeError(
            f"TCN parameter mismatch: config={expected_tcn}, "
            f"reconstructed={actual_tcn}"
        )

    return {
        "CNN": cnn,
        "LSTM": lstm,
        "TCN": tcn,
    }


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def state_size_bytes(model: torch.nn.Module) -> int:
    total = 0
    for tensor in model.state_dict().values():
        total += tensor.numel() * tensor.element_size()
    return int(total)


def reset_runtime_state(model: torch.nn.Module) -> None:
    reset = getattr(model, "reset_hidden", None)
    if callable(reset):
        reset()


def synthetic_signal(
    audio_samples: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = 0.1 * torch.randn(
        audio_samples,
        generator=generator,
        dtype=torch.float32,
    )
    return x.to(device)


def cnn_receptive_field(model: torch.nn.Module) -> int:
    blocks = len(model.blocks) - 1
    dilation_sum = sum(
        model.dilation_growth ** layer
        for layer in range(model.layers)
    )
    return (
        1
        + blocks
        * (model.kernel_size - 1)
        * dilation_sum
    )


def architecture_context(
    architecture: str,
    model: torch.nn.Module,
) -> int:
    if architecture == "CNN":
        return cnn_receptive_field(model) - 1
    if architecture == "TCN":
        return int(model.receptive_field) - 1
    return 0


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_context_chunk(
    signal: torch.Tensor,
    start: int,
    end: int,
    context: int,
) -> torch.Tensor:
    left = start - context

    if left >= 0:
        return signal[left:end]

    pad = torch.zeros(
        -left,
        dtype=signal.dtype,
        device=signal.device,
    )
    return torch.cat((pad, signal[:end]), dim=0)


def infer_stream(
    architecture: str,
    model: torch.nn.Module,
    signal: torch.Tensor,
    chunk_size: int,
) -> int:
    """Process a complete mono signal using the validated streaming strategy."""
    total_samples = int(signal.numel())
    context = architecture_context(architecture, model)
    produced = 0

    reset_runtime_state(model)

    with torch.inference_mode():
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            expected = end - start

            if architecture == "LSTM":
                chunk = (
                    signal[start:end]
                    .view(-1, 1, 1)
                    .contiguous()
                )
                output = model(chunk)
                actual = int(output.shape[0])

            elif architecture == "CNN":
                input_chunk = make_context_chunk(
                    signal,
                    start,
                    end,
                    context,
                )
                x = (
                    input_chunk
                    .view(-1, 1, 1)
                    .contiguous()
                )
                full_output = model(x)
                output = full_output[-expected:]
                actual = int(output.shape[0])

            elif architecture == "TCN":
                input_chunk = make_context_chunk(
                    signal,
                    start,
                    end,
                    context,
                )
                x = (
                    input_chunk
                    .view(1, 1, -1)
                    .contiguous()
                )
                output = model(x)
                actual = int(output.shape[-1])

            else:
                raise ValueError(
                    f"Unsupported architecture: {architecture}"
                )

            if actual != expected:
                raise RuntimeError(
                    f"{architecture} output length mismatch in block "
                    f"{start}:{end}: {actual} != {expected}"
                )

            produced += actual
            del output

    reset_runtime_state(model)

    if produced != total_samples:
        raise RuntimeError(
            f"{architecture} produced {produced} samples, "
            f"expected {total_samples}"
        )

    return produced


def benchmark_one(
    architecture: str,
    base_model: torch.nn.Module,
    device: torch.device,
    audio_seconds: float,
    sample_rate: int,
    chunk_size: int,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict:
    model = base_model.to(device)
    model.eval()

    audio_samples = int(round(audio_seconds * sample_rate))
    signal = synthetic_signal(
        audio_samples,
        device,
        seed,
    )

    context = architecture_context(architecture, model)

    # CoreAudioML's GatedConvNet dynamically creates tensors without an
    # explicit device. torch.device(...) makes those allocations follow the
    # benchmark device without modifying the external submodule.
    with torch.device(device):
        for _ in range(warmup):
            infer_stream(
                architecture,
                model,
                signal,
                chunk_size,
            )
            sync(device)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            sync(device)
            baseline_allocated = torch.cuda.memory_allocated(device)
            baseline_reserved = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline_allocated = None
            baseline_reserved = None

        times = []
        for _ in range(repeats):
            sync(device)
            start_time = time.perf_counter()
            infer_stream(
                architecture,
                model,
                signal,
                chunk_size,
            )
            sync(device)
            elapsed = time.perf_counter() - start_time
            times.append(float(elapsed))

        if device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
        else:
            peak_allocated = None
            peak_reserved = None

    median_seconds = statistics.median(times)
    mean_seconds = statistics.mean(times)
    p95_seconds = percentile(times, 0.95)
    median_rtf = median_seconds / audio_seconds

    result = {
        "architecture": architecture,
        "device": str(device),
        "sample_rate_hz": sample_rate,
        "audio_seconds": audio_seconds,
        "audio_samples": audio_samples,
        "chunk_size_samples": chunk_size,
        "context_samples": context,
        "warmup_iterations": warmup,
        "timed_repetitions": repeats,
        "trainable_parameters": count_trainable_parameters(model),
        "all_parameters": count_all_parameters(model),
        "state_size_bytes": state_size_bytes(model),
        "state_size_mb": state_size_bytes(model) / (1024 ** 2),
        "median_inference_seconds": median_seconds,
        "mean_inference_seconds": mean_seconds,
        "p95_inference_seconds": p95_seconds,
        "min_inference_seconds": min(times),
        "max_inference_seconds": max(times),
        "median_rtf": median_rtf,
        "median_realtime_x": (
            float("inf") if median_rtf == 0 else 1.0 / median_rtf
        ),
        "timings_seconds": times,
        "cuda_baseline_allocated_bytes": baseline_allocated,
        "cuda_baseline_reserved_bytes": baseline_reserved,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_peak_increment_bytes": (
            None
            if peak_allocated is None
            else int(peak_allocated - baseline_allocated)
        ),
    }

    reset_runtime_state(model)
    model.to("cpu")
    del signal
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result

def hardware_metadata(args: argparse.Namespace) -> dict:
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": (
            os.environ.get("PROCESSOR_IDENTIFIER")
            or platform.processor()
            or "unknown"
        ),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if cuda_available
            else None
        ),
        "cpu_threads": args.cpu_threads,
        "dtype": "float32",
        "signal": "deterministic synthetic mono noise, amplitude scale 0.1",
    }


def write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "architecture",
        "device",
        "trainable_parameters",
        "state_size_bytes",
        "state_size_mb",
        "audio_seconds",
        "median_inference_seconds",
        "mean_inference_seconds",
        "p95_inference_seconds",
        "median_rtf",
        "median_realtime_x",
        "cuda_peak_allocated_bytes",
        "cuda_peak_increment_bytes",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def fmt(value, digits=4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def write_markdown(payload: dict, path: Path) -> None:
    rows = payload["results"]
    metadata = payload["environment"]

    lines = [
        "# Efficiency benchmark",
        "",
        "Generated by `scripts/benchmark_efficiency.py`.",
        "",
        "## Protocol",
        "",
        (
            f"- Signal: deterministic synthetic mono float32, "
            f"{payload['protocol']['audio_seconds']} s at "
            f"{payload['protocol']['sample_rate_hz']} Hz."
        ),
        (
            f"- Streaming block: "
            f"{payload['protocol']['chunk_size_samples']} output samples "
            f"(~{1000 * payload['protocol']['chunk_size_samples'] / payload['protocol']['sample_rate_hz']:.1f} ms)."
        ),
        (
            "- LSTM preserves recurrent hidden state between blocks; "
            "CNN and TCN receive the causal left context required to "
            "produce each output block."
        ),
        (
            f"- Warm-up: {payload['protocol']['warmup_iterations']} "
            f"iterations."
        ),
        (
            f"- Timed repetitions: "
            f"{payload['protocol']['timed_repetitions']}."
        ),
        (
            f"- CPU threads: {metadata['cpu_threads']}."
        ),
        "- Batch size: 1.",
        (
            "- RTF = inference time / audio duration; lower is better. "
            "RTF < 1 means faster than real time."
        ),
        (
            "- Model size is standardized as raw bytes in the PyTorch "
            "state_dict (parameters + persistent buffers), avoiding "
            "JSON-vs-PT serialization bias."
        ),
        (
            "- Architectures are reconstructed from versioned metadata; "
            "learned weight values are not required because this stage "
            "measures architecture cost rather than fidelity."
        ),
        "",
        "## Environment",
        "",
        f"- CPU: {metadata['cpu']}",
        f"- GPU: {metadata['gpu'] or 'N/A'}",
        f"- PyTorch: {metadata['torch']}",
        f"- CUDA runtime: {metadata['cuda_runtime'] or 'N/A'}",
        f"- Python: {metadata['python']}",
        "",
        "## Results",
        "",
        (
            "| Architecture | Device | Params | State size (MB) | "
            "Median time (s) | p95 (s) | Median RTF | Realtime x | "
            "CUDA peak alloc. (MB) |"
        ),
        (
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for row in rows:
        peak = row.get("cuda_peak_allocated_bytes")
        peak_mb = None if peak is None else peak / (1024 ** 2)
        lines.append(
            "| "
            + " | ".join(
                [
                    row["architecture"],
                    row["device"],
                    f"{row['trainable_parameters']:,}",
                    fmt(row["state_size_mb"], 4),
                    fmt(row["median_inference_seconds"], 6),
                    fmt(row["p95_inference_seconds"], 6),
                    fmt(row["median_rtf"], 6),
                    fmt(row["median_realtime_x"], 2),
                    fmt(peak_mb, 2),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            (
                "- Timing values are throughput measurements for a fixed "
                "audio block, not end-to-end audio-interface latency."
            ),
            (
                "- CUDA peak memory is reported only for GPU runs. "
                "CPU peak native memory is not estimated by this script."
            ),
            (
                "- Fidelity metrics remain in the common experiment "
                "registry and must be combined with these results only "
                "after the efficiency benchmark is complete."
            ),
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.seconds <= 0:
        raise ValueError("--seconds must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be >= 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)

    devices = []
    if args.device in ("cpu", "both"):
        devices.append(torch.device("cpu"))
    if args.device in ("cuda", "both"):
        if not torch.cuda.is_available():
            if args.device == "cuda":
                raise RuntimeError(
                    "CUDA requested but torch.cuda.is_available() is False."
                )
            print("WARNING: CUDA unavailable; running CPU benchmark only.")
        else:
            devices.append(torch.device("cuda"))

    models = build_models()

    protocol = {
        "audio_seconds": args.seconds,
        "sample_rate_hz": args.sample_rate,
        "chunk_size_samples": args.chunk_size,
        "warmup_iterations": args.warmup,
        "timed_repetitions": args.repeats,
        "seed": args.seed,
        "representative_models": {
            "CNN": str(CNN_MODEL.relative_to(ROOT)),
            "LSTM": str(LSTM_MODEL.relative_to(ROOT)),
            "TCN": str(TCN_CONFIG.relative_to(ROOT)),
        },
    }

    print("=" * 80)
    print("THESIS EFFICIENCY BENCHMARK")
    print("=" * 80)
    print(f"Audio duration : {args.seconds:.3f} s")
    print(f"Sample rate    : {args.sample_rate} Hz")
    print(f"Chunk size     : {args.chunk_size} samples")
    print(f"Warm-up        : {args.warmup}")
    print(f"Repeats        : {args.repeats}")
    print(f"CPU threads    : {args.cpu_threads}")
    print(f"Devices        : {[str(d) for d in devices]}")
    print("=" * 80)

    results = []

    for device in devices:
        print()
        print(f"--- Device: {device} ---")
        for architecture in ("CNN", "LSTM", "TCN"):
            print(f"Benchmarking {architecture}...", flush=True)
            result = benchmark_one(
                architecture=architecture,
                base_model=models[architecture],
                device=device,
                audio_seconds=args.seconds,
                sample_rate=args.sample_rate,
                chunk_size=args.chunk_size,
                warmup=args.warmup,
                repeats=args.repeats,
                seed=args.seed,
            )
            results.append(result)

            peak = result["cuda_peak_allocated_bytes"]
            peak_text = (
                "N/A"
                if peak is None
                else f"{peak / (1024 ** 2):.2f} MB"
            )
            print(
                f"  params={result['trainable_parameters']:,} | "
                f"state={result['state_size_mb']:.4f} MB | "
                f"median={result['median_inference_seconds']:.6f} s | "
                f"RTF={result['median_rtf']:.6f} | "
                f"xRT={result['median_realtime_x']:.2f} | "
                f"peak={peak_text}"
            )

    payload = {
        "schema_version": 1,
        "protocol": protocol,
        "environment": hardware_metadata(args),
        "results": results,
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    json_path = args.outdir / DEFAULT_JSON.name
    csv_path = args.outdir / DEFAULT_CSV.name

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    write_csv(results, csv_path)
    write_markdown(payload, DEFAULT_MD)

    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print(f"JSON:     {json_path}")
    print(f"CSV:      {csv_path}")
    print(f"Markdown: {DEFAULT_MD}")
    print("=" * 80)


if __name__ == "__main__":
    main()
