#!/usr/bin/env python
"""
Train a causal TCN for the thesis datasets.

Architecture: TCN-300-C-inspired backbone based on Steinmetz & Reiss (AES 2022),
adapted to the thesis setup:
- one separate model per pedal setting;
- no FiLM / parameter conditioning;
- causal valid convolutions with dilations 1, 10, 100, 1000;
- 32 channels, kernel size 13;
- same Wright/CoreAudioML-style training objective:
  0.75 * ESR with high-pass pre-emphasis + 0.25 * DC loss.

The script consumes the already processed/aligned WAV files and produces
artifacts compatible with the thesis evaluation pipeline, including
test_out_bestv.wav and training_stats.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train causal TCN thesis baseline.")
    p.add_argument("--effect", choices=["starcaster", "dynacomp"], required=True)
    p.add_argument("--setting", type=int, choices=[25, 50, 75, 100], required=True)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--segment-length", type=int, default=22050)
    p.add_argument("--learning-rate", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--kernel-size", type=int, default=13)
    p.add_argument("--dilations", type=int, nargs="+", default=[1, 10, 100, 1000])

    p.add_argument("--inference-chunk", type=int, default=20000)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--smoke-test", action="store_true")

    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_mono_wav(path: Path) -> tuple[int, np.ndarray]:
    fs, x = wavfile.read(path)

    if x.ndim == 2:
        if x.shape[1] != 1:
            raise ValueError(f"Expected mono WAV, got shape {x.shape}: {path}")
        x = x[:, 0]

    if x.dtype == np.int16:
        y = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        y = x.astype(np.float32) / 2147483648.0
    elif x.dtype in (np.float32, np.float64):
        y = x.astype(np.float32)
    else:
        raise TypeError(f"Unsupported WAV dtype {x.dtype}: {path}")

    return int(fs), np.ascontiguousarray(y)


def write_float_wav(path: Path, fs: int, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, fs, np.asarray(x, dtype=np.float32))


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.PReLU(out_ch)
        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res(x)
        y = self.act(self.bn(self.conv(x)))
        residual = residual[..., -y.shape[-1] :]
        return y + residual


class CausalTCN300(nn.Module):
    def __init__(
        self,
        channels: int = 32,
        kernel_size: int = 13,
        dilations: tuple[int, ...] = (1, 10, 100, 1000),
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)

        blocks = []
        in_ch = 1
        for dilation in self.dilations:
            blocks.append(TCNBlock(in_ch, channels, kernel_size, dilation))
            in_ch = channels
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Conv1d(channels, 1, kernel_size=1)

    @property
    def receptive_field(self) -> int:
        return 1 + sum((self.kernel_size - 1) * d for d in self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return torch.tanh(self.output(x))


def pre_emphasis(x: torch.Tensor) -> torch.Tensor:
    # Matches CoreAudioML PreEmph([-0.85, 1]):
    # y[n] = x[n] - 0.85 * x[n-1]
    x_prev = F.pad(x[..., :-1], (1, 0))
    return x - 0.85 * x_prev


def esr_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    eps = 1e-5
    err = torch.mean((target - output) ** 2)
    energy = torch.mean(target ** 2) + eps
    return err / energy


def dc_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    eps = 1e-5
    mean_err = torch.mean(target, dim=-1) - torch.mean(output, dim=-1)
    loss = torch.mean(mean_err ** 2)
    energy = torch.mean(target ** 2) + eps
    return loss / energy


def training_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    esr_pre = esr_loss(pre_emphasis(output), pre_emphasis(target))
    dc = dc_loss(output, target)
    return 0.75 * esr_pre + 0.25 * dc


def build_paths(effect: str, setting: int, run_name: str | None):
    if effect == "starcaster":
        folder = ROOT / "data" / "processed" / "starcaster" / str(setting)
        prefix = f"starcaster{setting}"
        default_run = f"starcaster{setting}_tcn300c_scratch_v1-"
    else:
        folder = ROOT / "data" / "processed" / "mxrdynacomp" / str(setting)
        prefix = f"mxrdynacomp{setting}"
        default_run = f"mxrdynacomp{setting}_tcn300c_scratch_v1-"

    run_dir = ROOT / "results" / (run_name or default_run)
    return folder, prefix, run_dir


def load_split(folder: Path, prefix: str, split: str):
    input_path = folder / split / f"{prefix}-input.wav"
    target_path = folder / split / f"{prefix}-target.wav"

    fs_x, x = read_mono_wav(input_path)
    fs_y, y = read_mono_wav(target_path)

    if fs_x != fs_y:
        raise ValueError(f"Sample-rate mismatch in {split}: {fs_x} vs {fs_y}")
    if len(x) != len(y):
        raise ValueError(f"Length mismatch in {split}: {len(x)} vs {len(y)}")

    return fs_x, x, y


def make_context_window(
    x: np.ndarray,
    y: np.ndarray,
    start: int,
    segment_length: int,
    context: int,
) -> tuple[np.ndarray, np.ndarray]:
    end = start + segment_length
    y_seg = y[start:end]

    left = start - context
    if left >= 0:
        x_seg = x[left:end]
    else:
        pad = np.zeros(-left, dtype=np.float32)
        x_seg = np.concatenate([pad, x[0:end]])

    expected = context + len(y_seg)
    if len(x_seg) != expected:
        raise RuntimeError(
            f"Unexpected context-window length: {len(x_seg)} != {expected}"
        )

    return x_seg, y_seg


def iter_training_batches(
    x: np.ndarray,
    y: np.ndarray,
    segment_length: int,
    context: int,
    batch_size: int,
):
    starts = list(range(0, len(y) - segment_length + 1, segment_length))
    if not starts:
        raise ValueError(
            f"Training file ({len(y)} samples) is shorter than segment length "
            f"({segment_length})."
        )

    random.shuffle(starts)

    for batch_start in range(0, len(starts), batch_size):
        selected = starts[batch_start : batch_start + batch_size]
        xs, ys = [], []
        for start in selected:
            x_seg, y_seg = make_context_window(
                x, y, start, segment_length, context
            )
            xs.append(x_seg)
            ys.append(y_seg)

        xb = torch.from_numpy(np.stack(xs)).unsqueeze(1)
        yb = torch.from_numpy(np.stack(ys)).unsqueeze(1)
        yield xb, yb


@torch.inference_mode()
def infer_full(
    model: CausalTCN300,
    x: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    model.eval()
    context = model.receptive_field - 1
    outputs = []

    for start in range(0, len(x), chunk_size):
        end = min(start + chunk_size, len(x))
        y_dummy = np.zeros(end - start, dtype=np.float32)
        x_seg, _ = make_context_window(
            x,
            y_dummy if start == 0 else np.zeros(len(x), dtype=np.float32),
            start,
            end - start,
            context,
        )

        # The helper above expects y indexed on the same global axis. For chunks
        # after zero, build the input context explicitly to avoid allocating
        # target content.
        if start > 0:
            left = start - context
            if left >= 0:
                x_seg = x[left:end]
            else:
                x_seg = np.concatenate(
                    [np.zeros(-left, dtype=np.float32), x[:end]]
                )

        xb = torch.from_numpy(
            np.ascontiguousarray(x_seg, dtype=np.float32)
        ).view(1, 1, -1).to(device)

        pred = model(xb)
        expected = end - start
        if pred.shape[-1] != expected:
            raise RuntimeError(
                f"Inference length mismatch: got {pred.shape[-1]}, "
                f"expected {expected}"
            )
        outputs.append(pred.squeeze(0).squeeze(0).cpu())

    return torch.cat(outputs, dim=0)


def evaluate_signal(
    model: CausalTCN300,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    chunk_size: int,
):
    pred = infer_full(model, x, device, chunk_size)
    target = torch.from_numpy(y).float()

    pred_bct = pred.view(1, 1, -1)
    target_bct = target.view(1, 1, -1)

    composite = training_loss(pred_bct, target_bct).item()
    esr = esr_loss(pred_bct, target_bct).item()
    return pred.numpy(), composite, esr


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: Path,
    model: CausalTCN300,
    args: argparse.Namespace,
    epoch: int,
    best_val: float,
):
    payload = {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val,
        "model_config": {
            "architecture": "TCN300CInspired",
            "channels": args.channels,
            "kernel_size": args.kernel_size,
            "dilations": list(args.dilations),
            "causal": True,
            "conditioning": False,
            "output_activation": "tanh",
        },
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    folder, prefix, run_dir = build_paths(
        args.effect, args.setting, args.run_name
    )
    fs_train, x_train, y_train = load_split(folder, prefix, "train")
    fs_val, x_val, y_val = load_split(folder, prefix, "val")
    fs_test, x_test, y_test = load_split(folder, prefix, "test")

    if not (fs_train == fs_val == fs_test):
        raise ValueError(
            f"Split sample rates differ: {fs_train}, {fs_val}, {fs_test}"
        )

    model = CausalTCN300(
        channels=args.channels,
        kernel_size=args.kernel_size,
        dilations=tuple(args.dilations),
    ).to(device)

    rf = model.receptive_field
    context = rf - 1
    params = count_parameters(model)

    print("=" * 72)
    print("TCN thesis baseline")
    print(f"Effect / setting : {args.effect} / {args.setting}")
    print(f"Device           : {device}")
    print(f"Sample rate      : {fs_train} Hz")
    print(f"Channels         : {args.channels}")
    print(f"Kernel size      : {args.kernel_size}")
    print(f"Dilations        : {list(args.dilations)}")
    print(f"Receptive field  : {rf} samples ({1000 * rf / fs_train:.2f} ms)")
    print(f"Trainable params : {params:,}")
    print(f"Run directory    : {run_dir}")
    print("=" * 72)

    if args.smoke_test:
        model.train()
        xb, yb = next(
            iter_training_batches(
                x_train,
                y_train,
                args.segment_length,
                context,
                min(args.batch_size, 2),
            )
        )
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        loss = training_loss(out, yb)
        loss.backward()

        print(f"Smoke input      : {tuple(xb.shape)}")
        print(f"Smoke output     : {tuple(out.shape)}")
        print(f"Smoke target     : {tuple(yb.shape)}")
        print(f"Smoke loss       : {loss.item():.6f}")
        print("SMOKE TEST OK")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "architecture": "TCN300CInspired",
        "effect": args.effect,
        "setting": args.setting,
        "sample_rate": fs_train,
        "channels": args.channels,
        "kernel_size": args.kernel_size,
        "dilations": list(args.dilations),
        "receptive_field_samples": rf,
        "receptive_field_ms": 1000 * rf / fs_train,
        "causal": True,
        "conditioning": False,
        "loss": {
            "ESRPre": 0.75,
            "DC": 0.25,
            "pre_filter": [-0.85, 1.0],
        },
        "segment_length": args.segment_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "trainable_parameters": params,
    }
    with open(run_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    stats = {
        "current_epoch": 0,
        "training_losses": [],
        "validation_losses": [],
        "best_val_loss": float("inf"),
        "best_epoch": None,
        "total_time": 0.0,
        "test_loss_final": None,
        "test_lossESR_final": None,
        "test_loss_best": None,
        "test_lossESR_best": None,
    }

    best_path = run_dir / "model_best.pt"
    final_path = run_dir / "model_final.pt"
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []

        for xb, yb in iter_training_batches(
            x_train,
            y_train,
            args.segment_length,
            context,
            args.batch_size,
        ):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            if out.shape != yb.shape:
                raise RuntimeError(
                    f"Training shape mismatch: output={out.shape}, "
                    f"target={yb.shape}"
                )

            loss = training_loss(out, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses))

        _, val_loss, _ = evaluate_signal(
            model, x_val, y_val, device, args.inference_chunk
        )
        scheduler.step(val_loss)

        stats["current_epoch"] = epoch
        stats["training_losses"].append(train_loss)
        stats["validation_losses"].append(float(val_loss))
        stats["total_time"] = (time.time() - start_time) / 3600.0

        improved = val_loss < stats["best_val_loss"]
        if improved:
            stats["best_val_loss"] = float(val_loss)
            stats["best_epoch"] = epoch
            patience_counter = 0
            save_checkpoint(best_path, model, args, epoch, float(val_loss))
        else:
            patience_counter += 1

        with open(run_dir / "training_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        lr = optimizer.param_groups[0]["lr"]
        marker = " *BEST*" if improved else ""
        print(
            f"Epoch {epoch:03d} | train={train_loss:.6f} | "
            f"val={val_loss:.6f} | lr={lr:.6g} | "
            f"patience={patience_counter}/{args.patience}{marker}"
        )

        if args.patience and patience_counter > args.patience:
            print("Early stopping: validation patience limit reached.")
            break

    save_checkpoint(
        final_path,
        model,
        args,
        stats["current_epoch"],
        float(stats["best_val_loss"]),
    )

    final_pred, final_loss, final_esr = evaluate_signal(
        model, x_test, y_test, device, args.inference_chunk
    )
    write_float_wav(run_dir / "test_out_final.wav", fs_test, final_pred)
    stats["test_loss_final"] = float(final_loss)
    stats["test_lossESR_final"] = float(final_esr)

    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])

    best_pred, best_loss, best_esr = evaluate_signal(
        model, x_test, y_test, device, args.inference_chunk
    )
    write_float_wav(run_dir / "test_out_bestv.wav", fs_test, best_pred)

    val_pred = infer_full(model, x_val, device, args.inference_chunk).numpy()
    write_float_wav(run_dir / "val_out_bestv.wav", fs_val, val_pred)

    stats["test_loss_best"] = float(best_loss)
    stats["test_lossESR_best"] = float(best_esr)
    stats["best_val_loss"] = float(best_ckpt["best_val_loss"])
    stats["best_epoch"] = int(best_ckpt["epoch"])

    with open(run_dir / "training_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(device))
        with open(run_dir / "maxmemusage.txt", "w", encoding="utf-8") as f:
            f.write(str(peak))

    print("=" * 72)
    print("TRAINING COMPLETE")
    print(f"Best epoch       : {stats['best_epoch']}")
    print(f"Best val loss    : {stats['best_val_loss']:.6f}")
    print(f"Test loss best   : {stats['test_loss_best']:.6f}")
    print(f"Test ESR best    : {stats['test_lossESR_best']:.6f}")
    print(f"Output           : {run_dir / 'test_out_bestv.wav'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
