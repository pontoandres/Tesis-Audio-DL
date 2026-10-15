import argparse
import os
import numpy as np
from scipy.io import wavfile


def to_float(x):
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        scale = max(abs(info.min), abs(info.max))
        return x.astype(np.float64) / scale

    return x.astype(np.float64)


def get_level_stats(x):
    x = to_float(x)

    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))

    rms_db = 20 * np.log10(rms + 1e-12)
    peak_db = 20 * np.log10(peak + 1e-12)

    clipping = int(np.sum(np.abs(x) >= 0.999))

    return {
        "rms": rms,
        "rms_db": rms_db,
        "peak": peak,
        "peak_db": peak_db,
        "clipping": clipping
    }


def analyze(path):
    fs, raw = wavfile.read(path)

    original_shape = raw.shape

    if raw.ndim == 1:
        channels = 1
        is_mono = True

    elif raw.ndim == 2:
        channels = raw.shape[1]
        is_mono = channels == 1

    else:
        raise ValueError(f"Unexpected audio shape: {raw.shape}")

    duration = len(raw) / fs

    # Overall stats.
    # For stereo this combines both channels only as a rough global number.
    overall_stats = get_level_stats(raw.reshape(-1))

    channel_stats = []

    if raw.ndim == 1:
        stats = get_level_stats(raw)

        channel_stats.append({
            "channel": 0,
            **stats
        })

    else:
        for ch in range(raw.shape[1]):
            stats = get_level_stats(raw[:, ch])

            channel_stats.append({
                "channel": ch,
                **stats
            })

    stereo_info = None

    if raw.ndim == 2 and raw.shape[1] == 2:
        left = to_float(raw[:, 0])
        right = to_float(raw[:, 1])

        left_centered = left - np.mean(left)
        right_centered = right - np.mean(right)

        denom = np.sqrt(
            np.sum(left_centered ** 2)
            * np.sum(right_centered ** 2)
        )

        if denom > 0:
            correlation = np.sum(
                left_centered * right_centered
            ) / denom
        else:
            correlation = 0.0

        difference = left - right

        diff_rms = np.sqrt(np.mean(difference ** 2))
        diff_rms_db = 20 * np.log10(diff_rms + 1e-12)

        rms_difference_db = (
            channel_stats[1]["rms_db"]
            - channel_stats[0]["rms_db"]
        )

        stereo_info = {
            "correlation": correlation,
            "diff_rms_db": diff_rms_db,
            "rms_difference_db": rms_difference_db
        }

    return {
        "path": path,
        "fs": fs,
        "shape": original_shape,
        "channels": channels,
        "is_mono": is_mono,
        "duration": duration,
        "samples": len(raw),
        "rms": overall_stats["rms"],
        "rms_db": overall_stats["rms_db"],
        "peak": overall_stats["peak"],
        "peak_db": overall_stats["peak_db"],
        "clipping": overall_stats["clipping"],
        "channel_stats": channel_stats,
        "stereo_info": stereo_info
    }


def print_info(name, data):
    print()
    print("=" * 65)
    print(name)
    print("=" * 65)

    print(f"File:        {data['path']}")
    print(f"Sample rate: {data['fs']} Hz")
    print(f"Shape:       {data['shape']}")
    print(f"Channels:    {data['channels']}")

    if data["is_mono"]:
        print("MONO CHECK:  PASS - MONO")
    else:
        print("MONO CHECK:  FAIL - NOT MONO")

    print(f"Duration:    {data['duration']:.3f} s")
    print(f"Samples:     {data['samples']}")
    print(f"Overall RMS: {data['rms_db']:.2f} dBFS")
    print(f"Overall peak:{data['peak_db']:.2f} dBFS")
    print(f"Clipping:    {data['clipping']} samples")

    print()
    print("PER-CHANNEL LEVELS")

    for ch in data["channel_stats"]:
        print(
            f"Channel {ch['channel']}: "
            f"RMS {ch['rms_db']:.2f} dBFS | "
            f"Peak {ch['peak_db']:.2f} dBFS | "
            f"Clipping {ch['clipping']}"
        )

    if data["stereo_info"] is not None:
        info = data["stereo_info"]

        print()
        print("STEREO CHANNEL COMPARISON")

        print(
            f"L/R correlation:       "
            f"{info['correlation']:+.6f}"
        )

        print(
            f"RMS difference R - L:  "
            f"{info['rms_difference_db']:+.2f} dB"
        )

        print(
            f"L-R difference RMS:    "
            f"{info['diff_rms_db']:.2f} dBFS"
        )

        print()

        if (
            info["correlation"] > 0.999
            and abs(info["rms_difference_db"]) < 0.1
        ):
            print(
                "STEREO DIAGNOSIS: channels are effectively "
                "identical / duplicated mono."
            )
            print(
                "Keeping either channel as mono should be safe."
            )

        elif abs(info["rms_difference_db"]) > 20:
            print(
                "STEREO DIAGNOSIS: one channel is much quieter "
                "than the other."
            )
            print(
                "Use the stronger channel; do NOT average blindly."
            )

        else:
            print(
                "STEREO DIAGNOSIS: channels are not obviously identical."
            )
            print(
                "Inspect before converting to mono."
            )


def save_mono_copy(input_path, output_path, channel):
    fs, raw = wavfile.read(input_path)

    if raw.ndim == 1:
        print()
        print("Target is already mono.")
        print("No conversion necessary.")
        return

    if raw.ndim != 2:
        raise ValueError(
            f"Cannot convert unexpected audio shape: {raw.shape}"
        )

    if channel < 0 or channel >= raw.shape[1]:
        raise ValueError(
            f"Requested channel {channel}, "
            f"but file has {raw.shape[1]} channels."
        )

    mono = raw[:, channel]

    folder = os.path.dirname(output_path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    wavfile.write(
        output_path,
        fs,
        mono
    )

    print()
    print("=" * 65)
    print("MONO COPY SAVED")
    print("=" * 65)
    print(f"Source:  {input_path}")
    print(f"Channel: {channel}")
    print(f"Output:  {output_path}")
    print(f"Shape:   {mono.shape}")
    print(f"Dtype:   {mono.dtype}")
    print()
    print(
        "Original file was NOT modified."
    )


parser = argparse.ArgumentParser()

parser.add_argument(
    "input",
    help="Dry/input WAV"
)

parser.add_argument(
    "target",
    help="Pedal/target WAV"
)

parser.add_argument(
    "--save-target-mono",
    default=None,
    help=(
        "Optional output WAV path for a mono copy "
        "of the target"
    )
)

parser.add_argument(
    "--mono-channel",
    type=int,
    default=0,
    help=(
        "Channel to keep when saving mono. "
        "0 = first/left, 1 = second/right. Default: 0"
    )
)

args = parser.parse_args()


input_data = analyze(args.input)
target_data = analyze(args.target)


print_info(
    "INPUT / DRY",
    input_data
)

print_info(
    "TARGET / PEDAL",
    target_data
)


print()
print("=" * 65)
print("COMPARISON")
print("=" * 65)

# If target is stereo, channel 0 is shown separately because
# the overall stereo RMS may be misleading.
target_reference_rms = target_data["channel_stats"][0]["rms_db"]
target_reference_peak = target_data["channel_stats"][0]["peak_db"]

input_reference_rms = input_data["channel_stats"][0]["rms_db"]
input_reference_peak = input_data["channel_stats"][0]["peak_db"]

print(
    f"Target Ch0 - Input RMS:  "
    f"{target_reference_rms - input_reference_rms:+.2f} dB"
)

print(
    f"Target Ch0 - Input peak: "
    f"{target_reference_peak - input_reference_peak:+.2f} dB"
)


print()
print("=" * 65)
print("SANITY CHECK")
print("=" * 65)

ok = True


if input_data["fs"] != target_data["fs"]:
    print(
        f"FAIL: sample rates differ "
        f"({input_data['fs']} vs {target_data['fs']})"
    )
    ok = False

else:
    print(
        f"PASS: same sample rate "
        f"({input_data['fs']} Hz)"
    )


if not input_data["is_mono"]:
    print("FAIL: input is not mono")
    ok = False

else:
    print("PASS: input is mono")


if not target_data["is_mono"]:
    print(
        "WARNING: target is stereo - "
        "inspect channel comparison above"
    )
    ok = False

else:
    print("PASS: target is mono")


if input_data["clipping"] > 0:
    print(
        "WARNING: input contains samples at/near clipping"
    )

else:
    print(
        "PASS: no input clipping detected"
    )


if target_data["clipping"] > 0:
    print(
        "WARNING: target contains samples at/near clipping"
    )

else:
    print(
        "PASS: no target clipping detected"
    )


print()

if ok:
    print(
        "OVERALL: BASIC FORMAT CHECK PASSED"
    )

else:
    print(
        "OVERALL: CHECK NEEDS ATTENTION"
    )


# Optional mono conversion
if args.save_target_mono is not None:
    save_mono_copy(
        args.target,
        args.save_target_mono,
        args.mono_channel
    )

print()