from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags


ROOT = Path(__file__).resolve().parents[1]

PEDAL = "mxrdynacomp"

CONFIGS = (
    "25",
    "50",
    "75",
    "100",
)

SPLITS = (
    "train",
    "val",
    "test",
)


CURRENT_DELAY = {
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


TARGET_RESIDUAL = 128

WINDOW_S = 8

EXPECTED_LAG = 128
SEARCH_RADIUS = 48

POSITIONS = [
    0.05,
    0.25,
    0.50,
    0.75,
    0.95,
]


def mono_float(x):

    if x.ndim > 1:
        x = x[:, 0]

    if x.dtype == np.int16:
        return (
            x.astype(np.float64)
            / 32768.0
        )

    if x.dtype == np.int32:
        return (
            x.astype(np.float64)
            / 2147483648.0
        )

    return x.astype(np.float64)


def check(config, split):

    name = (
        f"MXR Dyna Comp "
        f"{config}% {split.upper()}"
    )

    base = (
        ROOT
        / "data"
        / "processed"
        / PEDAL
        / config
        / split
    )

    in_path = (
        base
        / f"{PEDAL}{config}-input.wav"
    )

    target_path = (
        base
        / f"{PEDAL}{config}-target.wav"
    )

    fs1, x = wavfile.read(
        in_path,
        mmap=True
    )

    fs2, y = wavfile.read(
        target_path,
        mmap=True
    )

    if fs1 != fs2:
        raise ValueError(
            f"{name}: sample rates differ: "
            f"{fs1} vs {fs2}"
        )

    n = min(
        len(x),
        len(y)
    )

    win = int(
        WINDOW_S * fs1
    )


    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Sample rate: {fs1}"
    )

    print(
        f"Duration: "
        f"{n / fs1:.3f} s"
    )

    print()

    print(
        "position_s   lag_samples   "
        "lag_ms      corr"
    )

    measured_lags = []

    for frac in POSITIONS:

        start = int(
            frac
            * max(
                0,
                n - win
            )
        )

        a = mono_float(
            np.asarray(
                x[start:start + win]
            )
        ).copy()

        b = mono_float(
            np.asarray(
                y[start:start + win]
            )
        ).copy()

        a -= np.mean(a)
        b -= np.mean(b)

        c = correlate(
            b,
            a,
            mode="full",
            method="fft"
        )

        lags = correlation_lags(
            len(b),
            len(a),
            mode="full"
        )

        mask = (
            (lags >= EXPECTED_LAG - SEARCH_RADIUS)
            & (lags <= EXPECTED_LAG + SEARCH_RADIUS)
        )

        c2 = c[mask]
        l2 = lags[mask]

        # IMPORTANTE:
        # no asumimos inversión ni no-inversión
        # de polaridad.
        # Elegimos el máximo por magnitud.
        j = int(
            np.argmax(
                np.abs(c2)
            )
        )

        lag = int(
            l2[j]
        )

        denom = (
            np.sqrt(
                np.sum(a * a)
                * np.sum(b * b)
            )
            + 1e-15
        )

        corr = float(
            c2[j]
            / denom
        )

        measured_lags.append(
            lag
        )

        print(
            f"{start/fs1:10.2f}   "
            f"{lag:11d}   "
            f"{1000*lag/fs1:8.3f}   "
            f"{corr:+.4f}"
        )

    median_lag = int(
        np.median(
            measured_lags
        )
    )

    current = CURRENT_DELAY[
        (config, split)
    ]

    suggested = (
        current
        + (
            TARGET_RESIDUAL
            - median_lag
        )
    )

    print()
    print(
        f"Lags: {measured_lags}"
    )

    print(
        f"Median residual lag: "
        f"{median_lag} samples"
    )

    print(
        f"Current causal delay: "
        f"{current}"
    )

    print(
        f"Target residual lag: "
        f"{TARGET_RESIDUAL}"
    )

    print(
        f"SUGGESTED CAUSAL DELAY: "
        f"{suggested}"
    )

    return {
        "config": config,
        "split": split,
        "lags": measured_lags,
        "median": median_lag,
        "current": current,
        "suggested": suggested,
    }


def main():

    results = []

    for config in CONFIGS:
        for split in SPLITS:

            result = check(
                config,
                split
            )

            results.append(
                result
            )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print()
    print(
        "config   split   median   "
        "current   suggested"
    )

    for r in results:

        print(
            f"{r['config']:>5}   "
            f"{r['split']:<5}   "
            f"{r['median']:>6}   "
            f"{r['current']:>7}   "
            f"{r['suggested']:>9}"
        )

    print()
    print(
        "Copy the suggested values into "
        "CAUSAL_DELAY in align_dataset_dynacomp.py, "
        "rerun preprocessing, and verify again."
    )


if __name__ == "__main__":
    main()