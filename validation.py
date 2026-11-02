"""Read impact-test accelerometer data and compare peaks with a modal prediction.

Input CSV has a header containing ``time_s`` and ``acceleration_m_s2``.
The analysis is intentionally small and auditable: it reports spectral peaks,
not a claim that every peak is a structural mode.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def impact_spectrum(path: str | Path, min_hz: float = 5.0, max_hz: float = 1000.0,
                    max_peaks: int = 12) -> dict:
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"time_s", "acceleration_m_s2"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires time_s and acceleration_m_s2 columns")
        rows = list(reader)
    if len(rows) < 128:
        raise ValueError("At least 128 samples are required")
    time = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
    signal = np.asarray([float(row["acceleration_m_s2"]) for row in rows], dtype=float)
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(signal)):
        raise ValueError("CSV contains non-finite values")
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("time_s must increase strictly")
    sample_dt = float(np.mean(dt))
    if float(np.max(np.abs(dt - sample_dt)) / sample_dt) > 0.01:
        raise ValueError("Samples must be uniformly spaced within 1%; resample first")
    nyquist = 0.5 / sample_dt
    if not 0 <= min_hz < max_hz <= nyquist:
        raise ValueError(f"Frequency window must lie below Nyquist ({nyquist:.3f} Hz)")
    detrended = signal - np.mean(signal)
    fft = np.fft.rfft(detrended * np.hanning(len(signal)))
    frequency = np.fft.rfftfreq(len(signal), sample_dt)
    amplitude = np.abs(fft)
    eligible = np.where((frequency >= min_hz) & (frequency <= max_hz))[0]
    local = [int(i) for i in eligible if 0 < i < len(amplitude)-1
             and amplitude[i] > amplitude[i-1] and amplitude[i] >= amplitude[i+1]]
    local.sort(key=lambda i: amplitude[i], reverse=True)
    strongest = local[:max_peaks]
    return {
        "sample_rate_hz": 1.0 / sample_dt,
        "duration_s": float(time[-1] - time[0]),
        "frequency_resolution_hz": float(1.0 / (len(signal) * sample_dt)),
        "peaks": [{"frequency_hz": float(frequency[i]),
                   "relative_amplitude": float(amplitude[i] / max(amplitude))}
                  for i in strongest],
    }


def compare_modal_peaks(predicted_hz: list[float], measured_peaks: list[dict],
                        max_difference_fraction: float = 0.10) -> list[dict]:
    """One-to-one nearest peak matching; unmatched predictions remain explicit."""
    remaining = {i for i in range(len(measured_peaks))}
    output = []
    for mode, predicted in enumerate(predicted_hz, 1):
        if predicted <= 0:
            raise ValueError("Predicted flexible frequencies must be positive")
        candidates = sorted(remaining,
                            key=lambda i: abs(measured_peaks[i]["frequency_hz"]-predicted))
        match = candidates[0] if candidates else None
        measured = measured_peaks[match]["frequency_hz"] if match is not None else None
        error = abs(measured-predicted)/predicted if measured is not None else None
        accepted = bool(error is not None and error <= max_difference_fraction)
        if accepted:
            remaining.remove(match)
        output.append({"mode": mode, "predicted_hz": float(predicted),
                       "measured_hz": float(measured) if accepted else None,
                       "relative_error": float(error) if accepted else None,
                       "status": "matched" if accepted else "unidentified"})
    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--min-hz", type=float, default=5.0)
    parser.add_argument("--max-hz", type=float, default=1000.0)
    parser.add_argument("--predicted-hz", nargs="*", type=float, default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = impact_spectrum(args.csv_path, args.min_hz, args.max_hz)
    if args.predicted_hz:
        report["comparison"] = compare_modal_peaks(args.predicted_hz, report["peaks"])
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
