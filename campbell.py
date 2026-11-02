"""Conservative 1P and blade-pass modal separation check with standalone SVG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from surrogate import predict


def check_design(model: dict[str, Any], design: dict[str, Any],
                 base_margin_fraction: float = 0.15) -> dict[str, Any]:
    frequencies = predict(model, design)
    rpm_min, rpm_max = float(design["rpm_min"]), float(design["rpm_max"])
    blades = int(design["blades"])
    if not (0 <= rpm_min < rpm_max) or blades < 1:
        raise ValueError("require 0 <= rpm_min < rpm_max and blades >= 1")
    if not (0 < base_margin_fraction < 1):
        raise ValueError("margin must be between 0 and 1")
    p95 = model["held_out_metrics"]["p95_ape_percent_by_mode"]
    if len(p95) != len(frequencies):
        raise ValueError("model error bound and target count differ")
    excitations = {"1P": 1, "blade_pass": blades}
    checks = []
    for mode_number, (frequency, p95_percent) in enumerate(zip(frequencies, p95), 1):
        for name, multiplier in excitations.items():
            low, high = rpm_min * multiplier / 60, rpm_max * multiplier / 60
            closest = min(max(frequency, low), high)
            observed_gap = abs(frequency - closest) / frequency
            required_gap = base_margin_fraction + float(p95_percent) / 100
            checks.append({
                "mode": mode_number, "excitation": name,
                "excitation_range_hz": [low, high],
                "closest_excitation_hz": closest,
                "minimum_separation_fraction": observed_gap,
                "required_separation_fraction": required_gap,
                "passes": bool(observed_gap > required_gap),
            })
    return {
        "source": model["source"], "predicted_frequencies_hz": frequencies,
        "rpm_range": [rpm_min, rpm_max], "blades": blades,
        "base_margin_fraction": base_margin_fraction,
        "held_out_p95_error_percent_by_mode": p95,
        "checks": checks, "passes": all(item["passes"] for item in checks),
        "interpretation": "Frequency separation screen only; modal amplitude and damping are not predicted.",
    }


def render_svg(report: dict[str, Any]) -> str:
    """Dependency-free Campbell diagram. Hz on Y, RPM on X."""
    min_rpm, max_rpm = report["rpm_range"]
    freqs = report["predicted_frequencies_hz"]
    blades = report["blades"]
    max_hz = max(max(freqs) * 1.25, max_rpm * blades / 60 * 1.1)
    width, height = 880, 540
    left, right, top, bottom = 75, 625, 55, 475
    legend_x = 660
    def x(value: float) -> float:
        return left + (value - min_rpm) / (max_rpm - min_rpm) * (right - left)
    def y(value: float) -> float:
        return bottom - value / max_hz * (bottom - top)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="75" y="30" font-family="sans-serif" font-size="20" font-weight="bold">Campbell frequency separation screen</text>',
             f'<text x="75" y="50" font-family="sans-serif" font-size="12" fill="#555">{"CalculiX concept" if report.get("source") == "calculix_ccx_native" else "Local beam FEM"} surrogate · {"PASS" if report["passes"] else "REVIEW"}</text>',
             f'<path d="M{left},{top}V{bottom}H{right}" stroke="#333" fill="none" stroke-width="2"/>']
    for step in range(5):
        freq = max_hz * step / 4
        py = y(freq)
        parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{right}" y2="{py:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-8}" y="{py+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{freq:.0f}</text>')
    for step in range(5):
        rpm = min_rpm + (max_rpm - min_rpm) * step / 4
        px = x(rpm)
        parts.append(f'<text x="{px:.1f}" y="{bottom+20}" text-anchor="middle" font-family="sans-serif" font-size="12">{rpm:.0f}</text>')
    parts += [f'<text x="{(left+right)/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14">Rotor speed (RPM)</text>',
              f'<text transform="translate(20 {(top+bottom)/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">Frequency (Hz)</text>',
              '<line x1="642" y1="72" x2="642" y2="475" stroke="#e5e7eb"/>',
              f'<text x="{legend_x}" y="91" font-family="sans-serif" font-size="14" font-weight="bold">Legend</text>']
    for mode, (frequency, error) in enumerate(zip(freqs, report["held_out_p95_error_percent_by_mode"]), 1):
        low, high = frequency * (1-error/100), frequency * (1+error/100)
        parts.append(f'<rect x="{left}" y="{y(high):.2f}" width="{right-left}" height="{y(low)-y(high):.2f}" fill="#3b82f6" opacity="0.08"/>')
        parts.append(f'<line x1="{left}" y1="{y(frequency):.2f}" x2="{right}" y2="{y(frequency):.2f}" stroke="#2563eb" stroke-width="2"/>')
        legend_y = 116 + 28 * (mode - 1)
        parts.append(f'<line x1="{legend_x}" y1="{legend_y-4}" x2="{legend_x+22}" y2="{legend_y-4}" stroke="#2563eb" stroke-width="2"/>')
        parts.append(f'<text x="{legend_x+30}" y="{legend_y}" font-family="sans-serif" font-size="12" fill="#1d4ed8">Mode {mode}: {frequency:.1f} Hz</text>')
    for index, (name, multiplier, color) in enumerate((("1P", 1, "#e11d48"), ("blade pass", blades, "#f97316"))):
        parts.append(f'<line x1="{left}" y1="{y(min_rpm*multiplier/60):.2f}" x2="{right}" y2="{y(max_rpm*multiplier/60):.2f}" stroke="{color}" stroke-width="3"/>')
        legend_y = 116 + 28 * (len(freqs) + index + 1)
        parts.append(f'<line x1="{legend_x}" y1="{legend_y-4}" x2="{legend_x+22}" y2="{legend_y-4}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+30}" y="{legend_y}" font-family="sans-serif" font-size="12" fill="{color}">{name}</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output directory for check.json and campbell.svg")
    parser.add_argument("--margin", type=float, default=0.15)
    args = parser.parse_args()
    report = check_design(json.loads(args.model.read_text()), json.loads(args.design.read_text()), args.margin)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "check.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "campbell.svg").write_text(render_svg(report))
    print(json.dumps({"passes": report["passes"], "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
