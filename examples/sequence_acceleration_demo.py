"""Demo: sequence acceleration -- pi and ln 2 from a handful of terms.

Sums the glacially slow Leibniz series for pi and accelerates it with Aitken and Wynn's epsilon
algorithm, showing a dozen terms beat a million raw ones. Draws the error-vs-terms curves for the raw
sum and the accelerated estimates.

    python examples/sequence_acceleration_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sequence_acceleration import (  # noqa: E402
    aitken_iterated,
    wynn_epsilon,
    partial_sums,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Sequence acceleration: extracting a limit from a slowly-converging series\n")

    def leibniz(k):
        return (-1) ** k / (2 * k + 1)  # sum = pi/4

    print("  Leibniz series 1 - 1/3 + 1/5 - ... = pi/4, accelerating pi:\n")
    print(f"  {'terms':>6}  {'raw sum error':>15}  {'Aitken error':>14}  {'Wynn error':>12}")
    curve = []
    for nterms in [5, 10, 15, 20, 25, 30]:
        raw = partial_sums(leibniz, nterms)
        raw_pi = raw[-1] * 4
        ait = aitken_iterated(raw) * 4
        wyn = wynn_epsilon(raw) * 4
        e_raw = abs(raw_pi - math.pi)
        e_ait = abs(ait - math.pi)
        e_wyn = abs(wyn - math.pi)
        curve.append((nterms, e_raw, e_ait, e_wyn))
        print(f"  {nterms:>6}  {e_raw:>15.2e}  {e_ait:>14.2e}  {e_wyn:>12.2e}")

    print("\n  30 raw terms give ~2 digits of pi; Wynn's epsilon algorithm on the same 30 partial")
    print("  sums gives full machine precision. The raw sum would need ~10^15 terms to match it.\n")

    # ln 2 as well
    def ln2(k):
        return (-1) ** k / (k + 1)
    raw = partial_sums(ln2, 25)
    print(f"  ln 2 series (25 terms): raw error {abs(raw[-1]-math.log(2)):.2e}, "
          f"Wynn error {abs(wynn_epsilon(raw)-math.log(2)):.2e}")

    _svg(os.path.join(outdir, "sequence_acceleration.svg"), curve)
    print(f"\n  wrote {os.path.join(outdir, 'sequence_acceleration.svg')}")


def _svg(path, curve, width=760, height=410):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Error in pi vs number of Leibniz terms (log scale): raw, Aitken, Wynn</text>',
    ]
    ox, oy, ow, oh = 60, 55, width - 100, height - 110
    ns = [c[0] for c in curve]
    # log10 errors, floored so machine-precision doesn't blow the axis
    def logf(e):
        return math.log10(max(e, 1e-16))
    series = {
        "raw": ("#ff6b6b", [logf(c[1]) for c in curve]),
        "Aitken": ("#ffd43b", [logf(c[2]) for c in curve]),
        "Wynn": ("#06d6a0", [logf(c[3]) for c in curve]),
    }
    allv = [v for _, ys in series.values() for v in ys]
    ymin, ymax = min(allv), max(allv)

    def px(i):
        return ox + ow * (ns[i] - ns[0]) / (ns[-1] - ns[0])

    def py(v):
        return oy + oh * (1 - (v - ymin) / (ymax - ymin or 1))

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    ly = oy + 12
    for name, (col, ys) in series.items():
        pts = " ".join(f"{px(i):.1f},{py(ys[i]):.1f}" for i in range(len(ns)))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        for i in range(len(ns)):
            parts.append(f'<circle cx="{px(i):.1f}" cy="{py(ys[i]):.1f}" r="2.5" fill="{col}"/>')
        parts.append(f'<text x="{ox+ow-90}" y="{ly}" fill="{col}" font-size="10">{name}</text>')
        ly += 15
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+22:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">number of terms</text>')
    parts.append(f'<text x="{ox-6}" y="{oy-4}" fill="#8b949e" font-size="10" text-anchor="end">'
                 f'log10 |error|</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
