"""Demo: Box-Muller -- uniform randomness into a bell curve.

Generates standard normals from uniforms, prints their moments and the 68-95-99.7 rule, and
draws the sample histogram against the analytic Gaussian density with the 1/2/3-sigma bands.

    python examples/box_muller_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from box_muller import (standard_normals, polar_normals, sample,  # noqa: E402
                        mean, variance, skewness, kurtosis, fraction_within)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    z = standard_normals(200000, seed=1)
    print("Box-Muller: two uniforms -> two independent standard normals\n")
    print(f"  sample of {len(z)} standard normals:")
    print(f"    mean     = {mean(z):+.4f}  (target 0)")
    print(f"    variance = {variance(z):.4f}  (target 1)")
    print(f"    skewness = {skewness(z):+.4f}  (target 0)")
    print(f"    kurtosis = {kurtosis(z):.4f}  (target 3)")
    print(f"\n  68-95-99.7 rule:")
    for k, tgt in ((1, 0.6827), (2, 0.9545), (3, 0.9973)):
        print(f"    within {k} sigma: {fraction_within(z, k):.4f}  (normal: {tgt})")

    p = polar_normals(50000, seed=2)
    print(f"\n  Marsaglia polar method (no trig): mean {mean(p):+.4f}, var {variance(p):.4f}, "
          f"kurtosis {kurtosis(p):.3f} -- same distribution, faster.")
    s = sample(50000, mu=100, sigma=15, seed=4)
    print(f"  scaled to N(100, 15^2) (e.g. IQ scores): mean {mean(s):.1f}, "
          f"sd {math.sqrt(variance(s)):.1f}")
    print("\n  The transform is exact: r = sqrt(-2 ln u1) makes r^2 exponential (the Gaussian")
    print("  radius), theta = 2 pi u2 is the uniform angle, and (r cos, r sin) is a 2D normal.")
    print("  Every simulation that needs noise -- Brownian motion, Monte-Carlo finance -- starts here.")

    _svg(os.path.join(outdir, "box_muller.svg"), z)
    print(f"\n  wrote {os.path.join(outdir, 'box_muller.svg')}")


def _svg(path, z, w=760, h=390):
    # histogram of z over [-4, 4]
    lo, hi, bins = -4.0, 4.0, 48
    width = (hi - lo) / bins
    hist = [0] * bins
    for x in z:
        if lo <= x < hi:
            hist[int((x - lo) / width)] += 1
    n = len(z)
    # normalize histogram to a density
    dens = [c / (n * width) for c in hist]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Box-Muller: the sample histogram is the Gaussian</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'{n} transformed uniforms (bars) against the analytic N(0,1) density (curve); '
        f'shaded 1/2/3-sigma bands</text>',
    ]

    x0, x1 = 50, w - 30
    y0, y1 = h - 50, 62
    dmax = 1.0 / math.sqrt(2 * math.pi) * 1.15   # peak of N(0,1)

    def X(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    def Y(d):
        return y0 - d / dmax * (y0 - y1)

    # sigma bands (shaded, lightest outermost)
    for k, col in ((3, "#161b22"), (2, "#1c2530"), (1, "#22303e")):
        parts.append(f'<rect x="{X(-k):.1f}" y="{y1:.1f}" width="{X(k)-X(-k):.1f}" '
                     f'height="{y0-y1:.1f}" fill="{col}"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>')

    # histogram bars
    for i, d in enumerate(dens):
        bx = X(lo + i * width)
        bw = X(lo + (i + 1) * width) - bx
        parts.append(f'<rect x="{bx:.1f}" y="{Y(d):.1f}" width="{max(bw-0.5,0.5):.1f}" '
                     f'height="{y0-Y(d):.1f}" fill="#4dabf7" opacity="0.75"/>')
    # analytic Gaussian curve
    pts = []
    steps = 200
    for i in range(steps + 1):
        xv = lo + (hi - lo) * i / steps
        d = math.exp(-0.5 * xv * xv) / math.sqrt(2 * math.pi)
        pts.append(f"{X(xv):.1f},{Y(d):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#ff6b6b" stroke-width="2.5"/>')

    for k in (-3, -2, -1, 0, 1, 2, 3):
        parts.append(f'<text x="{X(k):.1f}" y="{y0+15:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{k}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">standard deviations from the mean</text>')
    parts.append(f'<rect x="{x0+8}" y="{y1}" width="9" height="9" fill="#4dabf7"/>'
                 f'<text x="{x0+21}" y="{y1+8}" fill="#e6edf3" font-size="9">sample histogram</text>')
    parts.append(f'<rect x="{x0+8}" y="{y1+14}" width="9" height="9" fill="#ff6b6b"/>'
                 f'<text x="{x0+21}" y="{y1+22}" fill="#e6edf3" font-size="9">N(0,1) density</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
