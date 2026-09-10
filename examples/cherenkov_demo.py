"""Demo: Cherenkov radiation -- the blue glow of beating light in a medium.

Prints the Cherenkov threshold and cone angle across radiators (water, glass, aerogel), then
draws the cone half-angle versus particle speed for each medium (rising from threshold to its
maximum) plus a sketch of the radiation cone trailing a superluminal particle.

    python examples/cherenkov_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cherenkov import (threshold_beta, threshold_gamma, cone_angle,  # noqa: E402
                       max_cone_angle, photon_yield_factor, emits)


MEDIA = [
    ("water (n=1.33)", 1.333, "#4dabf7"),
    ("glass (n=1.52)", 1.52, "#06d6a0"),
    ("aerogel (n=1.05)", 1.05, "#ff922b"),
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Cherenkov radiation: cos(theta) = 1/(n beta), emit only if beta > 1/n\n")
    print(f"  {'medium':<20}{'beta_thr':>10}{'gamma_thr':>11}{'max cone':>12}")
    for name, n, _ in MEDIA:
        print(f"  {name:<20}{threshold_beta(n):>10.3f}{threshold_gamma(n):>11.2f}"
              f"{math.degrees(max_cone_angle(n)):>9.1f} deg")

    print("\n  Cone angle vs speed in water:")
    for beta in (0.76, 0.85, 0.95, 0.999):
        th = math.degrees(cone_angle(beta, 1.333))
        y = photon_yield_factor(beta, 1.333)
        print(f"    beta = {beta:<6} ->  theta = {th:>4.1f} deg,  rel. photon yield {y:.3f}")

    print("\n  A charged particle outrunning light-in-medium sheds an EM shock cone -- the")
    print("  optical sonic boom that glows blue in a reactor pool. The cone angle reads off")
    print("  the velocity, so ring-imaging Cherenkov detectors use it to identify particles.")

    _svg(os.path.join(outdir, "cherenkov.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'cherenkov.svg')}")


def _svg(path, size=720, pad=72):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Cherenkov radiation</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'cone half-angle vs speed (top); the radiation cone trailing a superluminal particle (bottom)</text>',
    ]

    x0, x1 = pad, size - pad
    mid = size * 0.54

    # --- top: theta_c vs beta ---
    ty0, ty1 = mid - 26, pad + 44
    def BX(beta):
        return x0 + (beta - 0.6) / (1.0 - 0.6) * (x1 - x0)
    def TY(deg):
        return ty0 - deg / 55.0 * (ty0 - ty1)
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x1}" y2="{ty0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x0}" y2="{ty1}" stroke="#8b949e" stroke-width="1.4"/>')
    for beta in (0.6, 0.7, 0.8, 0.9, 1.0):
        parts.append(f'<text x="{BX(beta):.1f}" y="{ty0+14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{beta:.1f}</text>')
    for deg in (0, 20, 40):
        parts.append(f'<text x="{x0-6:.1f}" y="{TY(deg)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{deg}</text>')
    for name, n, col in MEDIA:
        bt = threshold_beta(n)
        pts = []
        b = bt + 1e-4
        while b <= 1.0:
            pts.append(f"{BX(b):.1f},{TY(math.degrees(cone_angle(b, n))):.1f}")
            b += 0.004
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        # threshold marker on the axis
        parts.append(f'<circle cx="{BX(bt):.1f}" cy="{TY(0):.1f}" r="3.5" fill="{col}"/>')
        parts.append(f'<text x="{BX(1.0)-4:.1f}" y="{TY(math.degrees(max_cone_angle(n)))-4:.1f}" '
                     f'fill="{col}" font-size="10" text-anchor="end">{name.split(" ")[0]}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{ty0+28:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">particle speed beta = v/c</text>')
    parts.append(f'<text x="{x0-24:.1f}" y="{(ty0+ty1)/2:.1f}" fill="#8b949e" font-size="10" '
                 f'transform="rotate(-90 {x0-24:.1f} {(ty0+ty1)/2:.1f})" text-anchor="middle">cone angle (deg)</text>')

    # --- bottom: cone geometry (water, beta=0.95) ---
    n, beta = 1.333, 0.95
    th = cone_angle(beta, n)
    py = size * 0.82
    px = size * 0.5
    # particle at px moving right; wave cone trails left at half-angle th
    L = 200
    parts.append(f'<line x1="{px-260:.1f}" y1="{py:.1f}" x2="{px+40:.1f}" y2="{py:.1f}" '
                 f'stroke="#21262d" stroke-width="1"/>')
    # particle
    parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="#e6edf3"/>')
    parts.append(f'<polygon points="{px+30:.1f},{py:.1f} {px+14:.1f},{py-5:.1f} {px+14:.1f},{py+5:.1f}" '
                 f'fill="#e6edf3"/>')
    parts.append(f'<text x="{px+34:.1f}" y="{py-8:.1f}" fill="#e6edf3" font-size="11">particle, beta {beta}</text>')
    # cone edges trailing back
    dx = L * math.cos(th)
    dy = L * math.sin(th)
    parts.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px-dx:.1f}" y2="{py-dy:.1f}" '
                 f'stroke="#4dabf7" stroke-width="2.4"/>')
    parts.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px-dx:.1f}" y2="{py+dy:.1f}" '
                 f'stroke="#4dabf7" stroke-width="2.4"/>')
    # angle arc from the axis (pointing back along -x)
    parts.append(f'<path d="M {px-60:.1f} {py:.1f} A 60 60 0 0 0 '
                 f'{px-60*math.cos(th):.1f} {py-60*math.sin(th):.1f}" '
                 f'fill="none" stroke="#ffd43b" stroke-width="1.4"/>')
    parts.append(f'<text x="{px-92:.1f}" y="{py-16:.1f}" fill="#ffd43b" font-size="12">'
                 f'theta_c = {math.degrees(th):.0f} deg</text>')
    parts.append(f'<text x="{px-dx:.1f}" y="{py-dy-8:.1f}" fill="#4dabf7" font-size="11">'
                 f'blue Cherenkov wavefront</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
