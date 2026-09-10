"""Demo: the capillary length -- where surface tension gives way to gravity.

Prints the capillary length of several liquids, the Bond number across drop sizes, and the
Weber breakup velocity, then draws the drop-shape crossover: small drops are round spheres,
large ones flatten into gravity-squashed puddles, with the capillary length (Bo = 1) marked.

    python examples/capillary_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from capillary import (capillary_length, bond_number, breakup_velocity,  # noqa: E402
                       max_puddle_depth, GAMMA_WATER, RHO_WATER)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    lc = capillary_length(GAMMA_WATER)
    print("Capillary length l_c = sqrt(gamma / (rho g)): surface tension vs gravity\n")
    print(f"  {'liquid':<22}{'gamma (N/m)':>13}{'rho':>8}{'l_c (mm)':>11}")
    liquids = [
        ("water", 0.0728, 998.0),
        ("mercury", 0.487, 13534.0),
        ("ethanol", 0.0223, 789.0),
        ("liquid nitrogen", 0.0089, 807.0),
    ]
    for name, g, rho in liquids:
        print(f"  {name:<22}{g:>13.4f}{rho:>8.0f}{capillary_length(g, rho)*1000:>11.2f}")

    print(f"\n  Water crossover l_c = {lc*1000:.2f} mm.  Bond number Bo = (L/l_c)^2:")
    print(f"  {'drop / feature':<26}{'size':>10}{'Bo':>10}{'regime':>16}")
    for label, L in (("mist droplet", 20e-6), ("raindrop", 2e-3),
                     ("water strider foot dimple", 3e-3), ("coin of water", 1e-2),
                     ("spilled puddle", 5e-2)):
        Bo = bond_number(L, GAMMA_WATER)
        regime = "round (tension)" if Bo < 1 else "flat (gravity)"
        print(f"  {label:<26}{L*1000:>8.2f}mm{Bo:>10.2f}{regime:>16}")

    print("\n  Max non-wetting puddle depth = 2 l_c = %.1f mm (water can't pile higher)."
          % (max_puddle_depth(GAMMA_WATER) * 1000))
    print("  A moving 2 mm drop shatters once it tops the critical Weber number, at")
    print("  v = %.2f m/s -- why rain fragments and sprays atomize." % breakup_velocity(2e-3, GAMMA_WATER))

    _svg(os.path.join(outdir, "capillary.svg"), lc)
    print(f"\n  wrote {os.path.join(outdir, 'capillary.svg')}")


def _svg(path, lc, size=720, pad=70):
    # Row of drops of increasing radius; below l_c drawn as spheres, above as flattened
    # puddles capped at ~2 l_c tall, to show the Bond-number crossover.
    radii_mm = [0.3, 0.8, 1.5, 2.7, 5.0, 9.0, 16.0]
    lc_mm = lc * 1000.0
    hmax_mm = 2.0 * lc_mm            # puddle depth ceiling

    x0, x1 = pad, size - pad
    baseline = size * 0.62
    # horizontal scale: place drops evenly; vertical scale mm->px
    scale = 15.0                     # px per mm for drop rendering

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Drop shape across the capillary length</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'below l_c surface tension keeps drops round; above it gravity flattens them into puddles</text>',
    ]

    # ground line
    parts.append(f'<line x1="{x0-10:.1f}" y1="{baseline:.1f}" x2="{x1+10:.1f}" y2="{baseline:.1f}" '
                 f'stroke="#8b949e" stroke-width="1.5"/>')

    n = len(radii_mm)
    for i, r_mm in enumerate(radii_mm):
        cx = x0 + (x1 - x0) * (i + 0.5) / n
        Bo = bond_number(r_mm * 1e-3, GAMMA_WATER)
        col = "#4dabf7" if Bo < 1 else "#ff922b"
        if Bo <= 1.0:
            # round sphere sitting on the line
            rpx = r_mm * scale
            parts.append(f'<circle cx="{cx:.1f}" cy="{baseline - rpx:.1f}" r="{rpx:.1f}" '
                         f'fill="{col}" opacity="0.85"/>')
        else:
            # flattened puddle: height capped at 2 l_c, width grows to conserve the idea
            hpx = min(r_mm, hmax_mm) * scale
            # volume ~ r^3; width spreads as that volume at capped height -> ~ r^3/h
            wpx = (r_mm ** 3 / min(r_mm, hmax_mm)) * scale * 0.5
            wpx = min(wpx, (x1 - x0) / n * 0.95)
            parts.append(f'<ellipse cx="{cx:.1f}" cy="{baseline - hpx/2:.1f}" '
                         f'rx="{wpx:.1f}" ry="{hpx/2:.1f}" fill="{col}" opacity="0.85"/>')
        # label
        parts.append(f'<text x="{cx:.1f}" y="{baseline+18:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{r_mm:g} mm</text>')
        parts.append(f'<text x="{cx:.1f}" y="{baseline+31:.1f}" fill="{col}" font-size="9" '
                     f'text-anchor="middle">Bo={Bo:.1f}</text>')

    # capillary-length marker between the round and flat regimes
    # find split index
    split = cx0 = None
    for i, r_mm in enumerate(radii_mm):
        if bond_number(r_mm * 1e-3, GAMMA_WATER) > 1.0:
            split = i
            break
    if split:
        xsp = x0 + (x1 - x0) * split / n
        parts.append(f'<line x1="{xsp:.1f}" y1="{pad+60:.1f}" x2="{xsp:.1f}" y2="{baseline+40:.1f}" '
                     f'stroke="#ffd43b" stroke-width="1.5" stroke-dasharray="5 5"/>')
        parts.append(f'<text x="{xsp:.1f}" y="{pad+54:.1f}" fill="#ffd43b" font-size="11" '
                     f'text-anchor="middle">l_c = {lc*1000:.1f} mm  (Bo = 1)</text>')

    parts.append(f'<text x="{x0:.1f}" y="{size-24:.1f}" fill="#4dabf7" font-size="11">'
                 f'blue = surface tension wins (round)</text>')
    parts.append(f'<text x="{x1:.1f}" y="{size-24:.1f}" fill="#ff922b" font-size="11" '
                 f'text-anchor="end">orange = gravity wins (flat)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
