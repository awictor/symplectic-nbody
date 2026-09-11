"""Demo: Perlin noise -- smooth gradient fields and fractal terrain.

Shows the noise passing through zero on the lattice, its bounded organic variation, and fractal
Brownian motion layering octaves. Draws a 2-D fBm heightfield as a grayscale grid plus a 1-D fBm
'terrain' cross-section.

    python examples/perlin_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from perlin import Perlin  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Perlin noise: smooth pseudo-random gradient fields\n")

    p = Perlin(seed=42)

    print("  Gradient noise is exactly zero at integer lattice points:")
    print(f"    noise2(3,5) = {p.noise2(3, 5):.2e}, noise2(0,0) = {p.noise2(0, 0):.2e}")
    print(f"    but noise2(3.5, 5.5) = {p.noise2(3.5, 5.5):.4f} (smooth between)\n")

    # sample stats
    vals = [p.noise2(i * 0.1, j * 0.1) for i in range(100) for j in range(100)]
    print(f"  Over a 100x100 sample grid:")
    print(f"    range [{min(vals):.3f}, {max(vals):.3f}], mean {sum(vals)/len(vals):.4f}\n")

    print("  Fractal Brownian motion (summing octaves at doubling frequency, halving amplitude):")
    for oct in [1, 2, 4, 6]:
        row = [p.fbm2(i * 0.15, 0.0, octaves=oct) for i in range(20)]
        rng = max(row) - min(row)
        print(f"    {oct} octave(s): sample spread {rng:.3f} (more octaves -> more fine detail)")

    print("\n  Each lattice point holds a random gradient vector; the noise at a point is the")
    print("  smoothstep-interpolated dot product of the surrounding gradients with the offset")
    print("  vectors. Layering octaves (fBm) is the standard recipe for terrain, clouds, and fire.")

    _svg(os.path.join(outdir, "perlin.svg"), p)
    print(f"\n  wrote {os.path.join(outdir, 'perlin.svg')}")


def _svg(path, p, width=760, height=470):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Perlin fractal noise: a 2-D heightfield and a 1-D terrain slice</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'left: fBm heightfield (dark=low, bright=high); right: a 1-D fBm cross-section</text>',
    ]

    # left: 2-D fBm heightfield as a grid of colored cells
    grid = 48
    cell = 320 / grid
    ox, oy = 40, 70
    scale = 0.09
    for i in range(grid):
        for j in range(grid):
            v = p.fbm2(i * scale, j * scale, octaves=5)     # ~[-1,1]
            t = max(0.0, min(1.0, (v + 1) / 2))
            # terrain-ish palette: deep blue -> green -> tan -> white
            if t < 0.4:
                r, g, b = int(20 + t * 100), int(60 + t * 150), int(120 + t * 200)
            elif t < 0.6:
                r, g, b = int(40 + t * 80), int(140 + t * 120), int(70 + t * 60)
            elif t < 0.8:
                r, g, b = int(150 + t * 100), int(140 + t * 90), int(90 + t * 60)
            else:
                r, g, b = int(200 + t * 55), int(200 + t * 55), int(200 + t * 55)
            r, g, b = min(255, r), min(255, g), min(255, b)
            x = ox + i * cell
            y = oy + j * cell
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell+0.5:.1f}" '
                         f'height="{cell+0.5:.1f}" fill="rgb({r},{g},{b})"/>')

    # right: 1-D fBm cross-section
    gx0, gy0, gw, gh = 400, 90, 330, 300
    parts.append(f'<line x1="{gx0}" y1="{gy0+gh}" x2="{gx0+gw}" y2="{gy0+gh}" stroke="#484f58"/>')
    parts.append(f'<line x1="{gx0}" y1="{gy0}" x2="{gx0}" y2="{gy0+gh}" stroke="#484f58"/>')
    n = 300
    pts = []
    for i in range(n):
        x = i * 0.03
        v = p.fbm1(x, octaves=6)      # ~[-1,1]
        px = gx0 + i / n * gw
        py = gy0 + gh - (v + 1) / 2 * gh
        pts.append(f"{px:.1f},{py:.1f}")
    # fill under the curve for a terrain look
    fill = f"{gx0},{gy0+gh} " + " ".join(pts) + f" {gx0+gw},{gy0+gh}"
    parts.append(f'<polygon points="{fill}" fill="#06d6a0" fill-opacity="0.15"/>')
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#06d6a0" stroke-width="1.8"/>')
    parts.append(f'<text x="{gx0}" y="{gy0-6}" fill="#8b949e" font-size="11">1-D fBm terrain (6 octaves)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
