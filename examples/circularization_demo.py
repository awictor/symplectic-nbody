"""Demo: gravitational waves circularize binaries (Peters 1964).

Evolves the coupled Peters (a, e) equations for several starting eccentricities
and shows that every binary is driven toward e = 0 as it inspirals. Renders the
family of tracks in the (a, e) plane to SVG: all roads lead to a circular merger.

    python examples/circularization_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gravwave import peters_evolve  # noqa: E402

_PALETTE = ["#e63946", "#f4a261", "#2a9d8f", "#3a86ff", "#8338ec"]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    m1 = m2 = 0.5
    c = 8.0
    e0s = [0.2, 0.4, 0.6, 0.8, 0.9]
    print("Peters (1964): gravitational waves circularize binaries\n")
    print(f"{'e0':>6}{'a_final':>12}{'e_final':>12}{'e reduction':>14}")
    print("-" * 44)

    tracks = []
    for e0 in e0s:
        ts, a, e = peters_evolve(1.0, e0, m1, m2, c, dt=0.02,
                                 n_steps=20000, sample_every=1)
        tracks.append((e0, a, e))
        print(f"{e0:>6.2f}{a[-1]:>12.4f}{e[-1]:>12.4f}{e0/e[-1] if e[-1]>0 else float('inf'):>13.1f}x")

    # render (a, e) tracks
    size, pad = 720, 60

    def sx(a):  # a from 1 -> 0 left to right? use a on x, 0..1
        return pad + a * (size - 2 * pad)

    def sy(e):
        return size - pad - e * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        # axes
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    for k, (e0, a, e) in enumerate(tracks):
        col = _PALETTE[k % len(_PALETTE)]
        poly = " ".join(f"{sx(a[i]):.1f},{sy(e[i]):.1f}" for i in range(len(a)))
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" '
                     f'stroke-width="1.8"/>')
        parts.append(f'<circle cx="{sx(a[0]):.1f}" cy="{sy(e[0]):.1f}" r="4" '
                     f'fill="none" stroke="{col}" stroke-width="1.5"/>')
        parts.append(f'<text x="{sx(a[0])+8:.1f}" y="{sy(e[0]):.1f}" fill="{col}" '
                     f'font-size="12">e0={e0}</text>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'All binaries circularize as they inspiral</text>')
    parts.append(f'<text x="{pad}" y="54" fill="#8b949e" font-size="12">'
                 f'Peters (a,e) tracks; x = semi-major axis, y = eccentricity; '
                 f'hollow = start</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">a decreases -&gt; merger</text>')
    parts.append("</svg>")
    path = os.path.join(outdir, "circularization.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"\nwrote {path}")
    print("Every track bends toward e=0: by merger, even a wildly eccentric")
    print("binary is nearly circular. That's why LIGO templates start circular.")


if __name__ == "__main__":
    main()
