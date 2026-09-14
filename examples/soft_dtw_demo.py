"""Soft-DTW demo: the soft alignment matrix between two warped series, and the gamma -> DTW limit (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import soft_dtw as S


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
RED = "#ff6b6b"
YELLOW = "#ffd43b"


def _heat(t):
    t = max(0.0, min(1.0, t))
    # dark -> blue -> green -> yellow
    stops = [(0.0, (13, 17, 23)), (0.4, (30, 60, 120)), (0.7, (6, 160, 120)), (1.0, (255, 212, 59))]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            w = (t - t0) / (t1 - t0) if t1 > t0 else 0
            return f"#{int(c0[0]+w*(c1[0]-c0[0])):02x}{int(c0[1]+w*(c1[1]-c0[1])):02x}{int(c0[2]+w*(c1[2]-c0[2])):02x}"
    return "#ffd43b"


def main(outdir=None):
    # two similar shapes, one warped (stretched in the middle)
    x = [math.sin(2 * math.pi * k / 20) for k in range(20)]
    y = ([math.sin(2 * math.pi * k / 20) for k in range(7)]
         + [math.sin(2 * math.pi * 7 / 20)] * 4                     # a plateau: local time stretch
         + [math.sin(2 * math.pi * k / 20) for k in range(7, 20)])

    lines = []
    lines.append("Soft-DTW: differentiable dynamic time warping")
    lines.append("=" * 52)
    lines.append(f"series lengths: {len(x)} and {len(y)} (y has a time-stretched plateau)")
    lines.append("")
    hd = S.hard_dtw(x, y)
    lines.append(f"hard DTW cost: {hd:.5f}")
    lines.append("")
    lines.append("soft-DTW converges to hard DTW as the temperature gamma -> 0:")
    lines.append(f"{'gamma':>10}{'soft-DTW':>14}{'|soft - hard|':>16}")
    for g in (1.0, 0.3, 0.1, 0.03, 0.01, 0.001):
        sd = S.soft_dtw(x, y, g)
        lines.append(f"{g:>10}{sd:>14.5f}{abs(sd - hd):>16.2e}")
    lines.append("")
    lines.append("Large gamma averages over ALL alignment paths (smooth, differentiable);")
    lines.append("small gamma concentrates on the single optimal path (classic DTW).")
    lines.append("The soft alignment matrix is the gradient of soft-DTW w.r.t. the cost matrix,")
    lines.append("which is what lets soft-DTW be used as a loss and to average series under warping.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        E1 = S.alignment_matrix(x, y, 1.0)
        E2 = S.alignment_matrix(x, y, 0.02)
        n, m = len(x), len(y)
        W, H = 720, 380
        cell = 15
        ox1, oy = 60, 60
        ox2 = 400

        def draw(s, E, ox, title):
            mx = max(max(row) for row in E) or 1
            s.append(f'<text x="{ox}" y="{oy-10}" fill="{TEXT}" font-size="13">{title}</text>')
            for i in range(n):
                for j in range(m):
                    s.append(f'<rect x="{ox+j*cell}" y="{oy+i*cell}" width="{cell}" height="{cell}" '
                             f'fill="{_heat(E[i][j]/mx)}"/>')

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ox1}" y="30" fill="{TEXT}" font-size="15">'
                 f'Soft alignment matrices: diffuse (large gamma) vs sharp (small gamma)</text>')
        draw(s, E1, ox1, "gamma = 1.0  (soft, all paths)")
        draw(s, E2, ox2, "gamma = 0.02 (sharp, ~ DTW path)")
        s.append(f'<text x="{ox1}" y="{H-16}" fill="{GRAY}" font-size="10">'
                 f'Each cell is the expected occupancy of aligning x[i] with y[j]. Large gamma spreads '
                 f'the mass; small gamma collapses it onto the optimal warping path (the bright ridge).</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "soft_dtw.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
