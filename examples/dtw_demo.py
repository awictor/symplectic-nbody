"""Demo: dynamic time warping aligning two time-warped signals.

Aligns two versions of the same shape recorded at different speeds, shows DTW crushing the Euclidean
distance, and draws both series with the warping path connecting matched points.

    python examples/dtw_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dtw import dtw, dtw_path, euclidean_distance  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Dynamic time warping: aligning series that vary in speed\n")

    # two sine-like bumps at different speeds/phases
    a = [round(math.sin(i * 0.35), 3) for i in range(30)]
    # b is a time-warped version: slow start, fast middle
    b = []
    t = 0.0
    for i in range(30):
        b.append(round(math.sin(t), 3))
        t += 0.2 if i < 15 else 0.55
    n = min(len(a), len(b))

    d_dtw, path = dtw_path(a, b)
    d_euc = euclidean_distance(a[:n], b[:n])
    print(f"  two {len(a)}-point signals of the same shape at different speeds:")
    print(f"    DTW distance:       {d_dtw:.3f}")
    print(f"    Euclidean distance: {d_euc:.3f}   (naive point-by-point)")
    print(f"    -> DTW is {d_euc / d_dtw:.1f}x smaller: it warps the time axis to match the shapes")
    print(f"    warping path length: {len(path)} matched index pairs")

    # identical + stretched
    base = [1, 3, 2, 5, 4, 2]
    stretch = []
    for v in base:
        stretch += [v, v, v]
    print(f"\n  Stretch invariance: DTW(base, 3x-stretched-base) = {dtw(base, stretch)} (exactly zero)")

    # a Sakoe-Chiba band
    d_full = dtw(a, b)
    d_band = dtw(a, b, band=5)
    print(f"\n  Sakoe-Chiba band (limits the warp window):")
    print(f"    unconstrained DTW: {d_full:.3f}")
    print(f"    band=5 DTW:        {d_band:.3f}   (band restricts the path, so distance >= unconstrained)")

    print("\n  DTW fills a cost grid where cell (i,j) is the local distance plus the cheapest of its")
    print("  three predecessors (match / insert / delete). The bottom-right cell is the DTW distance,")
    print("  and backtracking the minimizing moves recovers the monotone warping path.")

    _svg(os.path.join(outdir, "dtw.svg"), a, b, path)
    print(f"\n  wrote {os.path.join(outdir, 'dtw.svg')}")


def _svg(path_file, a, b, warp, width=760, height=440):
    m_left, m_right = 50, 30
    pw = width - m_left - m_right
    # two stacked signal plots with connecting warp lines
    top_y = 90
    bot_y = 320
    amp = 60

    na, nb = len(a), len(b)
    amin = min(min(a), min(b))
    amax = max(max(a), max(b))
    rng = amax - amin or 1

    def ax(i):
        return m_left + i / (na - 1) * pw

    def bx(j):
        return m_left + j / (nb - 1) * pw

    def yy(base, v):
        return base - (v - amin) / rng * amp

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Dynamic time warping: two speed-varying signals aligned</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'blue = series A (top), green = series B (bottom), gray = warping-path correspondences</text>',
    ]

    # warp correspondence lines (draw first, behind)
    for (i, j) in warp:
        parts.append(f'<line x1="{ax(i):.1f}" y1="{yy(top_y, a[i]):.1f}" '
                     f'x2="{bx(j):.1f}" y2="{yy(bot_y, b[j]):.1f}" '
                     f'stroke="#484f58" stroke-width="0.6" opacity="0.6"/>')

    # series A
    pa = " ".join(f"{ax(i):.1f},{yy(top_y, a[i]):.1f}" for i in range(na))
    parts.append(f'<polyline points="{pa}" fill="none" stroke="#4dabf7" stroke-width="2"/>')
    for i in range(na):
        parts.append(f'<circle cx="{ax(i):.1f}" cy="{yy(top_y, a[i]):.1f}" r="2.5" fill="#4dabf7"/>')
    # series B
    pb = " ".join(f"{bx(j):.1f},{yy(bot_y, b[j]):.1f}" for j in range(nb))
    parts.append(f'<polyline points="{pb}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    for j in range(nb):
        parts.append(f'<circle cx="{bx(j):.1f}" cy="{yy(bot_y, b[j]):.1f}" r="2.5" fill="#06d6a0"/>')

    parts.append(f'<text x="{m_left}" y="{top_y-amp-8:.0f}" fill="#4dabf7" font-size="11">series A</text>')
    parts.append(f'<text x="{m_left}" y="{bot_y+18:.0f}" fill="#06d6a0" font-size="11">series B (time-warped)</text>')
    parts.append("</svg>")
    with open(path_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
