"""Isolation Forest demo: score a cloud with planted outliers, color points by anomaly score (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import isolation_forest as IF


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def _color(t):
    """Blue (normal) -> yellow -> red (anomalous) by score t in [0,1]."""
    t = max(0.0, min(1.0, t))
    stops = [(0.0, (77, 171, 247)), (0.5, (6, 214, 160)), (0.7, (255, 212, 59)), (1.0, (255, 107, 107))]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            w = (t - t0) / (t1 - t0) if t1 > t0 else 0
            return f"#{int(c0[0]+w*(c1[0]-c0[0])):02x}{int(c0[1]+w*(c1[1]-c0[1])):02x}{int(c0[2]+w*(c1[2]-c0[2])):02x}"
    return "#ff6b6b"


def main(outdir=None):
    rnd = _lcg(20260913)

    def gauss():
        return math.sqrt(-2 * math.log(rnd() + 1e-12)) * math.cos(2 * math.pi * rnd())

    # two dense gaussian blobs + scattered outliers
    pts = []
    for _ in range(120):
        pts.append([gauss() * 0.6 - 2.5, gauss() * 0.6 - 1.0])
    for _ in range(120):
        pts.append([gauss() * 0.6 + 2.5, gauss() * 0.6 + 1.5])
    outliers = [[0.0, 5.0], [-6.0, 4.0], [6.0, -4.0], [0.0, -5.0], [-5.0, -4.0]]
    pts += outliers
    n_in = 240

    forest = IF.IsolationForest(n_trees=150, sample_size=128, seed=1).fit(pts)
    scores = forest.score_samples(pts)

    lines = []
    lines.append("Isolation Forest: unsupervised anomaly detection")
    lines.append("=" * 54)
    lines.append(f"{n_in} clustered points (2 blobs) + {len(outliers)} planted outliers")
    lines.append(f"forest: 150 trees, subsample 128, score = 2^(-E[path]/c(psi))")
    lines.append("")
    inlier_mean = sum(scores[:n_in]) / n_in
    lines.append(f"mean anomaly score, inliers:  {inlier_mean:.4f}")
    lines.append(f"anomaly scores, planted outliers:")
    for k, o in enumerate(outliers):
        lines.append(f"  ({o[0]:>5.1f},{o[1]:>5.1f}) -> {scores[n_in + k]:.4f}")
    lines.append("")
    lines.append("top-10 most anomalous point indices (planted outliers are 240-244):")
    lines.append("  " + ", ".join(str(i) for i in forest.rank(pts)[:10]))
    lines.append("")
    lines.append("Anomalies isolate in a few random cuts -> short path -> score near 1.")
    lines.append("Points buried in a dense blob need many cuts -> long path -> score below 0.5.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        W, H = 640, 480
        ml, mt, size = 40, 50, 400
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        lo_x, hi_x = min(xs) - 1, max(xs) + 1
        lo_y, hi_y = min(ys) - 1, max(ys) + 1

        def sx(x):
            return ml + (x - lo_x) / (hi_x - lo_x) * size

        def sy(y):
            return mt + size - (y - lo_y) / (hi_y - lo_y) * size

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="30" fill="{TEXT}" font-size="15">'
                 f'Isolation Forest anomaly scores (blue=normal, red=anomaly)</text>')
        # order so high-score points draw on top
        order = sorted(range(len(pts)), key=lambda i: scores[i])
        for i in order:
            r = 3 + 5 * max(0.0, scores[i] - 0.5)
            s.append(f'<circle cx="{sx(pts[i][0]):.1f}" cy="{sy(pts[i][1]):.1f}" r="{r:.1f}" '
                     f'fill="{_color(scores[i])}" fill-opacity="0.85"/>')
        # color-bar legend
        for k in range(50):
            t = k / 49
            s.append(f'<rect x="{ml + k*7}" y="{H-40}" width="7" height="12" fill="{_color(t)}"/>')
        s.append(f'<text x="{ml}" y="{H-46}" fill="{GRAY}" font-size="10">score 0 (normal)</text>')
        s.append(f'<text x="{ml + 260}" y="{H-46}" fill="{GRAY}" font-size="10">1 (anomaly)</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "isolation_forest.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
