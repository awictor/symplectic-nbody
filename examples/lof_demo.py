"""LOF demo: a dense and a sparse cluster with a local outlier that global methods miss (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import lof as L


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


def _color(lof):
    """LOF ~1 blue (normal) -> >2 red (outlier)."""
    t = max(0.0, min(1.0, (lof - 1.0) / 2.0))
    stops = [(0.0, (77, 171, 247)), (0.5, (255, 212, 59)), (1.0, (255, 107, 107))]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            w = (t - t0) / (t1 - t0) if t1 > t0 else 0
            return f"#{int(c0[0]+w*(c1[0]-c0[0])):02x}{int(c0[1]+w*(c1[1]-c0[1])):02x}{int(c0[2]+w*(c1[2]-c0[2])):02x}"
    return "#ff6b6b"


def main(outdir=None):
    rnd = _lcg(20260913)

    def g():
        return math.sqrt(-2 * math.log(rnd() + 1e-12)) * math.cos(2 * math.pi * rnd())

    dense = [[g() * 0.4, g() * 0.4] for _ in range(90)]
    sparse = [[g() * 2.2 + 11, g() * 2.2 + 2] for _ in range(45)]
    local_out = [2.6, 2.6]                         # local outlier next to the dense cluster
    global_out = [6.0, -6.0]                       # obvious global outlier
    pts = dense + sparse + [local_out, global_out]
    n = len(pts)

    lof = L.LOF(pts, k=15)
    sc = lof.scores()

    lines = []
    lines.append("Local Outlier Factor: density-relative anomaly detection")
    lines.append("=" * 58)
    lines.append(f"{len(dense)} dense-cluster + {len(sparse)} sparse-cluster points, k = 15")
    lines.append("")
    lines.append(f"mean LOF, dense cluster:   {sum(sc[:90])/90:.3f}")
    lines.append(f"mean LOF, sparse cluster:  {sum(sc[90:135])/45:.3f}")
    lines.append(f"LOF, local outlier {local_out}:  {sc[135]:.3f}")
    lines.append(f"LOF, global outlier {global_out}: {sc[136]:.3f}")
    lines.append("")
    # show the local-outlier discriminator
    dist_to_dense = min(math.dist(local_out, dense[i]) for i in range(90))
    sparse_kdist = lof.k_distance(90)
    lines.append(f"local outlier's distance to the dense cluster: {dist_to_dense:.2f}")
    lines.append(f"a typical spacing INSIDE the sparse cluster:   {sparse_kdist:.2f}")
    lines.append("-> the local outlier's distance is normal by sparse-cluster standards,")
    lines.append("   so a global distance/isolation test would MISS it -- but LOF flags it,")
    lines.append("   because it compares each point's density to its neighbours' densities.")
    lines.append("")
    lines.append("top-5 most anomalous indices (135=local, 136=global outlier):")
    lines.append("  " + ", ".join(str(i) for i in lof.rank()[:5]))

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
                 f'Local Outlier Factor (blue=normal, red=local anomaly)</text>')
        order = sorted(range(n), key=lambda i: sc[i])
        for i in order:
            r = 3 + 3 * max(0.0, sc[i] - 1.0)
            s.append(f'<circle cx="{sx(pts[i][0]):.1f}" cy="{sy(pts[i][1]):.1f}" r="{r:.1f}" '
                     f'fill="{_color(sc[i])}" fill-opacity="0.85"/>')
        # annotate the two outliers
        for idx, lab in ((135, "local"), (136, "global")):
            s.append(f'<text x="{sx(pts[idx][0])+8:.1f}" y="{sy(pts[idx][1]):.1f}" '
                     f'fill="{TEXT}" font-size="11">{lab} (LOF {sc[idx]:.1f})</text>')
        s.append(f'<text x="{ml}" y="{H-16}" fill="{GRAY}" font-size="10">'
                 f'The dense blob (lower-left) and the loose blob (upper-right) both read as normal. '
                 f'The local outlier beside the dense blob is flagged though its distance is sparse-normal.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "lof.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
