"""RBF interpolation demo: reconstruct a surface from scattered samples, shown as a heatmap (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import rbf_interpolation as R


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
    """Map t in [0,1] to a blue->green->yellow->red heat color."""
    t = max(0.0, min(1.0, t))
    stops = [(0.0, (13, 27, 62)), (0.35, (77, 171, 247)), (0.6, (6, 214, 160)),
             (0.8, (255, 212, 59)), (1.0, (255, 107, 107))]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            w = (t - t0) / (t1 - t0) if t1 > t0 else 0
            r = int(c0[0] + w * (c1[0] - c0[0]))
            g = int(c0[1] + w * (c1[1] - c0[1]))
            b = int(c0[2] + w * (c1[2] - c0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#ff6b6b"


def _heatmap(s, rbf, ox, oy, w, h, title, lo_v, hi_v, res=32):
    s.append(f'<text x="{ox}" y="{oy-8}" fill="{TEXT}" font-size="13">{title}</text>')
    cw = w / res
    ch = h / res
    for i in range(res):
        for j in range(res):
            x = -1 + 2 * (i + 0.5) / res
            y = -1 + 2 * (j + 0.5) / res
            v = rbf.evaluate([x, y])
            t = (v - lo_v) / (hi_v - lo_v) if hi_v > lo_v else 0.5
            px = ox + i * cw
            py = oy + h - (j + 1) * ch
            s.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{cw+0.6:.1f}" height="{ch+0.6:.1f}" '
                     f'fill="{_color(t)}"/>')


def main(outdir=None):
    rnd = _lcg(20260913)
    # true surface: two Gaussian bumps
    def f(p):
        x, y = p
        return (math.exp(-3 * ((x - 0.4) ** 2 + (y - 0.4) ** 2))
                - 0.7 * math.exp(-4 * ((x + 0.4) ** 2 + (y + 0.3) ** 2)))

    n = 45
    pts = [[rnd() * 2 - 1, rnd() * 2 - 1] for _ in range(n)]
    vals = [f(p) for p in pts]

    lines = []
    lines.append("Radial basis function interpolation of scattered data")
    lines.append("=" * 56)
    lines.append(f"{n} scattered samples of a two-bump surface on [-1,1]^2")
    lines.append("")
    lines.append(f"{'kernel':>22}{'node err':>12}{'RMS error':>12}{'max error':>12}")
    # ground-truth error grid
    gq = []
    for i in range(30):
        for j in range(30):
            gq.append([-1 + 2 * i / 29, -1 + 2 * j / 29])
    gtrue = [f(q) for q in gq]

    best = None
    for kind, eps in (("gaussian", 2.0), ("multiquadric", 1.5),
                      ("inverse_multiquadric", 1.5), ("thin_plate", 1.0)):
        rbf = R.RBFInterpolator(pts, vals, kind, epsilon=eps)
        nerr = max(abs(rbf.evaluate(pts[i]) - vals[i]) for i in range(n))
        errs = [abs(rbf.evaluate(gq[k]) - gtrue[k]) for k in range(len(gq))]
        rms = math.sqrt(sum(e * e for e in errs) / len(errs))
        mx = max(errs)
        lines.append(f"{kind:>22}{nerr:>12.2e}{rms:>12.4f}{mx:>12.4f}")
        if best is None or rms < best[1]:
            best = (kind, rms, rbf)
    lines.append("")
    lines.append(f"All kernels interpolate the {n} nodes to machine precision; the RMS/max columns")
    lines.append(f"measure accuracy BETWEEN samples. Best here: {best[0]}.")
    lines.append("")
    lines.append("The interpolant needs no grid and no dimension-specific code -- s(x) = sum w_i "
                 "phi(|x - x_i|).")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # SVG: true surface (dense sample) vs RBF reconstruction, with sample points marked
        W, H = 720, 400
        vmin = min(min(gtrue), min(vals))
        vmax = max(max(gtrue), max(vals))
        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="30" y="26" fill="{TEXT}" font-size="15">'
                 f'RBF interpolation: {n} scattered samples reconstruct a surface</text>')

        class _True:
            def evaluate(self, p):
                return f(p)

        _heatmap(s, _True(), 40, 60, 300, 300, "true surface", vmin, vmax)
        _heatmap(s, best[2], 400, 60, 300, 300, f"RBF reconstruction ({best[0]})", vmin, vmax)
        # mark sample points on the reconstruction panel
        for p in pts:
            px = 400 + (p[0] + 1) / 2 * 300
            py = 60 + 300 - (p[1] + 1) / 2 * 300
            s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2" fill="#0d1117" '
                     f'stroke="#e6edf3" stroke-width="0.8"/>')
        s.append(f'<text x="40" y="{H-12}" fill="{GRAY}" font-size="10">'
                 f'White-ringed dots are the only data the reconstruction sees; everywhere else is '
                 f'interpolated from radial distances to them.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "rbf_interpolation.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
