"""PCA-whitening demo: an anisotropic cloud, its principal axes, and PCA vs ZCA whitening (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pca_whitening as P


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
PURPLE = "#b197fc"
YELLOW = "#ffd43b"
RED = "#ff6b6b"


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def _gauss(rnd):
    return math.sqrt(-2 * math.log(rnd() + 1e-12)) * math.cos(2 * math.pi * rnd())


def _panel(s, pts, ox, oy, w, h, title, color, axes=None):
    # fixed data range [-4,4] so the three panels share a scale
    lo, hi = -4.5, 4.5

    def sx(x):
        return ox + (x - lo) / (hi - lo) * w

    def sy(y):
        return oy + h - (y - lo) / (hi - lo) * h

    s.append(f'<text x="{ox+w/2:.0f}" y="{oy-8}" fill="{TEXT}" font-size="12" '
             f'text-anchor="middle">{title}</text>')
    s.append(f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" fill="none" stroke="{GRAY}" '
             f'stroke-width="0.6"/>')
    # origin cross
    s.append(f'<line x1="{sx(lo):.1f}" y1="{sy(0):.1f}" x2="{sx(hi):.1f}" y2="{sy(0):.1f}" '
             f'stroke="#21262d"/>')
    s.append(f'<line x1="{sx(0):.1f}" y1="{sy(lo):.1f}" x2="{sx(0):.1f}" y2="{sy(hi):.1f}" '
             f'stroke="#21262d"/>')
    for (x, y) in pts:
        s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="1.5" fill="{color}" '
                 f'fill-opacity="0.6"/>')
    if axes:
        for (vec, lam, col) in axes:
            L = math.sqrt(lam) * 2
            s.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" '
                     f'x2="{sx(vec[0]*L):.1f}" y2="{sy(vec[1]*L):.1f}" '
                     f'stroke="{col}" stroke-width="2.5"/>')


def _svg(path, raw, wp, wz, model):
    W, H = 760, 320
    pw = 220
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="monospace">']
    s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    axes = [(model["components"][0], model["eigenvalues"][0], YELLOW),
            (model["components"][1], model["eigenvalues"][1], RED)]
    _panel(s, raw, 30, 40, pw, pw, "raw (anisotropic) + PCA axes", BLUE, axes)
    _panel(s, wp, 270, 40, pw, pw, "PCA-whitened (rotated)", GREEN)
    _panel(s, wz, 510, 40, pw, pw, "ZCA-whitened (axis-aligned)", PURPLE)
    s.append(f'<text x="30" y="{H-14}" fill="{GRAY}" font-size="10">'
             f'Both whitenings give identity covariance (a round cloud); ZCA rotates back so the '
             f'cloud stays aligned with the original data, PCA leaves it on the principal axes.</text>')
    s.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(s))


def main(outdir=None):
    rnd = _lcg(20260913)
    # anisotropic cloud: std 3 along a 30-degree axis, std 0.8 across
    theta = math.radians(30)
    ct, st = math.cos(theta), math.sin(theta)
    raw = []
    for _ in range(400):
        a = 3.0 * _gauss(rnd)
        b = 0.8 * _gauss(rnd)
        raw.append([ct * a - st * b, st * a + ct * b])

    model = P.pca(raw)
    wp, _, _ = P.whiten(raw, "pca")
    wz, _, _ = P.whiten(raw, "zca")

    lines = []
    lines.append("PCA and whitening of an anisotropic 2D cloud")
    lines.append("=" * 52)
    lines.append(f"planted: std 3.0 along a 30-degree axis, std 0.8 across ({len(raw)} points)")
    lines.append("")
    lines.append(f"principal axis 1: ({model['components'][0][0]:+.3f}, "
                 f"{model['components'][0][1]:+.3f})   variance {model['eigenvalues'][0]:.3f}")
    lines.append(f"principal axis 2: ({model['components'][1][0]:+.3f}, "
                 f"{model['components'][1][1]:+.3f})   variance {model['eigenvalues'][1]:.3f}")
    lines.append(f"explained variance: {model['explained_variance_ratio'][0]*100:.1f}%, "
                 f"{model['explained_variance_ratio'][1]*100:.1f}%")
    lines.append(f"axis angle: {math.degrees(math.atan2(model['components'][0][1], model['components'][0][0])):.1f} "
                 f"degrees (planted 30)")
    lines.append("")

    def fmt_cov(data, label):
        c = P.covariance(data)
        return (f"{label} covariance: [[{c[0][0]:.3f}, {c[0][1]:+.3f}], "
                f"[{c[1][0]:+.3f}, {c[1][1]:.3f}]]")

    lines.append(fmt_cov(raw, "raw       "))
    lines.append(fmt_cov(wp, "PCA-white "))
    lines.append(fmt_cov(wz, "ZCA-white "))
    lines.append("Both whitenings drive the covariance to the identity (unit, uncorrelated variance).")
    lines.append("")

    cen, _ = P.center(raw)

    def sqdist(A, B):
        return sum((A[i][j] - B[i][j]) ** 2 for i in range(len(A)) for j in range(2))

    lines.append(f"distance from centered data:  PCA {sqdist(wp, cen):.1f}   "
                 f"ZCA {sqdist(wz, cen):.1f}")
    lines.append("ZCA is the minimal-distortion whitening: it rotates back onto the original axes.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        _svg(os.path.join(outdir, "pca_whitening.svg"), raw, wp, wz, model)

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
