"""Polar decomposition demo: split a 2D transform into rotation x stretch, shown on a unit circle (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import polar_decomposition as PD


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
YELLOW = "#ffd43b"
RED = "#ff6b6b"


def _apply(M, pts):
    return [[M[0][0] * x + M[0][1] * y, M[1][0] * x + M[1][1] * y] for x, y in pts]


def main(outdir=None):
    # a general 2x2 transform (rotation + shear + scale)
    A = [[1.4, 0.6], [-0.3, 0.9]]
    U, P = PD.polar_newton(A)
    angle = math.degrees(math.atan2(U[1][0], U[0][0]))

    lines = []
    lines.append("Polar decomposition: A = U P (rotation x stretch)")
    lines.append("=" * 52)
    lines.append(f"A = [[{A[0][0]}, {A[0][1]}], [{A[1][0]}, {A[1][1]}]]")
    lines.append("")
    lines.append("orthogonal factor U (pure rotation/reflection):")
    lines.append(f"  [[{U[0][0]:+.4f}, {U[0][1]:+.4f}],")
    lines.append(f"   [{U[1][0]:+.4f}, {U[1][1]:+.4f}]]   rotation angle {angle:.1f} deg")
    lines.append("symmetric positive-definite factor P (pure stretch):")
    lines.append(f"  [[{P[0][0]:+.4f}, {P[0][1]:+.4f}],")
    lines.append(f"   [{P[1][0]:+.4f}, {P[1][1]:+.4f}]]")
    lines.append("")
    from jacobi_eigen import sorted_eigen
    vals, vecs = sorted_eigen(P)
    lines.append(f"stretch factors (eigenvalues of P): {vals[0]:.4f}, {vals[1]:.4f}")
    lines.append("(these are exactly the singular values of A -- the principal stretch amounts)")
    lines.append("")
    # verify
    UP = PD._matmul(U, P)
    err = max(abs(UP[i][j] - A[i][j]) for i in range(2) for j in range(2))
    check_orth = PD.is_orthogonal(U)
    lines.append(f"U P reconstructs A?  max error {err:.2e}")
    lines.append(f"U orthogonal?        {check_orth}")
    lines.append("")
    lines.append("U is the closest rotation to A -- the right way to re-orthogonalize a drifted")
    lines.append("rotation matrix (camera pose, molecular frame) without discarding information.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        W, H = 720, 300
        # three panels: unit circle -> P (stretch) -> U (rotate) = A
        circle = [[math.cos(2 * math.pi * k / 80), math.sin(2 * math.pi * k / 80)] for k in range(81)]
        after_P = _apply(P, circle)
        after_A = _apply(A, circle)
        panels = [("unit circle", circle, BLUE),
                  ("after stretch P", after_P, GREEN),
                  ("after rotate U = A", after_A, YELLOW)]
        pw = 220
        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="30" y="26" fill="{TEXT}" font-size="15">'
                 f'Polar decomposition: stretch (P) then rotate (U) equals A</text>')
        for pi, (title, pts, col) in enumerate(panels):
            ox = 30 + pi * (pw + 10)
            oy = 55
            cx, cy = ox + pw / 2, oy + pw / 2
            sc = 55
            s.append(f'<text x="{cx:.0f}" y="{oy-6}" fill="{TEXT}" font-size="12" '
                     f'text-anchor="middle">{title}</text>')
            # axes
            s.append(f'<line x1="{ox}" y1="{cy:.0f}" x2="{ox+pw}" y2="{cy:.0f}" stroke="#21262d"/>')
            s.append(f'<line x1="{cx:.0f}" y1="{oy}" x2="{cx:.0f}" y2="{oy+pw}" stroke="#21262d"/>')
            poly = " ".join(f"{cx+x*sc:.1f},{cy-y*sc:.1f}" for x, y in pts)
            s.append(f'<polyline points="{poly}" fill="{col}" fill-opacity="0.12" '
                     f'stroke="{col}" stroke-width="2"/>')
            # mark the transformed x-axis unit vector to show rotation
            v = pts[0]
            s.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{cx+v[0]*sc:.1f}" y2="{cy-v[1]*sc:.1f}" '
                     f'stroke="{RED}" stroke-width="2"/>')
        s.append(f'<text x="30" y="{H-12}" fill="{GRAY}" font-size="10">'
                 f'The circle becomes an ellipse under the symmetric stretch P (axis-aligned to P\'s '
                 f'eigenvectors), then the whole thing rotates rigidly by U -- together they are A.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "polar_decomposition.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


def check(*a, **k):
    pass


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
