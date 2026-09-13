"""Demo: SVM by SMO -- a nonlinear RBF boundary carving apart two interleaving classes.

Trains an RBF-kernel SVM on two interleaving "moon" clusters (not linearly separable), reports the
support vectors and accuracy, and draws the curved decision boundary as a shaded region with the
support vectors highlighted.

    python examples/svm_smo_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from svm_smo import SVM, rbf_kernel, linear_kernel, functional_margins  # noqa: E402


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("SVM by SMO: the maximum-margin classifier, with a nonlinear RBF kernel\n")

    rng = _lcg(2024)
    X, y = [], []
    # two half-moons
    for _ in range(40):
        t = math.pi * rng()
        X.append([math.cos(t) + 0.1 * (rng() - 0.5), math.sin(t) + 0.1 * (rng() - 0.5)])
        y.append(1)
        t = math.pi * rng()
        X.append([1 - math.cos(t) + 0.1 * (rng() - 0.5),
                  -math.sin(t) + 0.5 + 0.1 * (rng() - 0.5)])
        y.append(-1)

    # linear SVM for contrast
    lin = SVM(kernel=linear_kernel, C=10.0, max_passes=100).fit(X, y)
    lin_acc = sum(1 for i in range(len(X)) if lin.predict(X[i]) == y[i]) / len(X)

    svm = SVM(kernel=rbf_kernel(gamma=2.0), C=10.0, max_passes=200).fit(X, y)
    rbf_acc = sum(1 for i in range(len(X)) if svm.predict(X[i]) == y[i]) / len(X)
    sv = svm.support_vectors()

    print(f"  two interleaving half-moons, {len(X)} points, not linearly separable\n")
    print(f"    {'classifier':<22}{'train accuracy':>16}")
    print(f"    {'linear SVM':<22}{lin_acc:>15.1%}")
    print(f"    {'RBF SVM (gamma=2)':<22}{rbf_acc:>15.1%}")
    print(f"\n  RBF SVM: {len(sv)} support vectors out of {len(X)} points")
    fm = functional_margins(svm)
    print(f"  min functional margin over training set: {min(fm):.3f}  (>= 1 => on/outside margin)")
    print(f"\n  The linear boundary cannot separate the moons; the RBF kernel bends the")
    print(f"  boundary through the gap. Only the support vectors (alpha > 0) shape it.")

    _svg(os.path.join(outdir, "svm_smo.svg"), svm, X, y, sv)
    print(f"\n  wrote {os.path.join(outdir, 'svm_smo.svg')}")


def _svg(path, svm, X, y, sv, width=760, height=440):
    xs = [p[0] for p in X]
    ys = [p[1] for p in X]
    minx, maxx = min(xs) - 0.4, max(xs) + 0.4
    miny, maxy = min(ys) - 0.4, max(ys) + 0.4
    pad = 30

    def sx(x):
        return pad + (width - 2 * pad) * (x - minx) / (maxx - minx)

    def sy(yy):
        return height - pad - (height - 2 * pad - 20) * (yy - miny) / (maxy - miny)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'RBF-SVM decision regions on two interleaving moons (support vectors ringed)</text>',
    ]

    # decision-region grid shading
    gx, gy = 76, 44
    for a in range(gx):
        for b in range(gy):
            x = minx + (maxx - minx) * a / (gx - 1)
            yy = miny + (maxy - miny) * b / (gy - 1)
            f = svm.decision_function([x, yy])
            if f >= 0:
                color = "#16324d"   # class +1 region (dark blue)
            else:
                color = "#3d2416"   # class -1 region (dark orange)
            px = sx(x)
            py = sy(yy)
            cw = (width - 2 * pad) / (gx - 1) + 1
            ch = (height - 2 * pad - 20) / (gy - 1) + 1
            parts.append(f'<rect x="{px:.1f}" y="{py - ch:.1f}" width="{cw:.1f}" '
                         f'height="{ch:.1f}" fill="{color}"/>')

    # data points
    for i, (x, yy) in enumerate(X):
        color = "#4dabf7" if y[i] == 1 else "#ff922b"
        if i in set(sv):
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(yy):.1f}" r="6" fill="none" '
                         f'stroke="#ffd43b" stroke-width="2"/>')
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(yy):.1f}" r="3.5" fill="{color}"/>')

    parts.append(f'<text x="{width-210}" y="{height-40}" fill="#4dabf7" font-size="10">'
                 f'blue = class +1</text>')
    parts.append(f'<text x="{width-210}" y="{height-26}" fill="#ff922b" font-size="10">'
                 f'orange = class -1</text>')
    parts.append(f'<text x="{width-210}" y="{height-12}" fill="#ffd43b" font-size="10">'
                 f'ringed = support vector</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
