"""Demo: Gaussian mixture models by EM -- soft, probabilistic clustering.

Fits a diagonal-covariance mixture to three 2-D clusters of unequal spread, showing that EM
recovers the true means, weights, and variances, that its log-likelihood climbs monotonically, and
that BIC selects the right number of components. Draws each point tinted by its soft responsibility
and the fitted component ellipses, beside the BIC-vs-k model-selection curve.

    python examples/gmm_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gmm import GaussianMixture, fit_best  # noqa: E402


def _data(seed=42):
    state = seed

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    def gauss(mu, sd):
        u1 = max(1e-9, rng())
        u2 = rng()
        return mu + sd * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    specs = [((2.0, 2.0), 0.5, 120), ((8.0, 8.0), 1.4, 120), ((2.0, 8.0), 0.9, 120)]
    X = []
    for (cx, cy), sd, n in specs:
        for _ in range(n):
            X.append([gauss(cx, sd), gauss(cy, sd)])
    return X


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    X = _data()
    g = fit_best(X, 3, restarts=8, seed=1)

    print("Gaussian mixture model (EM): soft, probabilistic clustering\n")
    print(f"  {len(X)} points, 3 clusters of unequal spread, 2-D\n")
    print(f"  converged in {g.n_iter_} iterations, log-likelihood {g.log_likelihood_:.1f}\n")

    print("  Recovered components (true means (2,2) sd .5, (8,8) sd 1.4, (2,8) sd .9):")
    order = sorted(range(3), key=lambda c: (g.means[c][0], g.means[c][1]))
    for c in order:
        m = g.means[c]
        sd = [math.sqrt(v) for v in g.variances[c]]
        print(f"    mean ({m[0]:5.2f},{m[1]:5.2f})  sd ({sd[0]:.2f},{sd[1]:.2f})  "
              f"weight {g.weights[c]:.2f}")

    print("\n  EM climbs the log-likelihood monotonically (first / last 4 iterations):")
    h = g.history_
    show = list(range(min(4, len(h)))) + list(range(max(0, len(h) - 4), len(h)))
    for i in sorted(set(show)):
        print(f"    iter {i:>2}: {h[i]:.1f}")

    print("\n  BIC selects the number of components (lower is better):")
    bics = {}
    for k in (1, 2, 3, 4, 5):
        gk = fit_best(X, k, restarts=5, seed=2)
        bics[k] = gk.bic(X)
        print(f"    k={k}: BIC {bics[k]:8.1f}  AIC {gk.aic(X):8.1f}")
    best_k = min(bics, key=bics.get)
    print(f"    BIC minimized at k={best_k} (true is 3)")

    print("\n  Unlike k-means' hard nearest-centroid assignment, a mixture gives each point a")
    print("  probability of belonging to each component, so clusters can differ in size, weight,")
    print("  and spread -- and the likelihood lets BIC/AIC choose k in a principled way.")

    _svg(os.path.join(outdir, "gmm.svg"), X, g, bics)
    print(f"\n  wrote {os.path.join(outdir, 'gmm.svg')}")


def _svg(path, X, g, bics, width=760, height=430):
    # base RGB for each component; a point's colour blends by its responsibilities
    base = [(77, 171, 247), (255, 212, 59), (255, 107, 107)]  # blue, yellow, red
    proba = g.predict_proba(X)
    xs = [p[0] for p in X]
    ys = [p[1] for p in X]
    xa, xb = min(xs) - 0.5, max(xs) + 0.5
    ya, yb = min(ys) - 0.5, max(ys) + 0.5
    lx0, lx1 = 45, width // 2 - 10
    y0, y1 = height - 45, 65

    def LX(x):
        return lx0 + (x - xa) / (xb - xa) * (lx1 - lx0)

    def LY(y):
        return y0 - (y - ya) / (yb - ya) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Gaussian mixture (EM): points tinted by soft responsibility, and BIC picking k</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'colour blends the P(component) of each point; ellipses are the fitted 2-sigma '
        f'components</text>',
        f'<line x1="{lx0}" y1="{y0}" x2="{lx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>',
        f'<line x1="{lx0}" y1="{y0}" x2="{lx0}" y2="{y1}" stroke="#8b949e" stroke-width="1.2"/>',
    ]

    # points, colour = responsibility-weighted blend of component colours
    k = g.k
    for i, (px, py) in enumerate(X):
        r = proba[i]
        cr = int(sum(r[c] * base[c % 3][0] for c in range(k)))
        cg = int(sum(r[c] * base[c % 3][1] for c in range(k)))
        cb = int(sum(r[c] * base[c % 3][2] for c in range(k)))
        parts.append(f'<circle cx="{LX(px):.1f}" cy="{LY(py):.1f}" r="2.6" '
                     f'fill="rgb({cr},{cg},{cb})" opacity="0.85"/>')

    # 2-sigma ellipses (axis-aligned: diagonal covariance)
    for c in range(k):
        m = g.means[c]
        sx = math.sqrt(g.variances[c][0])
        sy = math.sqrt(g.variances[c][1])
        rx = abs(LX(m[0] + 2 * sx) - LX(m[0]))
        ry = abs(LY(m[1] + 2 * sy) - LY(m[1]))
        col = f"rgb{base[c % 3]}"
        parts.append(f'<ellipse cx="{LX(m[0]):.1f}" cy="{LY(m[1]):.1f}" rx="{rx:.1f}" '
                     f'ry="{ry:.1f}" fill="none" stroke="{col}" stroke-width="1.6" '
                     f'stroke-dasharray="4 3"/>')
        parts.append(f'<circle cx="{LX(m[0]):.1f}" cy="{LY(m[1]):.1f}" r="3" fill="{col}"/>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{y0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">soft clusters + 2-sigma ellipses</text>')

    # right panel: BIC vs k curve
    rx0, rx1 = width // 2 + 45, width - 30
    ks = sorted(bics)
    bvals = [bics[kk] for kk in ks]
    bmin, bmax = min(bvals), max(bvals)

    def RX(idx):
        return rx0 + idx / (len(ks) - 1) * (rx1 - rx0)

    def RY(v):
        return y0 - (v - bmin) / (bmax - bmin + 1e-9) * (y0 - y1)

    parts.append(f'<line x1="{rx0}" y1="{y0}" x2="{rx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{y0}" x2="{rx0}" y2="{y1}" stroke="#8b949e" stroke-width="1.2"/>')
    poly = " ".join(f"{RX(i):.1f},{RY(bvals[i]):.1f}" for i in range(len(ks)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#06d6a0" stroke-width="2.5"/>')
    best_i = bvals.index(min(bvals))
    for i, kk in enumerate(ks):
        col = "#ffd43b" if i == best_i else "#06d6a0"
        parts.append(f'<circle cx="{RX(i):.1f}" cy="{RY(bvals[i]):.1f}" r="4" fill="{col}"/>')
        parts.append(f'<text x="{RX(i):.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">k={kk}</text>')
    parts.append(f'<text x="{RX(best_i):.1f}" y="{RY(bvals[best_i])-10:.1f}" fill="#ffd43b" '
                 f'font-size="10" text-anchor="middle">min BIC</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{y0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">BIC vs number of components</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
