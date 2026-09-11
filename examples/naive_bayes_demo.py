"""Demo: naive Bayes classification, Gaussian and multinomial.

Fits a Gaussian naive Bayes to three 2-D classes and draws the decision regions its per-class
Gaussians carve out, then trains a multinomial naive Bayes spam filter on tiny bag-of-word
documents and shows the log-odds each word contributes. Illustrates that a crude independence
assumption still yields a strong, instant classifier.

    python examples/naive_bayes_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from naive_bayes import GaussianNB, MultinomialNB  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    state = 3

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    # three 2-D classes, deliberately different spreads
    specs = [((0.0, 0.0), 1.0), ((6.0, 1.0), 1.5), ((2.0, 6.0), 0.9)]
    X, y = [], []
    for c, ((cx, cy), sd) in enumerate(specs):
        for _ in range(35):
            X.append([cx + (rng() - 0.5) * 2 * sd, cy + (rng() - 0.5) * 2 * sd])
            y.append(c)
    gnb = GaussianNB().fit(X, y)

    print("Gaussian naive Bayes: continuous features, one Gaussian per class per feature\n")
    print(f"  {len(X)} points, 3 classes, training accuracy {gnb.accuracy(X, y):.1%}\n")
    print("  Learned per-class feature statistics:")
    for c in gnb.classes:
        mean = gnb.theta[c]
        sd = [math.sqrt(v) for v in gnb.var[c]]
        print(f"    class {c}: mean ({mean[0]:5.2f},{mean[1]:5.2f})  "
              f"sd ({sd[0]:.2f},{sd[1]:.2f})  prior {math.exp(gnb.log_prior[c]):.2f}")

    probe = [[0.0, 0.0], [3.0, 3.0], [6.0, 1.0]]
    print("\n  Posterior class probabilities at a few points:")
    for x, p in zip(probe, gnb.predict_proba(probe)):
        pretty = ", ".join(f"c{c}={p[c]:.2f}" for c in gnb.classes)
        print(f"    {x}: {pretty}")

    # --- multinomial spam filter ---
    vocab = ["free", "money", "offer", "meeting", "report", "project"]
    Xt = [[3, 2, 2, 0, 0, 0], [2, 3, 1, 0, 1, 0], [4, 1, 3, 0, 0, 0], [3, 2, 2, 0, 0, 1],
          [0, 0, 0, 3, 2, 2], [0, 1, 0, 2, 3, 1], [0, 0, 1, 4, 1, 2], [0, 0, 0, 2, 2, 3]]
    yt = ["spam"] * 4 + ["ham"] * 4
    mnb = MultinomialNB(alpha=1.0).fit(Xt, yt)

    print("\nMultinomial naive Bayes: word counts, a classic spam filter\n")
    print(f"  {len(Xt)} documents, vocab {vocab}, training accuracy {mnb.accuracy(Xt, yt):.1%}\n")
    print("  Per-word evidence, log P(word|spam) - log P(word|ham) (positive => spammy):")
    weights = []
    for f, w in enumerate(vocab):
        lo = mnb.log_prob["spam"][f] - mnb.log_prob["ham"][f]
        weights.append((w, lo))
        bar = ("+" if lo >= 0 else "-") * min(20, int(abs(lo) * 6))
        print(f"    {w:>8}: {lo:+.2f} {bar}")

    tests = [[2, 1, 1, 0, 0, 0], [0, 0, 0, 2, 1, 1], [1, 0, 0, 1, 1, 0]]
    print("\n  Classifying new documents:")
    for x, pred, p in zip(tests, mnb.predict(tests), mnb.predict_proba(tests)):
        present = ", ".join(vocab[f] for f in range(len(vocab)) if x[f])
        print(f"    [{present}] -> {pred} (P(spam)={p['spam']:.2f})")

    print("\n  The independence assumption is false -- words co-occur -- yet summing per-feature")
    print("  log-evidence gives a fast, strong classifier trained in a single counting pass. It is")
    print("  the baseline every fancier text or tabular model has to beat.")

    _svg(os.path.join(outdir, "naive_bayes.svg"), X, y, gnb, vocab, weights)
    print(f"\n  wrote {os.path.join(outdir, 'naive_bayes.svg')}")


def _svg(path, X, y, gnb, vocab, weights, width=760, height=430):
    palette = ["#4dabf7", "#ff6b6b", "#ffd43b"]
    fills = ["#16324f", "#3d1d1d", "#3d3410"]
    xs = [p[0] for p in X]
    ys = [p[1] for p in X]
    xa, xb = min(xs) - 1, max(xs) + 1
    ya, yb = min(ys) - 1, max(ys) + 1
    lx0, lx1 = 45, width // 2 - 15
    y0, y1 = height - 45, 65

    def LX(x):
        return lx0 + (x - xa) / (xb - xa) * (lx1 - lx0)

    def LY(v):
        return y0 - (v - ya) / (yb - ya) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Naive Bayes: Gaussian decision regions (left), spam word-evidence (right)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'left: regions coloured by predicted class with the training points; right: each '
        f'word’s log-odds</text>',
    ]

    # decision regions by sampling a grid
    gx, gy = 70, 58
    cw = (lx1 - lx0) / gx
    ch = (y0 - y1) / gy
    for i in range(gx):
        for j in range(gy):
            xv = xa + (i + 0.5) / gx * (xb - xa)
            yv = ya + (j + 0.5) / gy * (yb - ya)
            cls = gnb.predict([[xv, yv]])[0]
            parts.append(f'<rect x="{lx0 + i*cw:.1f}" y="{y0 - (j+1)*ch:.1f}" '
                         f'width="{cw+0.6:.1f}" height="{ch+0.6:.1f}" fill="{fills[cls % 3]}"/>')
    # points
    for i, (px, pv) in enumerate(X):
        parts.append(f'<circle cx="{LX(px):.1f}" cy="{LY(pv):.1f}" r="3.2" '
                     f'fill="{palette[y[i] % 3]}" stroke="#0d1117" stroke-width="0.6"/>')
    # class-mean crosses
    for c in gnb.classes:
        mx, my = LX(gnb.theta[c][0]), LY(gnb.theta[c][1])
        parts.append(f'<line x1="{mx-5:.1f}" y1="{my:.1f}" x2="{mx+5:.1f}" y2="{my:.1f}" '
                     f'stroke="#e6edf3" stroke-width="1.6"/>')
        parts.append(f'<line x1="{mx:.1f}" y1="{my-5:.1f}" x2="{mx:.1f}" y2="{my+5:.1f}" '
                     f'stroke="#e6edf3" stroke-width="1.6"/>')

    # right: word log-odds as a diverging bar chart
    rx_axis = width // 2 + 130
    rtop = 90
    row_h = 42
    wmax = max(abs(w) for _, w in weights) or 1.0
    scale = 90 / wmax
    for k, (w, lo) in enumerate(weights):
        yy = rtop + k * row_h
        length = lo * scale
        col = "#ff6b6b" if lo >= 0 else "#4dabf7"
        x2 = rx_axis + length
        parts.append(f'<rect x="{min(rx_axis, x2):.1f}" y="{yy-8:.1f}" '
                     f'width="{abs(length):.1f}" height="14" fill="{col}"/>')
        parts.append(f'<text x="{rx_axis - 100:.1f}" y="{yy+4:.1f}" fill="#e6edf3" '
                     f'font-size="11">{w}</text>')
    parts.append(f'<line x1="{rx_axis}" y1="{rtop-16}" x2="{rx_axis}" y2="{rtop + len(weights)*row_h-20}" '
                 f'stroke="#8b949e" stroke-width="1.1"/>')
    parts.append(f'<text x="{rx_axis:.1f}" y="{rtop + len(weights)*row_h -2:.1f}" fill="#8b949e" '
                 f'font-size="9" text-anchor="middle">0 (spam right, ham left)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
