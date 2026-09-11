"""Demo: mutual information as a dependency measure.

Shows mutual information falling from H(X) (a perfect copy) to zero (independent) as a channel gets
noisier, ranks candidate features by their information gain about a label, and illustrates the
KL divergence between distributions. Contrasts MI with correlation on a nonlinear relationship.

    python examples/mutual_information_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mutual_information as mi  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    state = 1

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    print("Mutual information: how much one variable tells you about another\n")

    # MI vs channel noise: Y = X flipped with probability f
    print("  Binary symmetric channel, Y = X flipped with probability f:")
    print("  (I falls from H(X)=1 bit at f=0 to 0 at f=0.5, the fully-noisy channel)\n")
    flips = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    mi_curve = []
    N = 8000
    for f in flips:
        xs = [int(rng() * 2) for _ in range(N)]
        ys = [(x if rng() > f else 1 - x) for x in xs]
        info = mi.mutual_information(xs, ys)
        mi_curve.append(info)
        bar = "#" * int(round(info * 40))
        print(f"    f = {f:.2f}: I(X;Y) = {info:.3f} bits {bar}")

    # MI catches a nonlinear relation where correlation is blind: Y = X^2 mod-ish
    print("\n  MI sees nonlinear dependence that (linear) correlation misses:")
    xs = [int(rng() * 5) - 2 for _ in range(6000)]     # -2..2
    ys = [x * x for x in xs]                             # deterministic but nonmonotone
    # Pearson correlation
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    corr = cov / (sx * sy) if sx * sy > 0 else 0.0
    print(f"    Y = X^2 (deterministic): correlation {corr:+.3f} (near 0, looks 'unrelated')")
    print(f"                             mutual information {mi.mutual_information(xs, ys):.3f} bits "
          f"(= H(X), fully dependent)")

    # feature selection by information gain
    print("\n  Feature selection -- rank features by information gain about the label:")
    labels = [int(rng() * 2) for _ in range(4000)]
    good = [(l if rng() > 0.1 else 1 - l) for l in labels]      # 90% agrees with label
    weak = [(l if rng() > 0.35 else 1 - l) for l in labels]     # 65% agrees
    noise = [int(rng() * 2) for _ in range(4000)]               # independent
    for name, feat in [("informative", good), ("weak", weak), ("noise", noise)]:
        print(f"    {name:>11}: information gain {mi.information_gain(feat, labels):.4f} bits")

    # KL divergence between a biased coin and fair
    print("\n  KL divergence D(p||q) -- extra bits to code p-samples with a q-code:")
    for p1 in (0.5, 0.7, 0.9, 0.99):
        d = mi.kl_divergence([p1, 1 - p1], [0.5, 0.5])
        print(f"    D([{p1},{1-p1:.2f}] || fair) = {d:.4f} bits")

    print("\n  Mutual information is the shared information I(X;Y) = H(X)+H(Y)-H(X,Y): zero iff")
    print("  independent, maximal for a deterministic relationship, and blind to nothing -- it")
    print("  catches any dependency, which is why it drives feature selection and tree splits.")

    _svg(os.path.join(outdir, "mutual_information.svg"), flips, mi_curve)
    print(f"\n  wrote {os.path.join(outdir, 'mutual_information.svg')}")


def _svg(path, flips, mi_curve, width=760, height=400):
    lx0, lx1 = 60, width - 40
    y0, y1 = height - 55, 70

    def X(f):
        return lx0 + f / 0.5 * (lx1 - lx0)

    def Y(v):
        return y0 - v / 1.05 * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Mutual information vs channel noise</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'Y = X flipped with probability f; I(X;Y) falls from 1 bit (perfect copy) to 0 '
        f'(pure noise at f=0.5)</text>',
        f'<line x1="{lx0}" y1="{y0}" x2="{lx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>',
        f'<line x1="{lx0}" y1="{y0}" x2="{lx0}" y2="{y1}" stroke="#8b949e" stroke-width="1.2"/>',
    ]
    # gridlines at 0.5 and 1.0 bits
    for v in (0.5, 1.0):
        parts.append(f'<line x1="{lx0}" y1="{Y(v):.1f}" x2="{lx1}" y2="{Y(v):.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{lx0-6:.1f}" y="{Y(v)+4:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{v:.1f}</text>')
    # theoretical curve I = 1 - H(f) (binary entropy)
    theo = []
    steps = 100
    for i in range(steps + 1):
        f = 0.5 * i / steps
        if f in (0.0,):
            hf = 0.0
        elif f >= 0.5:
            hf = 1.0
        else:
            hf = -(f * math.log2(f) + (1 - f) * math.log2(1 - f))
        theo.append(f"{X(f):.1f},{Y(1 - hf):.1f}")
    parts.append(f'<polyline points="{" ".join(theo)}" fill="none" stroke="#06d6a0" '
                 f'stroke-width="1.6" stroke-dasharray="5 3"/>')
    # measured points + line
    pts = " ".join(f"{X(flips[i]):.1f},{Y(mi_curve[i]):.1f}" for i in range(len(flips)))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="2.2"/>')
    for i in range(len(flips)):
        parts.append(f'<circle cx="{X(flips[i]):.1f}" cy="{Y(mi_curve[i]):.1f}" r="3.5" '
                     f'fill="#ffd43b"/>')
        parts.append(f'<text x="{X(flips[i]):.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{flips[i]:.2f}</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{y0+34:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">flip probability f</text>')
    parts.append(f'<text x="{lx1-4:.1f}" y="{y1+2:.1f}" fill="#06d6a0" font-size="9" '
                 f'text-anchor="end">theory I = 1 - H(f)</text>')
    parts.append(f'<text x="{lx1-4:.1f}" y="{y1+16:.1f}" fill="#4dabf7" font-size="9" '
                 f'text-anchor="end">measured (8000 samples)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
