"""Demo: reverse-mode automatic differentiation computing exact gradients.

Shows autodiff matching hand-derived and finite-difference gradients, then uses it to train a small
model by gradient descent -- no gradient formulas written by hand. Draws the descent trajectory on a
loss surface.

    python examples/autodiff_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from autodiff import Value, grad, finite_diff_grad  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Reverse-mode automatic differentiation: exact gradients through any expression\n")

    # a nonlinear function of three variables
    def f_ad(v):
        x, y, z = v
        return (x * y).sin() + (z ** 2).exp() * x - y / z

    def f_pl(v):
        x, y, z = v
        return math.sin(x * y) + math.exp(z ** 2) * x - y / z

    pt = [0.8, 1.2, 0.5]
    val, g = grad(f_ad, pt)
    fd = finite_diff_grad(f_pl, pt)
    print(f"  f(x,y,z) = sin(xy) + e^(z^2)*x - y/z  at {pt}")
    print(f"    value: {val:.6f}")
    print(f"    autodiff gradient:      [{', '.join(f'{v:+.5f}' for v in g)}]")
    print(f"    finite-diff gradient:   [{', '.join(f'{v:+.5f}' for v in fd)}]")
    print(f"    max difference: {max(abs(g[i]-fd[i]) for i in range(3)):.2e} (autodiff is exact)")

    print("\n  One backward pass computes the WHOLE gradient vector, regardless of the number of")
    print("  inputs -- the property that makes it the engine of deep learning.")

    # train a tiny model: fit y = a*sin(b*x) + c to data by autodiff gradient descent
    print("\n  Training a 3-parameter model y = a*sin(b*x) + c by autodiff gradient descent:")
    true_a, true_b, true_c = 2.0, 1.5, 0.5
    xs = [i * 0.3 for i in range(25)]
    ys = [true_a * math.sin(true_b * x) + true_c for x in xs]

    params = [1.0, 1.0, 0.0]      # a, b, c
    lr = 0.02
    losses = []
    for epoch in range(2000):
        # build the loss as a Value graph over the parameters (autodiff through sin(b*x))
        a, b, c = Value(params[0]), Value(params[1]), Value(params[2])
        loss = Value(0.0)
        for x, y in zip(xs, ys):
            pred = a * (b * x).sin() + c
            loss = loss + (pred - y) ** 2
        loss = loss * (1.0 / len(xs))
        loss.backward()
        params = [params[0] - lr * a.grad, params[1] - lr * b.grad, params[2] - lr * c.grad]
        losses.append(loss.data)

    print(f"    true params:    a={true_a}, b={true_b}, c={true_c}")
    print(f"    learned params: a={params[0]:.3f}, b={params[1]:.3f}, c={params[2]:.3f}")
    print(f"    loss {losses[0]:.4f} -> {losses[-1]:.6f}")

    _svg(os.path.join(outdir, "autodiff.svg"), losses, xs, ys, params, true_a, true_b, true_c)
    print(f"\n  wrote {os.path.join(outdir, 'autodiff.svg')}")


def _svg(path, losses, xs, ys, params, ta, tb, tc, width=760, height=430):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Autodiff-trained model: loss curve (left), fit (right)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'gradients computed by reverse-mode autodiff -- no derivative formulas written by hand</text>',
    ]

    # left: loss curve (log)
    lx0, ly0, lw, lh = 55, 90, 300, 290
    parts.append(f'<line x1="{lx0}" y1="{ly0+lh}" x2="{lx0+lw}" y2="{ly0+lh}" stroke="#484f58"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly0+lh}" stroke="#484f58"/>')
    lmax = math.log10(max(losses))
    lmin = math.log10(max(min(losses), 1e-8))
    n = len(losses)
    pts = []
    for i, v in enumerate(losses):
        px = lx0 + i / n * lw
        py = ly0 + lh - (math.log10(max(v, 1e-8)) - lmin) / (lmax - lmin) * lh
        pts.append(f"{px:.1f},{py:.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4dabf7" stroke-width="1.8"/>')
    parts.append(f'<text x="{lx0}" y="{ly0-6}" fill="#8b949e" font-size="11">training loss (log scale)</text>')

    # right: data + fitted curve
    gx0, gy0, gw, gh = 420, 90, 310, 290
    xmin, xmax = min(xs), max(xs)
    yall = ys + [params[0] * math.sin(params[1] * x) + params[2] for x in xs]
    ymin, ymax = min(yall), max(yall)
    yr = ymax - ymin or 1

    def px(x):
        return gx0 + (x - xmin) / (xmax - xmin) * gw

    def py(y):
        return gy0 + gh - (y - ymin) / yr * gh

    # data points
    for x, y in zip(xs, ys):
        parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.5" fill="#8b949e"/>')
    # fitted curve
    fit = " ".join(f"{px(x):.1f},{py(params[0]*math.sin(params[1]*x)+params[2]):.1f}"
                   for x in [xmin + (xmax - xmin) * i / 200 for i in range(201)])
    parts.append(f'<polyline points="{fit}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    parts.append(f'<text x="{gx0}" y="{gy0-6}" fill="#8b949e" font-size="11">'
                 f'gray = data, green = learned a sin(bx)+c</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
