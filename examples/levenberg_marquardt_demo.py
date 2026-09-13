"""Demo: Levenberg-Marquardt fitting a noisy Gaussian peak, with the damped descent in action.

Generates noisy samples of a Gaussian, fits it with Levenberg-Marquardt from a poor initial guess,
prints the parameter recovery and the cost dropping each iteration, and draws the noisy data with the
initial guess and the fitted curve.

    python examples/levenberg_marquardt_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from levenberg_marquardt import (  # noqa: E402
    levenberg_marquardt,
    make_residual,
    gaussian_model,
)


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

    print("Levenberg-Marquardt: nonlinear least-squares fit of a Gaussian peak\n")

    true_p = [4.0, 2.0, 0.7]   # amplitude, center, width
    rng = _lcg(2024)
    xs = [i * 0.1 for i in range(50)]
    ys = [gaussian_model(x, true_p) + 0.15 * (rng() - 0.5) for x in xs]

    guess = [1.0, 1.0, 1.5]
    res = make_residual(gaussian_model, xs, ys)
    out = levenberg_marquardt(res, guess)

    print(f"  true parameters:  amp={true_p[0]:.3f}  center={true_p[1]:.3f}  width={true_p[2]:.3f}")
    print(f"  initial guess:    amp={guess[0]:.3f}  center={guess[1]:.3f}  width={guess[2]:.3f}")
    print(f"  LM fit:           amp={out['params'][0]:.3f}  center={out['params'][1]:.3f}  "
          f"width={out['params'][2]:.3f}")
    print(f"  converged in {out['iterations']} iterations, final SSR = {out['cost']:.4f}\n")

    print(f"  sum of squared residuals per accepted step:")
    h = out["history"]
    for i, s in enumerate(h):
        if i < 8 or i == len(h) - 1:
            bar = "#" * int(40 * s / h[0]) if h[0] > 0 else ""
            print(f"    step {i:>2}: {s:>10.4f}  {bar}")

    if out["covariance"]:
        stderr = [math.sqrt(abs(out["covariance"][i][i])) for i in range(3)]
        print(f"\n  parameter std-errors from the covariance: "
              f"amp +/-{stderr[0]:.3f}, center +/-{stderr[1]:.3f}, width +/-{stderr[2]:.3f}")

    _svg(os.path.join(outdir, "levenberg_marquardt.svg"), xs, ys, guess, out["params"])
    print(f"\n  wrote {os.path.join(outdir, 'levenberg_marquardt.svg')}")


def _svg(path, xs, ys, guess, fit, width=760, height=400):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Levenberg-Marquardt: noisy data, poor initial guess, converged Gaussian fit</text>',
    ]
    ox, oy, ow, oh = 50, 45, width - 90, height - 100
    xmin, xmax = min(xs), max(xs)
    ymin = min(min(ys), 0.0)
    ymax = max(max(ys), max(gaussian_model(x, fit) for x in xs)) * 1.1

    def sx(x):
        return ox + ow * (x - xmin) / (xmax - xmin)

    def sy(y):
        return oy + oh * (1 - (y - ymin) / (ymax - ymin))

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')

    # smooth curves
    def curve(params, color, dash=""):
        pts = []
        k = 0
        while k <= 200:
            x = xmin + (xmax - xmin) * k / 200
            pts.append(f"{sx(x):.1f},{sy(gaussian_model(x, params)):.1f}")
            k += 1
        da = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
                     f'stroke-width="2"{da}/>')

    curve(guess, "#8b949e", "5 4")   # initial guess
    curve(fit, "#06d6a0")            # fitted

    # data points
    for i in range(len(xs)):
        parts.append(f'<circle cx="{sx(xs[i]):.1f}" cy="{sy(ys[i]):.1f}" r="2.5" fill="#4dabf7"/>')

    parts.append(f'<circle cx="{ox+ow-150}" cy="{oy+8}" r="2.5" fill="#4dabf7"/>')
    parts.append(f'<text x="{ox+ow-142}" y="{oy+11}" fill="#e6edf3" font-size="10">noisy data</text>')
    parts.append(f'<rect x="{ox+ow-152}" y="{oy+22}" width="12" height="3" fill="#8b949e"/>')
    parts.append(f'<text x="{ox+ow-138}" y="{oy+27}" fill="#e6edf3" font-size="10">initial guess</text>')
    parts.append(f'<rect x="{ox+ow-152}" y="{oy+38}" width="12" height="3" fill="#06d6a0"/>')
    parts.append(f'<text x="{ox+ow-138}" y="{oy+43}" fill="#e6edf3" font-size="10">LM fit</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
