"""Demo: total-variation denoising -- removing noise while keeping edges sharp.

Denoises a noisy piecewise-constant signal with TV denoising and, for contrast, a moving average,
showing TV keeps the step edges crisp where the average smears them. Sweeps lambda to show the
fidelity/simplicity trade-off, and draws the noisy data with both reconstructions.

    python examples/tv_denoise_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tv_denoise import tv_denoise, total_variation, n_plateaus, lambda_path  # noqa: E402


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def moving_average(y, w):
    n = len(y)
    out = []
    for i in range(n):
        lo = max(0, i - w)
        hi = min(n, i + w + 1)
        out.append(sum(y[lo:hi]) / (hi - lo))
    return out


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Total-variation denoising: edge-preserving, unlike a blur\n")

    rng = _lcg(2024)
    clean = [0.0] * 25 + [4.0] * 25 + [1.5] * 25 + [3.0] * 25
    noisy = [c + (rng() - 0.5) * 1.2 for c in clean]

    tv = tv_denoise(noisy, lam=3.0)
    avg = moving_average(noisy, 4)

    def rmse(a):
        return (sum((a[i] - clean[i]) ** 2 for i in range(len(clean))) / len(clean)) ** 0.5

    print(f"  Signal: 4 constant levels + noise (100 samples).\n")
    print(f"  {'method':>18}  {'RMSE to truth':>14}  {'plateaus':>9}")
    print(f"  {'noisy input':>18}  {rmse(noisy):>14.4f}  {'--':>9}")
    print(f"  {'moving average':>18}  {rmse(avg):>14.4f}  {'--':>9}")
    print(f"  {'TV denoise':>18}  {rmse(tv):>14.4f}  {n_plateaus(tv, tol=0.05):>9}")
    print("\n  TV denoising recovers the flat levels with sharp edges; the moving average rounds")
    print("  every step into a ramp, so its error is higher near the jumps.\n")

    print("  Lambda trade-off (more smoothing -> fewer plateaus, lower total variation):")
    print(f"    {'lambda':>7}  {'total variation':>16}  {'plateaus':>9}")
    for lam, tvval, plat, _ in lambda_path(noisy, [0.5, 1.0, 3.0, 6.0, 12.0]):
        print(f"    {lam:>7.1f}  {tvval:>16.3f}  {plat:>9}")

    _svg(os.path.join(outdir, "tv_denoise.svg"), clean, noisy, tv, avg)
    print(f"\n  wrote {os.path.join(outdir, 'tv_denoise.svg')}")


def _svg(path, clean, noisy, tv, avg, width=760, height=400):
    n = len(clean)
    lo = min(min(noisy), 0) - 0.5
    hi = max(max(noisy), max(clean)) + 0.5
    ox, oy, ow, oh = 40, 55, width - 80, height - 100

    def px(i):
        return ox + ow * i / (n - 1)

    def py(v):
        return oy + oh * (1 - (v - lo) / (hi - lo))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Noisy dots, true steps (gray), TV denoise (green, sharp), moving average (orange, smeared)</text>',
    ]
    # noisy points
    for i in range(n):
        parts.append(f'<circle cx="{px(i):.1f}" cy="{py(noisy[i]):.1f}" r="1.6" fill="#8b949e" '
                     f'opacity="0.6"/>')
    # true
    cp = " ".join(f"{px(i):.1f},{py(clean[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{cp}" fill="none" stroke="#484f58" stroke-width="1.5" '
                 f'stroke-dasharray="4 3"/>')
    # moving average
    ap = " ".join(f"{px(i):.1f},{py(avg[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{ap}" fill="none" stroke="#ff922b" stroke-width="1.5"/>')
    # tv
    tp = " ".join(f"{px(i):.1f},{py(tv[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{tp}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
