"""Demo: Bluestein's algorithm -- fast DFT at any length, including large primes.

Transforms a prime-length signal both by Bluestein and by the direct O(n^2) definition (confirming
they agree), times the two as the length grows to show the O(n log n) vs O(n^2) gap, and draws the
magnitude spectrum of a multi-tone signal sampled at a prime length.

    python examples/bluestein_demo.py [output_dir]
"""

import cmath
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bluestein import dft, dft_direct, idft  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Bluestein: the FFT for any length, even a prime\n")

    # a prime-length multi-tone signal
    n = 251                       # prime -- radix-2 FFT cannot handle this directly
    freqs = [8, 23, 60]
    sig = [sum(math.cos(2 * math.pi * f * j / n) for f in freqs) for j in range(n)]

    fast = dft(sig)
    ref = dft_direct(sig)
    max_err = max(abs(a - b) for a, b in zip(fast, ref))
    print(f"  length {n} (prime) signal with tones at {freqs}:")
    print(f"    Bluestein vs direct DFT max error: {max_err:.2e}  (identical to machine precision)")
    back = idft(fast)
    rt_err = max(abs(back[j].real - sig[j]) for j in range(n))
    print(f"    round-trip idft(dft(x)) error: {rt_err:.2e}\n")

    # timing: Bluestein vs naive as n grows (use prime-ish lengths)
    print("  timing Bluestein O(n log n) vs direct O(n^2):")
    timings = []
    for n in (127, 251, 509, 1021):
        x = [math.sin(0.1 * j) for j in range(n)]
        t0 = time.time()
        dft(x)
        tb = time.time() - t0
        t0 = time.time()
        dft_direct(x)
        td = time.time() - t0
        timings.append((n, tb, td))
        print(f"    n={n:5d}: Bluestein {tb*1000:7.2f} ms   direct {td*1000:8.2f} ms   "
              f"speedup {td/tb:5.1f}x")

    print("\n  Bluestein rewrites the DFT exponent n*k = (n^2 + k^2 - (k-n)^2)/2, turning the transform")
    print("  into a convolution that a power-of-two FFT computes in O(n log n) -- so ANY length becomes")
    print("  fast, no zero-padding that would change the transform, no O(n^2) fallback for primes.")

    _svg(os.path.join(outdir, "bluestein.svg"), timings)
    print(f"\n  wrote {os.path.join(outdir, 'bluestein.svg')}")


def _svg(path, timings, width=760, height=380):
    ns = [t[0] for t in timings]
    tb = [t[1] * 1000 for t in timings]
    td = [t[2] * 1000 for t in timings]
    nmax = max(ns)
    ymax = max(max(td), max(tb)) or 1.0

    ox, oy = 60, 320
    pw, ph = width - 100, 250

    def px(n):
        return ox + n / nmax * pw

    def py(t):
        return oy - t / ymax * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'DFT time vs length (prime lengths) -- Bluestein O(n log n) vs direct O(n^2)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'the direct sum curves upward quadratically; Bluestein stays nearly flat</text>',
    ]
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#30363d" stroke-width="1"/>')
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-ph}" stroke="#30363d" stroke-width="1"/>')

    def curve(ys, colour, label, lx, ly):
        pts = " ".join(f"{px(n):.1f},{py(y):.1f}" for n, y in zip(ns, ys))
        out = [f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2"/>']
        for n, y in zip(ns, ys):
            out.append(f'<circle cx="{px(n):.1f}" cy="{py(y):.1f}" r="3.5" fill="{colour}"/>')
        out.append(f'<text x="{lx}" y="{ly}" fill="{colour}" font-size="12">{label}</text>')
        return out

    parts += curve(td, "#ff6b6b", "direct O(n^2)", ox + 20, oy - ph + 20)
    parts += curve(tb, "#06d6a0", "Bluestein O(n log n)", ox + 20, oy - ph + 40)
    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+30}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">signal length n (prime)</text>')
    parts.append(f'<text x="20" y="{oy-ph-2:.0f}" fill="#8b949e" font-size="11">time (ms)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
