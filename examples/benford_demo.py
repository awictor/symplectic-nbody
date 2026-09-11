"""Demo: Benford's law -- the leading digit is not uniform.

Prints the Benford probabilities against the observed leading-digit frequencies of the first
1000 Fibonacci numbers and a uniform control, then draws both histograms over the Benford
curve so the multiplicative sequence hugs it while the uniform data does not.

    python examples/benford_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benford import (benford_distribution, benford_probability,  # noqa: E402
                     leading_digit_frequencies, chi_square, total_variation_distance,
                     fibonacci)


def _uniform_dataset():
    """A dataset whose leading digits are (roughly) uniform 1..9 -- a Benford control."""
    data = []
    for decade in (1, 10, 100, 1000):
        for d in range(1, 10):
            data += [d * decade + k for k in range(30)]
    return data


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    fib = fibonacci(1000)
    uni = _uniform_dataset()
    ben = benford_distribution()
    fib_f = leading_digit_frequencies(fib)
    uni_f = leading_digit_frequencies(uni)

    print("Benford's law: P(d) = log10(1 + 1/d) -- 1 leads ~30% of the time, 9 only ~4.6%\n")
    print(f"  {'digit':>6}{'Benford':>10}{'Fibonacci':>12}{'uniform':>10}")
    for i, d in enumerate(range(1, 10)):
        print(f"  {d:>6}{ben[i]:>10.3f}{fib_f[i]:>12.3f}{uni_f[i]:>10.3f}")

    print(f"\n  Fibonacci:  chi2 = {chi_square(fib):6.2f}   TV dist = {total_variation_distance(fib):.4f}   -> follows Benford")
    print(f"  uniform:    chi2 = {chi_square(uni):6.1f}   TV dist = {total_variation_distance(uni):.4f}   -> does NOT")
    print("\n  Multiplicative data (Fibonacci, powers, factorials, populations) spans many")
    print("  orders of magnitude, so its mantissa is uniform in log space -- and that maps")
    print("  to the logarithmic digit law. Forensic auditors flag fabricated ledgers and")
    print("  election tallies by their departure from it.")

    _svg(os.path.join(outdir, "benford.svg"), ben, fib_f, uni_f)
    print(f"\n  wrote {os.path.join(outdir, 'benford.svg')}")


def _svg(path, ben, fib_f, uni_f, w=760, h=420):
    x0, y0 = 60, 40
    pw, ph = w - 100, h - 110
    ymax = 0.35

    def BX(i):
        return x0 + (i + 0.5) / 9 * pw

    def BY(p):
        return y0 + ph - p / ymax * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Benford\'s law: leading digits follow log10(1 + 1/d)</text>',
    ]
    # axes + gridlines
    for p in (0.0, 0.1, 0.2, 0.3):
        yy = BY(p)
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" '
                     f'stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{x0-8}" y="{yy+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{p:.1f}</text>')

    bw = pw / 9 * 0.28
    for i in range(9):
        cx = BX(i)
        # Fibonacci bars (blue) and uniform bars (orange), side by side
        parts.append(f'<rect x="{cx-bw-1:.1f}" y="{BY(fib_f[i]):.1f}" width="{bw:.1f}" '
                     f'height="{y0+ph-BY(fib_f[i]):.1f}" fill="#4dabf7"/>')
        parts.append(f'<rect x="{cx+1:.1f}" y="{BY(uni_f[i]):.1f}" width="{bw:.1f}" '
                     f'height="{y0+ph-BY(uni_f[i]):.1f}" fill="#ff922b"/>')
        parts.append(f'<text x="{cx:.1f}" y="{y0+ph+16:.1f}" fill="#8b949e" font-size="11" '
                     f'text-anchor="middle">{i+1}</text>')
    # Benford curve (green) with markers
    curve = " ".join(f"{BX(i):.1f},{BY(ben[i]):.1f}" for i in range(9))
    parts.append(f'<polyline points="{curve}" fill="none" stroke="#06d6a0" stroke-width="2.5"/>')
    for i in range(9):
        parts.append(f'<circle cx="{BX(i):.1f}" cy="{BY(ben[i]):.1f}" r="3" fill="#06d6a0"/>')

    parts.append(f'<text x="{x0+pw/2:.1f}" y="{h-30:.1f}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">leading digit</text>')
    # legend
    lx, ly = x0 + pw - 150, y0 + 6
    parts.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" fill="#06d6a0"/>'
                 f'<text x="{lx+15}" y="{ly+9}" fill="#e6edf3" font-size="10">Benford P(d)</text>')
    parts.append(f'<rect x="{lx}" y="{ly+16}" width="10" height="10" fill="#4dabf7"/>'
                 f'<text x="{lx+15}" y="{ly+25}" fill="#e6edf3" font-size="10">Fibonacci (fits)</text>')
    parts.append(f'<rect x="{lx}" y="{ly+32}" width="10" height="10" fill="#ff922b"/>'
                 f'<text x="{lx+15}" y="{ly+41}" fill="#e6edf3" font-size="10">uniform (fails)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
