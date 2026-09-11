"""Demo: Pollard's rho factorization, and why it crushes trial division.

Factors a range of numbers including RSA-style semiprimes, shows Pollard's rho finding factors in
about sqrt(p) iterations where trial division would take p, and draws the iteration count against the
sqrt(smallest factor) it tracks.

    python examples/pollard_rho_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pollard_rho import factorize, is_prime, factor_pairs, euler_phi, num_divisors  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Pollard's rho: factoring integers in ~sqrt(smallest factor) steps\n")

    examples = [600851475143, 1000000007 * 1000000009, 720720, 2 ** 20 * 3 ** 5,
                1000003 * 1000033]
    for n in examples:
        pairs = factor_pairs(n)
        pretty = " * ".join(f"{p}^{e}" if e > 1 else f"{p}" for p, e in pairs)
        print(f"  {n} = {pretty}")

    print("\n  RSA-style semiprimes (product of two primes) -- the case RSA relies on being hard:")
    for p, q in [(101, 103), (10007, 10009), (1000003, 1000033), (100000007, 100000037)]:
        n = p * q
        fs = sorted(factorize(n))
        print(f"    {n:>22d} = {fs[0]} * {fs[1]}")

    # iterations vs sqrt(smallest factor): time a few factorizations by counting rho steps
    print("\n  Pollard's rho finds a factor p in about sqrt(p) iterations (birthday paradox),")
    print("  where trial division would need p/2. For a 15-digit semiprime that is the")
    print("  difference between a few thousand steps and a hundred million.")

    # totient / divisor count fall out of the factorization
    print("\n  Number-theoretic functions from the factorization:")
    for n in [36, 100, 720720]:
        print(f"    n={n}: phi(n)={euler_phi(n)}, number of divisors={num_divisors(n)}")

    # data for the plot: iteration proxy = product of factor square roots
    _svg(os.path.join(outdir, "pollard_rho.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'pollard_rho.svg')}")


def _svg(path, width=760, height=430):
    # show iterations-to-factor vs sqrt(smallest prime factor) for a set of semiprimes:
    # empirically count the rho steps.
    from pollard_rho import is_prime
    from math import gcd

    def rho_steps(n):
        if n % 2 == 0:
            return 1
        x = y = 2
        c = 1
        d = 1
        steps = 0
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = gcd(abs(x - y), n)
            steps += 1
            if steps > 10 ** 6:
                break
        return steps

    # build semiprimes p*q with a small factor p of increasing size
    small_primes = [p for p in range(50, 100000) if is_prime(p)]
    data = []
    big = 100000007
    for target in [60, 200, 700, 2500, 9000, 30000]:
        # pick the smallest prime >= target
        p = next(pp for pp in small_primes if pp >= target)
        n = p * big
        steps = rho_steps(n)
        data.append((p, steps))

    m_left, m_bot, m_top, m_right = 70, 60, 80, 40
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    xs = [math.sqrt(d[0]) for d in data]
    ys = [d[1] for d in data]
    xmax = max(xs) * 1.1
    ymax = max(ys) * 1.15

    def px(x):
        return m_left + x / xmax * pw

    def py(y):
        return m_top + ph - y / ymax * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Pollard rho: iterations to find factor p scale with sqrt(p)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'blue points = measured rho iterations; yellow dashed = the sqrt(p) trend line</text>',
    ]

    parts.append(f'<line x1="{m_left}" y1="{m_top+ph}" x2="{m_left+pw}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<text x="{m_left+pw/2:.0f}" y="{height-14}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">sqrt(smallest prime factor)</text>')

    # trend line y = c * x fit through the points (least-ish squares slope)
    slope = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    tx = [0, xmax]
    ty = [slope * x for x in tx]
    parts.append(f'<line x1="{px(tx[0]):.1f}" y1="{py(ty[0]):.1f}" x2="{px(tx[1]):.1f}" '
                 f'y2="{py(min(ty[1], ymax)):.1f}" stroke="#ffd43b" stroke-width="1.8" '
                 f'stroke-dasharray="6,4"/>')

    for x, y in zip(xs, ys):
        parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="5" fill="#4dabf7"/>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
