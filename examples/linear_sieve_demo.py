"""Demo: the linear sieve -- primes, factorizations, totient, and Mobius in O(N).

Sieves up to N in linear time, shows the smallest-prime-factor factorization of a number, and plots
the Euler totient and Mobius functions, confirming the classic divisor identities.

    python examples/linear_sieve_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from linear_sieve import (linear_sieve, factorize_with_spf, brute_totient, brute_mobius)  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    N = 100
    S = linear_sieve(N)

    print("Linear sieve: primes + factorization + totient + Mobius, all in O(N)\n")
    print(f"  primes up to {N} ({len(S['primes'])} of them):")
    print(f"    {S['primes']}\n")

    for x in [360, 84, 97, 1000]:
        f = factorize_with_spf(x, linear_sieve(x)["spf"])
        s = " * ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(f.items()))
        print(f"  {x} = {s}   (via the smallest-prime-factor table, O(log n))")

    print(f"\n  Euler totient phi(n) -- integers below n coprime to it:")
    print(f"    n : {list(range(1, 13))}")
    print(f"    phi: {[S['phi'][i] for i in range(1, 13)]}")
    print(f"    verified vs brute coprime count: "
          f"{all(S['phi'][i] == brute_totient(i) for i in range(1, N + 1))}")

    print(f"\n  Mobius mu(n) -- 0 if square factor, else (-1)^(#distinct primes):")
    print(f"    n : {list(range(1, 13))}")
    print(f"    mu: {[S['mu'][i] for i in range(1, 13)]}")
    print(f"    verified vs brute: "
          f"{all(S['mu'][i] == brute_mobius(i) for i in range(1, N + 1))}")

    # classic identities
    id1 = all(sum(S['phi'][d] for d in range(1, n + 1) if n % d == 0) == n for n in range(1, 60))
    id2 = all(sum(S['mu'][d] for d in range(1, n + 1) if n % d == 0) == (1 if n == 1 else 0)
              for n in range(1, 60))
    print(f"\n  identity: sum of phi(d) over divisors of n equals n -> {id1}")
    print(f"  identity: sum of mu(d) over divisors of n is [n==1]  -> {id2}")

    print("\n  Each composite is struck exactly once, by its smallest prime factor -- the inner loop")
    print("  breaks the moment that prime divides i, avoiding the double-marking of Eratosthenes and")
    print("  making the whole sieve, plus the multiplicative phi and mu arrays, truly O(N).")

    _svg(os.path.join(outdir, "linear_sieve.svg"), S, N)
    print(f"\n  wrote {os.path.join(outdir, 'linear_sieve.svg')}")


def _svg(path, S, N, width=780, height=430):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'Euler totient phi(n) (blue) and Mobius mu(n) (bars) up to {N}</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'phi hugs n for primes and dips for smooth numbers; mu is +1/0/-1</text>',
    ]

    ox, oy = 50, 250
    plot_w = width - 100
    # phi plot (top region)
    max_phi = max(S['phi'][1:N + 1])

    def px(n):
        return ox + (n - 1) / (N - 1) * plot_w

    def py_phi(v):
        return oy - v / max_phi * 150

    pts = " ".join(f"{px(n):.1f},{py_phi(S['phi'][n]):.1f}" for n in range(1, N + 1))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="1.3"/>')
    # the n line (phi <= n-1)
    parts.append(f'<line x1="{px(1):.1f}" y1="{py_phi(1):.1f}" x2="{px(N):.1f}" '
                 f'y2="{py_phi(max_phi):.1f}" stroke="#8b949e" stroke-width="0.8" '
                 f'stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{ox}" y="{oy+18}" fill="#4dabf7" font-size="11">phi(n)</text>')

    # mu bars (bottom region)
    mu_base = oy + 100
    parts.append(f'<line x1="{ox}" y1="{mu_base}" x2="{ox+plot_w}" y2="{mu_base}" '
                 f'stroke="#30363d"/>')
    for n in range(1, N + 1):
        m = S['mu'][n]
        if m == 0:
            continue
        x = px(n)
        h = 28 * m
        col = "#06d6a0" if m > 0 else "#ff6b6b"
        parts.append(f'<line x1="{x:.1f}" y1="{mu_base}" x2="{x:.1f}" y2="{mu_base-h:.1f}" '
                     f'stroke="{col}" stroke-width="1.5"/>')
    parts.append(f'<text x="{ox}" y="{mu_base+42}" fill="#8b949e" font-size="11">'
                 f'mu(n): green +1, red -1, gap 0 (has a square factor)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
