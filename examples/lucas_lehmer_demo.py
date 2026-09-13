"""Demo: the Lucas-Lehmer test hunting Mersenne primes -- how the largest known primes are found.

Runs the Lucas-Lehmer test over exponents, listing the Mersenne primes M_p = 2^p - 1 it certifies and
the composites it rejects, shows the s_k recurrence for a small case, and reports the digit sizes of
the larger Mersenne primes. Draws which exponents yield Mersenne primes.

    python examples/lucas_lehmer_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lucas_lehmer import lucas_lehmer, mersenne_prime_exponents, _mersenne_mod  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Lucas-Lehmer: the deterministic test behind every record-breaking prime\n")

    # show the s_k recurrence for M_7 = 127
    p = 7
    M = (1 << p) - 1
    print(f"  M_{p} = 2^{p} - 1 = {M}")
    s = 4
    seq = [s]
    for _ in range(p - 2):
        s = _mersenne_mod(s * s - 2, p)
        seq.append(s)
    print(f"  s_0=4, s_(k+1)=s_k^2-2 mod {M}:  {seq}")
    print(f"  s_(p-2) = s_{p-2} = {seq[-1]}  ->  {'PRIME' if seq[-1] == 0 else 'composite'}\n")

    # scan exponents
    exps = mersenne_prime_exponents(150)
    print(f"  Mersenne primes M_p for p <= 150:  p in {exps}")
    print(f"  (only {len(exps)} of the primes up to 150 yield a Mersenne prime -- they are rare)\n")

    print(f"  digit sizes of the Mersenne primes found:")
    print(f"    {'p':>5}{'digits of M_p':>16}")
    for p in exps:
        digits = int(p * math.log10(2)) + 1
        print(f"    {p:>5}{digits:>16}")

    print(f"\n  The test is p-2 squarings mod (2^p - 1) -- and the modular reduction is a cheap")
    print(f"  bit fold, so it scales to exponents in the tens of millions. That is exactly what")
    print(f"  the GIMPS project runs to find primes with over 20 million digits.")

    _svg(os.path.join(outdir, "lucas_lehmer.svg"), exps, 150)
    print(f"\n  wrote {os.path.join(outdir, 'lucas_lehmer.svg')}")


def _svg(path, exps, limit, width=760, height=300):
    from pollard_rho import is_prime
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Prime exponents p: which give a Mersenne prime M_p = 2^p - 1 (green)?</text>',
    ]
    primes = [p for p in range(2, limit + 1) if is_prime(p)]
    ox, oy, ow = 40, 90, width - 80
    mset = set(exps)
    n = len(primes)
    for i, p in enumerate(primes):
        x = ox + ow * i / (n - 1)
        is_mp = p in mset
        color = "#06d6a0" if is_mp else "#30363d"
        r = 9 if is_mp else 5
        parts.append(f'<circle cx="{x:.1f}" cy="{oy:.1f}" r="{r}" fill="{color}"/>')
        if is_mp:
            parts.append(f'<text x="{x:.1f}" y="{oy-14:.1f}" fill="#06d6a0" font-size="10" '
                         f'text-anchor="middle">{p}</text>')
        if p in (2, 31, 127, 149) or i == n - 1:
            parts.append(f'<text x="{x:.1f}" y="{oy+20:.1f}" fill="#8b949e" font-size="8" '
                         f'text-anchor="middle">{p}</text>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+55:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">each dot is a prime p; green = M_p is also prime (a Mersenne prime)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
