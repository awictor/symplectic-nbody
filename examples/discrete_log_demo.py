"""Demo: baby-step giant-step discrete logarithm, and breaking a toy Diffie-Hellman.

Solves discrete logs by meet-in-the-middle, breaks a small Diffie-Hellman key exchange by recovering
the secret exponent, and shows the O(sqrt(n)) work versus brute force. Draws the work saved.

    python examples/discrete_log_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from discrete_log import discrete_log, brute_discrete_log, multiplicative_order  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Baby-step giant-step: discrete logarithm in O(sqrt(n))\n")

    print("  Solving g^x = h (mod p):")
    for g, h, p in [(3, 13, 17), (2, 22, 29), (7, 6, 41)]:
        x = discrete_log(g, h, p)
        print(f"    {g}^x = {h} (mod {p})  ->  x = {x}  (check: {g}^{x} = {pow(g, x, p)})")

    # break a toy Diffie-Hellman
    print("\n  Breaking a toy Diffie-Hellman key exchange:")
    p, g = 7919, 7
    alice_secret = 5555
    bob_secret = 1234
    A = pow(g, alice_secret, p)      # Alice's public value
    B = pow(g, bob_secret, p)        # Bob's public value
    shared = pow(B, alice_secret, p)
    print(f"    public: p={p}, g={g}, A={A}, B={B}")
    print(f"    (Alice and Bob's shared secret: {shared})")
    # an eavesdropper recovers a secret from a public value
    recovered_a = discrete_log(g, A, p)
    eaves_shared = pow(B, recovered_a, p)
    print(f"    eavesdropper solves g^x = A -> x = {recovered_a} (Alice's secret was {alice_secret})")
    print(f"    -> reconstructs the shared secret {eaves_shared}: {eaves_shared == shared}")
    print(f"    (this works only because p is tiny; real DH uses 2048+ bit primes)")

    # work comparison
    print("\n  Baby-step giant-step vs brute force (steps to solve):")
    for p in [101, 1009, 10007, 100003]:
        g = 5
        n = p - 1
        bsgs_steps = 2 * (int(math.isqrt(n)) + 1)
        print(f"    p={p:>7}: BSGS ~ 2*sqrt(n) = {bsgs_steps:>4} steps   vs brute force n = {n} steps")

    print("\n  Write x = i*N + j with N = ceil(sqrt(n)). Precompute the baby steps g^j in a hash map,")
    print("  then take giant steps h*(g^-N)^i and look each up. A hit gives x = i*N + j. Both loops")
    print("  run sqrt(n) times -- exponentially faster than brute force, but still exponential in bits.")

    _svg(os.path.join(outdir, "discrete_log.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'discrete_log.svg')}")


def _svg(path, width=760, height=420):
    m_left, m_bot, m_top, m_right = 70, 60, 80, 40
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    ns = [100, 1000, 10000, 100000, 1000000]
    brute = ns
    bsgs = [2 * (int(math.isqrt(n)) + 1) for n in ns]

    xs = [math.log10(n) for n in ns]
    xmin, xmax = min(xs), max(xs)
    ymax = math.log10(max(brute))

    def px(lx):
        return m_left + (lx - xmin) / (xmax - xmin) * pw

    def py(v):
        return m_top + ph - math.log10(v) / ymax * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Discrete log: baby-step giant-step vs brute force (log-log)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'green = O(sqrt(n)) BSGS steps, red = O(n) brute force; the gap widens as the group grows</text>',
    ]
    parts.append(f'<line x1="{m_left}" y1="{m_top+ph}" x2="{m_left+pw}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')

    bpts = " ".join(f"{px(lx):.1f},{py(v):.1f}" for lx, v in zip(xs, brute))
    gpts = " ".join(f"{px(lx):.1f},{py(v):.1f}" for lx, v in zip(xs, bsgs))
    parts.append(f'<polyline points="{bpts}" fill="none" stroke="#ff6b6b" stroke-width="2"/>')
    parts.append(f'<polyline points="{gpts}" fill="none" stroke="#06d6a0" stroke-width="2.2"/>')
    for lx, v in zip(xs, brute):
        parts.append(f'<circle cx="{px(lx):.1f}" cy="{py(v):.1f}" r="3" fill="#ff6b6b"/>')
    for lx, v in zip(xs, bsgs):
        parts.append(f'<circle cx="{px(lx):.1f}" cy="{py(v):.1f}" r="3" fill="#06d6a0"/>')
    for lx, n in zip(xs, ns):
        parts.append(f'<text x="{px(lx):.0f}" y="{m_top+ph+18:.0f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{n}</text>')
    parts.append(f'<text x="{m_left+pw/2:.0f}" y="{height-12}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">group order n (log scale)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
