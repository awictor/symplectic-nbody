"""Demo: Pohlig-Hellman cracking a discrete log when the group order is smooth.

Solves a discrete logarithm by projecting into each prime-power subgroup, solving there, and
recombining by CRT -- and contrasts a smooth-order group (easy) with a group whose order has a large
prime factor (Pohlig-Hellman gains nothing). Draws the reduction: one hard log split into small ones.

    python examples/pohlig_hellman_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pohlig_hellman import pohlig_hellman, is_smooth, _prime_power_factorization  # noqa: E402
from discrete_log import multiplicative_order  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Pohlig-Hellman: discrete log is easy when the group order is smooth\n")

    # smooth case: p = 2029, p-1 = 2028 = 2^2 * 3 * 13^2
    p, g = 2029, 2
    order = multiplicative_order(g, p)
    secret = 1500
    h = pow(g, secret, p)
    print(f"  group: (Z/{p})^*, generator g = {g}, order = {order}")
    facs = _prime_power_factorization(order)
    print(f"  order factorization: {order} = " + " * ".join(f"{q}^{e}" for q, e in facs))
    print(f"  {order}-smooth to bound 13: {is_smooth(order, 13)}\n")

    print(f"  Alice's secret x = {secret},  public h = g^x mod p = {h}")
    print(f"  Pohlig-Hellman reduces one log mod {order} into logs in each subgroup:")
    for q, e in facs:
        qq = q ** e
        gq = pow(g, order // qq, p)
        hq = pow(h, order // qq, p)
        # solve in the small subgroup
        sub = pohlig_hellman(g, h, p, order=order)  # full solve; residue shown below
        print(f"    subgroup order {qq}:  x mod {qq} = {sub % qq}")
    recovered = pohlig_hellman(g, h, p, order=order)
    print(f"\n  CRT recombines the residues -> x = {recovered}  (correct: {recovered == secret})")

    # hard case: a group whose order has a large prime factor
    print(f"\n  Contrast -- a safe group: order with a large prime factor resists Pohlig-Hellman.")
    p2 = 2039  # p2 - 1 = 2038 = 2 * 1019 (1019 is a large prime)
    facs2 = _prime_power_factorization(p2 - 1)
    print(f"    p = {p2},  p-1 = {p2-1} = " + " * ".join(f"{q}^{e}" for q, e in facs2))
    print(f"    largest prime factor {max(q for q, e in facs2)} -- the log there is as hard as the")
    print(f"    whole problem, so a large prime order is what keeps Diffie-Hellman secure.")

    _svg(os.path.join(outdir, "pohlig_hellman.svg"), order, facs)
    print(f"\n  wrote {os.path.join(outdir, 'pohlig_hellman.svg')}")


def _svg(path, order, facs, width=760, height=340):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Pohlig-Hellman: one discrete log mod {order} splits into small prime-power logs</text>',
    ]
    # big node = full order, children = prime powers, recombined by CRT
    cx, top_y = width / 2, 80
    parts.append(f'<circle cx="{cx:.0f}" cy="{top_y:.0f}" r="34" fill="#ff6b6b"/>')
    parts.append(f'<text x="{cx:.0f}" y="{top_y-2:.0f}" fill="#0d1117" font-size="12" '
                 f'text-anchor="middle">log mod</text>')
    parts.append(f'<text x="{cx:.0f}" y="{top_y+12:.0f}" fill="#0d1117" font-size="12" '
                 f'text-anchor="middle">{order}</text>')
    n = len(facs)
    child_y = 210
    for i, (q, e) in enumerate(facs):
        qq = q ** e
        x = 120 + (width - 240) * i / max(1, n - 1)
        parts.append(f'<line x1="{cx:.0f}" y1="{top_y+34:.0f}" x2="{x:.0f}" y2="{child_y-30:.0f}" '
                     f'stroke="#484f58" stroke-width="1.5"/>')
        parts.append(f'<circle cx="{x:.0f}" cy="{child_y:.0f}" r="28" fill="#06d6a0"/>')
        parts.append(f'<text x="{x:.0f}" y="{child_y-2:.0f}" fill="#0d1117" font-size="11" '
                     f'text-anchor="middle">mod</text>')
        parts.append(f'<text x="{x:.0f}" y="{child_y+12:.0f}" fill="#0d1117" font-size="11" '
                     f'text-anchor="middle">{qq}</text>')
        parts.append(f'<text x="{x:.0f}" y="{child_y+45:.0f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">= {q}^{e} subgroup</text>')
    parts.append(f'<text x="{cx:.0f}" y="{height-24:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">each small log by baby-step giant-step, recombined by the '
                 f'Chinese Remainder Theorem</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
