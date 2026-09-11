"""Demo: Diffie-Hellman key exchange -- a shared secret over an open line.

Walks Alice and Bob through the exchange (each ends with the same secret while the eavesdropper
sees only the public values), then shows the attacker's cost: breaking it means a discrete log,
whose baby-step/giant-step cost grows like sqrt(p) while the honest parties' work grows only
like log(p).

    python examples/diffie_hellman_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffie_hellman import (make_parameters, public_key, shared_secret,  # noqa: E402
                            exchange, discrete_log_bsgs, modexp)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    p, g = make_parameters(nbits=16, seed=3)
    a, b = 12345, 54321
    A, B, sa, sb = exchange(a, b, p, g)

    print("Diffie-Hellman: agree on a secret while an eavesdropper listens\n")
    print(f"  public parameters:  p = {p}  (safe prime),  g = {g}  (generator)\n")
    print(f"  {'':>8}{'secret':>10}{'sends g^secret mod p':>24}{'computes':>26}")
    print(f"  {'Alice':>8}{a:>10}{A:>24}{'B^a mod p = ' + str(sa):>26}")
    print(f"  {'Bob':>8}{b:>10}{B:>24}{'A^b mod p = ' + str(sb):>26}")
    print(f"\n  shared secret g^(ab) mod p = {sa}   (both match: {sa == sb})")
    print(f"  the wire carried only p, g, {A}, {B} -- never a or b.\n")

    # the attacker: recover a from A by discrete log
    cracked = discrete_log_bsgs(g, A, p)
    print(f"  An eavesdropper must solve g^x = {A} (mod {p}) for x. Baby-step/giant-step finds")
    print(f"  x = {cracked} in ~sqrt(p) = {int(math.isqrt(p))} steps -- feasible ONLY because p is tiny here.")
    print("  Scale p to 2048 bits and sqrt(p) is ~10^308 steps: the secret is safe, even though")
    print("  Alice and Bob each did only a few thousand multiplications. That gap is the point.")

    _svg(os.path.join(outdir, "diffie_hellman.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'diffie_hellman.svg')}")


def _svg(path, w=760, h=390):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Diffie-Hellman: a shared secret over an open channel</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'the exchange both sides can complete (left); the attacker\'s '
        f'sqrt(p) vs honest log(p) cost (right)</text>',
    ]

    # left: exchange diagram
    def box(x, y, wd, ht, stroke, lines):
        s = [f'<rect x="{x}" y="{y}" width="{wd}" height="{ht}" rx="6" fill="#161b22" '
             f'stroke="{stroke}" stroke-width="1.5"/>']
        for i, ln in enumerate(lines):
            s.append(f'<text x="{x + 8:.1f}" y="{y + 17 + i*14:.1f}" fill="#e6edf3" '
                     f'font-size="9">{ln}</text>')
        return s

    parts += box(35, 75, 165, 66, "#4dabf7",
                 ["Alice", "secret a", "sends A = g^a mod p", "gets B -> B^a = g^ab"])
    parts += box(35, 240, 165, 66, "#06d6a0",
                 ["Bob", "secret b", "sends B = g^b mod p", "gets A -> A^b = g^ab"])
    # exchange arrows through the open channel
    parts.append('<defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" '
                 'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#ffd43b"/></marker></defs>')
    parts.append(f'<line x1="120" y1="141" x2="120" y2="240" stroke="#ffd43b" stroke-width="1.5" '
                 f'marker-end="url(#ar)"/>')
    parts.append(f'<line x1="150" y1="240" x2="150" y2="141" stroke="#ffd43b" stroke-width="1.5" '
                 f'marker-end="url(#ar)"/>')
    parts.append(f'<text x="128" y="175" fill="#ffd43b" font-size="9">A</text>')
    parts.append(f'<text x="156" y="205" fill="#ffd43b" font-size="9">B</text>')
    # eavesdropper
    parts += box(240, 150, 175, 66, "#ff6b6b",
                 ["Eve sees p, g, A, B", "needs a = log_g A mod p", "discrete log: ~sqrt(p) hard"])
    parts.append(f'<text x="118" y="335" fill="#8b949e" font-size="10">Both reach g^ab; Eve never does.</text>')

    # right: cost curves -- sqrt(p) attacker vs log2(p) honest, over bit-length
    rx0, rx1 = w // 2 + 55, w - 25
    ry0, ry1 = h - 60, 75
    bits = list(range(16, 2049, 128))
    # work in log10(steps) so both fit: attacker ~ 10^(bits/2 * log10(2)), honest ~ bits
    atk_log10 = [b * math.log10(2) / 2 for b in bits]   # log10(sqrt(2^b)) = (b/2) log10 2
    hon_log10 = [math.log10(2 * b) for b in bits]        # ~2b multiplications
    ymax = max(atk_log10)

    def RX(b):
        return rx0 + b / bits[-1] * (rx1 - rx0)

    def RY(v):
        return ry0 - v / ymax * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    atk = " ".join(f"{RX(b):.1f},{RY(v):.1f}" for b, v in zip(bits, atk_log10))
    hon = " ".join(f"{RX(b):.1f},{RY(v):.1f}" for b, v in zip(bits, hon_log10))
    parts.append(f'<polyline points="{atk}" fill="none" stroke="#ff6b6b" stroke-width="2.5"/>')
    parts.append(f'<polyline points="{hon}" fill="none" stroke="#06d6a0" stroke-width="2.5"/>')
    parts.append(f'<text x="{RX(2048):.1f}" y="{RY(atk_log10[-1])+4:.1f}" fill="#ff6b6b" '
                 f'font-size="9" text-anchor="end">attacker ~sqrt(p)</text>')
    parts.append(f'<text x="{RX(2048):.1f}" y="{RY(hon_log10[-1])-5:.1f}" fill="#06d6a0" '
                 f'font-size="9" text-anchor="end">honest ~log(p)</text>')
    # mark 2048-bit attacker cost
    parts.append(f'<text x="{rx0+6:.1f}" y="{ry1+4:.1f}" fill="#8b949e" font-size="9">'
                 f'log10(steps)</text>')
    parts.append(f'<text x="{RX(2048):.1f}" y="{RY(atk_log10[-1])-6:.1f}" fill="#ff6b6b" '
                 f'font-size="9" text-anchor="end">~10^308</text>')
    for b in (16, 1024, 2048):
        parts.append(f'<text x="{RX(b):.1f}" y="{ry0+15:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{b}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">prime size (bits)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
