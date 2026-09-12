"""Demo: elliptic-curve cryptography -- the group law, ECDH key exchange, and ECDSA signatures.

Shows the point-addition group law on a small curve, runs a full ECDH exchange (Alice and Bob agree on
a secret an eavesdropper cannot compute), signs and verifies a message with ECDSA, and draws the points
of the small curve with a chord-and-tangent addition.

    python examples/elliptic_curve_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from elliptic_curve import (curve_f17, curve_toy, generate_keypair,  # noqa: E402
                            ecdh_shared_secret, ecdsa_sign, ecdsa_verify, _LCG)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Elliptic-curve cryptography: a group law that hides a one-way function\n")

    c = curve_f17()
    print(f"  curve y^2 = x^3 + {c.a}x + {c.b} over F_{c.p}, base G = {c.G} (order {c.n})")
    P = c.G
    Q = c.scalar_mult(2, c.G)
    print(f"    G + G     = {Q}   (point doubling via the tangent line)")
    print(f"    G + 2G    = {c.add(P, Q)}")
    print(f"    3G        = {c.scalar_mult(3, c.G)}   (consistent)\n")

    # ECDH
    t = curve_toy()
    rng_a = _LCG(11)
    rng_b = _LCG(29)
    da, Qa = generate_keypair(t, rng_a)
    db, Qb = generate_keypair(t, rng_b)
    sa = ecdh_shared_secret(t, da, Qb)
    sb = ecdh_shared_secret(t, db, Qa)
    print(f"  ECDH key exchange on the toy curve (F_{t.p}, prime order {t.n}):")
    print(f"    Alice secret a={da}, publishes A = a*G = {Qa}")
    print(f"    Bob   secret b={db}, publishes B = b*G = {Qb}")
    print(f"    Alice computes a*B = {sa}")
    print(f"    Bob   computes b*A = {sb}")
    print(f"    shared secret agrees: {sa == sb}  (eavesdropper sees only A, B, G)\n")

    # ECDSA
    d, Q = generate_keypair(t, _LCG(7))
    msg = b"pay bob 10 coins"
    sig = ecdsa_sign(t, d, msg, _LCG(99))
    print(f"  ECDSA signature on {msg!r}:")
    print(f"    signature (r, s) = {sig}")
    print(f"    verifies with the correct key:  {ecdsa_verify(t, Q, msg, sig)}")
    print(f"    verifies on a tampered message: {ecdsa_verify(t, Q, b'pay bob 99 coins', sig)}")
    d2, Q2 = generate_keypair(t, _LCG(555))
    print(f"    verifies with the wrong key:    {ecdsa_verify(t, Q2, msg, sig)}\n")

    print("  Scalar multiplication k*G is fast by double-and-add, but recovering k from k*G and G is")
    print("  the elliptic-curve discrete logarithm problem -- believed exponentially hard. That")
    print("  asymmetry gives 256-bit curves the security of 3072-bit RSA: the backbone of modern TLS.")

    _svg(os.path.join(outdir, "elliptic_curve.svg"), c)
    print(f"\n  wrote {os.path.join(outdir, 'elliptic_curve.svg')}")


def _svg(path, c, size=480, pad=40):
    p = c.p
    pts = [(x, y) for x in range(p) for y in range(p) if c.is_on_curve((x, y))]
    scale = (size - 2 * pad) / (p - 1)

    def sx(x):
        return pad + x * scale

    def sy(y):
        return size - pad - y * scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size+220}" height="{size}" '
        f'viewBox="0 0 {size+220} {size}" font-family="monospace">',
        f'<rect width="{size+220}" height="{size}" fill="#0d1117"/>',
        f'<text x="{pad}" y="24" fill="#e6edf3" font-size="15">'
        f'Points of y^2 = x^3 + {c.a}x + {c.b} over F_{p}</text>',
        f'<rect x="{pad}" y="{pad}" width="{size-2*pad}" height="{size-2*pad}" '
        f'fill="#010409" stroke="#30363d"/>',
    ]
    # symmetry axis y = p/2
    parts.append(f'<line x1="{pad}" y1="{sy((p-1)/2):.1f}" x2="{size-pad}" y2="{sy((p-1)/2):.1f}" '
                 f'stroke="#30363d" stroke-width="0.7" stroke-dasharray="3 3"/>')
    # all points
    for x, y in pts:
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="#4dabf7"/>')
    # highlight the multiples of G in a colour cycle
    P = None
    cols = ["#ffd43b", "#06d6a0", "#ff922b", "#b197fc", "#ff6b6b"]
    for k in range(1, 8):
        P = c.add(P, c.G)
        if P is None:
            break
        parts.append(f'<circle cx="{sx(P[0]):.1f}" cy="{sy(P[1]):.1f}" r="5.5" fill="none" '
                     f'stroke="{cols[k % len(cols)]}" stroke-width="2"/>')
        parts.append(f'<text x="{sx(P[0])+7:.0f}" y="{sy(P[1])-6:.0f}" fill="{cols[k%len(cols)]}" '
                     f'font-size="10">{k}G</text>')

    lx = size + 10
    parts.append(f'<text x="{lx}" y="60" fill="#4dabf7" font-size="12">blue: all curve points</text>')
    parts.append(f'<text x="{lx}" y="82" fill="#ffd43b" font-size="12">rings: 1G, 2G, 3G, ...</text>')
    parts.append(f'<text x="{lx}" y="108" fill="#8b949e" font-size="11">points reflect across</text>')
    parts.append(f'<text x="{lx}" y="124" fill="#8b949e" font-size="11">the dashed axis: that</text>')
    parts.append(f'<text x="{lx}" y="140" fill="#8b949e" font-size="11">reflection IS negation</text>')
    parts.append(f'<text x="{lx}" y="166" fill="#8b949e" font-size="11">{len(pts)} points, base</text>')
    parts.append(f'<text x="{lx}" y="182" fill="#8b949e" font-size="11">order {c.n}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
