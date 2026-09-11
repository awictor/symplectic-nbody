"""Demo: RSA public-key cryptography from scratch.

Generates a small keypair, walks a message through encrypt -> decrypt and sign -> verify, and
shows the square-and-multiply ladder that makes modular exponentiation fast. Draws the
key-exchange flow and the exponentiation cost (linear in bits, not in the exponent).

    python examples/rsa_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rsa import (generate_keypair, encrypt, decrypt, sign, verify,  # noqa: E402
                 encrypt_bytes, decrypt_bytes, modexp)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    pub, priv = generate_keypair(nbits=128, seed=7)
    e, n = pub
    d, _ = priv
    print("RSA from scratch: two primes -> a public key anyone can encrypt to\n")
    print(f"  public key  (e, n): e = {e}")
    print(f"                      n = {n}")
    print(f"  private key (d, n): d = {d}\n")

    m = 42424242
    c = encrypt(m, pub)
    back = decrypt(c, priv)
    print(f"  message   m = {m}")
    print(f"  encrypt   c = m^e mod n = {c}")
    print(f"  decrypt   m'= c^d mod n = {back}   -> round-trip {'OK' if back == m else 'FAIL'}")

    s = sign(m, priv)
    print(f"\n  sign      s = m^d mod n = {s}")
    print(f"  verify    s^e mod n = m ? {verify(m, s, pub)}   (tampered: {verify(m + 1, s, pub)})")

    text = b"public-key crypto, no shared secret"
    blocks = encrypt_bytes(text, pub)
    dec = decrypt_bytes(blocks, priv)
    print(f"\n  bytes: {text!r}")
    print(f"    -> {len(blocks)} ciphertext blocks -> decrypts to {dec!r}  "
          f"({'OK' if dec == text else 'FAIL'})")
    print("\n  Multiplying the primes is easy; factoring n back apart is not -- that gap is the")
    print("  whole of RSA's security. Modular exponentiation costs ~log2(e) multiplications by")
    print("  square-and-multiply, so even a 2048-bit exponent is fast.")

    _svg(os.path.join(outdir, "rsa.svg"), pub, priv)
    print(f"\n  wrote {os.path.join(outdir, 'rsa.svg')}")


def _svg(path, pub, priv, w=760, h=380):
    e, n = pub
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'RSA: encrypt with the public key, decrypt with the private</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'the one-way key flow (left); modular-exponentiation cost is linear in bits (right)</text>',
    ]

    # left: flow diagram Alice -> channel -> Bob
    def box(x, y, wd, ht, fill, stroke, lines):
        s = [f'<rect x="{x}" y="{y}" width="{wd}" height="{ht}" rx="6" fill="{fill}" '
             f'stroke="{stroke}" stroke-width="1.5"/>']
        for i, ln in enumerate(lines):
            s.append(f'<text x="{x + wd/2:.1f}" y="{y + 18 + i*15:.1f}" fill="#e6edf3" '
                     f'font-size="10" text-anchor="middle">{ln}</text>')
        return s

    lx = 40
    parts += box(lx, 80, 150, 60, "#161b22", "#4dabf7",
                 ["Alice", "has public (e, n)", "c = m^e mod n"])
    parts += box(lx, 220, 150, 60, "#161b22", "#06d6a0",
                 ["Bob", "has private (d, n)", "m = c^d mod n"])
    # arrow with ciphertext
    ax = lx + 75
    parts.append(f'<line x1="{ax}" y1="140" x2="{ax}" y2="220" stroke="#ffd43b" '
                 f'stroke-width="2" marker-end="url(#ah)"/>')
    parts.append('<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="3" '
                 'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#ffd43b"/></marker></defs>')
    parts.append(f'<text x="{ax + 8}" y="185" fill="#ffd43b" font-size="10">ciphertext c</text>')
    parts.append(f'<text x="{ax + 8}" y="200" fill="#8b949e" font-size="8">(safe on open channel)</text>')
    # eavesdropper
    parts += box(lx + 210, 150, 150, 60, "#161b22", "#ff6b6b",
                 ["Eve (attacker)", "sees c, e, n", "must factor n: hard"])

    # right: cost of modexp -- number of multiplications ~ 2*log2(e) vs the exponent size
    rx0, rx1 = w // 2 + 40, w - 30
    ry0, ry1 = h - 60, 80
    bits = list(range(8, 2049, 128))
    mults = [2 * b for b in bits]  # ~2 per bit (one square + up to one multiply)
    mmax = max(mults)

    def RX(b):
        return rx0 + b / bits[-1] * (rx1 - rx0)

    def RY(v):
        return ry0 - v / mmax * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    pts = " ".join(f"{RX(b):.1f},{RY(m):.1f}" for b, m in zip(bits, mults))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="2.5"/>')
    for b in (bits[0], 1024, 2048):
        parts.append(f'<circle cx="{RX(b):.1f}" cy="{RY(2*b):.1f}" r="2.5" fill="#4dabf7"/>')
        parts.append(f'<text x="{RX(b):.1f}" y="{ry0+15:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{b}</text>')
    parts.append(f'<text x="{RX(2048):.1f}" y="{RY(2*2048)-6:.1f}" fill="#4dabf7" font-size="9" '
                 f'text-anchor="end">~2 log2(e) mults</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">exponent size (bits)</text>')
    parts.append(f'<text x="{rx0-6:.1f}" y="{ry1-2:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">multiplications</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry1-14:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="middle">(vs 2^bits for naive repeated multiply)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
