"""Demo: SHA-256 from scratch -- digests, the avalanche effect, and HMAC.

Hashes some inputs (matching hashlib exactly), demonstrates the avalanche effect (a one-bit change
scrambles about half the output), and shows HMAC-SHA256. Draws the digest bytes as a grid and the
avalanche bit-difference.

    python examples/sha256_demo.py [output_dir]
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sha256 import sha256, sha256_hex, hmac_sha256_hex  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("SHA-256 from scratch: the hash behind TLS, git, and Bitcoin\n")

    for msg in (b"", b"abc", b"The quick brown fox jumps over the lazy dog"):
        mine = sha256_hex(msg)
        ref = hashlib.sha256(msg).hexdigest()
        print(f"  sha256({msg!r:48}) = {mine[:32]}...")
        print(f"    matches hashlib: {mine == ref}")
    print()

    # avalanche
    a = sha256(b"avalanche")
    b = sha256(b"avalanchf")     # one character different
    diff = sum(bin(a[i] ^ b[i]).count("1") for i in range(32))
    print("  avalanche effect (change one character of the input):")
    print(f"    sha256('avalanche') vs sha256('avalanchf')")
    print(f"    {diff} of 256 output bits differ ({100*diff/256:.0f}% -- a total scramble)\n")

    # HMAC
    key = b"secret-key"
    print("  HMAC-SHA256 (keyed message authentication):")
    print(f"    HMAC(key, 'message 1') = {hmac_sha256_hex(key, b'message 1')[:32]}...")
    print(f"    HMAC(key, 'message 2') = {hmac_sha256_hex(key, b'message 2')[:32]}...")
    print("    (unforgeable without the key; how APIs sign requests)\n")

    print("  The message is padded to a multiple of 512 bits, split into blocks, and each block is")
    print("  folded into a 256-bit state by 64 rounds of bitwise mixing (choice, majority, rotations,")
    print("  and constants from the cube roots of primes). Every implementation must agree to the bit,")
    print("  which is why the whole internet can share one hash function.")

    _svg(os.path.join(outdir, "sha256.svg"), a, b)
    print(f"\n  wrote {os.path.join(outdir, 'sha256.svg')}")


def _svg(path, digest_a, digest_b, width=760, height=360):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'The avalanche effect: two inputs differing by one character</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'each cell is one digest byte (brightness = value); red marks bytes that changed</text>',
    ]

    def grid(digest, other, y0, label):
        out = [f'<text x="30" y="{y0-6}" fill="#8b949e" font-size="12">{label}</text>']
        cols = 16
        cell = 40
        ox = 30
        for i in range(32):
            r, c = divmod(i, cols)
            x = ox + c * cell
            y = y0 + r * cell
            v = digest[i]
            changed = digest[i] != other[i]
            shade = 30 + int(v / 255 * 200)
            fill = f"rgb({shade},{shade+20 if not changed else 40},{shade+40 if not changed else 40})"
            border = "#ff6b6b" if changed else "#30363d"
            out.append(f'<rect x="{x}" y="{y}" width="{cell-3}" height="{cell-3}" fill="{fill}" '
                       f'stroke="{border}" stroke-width="{2 if changed else 0.5}"/>')
            out.append(f'<text x="{x+(cell-3)/2:.0f}" y="{y+(cell-3)/2+4:.0f}" fill="#e6edf3" '
                       f'font-size="9" text-anchor="middle">{v:02x}</text>')
        return out

    parts += grid(digest_a, digest_b, 75, "sha256('avalanche')")
    parts += grid(digest_b, digest_a, 220, "sha256('avalanchf')  (one letter changed)")
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
