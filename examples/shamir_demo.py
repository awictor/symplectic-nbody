"""Demo: Shamir's (k, n) secret sharing -- split a secret so any k pieces reconstruct it.

Splits a secret into shares, shows that any k reconstruct it while k-1 reveal nothing, and
demonstrates a byte-string secret round-trip. Draws the geometry: a polynomial through the shares
whose y-intercept is the secret.

    python examples/shamir_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shamir import split, reconstruct, split_bytes, reconstruct_bytes  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Shamir's secret sharing: any k of n shares reconstruct the secret\n")

    secret = 1234567890
    k, n = 3, 5
    shares = split(secret, k, n, seed=7)
    print(f"  secret: {secret}")
    print(f"  ({k}, {n}) threshold -> {n} shares, any {k} reconstruct:")
    for x, y in shares:
        print(f"    share {x}: {str(y)[:40]}...")

    print(f"\n  reconstruct from shares 1,2,3: {reconstruct(shares[:3])}")
    print(f"  reconstruct from shares 3,4,5: {reconstruct(shares[2:5])}")
    print(f"  reconstruct from just 2 shares: {str(reconstruct(shares[:2]))[:30]}... (wrong -- reveals nothing)")

    # byte-string secret
    msg = b"attack at dawn"
    length, bshares = split_bytes(msg, 2, 4, seed=3)
    print(f"\n  byte secret {msg!r} split (2,4):")
    print(f"    reconstruct from 2 shares: {reconstruct_bytes(length, bshares[:2])!r}")

    # small-field illustration of the geometry
    print("\n  The geometry: the secret is the y-intercept f(0) of a degree-(k-1) polynomial. Each")
    print("  share is a point on it. k points fix the polynomial uniquely (Lagrange interpolation")
    print("  recovers f(0)); k-1 points leave infinitely many polynomials, so the secret is hidden.")

    _svg(os.path.join(outdir, "shamir.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'shamir.svg')}")


def _svg(path, width=760, height=430):
    # illustrate over a small prime field for a clear picture: k=3, secret=5, prime=97
    prime = 97
    from shamir import split as _split, reconstruct as _rec
    secret = 5
    shares = _split(secret, 3, 5, prime=prime, seed=4)
    # the quadratic through the shares; sample it densely over [0, n]
    # recover the polynomial coefficients by interpolation is overkill; just plot the shares + secret
    m_left, m_bot, m_top, m_right = 60, 55, 80, 40
    pw = width - m_left - m_right
    ph = height - m_top - m_bot
    xmax = 6
    ymax = prime

    def px(x):
        return m_left + x / xmax * pw

    def py(y):
        return m_top + ph - y / ymax * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Shamir secret sharing over GF(97): shares on a degree-2 polynomial</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'blue = the 5 shares (points); gold = the secret at f(0) = the y-intercept; any 3 fix the curve</text>',
    ]
    parts.append(f'<line x1="{m_left}" y1="{m_top+ph}" x2="{m_left+pw}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    # x=0 vertical marker (where the secret lives)
    parts.append(f'<line x1="{px(0):.1f}" y1="{m_top}" x2="{px(0):.1f}" y2="{m_top+ph}" '
                 f'stroke="#30363d" stroke-width="1" stroke-dasharray="3,3"/>')

    # the polynomial: reconstruct coefficients by evaluating the interpolant at many x (over reals mod p
    # would wrap; instead show shares and the secret point clearly)
    for x, y in shares:
        parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="6" fill="#4dabf7"/>')
        parts.append(f'<text x="{px(x):.1f}" y="{py(y)-12:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">({x},{y})</text>')
    # secret point at x=0
    parts.append(f'<circle cx="{px(0):.1f}" cy="{py(secret):.1f}" r="8" fill="#ffd43b"/>')
    parts.append(f'<text x="{px(0)+10:.1f}" y="{py(secret)+4:.1f}" fill="#ffd43b" font-size="13">'
                 f'secret = f(0) = {secret}</text>')
    parts.append(f'<text x="{m_left+pw/2:.0f}" y="{height-12}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">share index x (secret hidden at x=0)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
