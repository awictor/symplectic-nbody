"""Demo: Lyndon words and Duval's algorithm -- the prime factorisation of strings.

Factorises words into their unique non-increasing Lyndon decomposition, canonicalises necklaces to
their least rotation, and lists the Lyndon words that generate the De Bruijn sequence. Draws the
Duval factorisation as coloured segments.

    python examples/lyndon_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lyndon import duval, least_rotation, lyndon_words_up_to, is_lyndon  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Lyndon words: the primes of string concatenation\n")

    print("  A Lyndon word is strictly smaller than all its rotations. Membership:")
    for w in ["aab", "aba", "abcab", "aabb", "z", "aa"]:
        print(f"    {w:6s} -> {'Lyndon' if is_lyndon(w) else 'not Lyndon'}")

    print("\n  Chen-Fox-Lyndon factorisation (unique, non-increasing) via Duval O(n):")
    for w in ["banana", "bbababaab", "abcabcabc", "zyxabc"]:
        factors = duval(w)
        print(f"    {w:12s} = {' | '.join(factors)}")

    print("\n  Least rotation (necklace canonicalisation) via Duval over the doubled string:")
    for w in ["cabab", "bca", "googgle", "tobeornottobe"]:
        idx, rot = least_rotation(w)
        print(f"    {w:14s} -> '{rot}'  (rotate by {idx})")

    print("\n  Lyndon words over {a,b} up to length 4 (FKM order -- concatenated they form the")
    print("  De Bruijn sequence B(2,4)):")
    lw = lyndon_words_up_to(2, 4)
    as_str = ["".join("ab"[c] for c in w) for w in lw]
    print(f"    {as_str}")
    debruijn = "".join(s for w, s in zip(lw, as_str) if 4 % len(w) == 0)
    print(f"    De Bruijn B(2,4) = {debruijn}  (length {len(debruijn)} = 2^4)")

    _svg(os.path.join(outdir, "lyndon.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'lyndon.svg')}")


_PAL = ["#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#ff6b6b", "#b197fc", "#e6edf3", "#8b949e"]


def _svg(path, width=760, height=380):
    examples = [
        ("banana", duval("banana")),
        ("bbababaab", duval("bbababaab")),
        ("aababcabcd", duval("aababcabcd")),
        ("dcbaabcd", duval("dcbaabcd")),
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'Duval factorisation: each string split into non-increasing Lyndon words</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'coloured blocks are the unique Lyndon factors, left to right in non-increasing order</text>',
    ]

    cell = 34
    ox = 60
    y = 90
    for name, factors in examples:
        x = ox
        parts.append(f'<text x="{ox-40}" y="{y+cell/2+5:.0f}" fill="#8b949e" font-size="11" '
                     f'text-anchor="end">{name}</text>')
        for fi, f in enumerate(factors):
            col = _PAL[fi % len(_PAL)]
            for ch in f:
                parts.append(f'<rect x="{x}" y="{y}" width="{cell-3}" height="{cell-3}" '
                             f'fill="{col}"/>')
                parts.append(f'<text x="{x+(cell-3)/2:.0f}" y="{y+cell/2+4:.0f}" fill="#0d1117" '
                             f'font-size="15" text-anchor="middle" font-weight="bold">{ch}</text>')
                x += cell
            x += 8      # gap between factors
        y += cell + 24

    parts.append(f'<text x="{ox}" y="{y+10}" fill="#8b949e" font-size="11">'
                 f'gaps separate the Lyndon factors; note each factor is <= the one before it</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
