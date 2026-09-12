"""Demo: the eertree (palindromic tree) -- every distinct palindrome of a string in O(n).

Counts and lists the distinct palindromic substrings of a word, shows how many occur and how often,
and finds the palindromic richness. Draws the palindromes grouped by length as coloured tiles.

    python examples/eertree_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eertree import build, brute_distinct_palindromes  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    word = "abacabadabacaba"          # a rich, highly palindromic word
    et = build(word)

    print("Eertree: all distinct palindromic substrings in one linear structure\n")
    print(f"  word: '{word}' ({len(word)} characters)")
    print(f"  tree: {len(et.nodes)} nodes (2 roots + one per distinct palindrome)\n")

    distinct = et.count_distinct()
    print(f"  distinct palindromic substrings: {distinct}")
    print(f"  verified against brute enumeration: "
          f"{distinct == len(brute_distinct_palindromes(word))}")
    print(f"  classical bound (<= n = {len(word)}): {'satisfied' if distinct <= len(word) else 'VIOLATED'}")

    occ = {"".join(k): v for k, v in et.occurrence_counts().items()}
    by_len = {}
    for pal in occ:
        by_len.setdefault(len(pal), []).append(pal)

    print("\n  distinct palindromes by length:")
    for L in sorted(by_len):
        pals = sorted(set(by_len[L]))
        print(f"    len {L}: {pals}")

    print("\n  most frequent palindromes:")
    for pal, c in sorted(occ.items(), key=lambda kv: (-kv[1], len(kv[0])))[:5]:
        print(f"    '{pal}' occurs {c} times")

    # palindromic richness: a string of length n is "rich" if it has exactly n distinct palindromes
    rich = distinct == len(word)
    print(f"\n  palindromic richness: {distinct}/{len(word)} distinct -> "
          f"{'RICH (maximal)' if rich else 'not maximal'}")
    print("  The eertree adds each character in amortised O(1) by walking suffix links to the longest")
    print("  extendable palindromic suffix -- capturing all palindromes in O(n), the palindrome")
    print("  analogue of the suffix automaton.")

    _svg(os.path.join(outdir, "eertree.svg"), by_len, occ, word)
    print(f"\n  wrote {os.path.join(outdir, 'eertree.svg')}")


_PAL = ["#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#ff6b6b", "#b197fc", "#e6edf3"]


def _svg(path, by_len, occ, word, width=780, height=430):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f"Distinct palindromes of '{word}', grouped by length</text>",
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'each tile is one distinct palindromic substring; brighter = occurs more often</text>',
    ]

    maxocc = max(occ.values()) if occ else 1
    y = 80
    row_h = 40
    for Li, L in enumerate(sorted(by_len)):
        pals = sorted(set(by_len[L]))
        parts.append(f'<text x="30" y="{y+row_h/2+4:.0f}" fill="#8b949e" font-size="12" '
                     f'text-anchor="start">len {L}</text>')
        x = 110
        base_col = _PAL[Li % len(_PAL)]
        for pal in pals:
            w = max(30, 11 * len(pal) + 8)
            # opacity by occurrence frequency
            op = 0.35 + 0.65 * (occ[pal] / maxocc)
            parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{row_h-6}" '
                         f'fill="{base_col}" opacity="{op:.2f}" rx="3"/>')
            parts.append(f'<text x="{x+w/2:.0f}" y="{y+row_h/2+3:.0f}" fill="#0d1117" '
                         f'font-size="12" text-anchor="middle" font-weight="bold">{pal}</text>')
            x += w + 8
            if x > width - 80:
                break
        y += row_h
        if y > height - 40:
            break

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
