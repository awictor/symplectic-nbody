"""Demo: suffix arrays -- a compact string index.

Builds the suffix array and LCP array of a string, prints the sorted suffixes with their common-
prefix lengths, searches for patterns by binary search, and pulls out the longest repeated substring
(the largest LCP) and the longest substring common to two strings.

    python examples/suffix_array_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from suffix_array import (build_suffix_array, build_lcp, search,  # noqa: E402
                          longest_repeated_substring, longest_common_substring)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    s = "mississippi"
    sa = build_suffix_array(s)
    lcp = build_lcp(s, sa)

    print("Suffix array: the sorted order of all suffixes, as an integer index\n")
    print(f"  string: {s!r}\n")
    print(f"    {'rank':>4} {'SA':>3} {'LCP':>4}  suffix")
    for r in range(len(sa)):
        print(f"    {r:>4} {sa[r]:>3} {lcp[r]:>4}  {s[sa[r]:]}")

    print("\n  Substring search is a binary search over the sorted suffixes (O(m log n)):")
    for pat in ["issi", "ss", "ppi", "xyz"]:
        occ = search(s, sa, pat)
        print(f"    {pat!r:>7}: {'occurs at ' + str(occ) if occ else 'not found'}")

    print(f"\n  Longest repeated substring (the largest LCP value): "
          f"{longest_repeated_substring(s)!r}")

    print("\n  Longest repeated substring of a few strings:")
    for word in ["banana", "abracadabra", "the theme of these theses", "aaaa"]:
        print(f"    {word!r:>28} -> {longest_repeated_substring(word)!r}")

    print("\n  Longest common substring of two strings (via a combined suffix array):")
    pairs = [("dogandcat", "thecatsat"), ("bioinformatics", "informant"),
             ("GATTACAGG", "TTGATTACA")]
    for a, b in pairs:
        print(f"    {a!r} & {b!r} -> {longest_common_substring(a, b)!r}")

    print("\n  The suffix array packs a suffix tree's power into one length-n array: sorted")
    print("  suffixes make substring search a binary search, every occurrence is a contiguous")
    print("  run, and the LCP array turns tree queries into array scans -- the largest LCP is the")
    print("  longest repeated substring. Built by prefix doubling; LCP by Kasai in linear time.")

    _svg(os.path.join(outdir, "suffix_array.svg"), s, sa, lcp)
    print(f"\n  wrote {os.path.join(outdir, 'suffix_array.svg')}")


def _svg(path, s, sa, lcp, width=760, height=430):
    n = len(sa)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Suffix array of {s!r}: sorted suffixes with LCP bars</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'each row is a suffix in sorted order; the orange bar is its common-prefix length with '
        f'the row above (the largest = longest repeated substring)</text>',
    ]
    row_h = min(28, (height - 90) // n)
    ox, oy = 60, 75
    maxlcp = max(lcp) or 1
    char_w = 15
    for r in range(n):
        y = oy + r * row_h
        # LCP bar
        bar_w = lcp[r] / maxlcp * 60
        parts.append(f'<rect x="{ox - 66:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{row_h - 3}" '
                     f'fill="#ff922b" opacity="0.7"/>')
        parts.append(f'<text x="{ox - 66:.1f}" y="{y + row_h - 6:.1f}" fill="#8b949e" '
                     f'font-size="9">{lcp[r]}</text>')
        # the suffix, with the shared prefix (length lcp[r]) highlighted
        suffix = s[sa[r]:]
        for c, ch in enumerate(suffix):
            col = "#ffd43b" if c < lcp[r] else "#c9d1d9"
            parts.append(f'<text x="{ox + c * char_w:.1f}" y="{y + row_h - 6:.1f}" fill="{col}" '
                         f'font-size="13">{ch}</text>')
    parts.append(f'<text x="{ox - 66:.1f}" y="{oy - 6:.1f}" fill="#ff922b" font-size="10">LCP</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
