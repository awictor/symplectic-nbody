"""Demo: linear-time string matching -- KMP, the Z-algorithm, and Manacher.

Shows the KMP prefix function that lets the search skip re-scanning, finds a pattern's (overlapping)
occurrences by KMP and confirms the Z-algorithm agrees, and picks out the longest palindromic
substring with Manacher -- all in linear time where the naive method is quadratic.

    python examples/string_matching_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from string_matching import (prefix_function, kmp_search, z_array, z_search,  # noqa: E402
                             brute_force_search, longest_palindrome, count_occurrences)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Linear-time string matching: KMP, Z-algorithm, Manacher\n")

    pattern = "ababaca"
    pi = prefix_function(pattern)
    print(f"  KMP prefix function of {pattern!r}:")
    print("    index:  " + " ".join(f"{i}" for i in range(len(pattern))))
    print("    char:   " + " ".join(pattern))
    print("    pi:     " + " ".join(str(v) for v in pi))
    print("    (pi[i] = longest proper prefix that is also a suffix of the first i+1 chars --")
    print("     on a mismatch the pattern shifts by more than one without rescanning the text)\n")

    text = "abababacabababacaba"
    pat = "ababaca"
    km = kmp_search(text, pat)
    zm = z_search(text, pat)
    print(f"  Searching for {pat!r} in {text!r}:")
    print(f"    KMP occurrences:      {km}")
    print(f"    Z-algorithm:          {zm}")
    print(f"    brute-force (check):  {brute_force_search(text, pat)}")
    print(f"    all three agree: {km == zm == brute_force_search(text, pat)}\n")

    # overlapping matches
    print(f"  Overlapping matches -- 'aa' in 'aaaaa': {kmp_search('aaaaa', 'aa')} "
          f"({count_occurrences('aaaaa', 'aa')} occurrences)\n")

    # Z-array
    zs = "aabxaabxcaabxaab"
    print(f"  Z-array of {zs!r} (z[i] = match length with the prefix):")
    print("    " + " ".join(f"{v}" for v in z_array(zs)) + "\n")

    # Manacher
    print("  Longest palindromic substring (Manacher, O(n)):")
    for s in ["babad", "forgeeksskeegfor", "abacdfgdcaba", "racecar", "banana"]:
        print(f"    {s:>16} -> {longest_palindrome(s)!r}")

    print("\n  All three exploit precomputed self-overlap to skip redundant comparisons: KMP's")
    print("  prefix function, Z's prefix-match array, and Manacher's mirror reuse. Naive matching")
    print("  is O(n*m) and naive palindrome search O(n^2); these are all O(n) -- the difference")
    print("  between scanning a genome once and scanning it thousands of times.")

    _svg(os.path.join(outdir, "string_matching.svg"), pattern, pi, text, pat, km)
    print(f"\n  wrote {os.path.join(outdir, 'string_matching.svg')}")


def _svg(path, pattern, pi, text, pat, matches, width=760, height=400):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'KMP: the prefix function (top) and pattern matches in the text (bottom)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'prefix-function bars show self-overlap; highlighted spans are pattern occurrences</text>',
    ]
    # prefix function as a bar chart over the pattern characters
    cw = min(46, (width - 90) // len(pattern))
    ox, oy = 60, 90
    maxpi = max(pi) or 1
    bar_h = 60
    for i, ch in enumerate(pattern):
        x = ox + i * cw
        h = pi[i] / maxpi * bar_h
        parts.append(f'<rect x="{x:.1f}" y="{oy + bar_h - h:.1f}" width="{cw - 4}" height="{h:.1f}" '
                     f'fill="#4dabf7"/>')
        parts.append(f'<text x="{x + (cw - 4) / 2:.1f}" y="{oy + bar_h + 16:.1f}" fill="#e6edf3" '
                     f'font-size="13" text-anchor="middle">{ch}</text>')
        parts.append(f'<text x="{x + (cw - 4) / 2:.1f}" y="{oy + bar_h - h - 4:.1f}" fill="#8b949e" '
                     f'font-size="9" text-anchor="middle">{pi[i]}</text>')
    parts.append(f'<text x="20" y="{oy + bar_h / 2:.1f}" fill="#8b949e" font-size="10">pi</text>')

    # the text with matched spans highlighted
    tcw = min(30, (width - 90) // len(text))
    ty = 250
    m = len(pat)
    match_cols = set()
    for start in matches:
        for k in range(start, start + m):
            match_cols.add(k)
    for i, ch in enumerate(text):
        x = ox + i * tcw
        inmatch = i in match_cols
        if inmatch:
            parts.append(f'<rect x="{x:.1f}" y="{ty - 14:.1f}" width="{tcw - 1}" height="20" '
                         f'fill="#06d6a0" opacity="0.3"/>')
        parts.append(f'<text x="{x + tcw / 2:.1f}" y="{ty:.1f}" '
                     f'fill="{"#06d6a0" if inmatch else "#c9d1d9"}" font-size="12" '
                     f'text-anchor="middle">{ch}</text>')
    parts.append(f'<text x="20" y="{ty:.1f}" fill="#8b949e" font-size="10">text</text>')
    parts.append(f'<text x="{ox}" y="{ty + 30:.1f}" fill="#8b949e" font-size="11">'
                 f'pattern {pat!r} found at {matches}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
