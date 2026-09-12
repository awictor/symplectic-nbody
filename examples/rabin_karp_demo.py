"""Demo: Rabin-Karp -- substring search by rolling polynomial hashes.

Finds all occurrences of a pattern, searches many patterns in one pass, and computes the longest
common substring of two strings by binary-searching the length with hashes. Draws the pattern's hash
matching windows across the text.

    python examples/rabin_karp_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rabin_karp import (search, multi_search, longest_common_substring,  # noqa: E402
                        brute_search, _hash)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Rabin-Karp: substring search by rolling hash\n")

    text = "abracadabra_abracadabra"
    pat = "abra"
    hits = search(text, pat)
    print(f"  text:    {text}")
    print(f"  pattern: '{pat}' found at {hits}  (brute: {brute_search(text, pat)})\n")

    print("  multi-pattern search in one pass:")
    patterns = ["abra", "cad", "dab", "ra"]
    res = multi_search(text, patterns)
    for p in patterns:
        print(f"    '{p}': {res[p]}")

    a = "the quick brown fox jumps"
    b = "a quick brown dog sits"
    lcs = longest_common_substring(a, b)
    print(f"\n  longest common substring of two sentences:")
    print(f"    A: '{a}'")
    print(f"    B: '{b}'")
    print(f"    LCS: '{lcs}' (length {len(lcs)})")

    print("\n  The rolling hash slides the window in O(1): subtract the departing character's")
    print("  contribution, multiply by the base, add the arriving character. Only windows whose hash")
    print("  matches the pattern's are compared in full, so search is O(n+m) on average -- and because")
    print("  every hash match is verified, the result is always exact, never fooled by a collision.")

    _svg(os.path.join(outdir, "rabin_karp.svg"), text, pat, hits)
    print(f"\n  wrote {os.path.join(outdir, 'rabin_karp.svg')}")


def _svg(path, text, pat, hits, width=780, height=300):
    n = len(text)
    m = len(pat)
    ph = _hash(pat)
    cell = min(30, (width - 80) / n)
    ox, oy = 40, 120
    hitset = set(hits)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="17">'
        f"Rolling-hash search for '{pat}' (hash {ph % 100000})</text>",
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'green windows are matches; the hash rolls across the text in O(1) per step</text>',
    ]

    # text characters
    for i, ch in enumerate(text):
        x = ox + i * cell
        parts.append(f'<rect x="{x:.1f}" y="{oy}" width="{cell-2:.1f}" height="{cell-2:.1f}" '
                     f'fill="#161b22" stroke="#30363d"/>')
        parts.append(f'<text x="{x+cell/2-1:.1f}" y="{oy+cell/2+3:.1f}" fill="#e6edf3" '
                     f'font-size="12" text-anchor="middle">{ch}</text>')

    # match windows as brackets below
    for h in hits:
        x1 = ox + h * cell
        x2 = ox + (h + m) * cell - 2
        parts.append(f'<rect x="{x1:.1f}" y="{oy+cell+6:.1f}" width="{x2-x1:.1f}" height="14" '
                     f'fill="#06d6a0" opacity="0.7" rx="3"/>')
        parts.append(f'<text x="{(x1+x2)/2:.0f}" y="{oy+cell+17:.0f}" fill="#0d1117" '
                     f'font-size="9" text-anchor="middle">match@{h}</text>')

    parts.append(f'<text x="{ox}" y="{oy+cell+50}" fill="#8b949e" font-size="12">'
                 f'{len(hits)} matches found by comparing only hash-equal windows</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
