"""Demo: Manacher's algorithm finding every palindrome in linear time.

Finds the longest palindromic substring and counts all palindromic substrings for several strings,
verifies against brute force, and shows the per-center palindrome radii. Draws the radius profile
that peaks at palindrome centers.

    python examples/manacher_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from manacher import (longest_palindrome, count_palindromic_substrings,
                      all_palindrome_radii)  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Manacher's algorithm: all palindromes in O(n)\n")

    examples = ["babad", "racecar", "abacaba", "mississippi", "aabaaacaaab"]
    for s in examples:
        lp = longest_palindrome(s)
        cnt = count_palindromic_substrings(s)
        print(f"  {s!r:16}: longest {lp!r:12} ({len(lp)} chars), {cnt} palindromic substrings")

    # brute-force confirmation on one
    s = "aabaaacaaab"
    def brute_count(s):
        return sum(1 for i in range(len(s)) for j in range(i + 1, len(s) + 1)
                   if s[i:j] == s[i:j][::-1])
    print(f"\n  Verify {s!r}: Manacher {count_palindromic_substrings(s)} == brute {brute_count(s)}")

    # radii profile for a demonstrative string
    demo = "abacabadabacaba"
    odd, even = all_palindrome_radii(demo)
    print(f"\n  Odd-length palindrome radius at each center of {demo!r}:")
    print("    " + " ".join(demo))
    print("    " + " ".join(str(r) for r in odd))
    print(f"    peak radius {max(odd)} at center '{demo[odd.index(max(odd))]}' "
          f"(index {odd.index(max(odd))}) -> the whole string is a palindrome")

    print("\n  The string is padded with separators so every palindrome is odd-length with one")
    print("  center. A running scan keeps the rightmost palindrome found; a new center inherits its")
    print("  mirror's radius for free and only expands beyond it -- so total work is linear.")

    _svg(os.path.join(outdir, "manacher.svg"), demo, odd)
    print(f"\n  wrote {os.path.join(outdir, 'manacher.svg')}")


def _svg(path, s, odd, width=760, height=400):
    n = len(s)
    m_left, m_bot, m_top, m_right = 40, 70, 90, 40
    pw = width - m_left - m_right
    ph = height - m_top - m_bot
    rmax = max(odd) or 1
    bar_w = pw / n

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Manacher: odd-palindrome radius at each character center</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'taller bar = longer palindrome centered on that character; the peak is the longest palindrome</text>',
    ]
    peak = odd.index(max(odd))
    for i in range(n):
        h = 10 + odd[i] / rmax * ph
        x = m_left + i * bar_w
        col = "#ffd43b" if i == peak else "#4dabf7"
        parts.append(f'<rect x="{x + 2:.1f}" y="{m_top + ph - h:.1f}" width="{bar_w - 4:.1f}" '
                     f'height="{h:.1f}" fill="{col}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{m_top + ph + 18:.1f}" fill="#e6edf3" '
                     f'font-size="13" text-anchor="middle">{s[i]}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{m_top + ph - h - 4:.1f}" fill="#8b949e" '
                     f'font-size="10" text-anchor="middle">{odd[i]}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
