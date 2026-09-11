"""Demo: Boyer-Moore string search -- skipping ahead.

Shows the algorithm finding all occurrences of a pattern, then how few character comparisons it
needs versus the naive scan (it is sublinear on real text). Draws the comparison-count advantage
across pattern lengths and the bad-character skip table.

    python examples/boyer_moore_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from boyer_moore import (find_all, _bad_char_table, _good_suffix_tables,  # noqa: E402
                         naive_find_all)


def _count_comparisons_bm(text, pattern):
    """Count character comparisons Boyer-Moore makes (instrumented copy of the search loop)."""
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return 0
    bad = _bad_char_table(pattern)
    good = _good_suffix_tables(pattern)
    comps = 0
    s = 0
    while s <= n - m:
        j = m - 1
        while j >= 0:
            comps += 1
            if pattern[j] != text[s + j]:
                break
            j -= 1
        if j < 0:
            s += good[0]
        else:
            s += max(j - bad.get(text[s + j], -1), good[j + 1], 1)
    return comps


def _count_comparisons_naive(text, pattern):
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return 0
    comps = 0
    for s in range(n - m + 1):
        for j in range(m):
            comps += 1
            if text[s + j] != pattern[j]:
                break
    return comps


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    text = "GCATCGCAGAGAGTATACAGTACG"
    pattern = "GCAGAGAG"
    print("Boyer-Moore: match right-to-left and jump ahead on a mismatch\n")
    print(f"  text    : {text}")
    print(f"  pattern : {pattern}")
    print(f"  found at: {find_all(text, pattern)}  (naive agrees: "
          f"{find_all(text, pattern) == naive_find_all(text, pattern)})\n")

    print("  bad-character table (last index of each pattern character):")
    bad = _bad_char_table(pattern)
    print("   " + "  ".join(f"{c}:{i}" for c, i in bad.items()))

    print("\n  Character comparisons, Boyer-Moore vs naive (English text, longer patterns win):")
    corpus = ("the quick brown fox jumps over the lazy dog. " * 60)
    print(f"  {'pattern':>22}{'BM comps':>10}{'naive':>9}{'speedup':>9}")
    for pat in ("fox", "lazy dog", "jumps over the", "the quick brown fox jumps"):
        bm = _count_comparisons_bm(corpus, pat)
        nv = _count_comparisons_naive(corpus, pat)
        print(f"  {pat[:20]:>22}{bm:>10}{nv:>9}{nv / bm:>8.1f}x")
    print("\n  A longer pattern with a rare mismatch character leaps forward by nearly its whole")
    print("  length, so most of the text is never even examined -- the search is sublinear.")

    _svg(os.path.join(outdir, "boyer_moore.svg"), corpus)
    print(f"\n  wrote {os.path.join(outdir, 'boyer_moore.svg')}")


def _svg(path, corpus, w=760, h=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Boyer-Moore: comparisons fall as the pattern grows</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'character comparisons vs pattern length -- naive rises, Boyer-Moore drops (sublinear)</text>',
    ]

    lengths = list(range(2, 26))
    base = "the quick brown fox jumps over the lazy dog"
    bm_counts, nv_counts = [], []
    for L in lengths:
        pat = base[:L]
        bm_counts.append(_count_comparisons_bm(corpus, pat))
        nv_counts.append(_count_comparisons_naive(corpus, pat))

    lx0, lx1 = 60, w - 40
    ly0, ly1 = h - 55, 62
    ymax = max(max(bm_counts), max(nv_counts)) * 1.05

    def LX(L):
        return lx0 + (L - lengths[0]) / (lengths[-1] - lengths[0]) * (lx1 - lx0)

    def LY(v):
        return ly0 - v / ymax * (ly0 - ly1)

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    nv = " ".join(f"{LX(L):.1f},{LY(v):.1f}" for L, v in zip(lengths, nv_counts))
    bm = " ".join(f"{LX(L):.1f},{LY(v):.1f}" for L, v in zip(lengths, bm_counts))
    parts.append(f'<polyline points="{nv}" fill="none" stroke="#ff6b6b" stroke-width="2.5"/>')
    parts.append(f'<polyline points="{bm}" fill="none" stroke="#06d6a0" stroke-width="2.5"/>')
    for L, v in zip(lengths, bm_counts):
        parts.append(f'<circle cx="{LX(L):.1f}" cy="{LY(v):.1f}" r="2" fill="#06d6a0"/>')
    parts.append(f'<text x="{LX(lengths[-1]):.1f}" y="{LY(nv_counts[-1])-6:.1f}" fill="#ff6b6b" '
                 f'font-size="10" text-anchor="end">naive O(nm)</text>')
    parts.append(f'<text x="{LX(lengths[-1]):.1f}" y="{LY(bm_counts[-1])+14:.1f}" fill="#06d6a0" '
                 f'font-size="10" text-anchor="end">Boyer-Moore (sublinear)</text>')
    for v in (0, int(ymax // 2), int(ymax)):
        parts.append(f'<text x="{lx0-6:.1f}" y="{LY(v)+3:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="end">{v}</text>')
    for L in (2, 12, 24):
        parts.append(f'<text x="{LX(L):.1f}" y="{ly0+15:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{L}</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">pattern length -> character comparisons</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
