"""Demo: the suffix automaton -- one tiny machine that knows every substring.

Builds the suffix automaton of a word, counts its distinct substrings against the raw O(n^2) total,
finds the longest repeated substring and the longest common substring with another word, and draws the
automaton's states laid out by the length of the substrings they represent.

    python examples/suffix_automaton_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from suffix_automaton import (build, count_distinct_substrings,  # noqa: E402
                              longest_repeated_substring, longest_common_substring,
                              brute_distinct_substrings)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    word = "abracadabra"
    sa = build(word)

    print("Suffix automaton: the smallest machine recognising every substring\n")
    print(f"  word: '{word}' ({len(word)} characters)")
    print(f"  automaton: {len(sa.states)} states (bound is 2n-1 = {2*len(word)-1})\n")

    distinct = count_distinct_substrings(word)
    raw = len(word) * (len(word) + 1) // 2
    print(f"  distinct substrings: {distinct}  (vs {raw} raw substrings with repeats)")
    print(f"  verified against brute enumeration: {distinct == len(brute_distinct_substrings(word))}")

    print(f"\n  occurrence counts:")
    for pat in ["abra", "a", "ra", "cad"]:
        print(f"    '{pat}' appears {sa.occurrences(pat)} time(s)")

    print(f"\n  longest repeated substring: '{longest_repeated_substring(word)}'")

    other = "cadabraxyz"
    lcs = longest_common_substring(word, other)
    print(f"  longest common substring of '{word}' and '{other}': '{lcs}'")

    print("\n  Built online in O(n): each character is appended by following suffix links and cloning")
    print("  a state when a transition would conflict. The result has at most 2n-1 states yet encodes")
    print("  all substrings -- every path from the start spells a distinct one. It answers substring")
    print("  membership in O(pattern), counts occurrences, and finds longest common/repeated pieces.")

    _svg(os.path.join(outdir, "suffix_automaton.svg"), sa, word)
    print(f"\n  wrote {os.path.join(outdir, 'suffix_automaton.svg')}")


def _svg(path, sa, word, width=800, height=430):
    n = len(sa.states)
    # layout: x by state length, y spread within each length band
    by_len = {}
    for i in range(n):
        by_len.setdefault(sa.states[i].length, []).append(i)
    maxlen = max(sa.states[i].length for i in range(n))

    pos = {}
    ox, oy = 60, 60
    span_x = width - 120
    for L, group in by_len.items():
        x = ox + (span_x * L / max(1, maxlen))
        k = len(group)
        for gi, state in enumerate(group):
            y = oy + (height - 140) * (gi + 1) / (k + 1)
            pos[state] = (x, y)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="17">'
        f"Suffix automaton of '{word}': {n} states, laid out left-to-right by substring length</text>",
        f'<text x="20" y="49" fill="#8b949e" font-size="12">'
        f'blue arrows are transitions (labelled by character); grey dashed arrows are suffix links'
        f'</text>',
    ]

    # suffix links (dashed grey, drawn first)
    for i in range(1, n):
        link = sa.states[i].link
        if link >= 0 and i in pos and link in pos:
            x1, y1 = pos[i]
            x2, y2 = pos[link]
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                         f'stroke="#484f58" stroke-width="0.8" stroke-dasharray="3 3"/>')

    # transitions (blue)
    for i in range(n):
        for ch, j in sa.states[i].trans.items():
            if i in pos and j in pos:
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                             f'stroke="#4dabf7" stroke-width="1.1" opacity="0.75"/>')
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                parts.append(f'<text x="{mx:.0f}" y="{my-2:.0f}" fill="#ffd43b" font-size="10" '
                             f'text-anchor="middle">{ch}</text>')

    # states
    for i in range(n):
        if i not in pos:
            continue
        x, y = pos[i]
        col = "#06d6a0" if i == 0 else "#161b22"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{col}" '
                     f'stroke="#4dabf7" stroke-width="1.3"/>')

    parts.append(f'<text x="{pos[0][0]:.0f}" y="{pos[0][1]+22:.0f}" fill="#06d6a0" '
                 f'font-size="10" text-anchor="middle">start</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
