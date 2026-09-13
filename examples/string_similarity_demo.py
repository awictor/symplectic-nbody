"""Demo: string similarity -- fuzzy name matching by Jaro-Winkler, q-grams, and Soundex.

Ranks a misspelled name against a candidate list by several similarity measures, and groups a set of
names by their Soundex phonetic code. Draws a similarity-score bar chart.

    python examples/string_similarity_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from string_similarity import jaro, jaro_winkler, jaccard, dice, soundex, rank  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("String similarity: fuzzy matching by Jaro-Winkler, q-grams, and Soundex\n")

    query = "Jonathon"
    candidates = ["Jonathan", "Johnathan", "Nathan", "Jon", "Jonas", "Nathaniel", "John"]
    print(f"  Ranking candidates against the query '{query}':\n")
    print(f"  {'candidate':>12}  {'Jaro':>6}  {'Jaro-Wink':>9}  {'Jaccard':>7}  {'Dice':>6}")
    scored = []
    for c in sorted(candidates, key=lambda x: -jaro_winkler(query, x)):
        j = jaro(query, c)
        jw = jaro_winkler(query, c)
        ja = jaccard(query, c)
        di = dice(query, c)
        scored.append((c, jw))
        print(f"  {c:>12}  {j:>6.3f}  {jw:>9.3f}  {ja:>7.3f}  {di:>6.3f}")

    print("\n  Jaro-Winkler favours a shared prefix; the top match is the true intended name.\n")

    # phonetic grouping by Soundex
    print("  Soundex phonetic codes (names that sound alike share a code):")
    names = ["Robert", "Rupert", "Rubin", "Smith", "Smyth", "Schmidt", "Ashcraft", "Ashcroft"]
    groups = {}
    for n in names:
        groups.setdefault(soundex(n), []).append(n)
    for code, members in sorted(groups.items()):
        print(f"    {code}: {', '.join(members)}")
    print("\n  'Robert'/'Rupert' collide (R163), as do 'Smith'/'Smyth' and 'Ashcraft'/'Ashcroft' --")
    print("  phonetic matching finds sound-alikes that edit distance would keep far apart.")

    _svg(os.path.join(outdir, "string_similarity.svg"), query, scored)
    print(f"\n  wrote {os.path.join(outdir, 'string_similarity.svg')}")


def _svg(path, query, scored, width=760, height=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="15">'
        f'Jaro-Winkler similarity of candidates to "{query}"</text>',
    ]
    ox, oy = 160, 60
    bw = width - ox - 60
    rowh = (height - oy - 40) / len(scored)
    for i, (name, score) in enumerate(scored):
        y = oy + i * rowh
        w = bw * score
        col = "#06d6a0" if score > 0.85 else ("#ffd43b" if score > 0.7 else "#4dabf7")
        parts.append(f'<text x="{ox-10}" y="{y+rowh/2+4:.0f}" fill="#e6edf3" font-size="11" '
                     f'text-anchor="end">{name}</text>')
        parts.append(f'<rect x="{ox}" y="{y+4:.0f}" width="{w:.1f}" height="{rowh-8:.0f}" '
                     f'fill="{col}"/>')
        parts.append(f'<text x="{ox+w+6:.0f}" y="{y+rowh/2+4:.0f}" fill="#8b949e" font-size="10">'
                     f'{score:.3f}</text>')
    # scale line at 1.0
    parts.append(f'<line x1="{ox+bw:.0f}" y1="{oy}" x2="{ox+bw:.0f}" y2="{height-40}" '
                 f'stroke="#30363d" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{ox+bw:.0f}" y="{height-25}" fill="#8b949e" font-size="9" '
                 f'text-anchor="middle">1.0</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
