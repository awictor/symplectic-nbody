"""Demo: a trie powering autocomplete and substring search.

Builds a trie from a small word list and shows prefix autocomplete (alphabetical and
frequency-ranked), longest-prefix matching, and O(len) membership; then indexes a text with a
suffix trie for substring queries. Draws the trie as a tree with word-ending nodes highlighted.

    python examples/trie_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trie import Trie, SuffixTrie  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    # a small autocomplete dictionary with realistic frequencies
    corpus = (["the"] * 9 + ["they"] * 3 + ["their"] * 4 + ["there"] * 5 + ["them"] * 2
              + ["to"] * 8 + ["too"] * 2 + ["top"] * 3 + ["tea"] * 1 + ["ten"] * 2)
    trie = Trie()
    for w in corpus:
        trie.insert(w)

    print("Trie: prefix tree for autocomplete and O(len) lookup\n")
    print(f"  dictionary: {len(trie)} distinct words, {len(corpus)} total insertions\n")

    print("  Autocomplete 'the' (alphabetical):")
    print(f"    {trie.autocomplete('the')}")
    print("  Autocomplete 't' by insertion frequency (what a search box would rank):")
    for w in trie.autocomplete("t", by_frequency=True, limit=5):
        print(f"    {w:<8} (typed {trie.count(w)}x)")

    print(f"\n  membership: 'the' stored = {trie.search('the')}, "
          f"'th' stored = {trie.search('th')} (prefix only)")
    print(f"  starts_with 'thei' = {trie.starts_with('thei')}, "
          f"'xyz' = {trie.starts_with('xyz')}")
    print(f"  longest stored prefix of 'thereafter' = '{trie.longest_prefix_of('thereafter')}'")

    print("\n  Deletion prunes dead branches but spares shared prefixes:")
    trie.delete("tea")
    print(f"    after deleting 'tea': autocomplete 'te' -> {trie.autocomplete('te')} "
          f"(ten kept)")

    # suffix trie substring index
    text = "abracadabra"
    st = SuffixTrie(text)
    print(f"\n  Suffix-trie substring index of {text!r}:")
    for pat in ("abra", "cad", "bra", "xyz"):
        occ = st.occurrences(pat)
        print(f"    {pat!r:>7}: {'present at ' + str(occ) if occ else 'not found'}")

    print("\n  Every operation is O(length of the key), independent of how many words are stored --")
    print("  words sharing a prefix share its path. That per-character walk is what powers")
    print("  autocomplete, spell-check, and longest-prefix IP routing.")

    _svg(os.path.join(outdir, "trie.svg"), trie)
    print(f"\n  wrote {os.path.join(outdir, 'trie.svg')}")


def _svg(path, trie, width=760, height=440):
    # layout the trie: assign each node an (depth, x) by an in-order-ish sweep of leaves
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Trie: each path spells a prefix; filled nodes end a word</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'words sharing a prefix share its path -- O(length) lookup, natural prefix '
        f'compression</text>',
    ]

    # count leaves under each node to space them out
    leaf_counter = [0]
    positions = {}     # id(node) -> (x, y)
    max_depth = [0]

    def width_of(node):
        if not node.children:
            return 1
        return sum(width_of(c) for c in node.children.values())

    x0, x1 = 50, width - 40
    top, bottom = 80, height - 40
    total_leaves = max(1, width_of(trie.root))

    slot = [0.0]

    def layout(node, depth, ch):
        max_depth[0] = max(max_depth[0], depth)
        if not node.children:
            x = x0 + (slot[0] + 0.5) / total_leaves * (x1 - x0)
            slot[0] += 1
        else:
            child_xs = []
            for c in sorted(node.children):
                child_xs.append(layout(node.children[c], depth + 1, c))
            x = sum(child_xs) / len(child_xs)
        positions[id(node)] = (x, depth, ch, node.is_word)
        return x

    layout(trie.root, 0, "")
    md = max(1, max_depth[0])

    def Y(depth):
        return top + depth / md * (bottom - top)

    # draw edges first
    def draw_edges(node):
        px, pd, _, _ = positions[id(node)]
        for c in sorted(node.children):
            child = node.children[c]
            cx, cd, _, _ = positions[id(child)]
            parts.append(f'<line x1="{px:.1f}" y1="{Y(pd):.1f}" x2="{cx:.1f}" y2="{Y(cd):.1f}" '
                         f'stroke="#30363d" stroke-width="1.2"/>')
            draw_edges(child)

    draw_edges(trie.root)

    # draw nodes
    for x, depth, ch, is_word in positions.values():
        y = Y(depth)
        if depth == 0:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#8b949e"/>')
            continue
        fill = "#06d6a0" if is_word else "#161b22"
        stroke = "#06d6a0" if is_word else "#4dabf7"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{fill}" '
                     f'stroke="{stroke}" stroke-width="1.4"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+3.5:.1f}" fill="#e6edf3" font-size="11" '
                     f'text-anchor="middle">{ch}</text>')

    parts.append(f'<circle cx="60" cy="{height-20}" r="7" fill="#06d6a0"/>')
    parts.append(f'<text x="74" y="{height-16}" fill="#8b949e" font-size="10">word end</text>')
    parts.append(f'<circle cx="170" cy="{height-20}" r="7" fill="#161b22" stroke="#4dabf7" stroke-width="1.4"/>')
    parts.append(f'<text x="184" y="{height-16}" fill="#8b949e" font-size="10">prefix node</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
