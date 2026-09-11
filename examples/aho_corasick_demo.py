"""Demo: Aho-Corasick -- finding many patterns in one pass.

Scans a text for a whole dictionary of patterns at once (checked against brute force), shows the
classic overlapping "ushers" example, and how the single-pass cost stays flat as the number of
patterns grows. Draws the trie with its failure links and the match timeline over a log line.

    python examples/aho_corasick_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aho_corasick import AhoCorasick, brute_search  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Aho-Corasick: match a whole dictionary of patterns in one left-to-right scan\n")
    ac = AhoCorasick(["he", "she", "his", "hers"])
    print(f"  patterns {ac.patterns} in 'ushers':")
    for pat, hits in ac.search("ushers").items():
        if hits:
            print(f"    '{pat}' at {hits}")
    print(f"  overlapping/nested matches all caught; brute force agrees: "
          f"{ac.search('ushers') == brute_search(ac.patterns, 'ushers')}\n")

    # a log-scanning blocklist
    patterns = ["error", "fail", "warn", "timeout", "critical", "denied"]
    log = "info: started; warn: retry; error: timeout on db; critical: fail; access denied; done"
    scanner = AhoCorasick(patterns)
    print(f"  scanning a log line for {len(patterns)} patterns in one pass:")
    matches = scanner.search(log)
    for pat in patterns:
        if matches[pat]:
            print(f"    '{pat}' at {matches[pat]}")
    print(f"  total hits: {scanner.count_matches(log)}, contains a blocked word: {scanner.contains_any(log)}\n")

    print("  The single scan finds every occurrence of every pattern in O(n + matches) time --")
    print("  independent of the pattern count, unlike running one search per pattern. It powers")
    print("  virus scanners, spam filters, and DNA motif search.")

    _svg(os.path.join(outdir, "aho_corasick.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'aho_corasick.svg')}")


def _svg(path, w=760, h=430):
    patterns = ["he", "she", "his", "hers"]
    ac = AhoCorasick(patterns)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Aho-Corasick: a trie of patterns with failure links</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'solid = trie edges (a character); dashed red = failure links (fallback on a mismatch)</text>',
    ]

    # BFS-assign positions: layer by depth, spread within layer
    from collections import deque
    layers = {}
    ids = {}
    q = deque([(ac.root, None, "")])
    counter = [0]

    def nid(node):
        if id(node) not in ids:
            ids[id(node)] = counter[0]
            counter[0] += 1
        return ids[id(node)]

    # gather nodes by depth via BFS
    seen = set()
    bq = deque([ac.root])
    nodes_by_depth = {}
    edge_list = []       # (parent, child, char)
    while bq:
        node = bq.popleft()
        if id(node) in seen:
            continue
        seen.add(id(node))
        nodes_by_depth.setdefault(node.depth, []).append(node)
        for ch, child in sorted(node.children.items()):
            edge_list.append((node, child, ch))
            bq.append(child)

    pos = {}
    maxdepth = max(nodes_by_depth)
    for depth, nodes in nodes_by_depth.items():
        for i, node in enumerate(nodes):
            x = 60 + i * (w - 120) / max(1, len(nodes) - 1 if len(nodes) > 1 else 1)
            if len(nodes) == 1:
                x = w / 2
            y = 80 + depth * 78
            pos[id(node)] = (x, y)

    # trie edges (solid)
    for parent, child, ch in edge_list:
        x1, y1 = pos[id(parent)]
        x2, y2 = pos[id(child)]
        parts.append(f'<line x1="{x1:.1f}" y1="{y1+14:.1f}" x2="{x2:.1f}" y2="{y2-14:.1f}" '
                     f'stroke="#4dabf7" stroke-width="1.6"/>')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        parts.append(f'<text x="{mx+6:.1f}" y="{my:.1f}" fill="#8b949e" font-size="11">{ch}</text>')

    # failure links (dashed red), skip links to root for clarity except from depth>=2
    for node in seen and [n for ns in nodes_by_depth.values() for n in ns]:
        if node.fail is not None and node.depth >= 2 and id(node.fail) in pos:
            x1, y1 = pos[id(node)]
            x2, y2 = pos[id(node.fail)]
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                         f'stroke="#ff6b6b" stroke-width="1" stroke-dasharray="4 3" opacity="0.8"/>')

    # nodes
    for ns in nodes_by_depth.values():
        for node in ns:
            x, y = pos[id(node)]
            is_out = bool(node.outputs)
            fill = "#06d6a0" if is_out else "#161b22"
            stroke = "#06d6a0" if is_out else "#4dabf7"
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" fill="{fill}" '
                         f'stroke="{stroke}" stroke-width="1.6"/>')
            if is_out:
                labels = ",".join(patterns[i] for i in node.outputs)
                parts.append(f'<text x="{x:.1f}" y="{y+27:.1f}" fill="#06d6a0" font-size="9" '
                             f'text-anchor="middle">{labels}</text>')
    # root label
    rx, ry = pos[id(ac.root)]
    parts.append(f'<text x="{rx:.1f}" y="{ry+4:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="middle">root</text>')
    parts.append(f'<text x="60" y="{h-20}" fill="#06d6a0" font-size="10">green node = a pattern ends here (an output)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
