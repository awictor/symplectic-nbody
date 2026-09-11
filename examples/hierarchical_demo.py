"""Demo: hierarchical agglomerative clustering and the dendrogram.

Builds the full merge tree of a small set of 2-D blobs, draws the dendrogram (whose branch heights
are the merge distances), cuts it into three clusters, and contrasts how single-linkage chaining
and complete-linkage compactness split a long chain of points differently.

    python examples/hierarchical_demo.py [output_dir]
"""

import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hierarchical import linkage, fcluster, merge_heights, is_monotone, LINKAGES  # noqa: E402


def _blobs(seed=7):
    state = seed

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    X, truth = [], []
    for c, (cx, cy) in enumerate([(0.0, 0.0), (6.0, 0.5), (3.0, 6.0)]):
        for _ in range(8):
            X.append([cx + (rng() - 0.5) * 1.4, cy + (rng() - 0.5) * 1.4])
            truth.append(c)
    return X, truth


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    X, truth = _blobs()
    merges = linkage(X, "ward")

    print("Hierarchical agglomerative clustering: a tree of nested groupings\n")
    print(f"  {len(X)} points, Ward linkage, {len(merges)} merges to one cluster\n")

    print("  Merge heights climb monotonically (no dendrogram inversions):")
    h = merge_heights(merges)
    for i in (0, 1, 2, len(h) // 2, len(h) - 2, len(h) - 1):
        print(f"    merge {i:>2}: height {h[i]:.3f}")
    print(f"  monotone: {is_monotone(merges)}\n")

    # the big gap in merge heights tells you the natural number of clusters
    gaps = [(h[i + 1] - h[i], i) for i in range(len(h) - 1)]
    gaps.sort(reverse=True)
    print(f"  Largest jump in merge height is before the final "
          f"{len(X) - gaps[0][1] - 1} merges -> suggests "
          f"{len(X) - gaps[0][1] - 1} clusters.\n")

    labels = fcluster(X, 3, "ward")
    sizes = Counter(labels)
    print(f"  Cutting into 3 clusters gives sizes {dict(sorted(sizes.items()))}")
    pure = all(
        max(sum(1 for i in range(len(labels)) if labels[i] == cl and truth[i] == t)
            for t in range(3)) / list(labels).count(cl) > 0.9
        for cl in set(labels))
    print(f"  clusters recover the true blobs: {pure}\n")

    print("  Linkage changes cluster shape -- on a long chain of 24 evenly-spaced points, cut in 2:")
    chain = [[i * 0.3, 0.0] for i in range(24)]
    for method in ("single", "complete", "average", "ward"):
        sz = sorted(Counter(fcluster(chain, 2, method)).values())
        note = "chains (peels an end)" if method == "single" else \
               "compact (even split)" if method == "complete" else ""
        print(f"    {method:>9}: split sizes {sz}  {note}")

    print("\n  Every merge is recorded with the distance at which it happened, so one run yields")
    print("  the entire family of clusterings: cut the tree low for many tight groups, high for a")
    print("  few broad ones. No k needed up front -- read it off the biggest gap in merge heights.")

    _svg(os.path.join(outdir, "hierarchical.svg"), X, merges, labels)
    print(f"\n  wrote {os.path.join(outdir, 'hierarchical.svg')}")


def _svg(path, X, merges, labels, width=760, height=440):
    palette = ["#4dabf7", "#ff6b6b", "#ffd43b", "#06d6a0", "#b197fc"]
    n = len(X)

    # ---- left panel: the scatter, coloured by the 3-cluster cut ----
    xs = [p[0] for p in X]
    ys = [p[1] for p in X]
    xa, xb = min(xs) - 0.6, max(xs) + 0.6
    ya, yb = min(ys) - 0.6, max(ys) + 0.6
    lx0, lx1 = 45, width // 2 - 20
    ly0, ly1 = height - 45, 70

    def LX(x):
        return lx0 + (x - xa) / (xb - xa) * (lx1 - lx0)

    def LY(v):
        return ly0 - (v - ya) / (yb - ya) * (ly0 - ly1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Hierarchical clustering: points cut into 3 (left), dendrogram (right)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'the dendrogram branch heights are merge distances; cutting at a height gives a '
        f'clustering</text>',
    ]
    for i, (px, pv) in enumerate(X):
        parts.append(f'<circle cx="{LX(px):.1f}" cy="{LY(pv):.1f}" r="4" '
                     f'fill="{palette[labels[i] % len(palette)]}" stroke="#0d1117" '
                     f'stroke-width="0.6"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.1"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.1"/>')

    # ---- right panel: the dendrogram ----
    # compute an x-position (leaf order) and merge y-height for each cluster node
    rx0, rx1 = width // 2 + 30, width - 25
    ry0, ry1 = height - 55, 70
    hmax = merges[-1][2]

    # leaf ordering via in-order traversal of the merge tree
    children = {}
    for s, (a, b, d, sz) in enumerate(merges):
        children[n + s] = (a, b)
    order = []

    def visit(node):
        if node < n:
            order.append(node)
        else:
            a, b = children[node]
            visit(a)
            visit(b)

    visit(n + len(merges) - 1)          # root
    leaf_x = {leaf: i for i, leaf in enumerate(order)}

    def NX(pos):
        return rx0 + pos / (n - 1) * (rx1 - rx0)

    def NY(dist):
        return ry0 - dist / hmax * (ry0 - ry1)

    # node x = mean of its members' leaf positions; drawn bottom (leaves) to top (root)
    node_x = {i: leaf_x[i] for i in range(n)}
    node_y = {i: ry0 for i in range(n)}
    for s, (a, b, d, sz) in enumerate(merges):
        new = n + s
        xa_, xb_ = node_x[a], node_x[b]
        ya_, yb_ = node_y[a], node_y[b]
        yc = NY(d)
        col = palette[s % len(palette)]
        # two verticals up to the merge height, joined by a horizontal
        parts.append(f'<line x1="{NX(xa_):.1f}" y1="{ya_:.1f}" x2="{NX(xa_):.1f}" y2="{yc:.1f}" '
                     f'stroke="#8b949e" stroke-width="1.3"/>')
        parts.append(f'<line x1="{NX(xb_):.1f}" y1="{yb_:.1f}" x2="{NX(xb_):.1f}" y2="{yc:.1f}" '
                     f'stroke="#8b949e" stroke-width="1.3"/>')
        parts.append(f'<line x1="{NX(xa_):.1f}" y1="{yc:.1f}" x2="{NX(xb_):.1f}" y2="{yc:.1f}" '
                     f'stroke="#4dabf7" stroke-width="1.3"/>')
        node_x[new] = (xa_ + xb_) / 2
        node_y[new] = yc
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.1"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.1"/>')

    # a cut line at the height that yields 3 clusters (between the 3rd- and 2nd-from-last merges)
    cut = (merges[-3][2] + merges[-2][2]) / 2
    parts.append(f'<line x1="{rx0}" y1="{NY(cut):.1f}" x2="{rx1}" y2="{NY(cut):.1f}" '
                 f'stroke="#ffd43b" stroke-width="1.2" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{rx1-2:.1f}" y="{NY(cut)-4:.1f}" fill="#ffd43b" font-size="9" '
                 f'text-anchor="end">cut -> 3 clusters</text>')
    parts.append(f'<text x="{rx0-4:.1f}" y="{ry1+4:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">{hmax:.1f}</text>')
    parts.append(f'<text x="{rx0-4:.1f}" y="{ry0:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">0</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
