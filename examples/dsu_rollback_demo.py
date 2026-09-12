"""Demo: rollback disjoint-set union for offline dynamic connectivity.

Adds edges one at a time tracking the component count, then rolls back through snapshots to earlier
states -- the undo that ordinary path-compressed union-find cannot do. Draws the component count over
a timeline of edge additions and retractions.

    python examples/dsu_rollback_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dsu_rollback import RollbackDSU  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Rollback disjoint-set union: undoable connectivity\n")

    n = 8
    d = RollbackDSU(n)
    print(f"  {n} isolated nodes -> {d.component_count()} components\n")

    edges = [(0, 1), (2, 3), (4, 5), (1, 2), (6, 7), (3, 4)]
    print("  Adding edges (recording a snapshot before each):")
    snaps = []
    timeline = [(0, d.component_count())]
    for step, (u, v) in enumerate(edges, 1):
        snaps.append(d.snapshot())
        merged = d.unite(u, v)
        timeline.append((step, d.component_count()))
        print(f"    edge {u}-{v}: {'merged' if merged else 'already connected'} "
              f"-> {d.component_count()} components")

    print(f"\n  Now {d.component_count()} components; 0-5 chained together, 6-7 a separate pair")

    # roll back to before the last 3 edges
    d.rollback(snaps[3])
    print(f"\n  Roll back 3 edges (undo 1-2, 6-7, 3-4):")
    print(f"    components: {d.component_count()}")
    print(f"    0 and 3 connected: {d.connected(0, 3)} (retracted -- the 1-2 bridge is gone)")
    print(f"    0 and 1 connected: {d.connected(0, 1)} (kept -- it was before the snapshot)")

    # full rollback
    d.rollback(snaps[0])
    print(f"\n  Full rollback to the start: {d.component_count()} components "
          f"(back to all singletons: {d.component_count() == n})")

    print("\n  Ordinary union-find uses path compression and cannot be undone. Rollback DSU uses only")
    print("  union by rank -- each union changes O(1) state (one parent, maybe one rank) -- and records")
    print("  those changes on a stack, so a snapshot is a stack length and rollback replays it in reverse.")

    _svg(os.path.join(outdir, "dsu_rollback.svg"), edges, snaps, n)
    print(f"\n  wrote {os.path.join(outdir, 'dsu_rollback.svg')}")


def _svg(path, edges, snaps, n, width=760, height=420):
    # replay to build a component-count timeline including the rollbacks
    d = RollbackDSU(n)
    counts = [d.component_count()]
    labels = ["start"]
    local_snaps = []
    for u, v in edges:
        local_snaps.append(d.snapshot())
        d.unite(u, v)
        counts.append(d.component_count())
        labels.append(f"+{u}{v}")
    # rollback to before last 3
    d.rollback(local_snaps[3])
    counts.append(d.component_count())
    labels.append("rollback")
    d.rollback(local_snaps[0])
    counts.append(d.component_count())
    labels.append("reset")

    m_left, m_bot, m_top, m_right = 60, 70, 80, 40
    pw = width - m_left - m_right
    ph = height - m_top - m_bot
    steps = len(counts)
    ymax = n

    def px(i):
        return m_left + i / (steps - 1) * pw

    def py(c):
        return m_top + ph - c / ymax * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Rollback DSU: component count as edges are added and retracted</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'each edge merges components (count drops); the rollback jumps back up -- the undo Bloom-free union-find enables</text>',
    ]
    parts.append(f'<line x1="{m_left}" y1="{m_top+ph}" x2="{m_left+pw}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    pts = " ".join(f"{px(i):.1f},{py(c):.1f}" for i, c in enumerate(counts))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="2.2"/>')
    for i, (c, lab) in enumerate(zip(counts, labels)):
        col = "#ff6b6b" if lab in ("rollback", "reset") else "#4dabf7"
        parts.append(f'<circle cx="{px(i):.1f}" cy="{py(c):.1f}" r="4" fill="{col}"/>')
        parts.append(f'<text x="{px(i):.0f}" y="{m_top+ph+18:.0f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{lab}</text>')
        parts.append(f'<text x="{px(i):.0f}" y="{py(c)-8:.0f}" fill="#e6edf3" font-size="10" '
                     f'text-anchor="middle">{c}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
