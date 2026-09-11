"""Demo: topological sort -- ordering tasks so prerequisites come first.

Sorts a build-dependency DAG with both Kahn's algorithm and DFS, detects a cycle, and computes
the critical path of a scheduled project. Draws the DAG laid out in topological layers with the
critical (longest) path highlighted.

    python examples/toposort_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from toposort import DAG, kahn_sort, dfs_sort, has_cycle, longest_path  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    build = DAG()
    for u, v in [("fetch", "compile"), ("compile", "link"), ("compile", "test"),
                 ("assets", "package"), ("link", "package"), ("test", "package"),
                 ("package", "deploy")]:
        build.add_edge(u, v)
    print("Topological sort: a build order where every dependency comes first\n")
    print(f"  Kahn:  {' -> '.join(kahn_sort(build))}")
    print(f"  DFS:   {' -> '.join(dfs_sort(build))}")
    print(f"  (both are valid orders; ties resolved differently)\n")

    cyc = DAG()
    for u, v in [("a", "b"), ("b", "c"), ("c", "a")]:
        cyc.add_edge(u, v)
    print(f"  a -> b -> c -> a has a cycle: {has_cycle(cyc)} (no order can exist)\n")

    # project scheduling: durations on each task, critical path = min completion time
    proj = DAG()
    for u, v in [("design", "backend"), ("design", "frontend"), ("backend", "integrate"),
                 ("frontend", "integrate"), ("integrate", "qa"), ("qa", "release"),
                 ("design", "docs"), ("docs", "release")]:
        proj.add_edge(u, v)
    dur = {"design": 5, "backend": 8, "frontend": 6, "integrate": 3,
           "qa": 4, "release": 2, "docs": 7}
    length, path = longest_path(proj, dur)
    print("  Project critical path (longest chain of durations = minimum completion time):")
    print(f"    {' -> '.join(path)}  =  {length} days")
    print(f"    (total task-days = {sum(dur.values())}, but the critical path bounds the schedule)")
    print("\n  The same ordering runs package managers, build systems, spreadsheet recompute,")
    print("  and course prerequisites -- and a cycle is a dependency that can never be resolved.")

    _svg(os.path.join(outdir, "toposort.svg"), proj, dur, path)
    print(f"\n  wrote {os.path.join(outdir, 'toposort.svg')}")


def _svg(path, graph, dur, critical, w=760, h=400):
    # assign each node a layer = longest distance from a source (topological depth)
    order = kahn_sort(graph)
    layer = {u: 0 for u in graph.adj}
    for u in order:
        for v in graph.adj[u]:
            layer[v] = max(layer[v], layer[u] + 1)
    # group nodes by layer
    layers = {}
    for u in order:
        layers.setdefault(layer[u], []).append(u)
    nlayers = max(layers) + 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Topological layers &amp; the critical path</text>',
        f'<text x="20" y="46" fill="#8b949e" font-size="12">'
        f'each column is a dependency layer; green = the critical (longest-duration) path</text>',
    ]

    # position nodes
    x0, y0 = 70, 90
    col_w = (w - 140) / max(1, nlayers - 1) if nlayers > 1 else 0
    pos = {}
    for lyr, nodes in layers.items():
        for i, u in enumerate(nodes):
            x = x0 + lyr * col_w
            y = y0 + i * 70 + (0 if len(nodes) > 1 else 60)
            pos[u] = (x, y)

    crit_set = set(critical)
    crit_edges = {(critical[i], critical[i + 1]) for i in range(len(critical) - 1)}

    # edges
    for u in graph.adj:
        for v in graph.adj[u]:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            on_crit = (u, v) in crit_edges
            col = "#06d6a0" if on_crit else "#30363d"
            sw = 2.5 if on_crit else 1
            parts.append(f'<line x1="{x1+22:.1f}" y1="{y1:.1f}" x2="{x2-22:.1f}" y2="{y2:.1f}" '
                         f'stroke="{col}" stroke-width="{sw}"/>')
    # nodes
    for u, (x, y) in pos.items():
        on_crit = u in crit_set
        stroke = "#06d6a0" if on_crit else "#4dabf7"
        parts.append(f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="26" ry="18" fill="#161b22" '
                     f'stroke="{stroke}" stroke-width="1.8"/>')
        parts.append(f'<text x="{x:.1f}" y="{y-1:.1f}" fill="{stroke}" font-size="9" '
                     f'text-anchor="middle">{u}</text>')
        parts.append(f'<text x="{x:.1f}" y="{y+10:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{dur[u]}d</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
