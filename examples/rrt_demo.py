"""Demo: RRT and RRT* motion planning through a cluttered 2D world.

Grows both a plain RRT and an RRT* through the same obstacle field, reports path lengths (showing RRT*'s
rewiring finds a shorter route), and draws the trees, obstacles, and the two solution paths.

    python examples/rrt_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rrt import World, rrt, rrt_star, path_length, path_is_valid  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("RRT / RRT*: sampling-based motion planning through obstacles\n")

    obstacles = [(30, 35, 13), (58, 62, 15), (48, 22, 11), (24, 68, 12),
                 (74, 38, 11), (68, 15, 9), (18, 42, 8)]
    world = World(100, 100, obstacles)
    start, goal = (5, 5), (95, 95)

    path_rrt, nodes_rrt, parents_rrt = rrt(world, start, goal, max_step=6, goal_bias=0.1,
                                           max_iters=8000, goal_tol=2.5, seed=7)
    path_star, nodes_star, parents_star = rrt_star(world, start, goal, max_step=6, goal_bias=0.1,
                                                   max_iters=3000, goal_tol=2.5, radius=12, seed=7)

    print(f"  world: {world.width}x{world.height} with {len(obstacles)} circular obstacles")
    print(f"  start {start} -> goal {goal}\n")
    print(f"  plain RRT : tree of {len(nodes_rrt)} nodes, path length "
          f"{path_length(path_rrt):.1f}, valid={path_is_valid(world, path_rrt, start, goal, 2.5)}")
    print(f"  RRT*      : tree of {len(nodes_star)} nodes, path length "
          f"{path_length(path_star):.1f}, valid={path_is_valid(world, path_star, start, goal, 2.5)}")
    print(f"  RRT* is {100*(1 - path_length(path_star)/path_length(path_rrt)):.1f}% shorter "
          f"(rewiring pays off)\n")

    print("  Each iteration samples a random point, steers the nearest tree node a step toward it, and")
    print("  keeps the move if collision-free. New samples fall mostly in unexplored space, so the tree")
    print("  is pulled outward and fills the free region fast. RRT* additionally rewires neighbours")
    print("  through cheaper routes, relaxing toward the shortest path as samples accumulate.")

    _svg(os.path.join(outdir, "rrt.svg"), world, start, goal,
         nodes_star, parents_star, path_rrt, path_star)
    print(f"\n  wrote {os.path.join(outdir, 'rrt.svg')}")


def _svg(path, world, start, goal, nodes, parents, path_rrt, path_star, size=520, pad=20):
    scale = (size - 2 * pad) / world.width

    def sx(x):
        return pad + x * scale

    def sy(y):
        return size - pad - y * scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size+220}" height="{size}" '
        f'viewBox="0 0 {size+220} {size}" font-family="monospace">',
        f'<rect width="{size+220}" height="{size}" fill="#0d1117"/>',
        f'<rect x="{pad}" y="{pad}" width="{size-2*pad}" height="{size-2*pad}" '
        f'fill="#010409" stroke="#30363d"/>',
        f'<text x="{pad}" y="16" fill="#e6edf3" font-size="15">RRT* tree and paths</text>',
    ]
    # obstacles
    for ox, oy, r in world.obstacles:
        parts.append(f'<circle cx="{sx(ox):.1f}" cy="{sy(oy):.1f}" r="{r*scale:.1f}" '
                     f'fill="#ff6b6b" opacity="0.25" stroke="#ff6b6b" stroke-width="1"/>')
    # RRT* tree edges (faint)
    for i, par in enumerate(parents):
        if par is not None:
            a, b = nodes[par], nodes[i]
            parts.append(f'<line x1="{sx(a[0]):.1f}" y1="{sy(a[1]):.1f}" x2="{sx(b[0]):.1f}" '
                         f'y2="{sy(b[1]):.1f}" stroke="#4dabf7" stroke-width="0.5" opacity="0.35"/>')
    # plain RRT path (orange)
    if path_rrt:
        pr = " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in path_rrt)
        parts.append(f'<polyline points="{pr}" fill="none" stroke="#ff922b" stroke-width="2" '
                     f'opacity="0.8"/>')
    # RRT* path (green)
    if path_star:
        ps = " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in path_star)
        parts.append(f'<polyline points="{ps}" fill="none" stroke="#06d6a0" stroke-width="2.5"/>')
    # start / goal
    parts.append(f'<circle cx="{sx(start[0]):.1f}" cy="{sy(start[1]):.1f}" r="6" fill="#ffd43b"/>')
    parts.append(f'<circle cx="{sx(goal[0]):.1f}" cy="{sy(goal[1]):.1f}" r="6" fill="#b197fc"/>')

    lx = size + 10
    parts.append(f'<text x="{lx}" y="60" fill="#ffd43b" font-size="12">yellow: start</text>')
    parts.append(f'<text x="{lx}" y="82" fill="#b197fc" font-size="12">purple: goal</text>')
    parts.append(f'<text x="{lx}" y="104" fill="#ff6b6b" font-size="12">red: obstacles</text>')
    parts.append(f'<text x="{lx}" y="126" fill="#4dabf7" font-size="12">blue: RRT* tree</text>')
    parts.append(f'<text x="{lx}" y="148" fill="#ff922b" font-size="12">orange: plain RRT path</text>')
    parts.append(f'<text x="{lx}" y="170" fill="#06d6a0" font-size="12">green: RRT* path</text>')
    parts.append(f'<text x="{lx}" y="200" fill="#8b949e" font-size="11">RRT* {100*(1-path_length(path_star)/path_length(path_rrt)):.0f}% shorter</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
