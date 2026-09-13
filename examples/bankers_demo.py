"""Demo: the Banker's algorithm -- granting resources while guaranteeing no deadlock.

Runs the classic safety check on a resource-allocation state, shows a safe sequence, demonstrates that
a dangerous request is refused while a safe one is granted, and detects a circular-wait deadlock. Draws
the allocation / max / available tables and the safe sequence.

    python examples/bankers_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bankers import is_safe, request_resources, detect_deadlock, _need  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Banker's algorithm: lend resources only if a safe finishing order still exists\n")

    procs = ["P0", "P1", "P2", "P3", "P4"]
    res = ["A", "B", "C"]
    alloc = [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]
    maxm = [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]
    avail = [3, 3, 2]

    print("     allocation    max        need")
    need = _need(alloc, maxm)
    for i, p in enumerate(procs):
        print(f"  {p}  {alloc[i]}     {maxm[i]}   {need[i]}")
    print(f"  available: {avail}\n")

    safe, seq = is_safe(alloc, maxm, avail)
    print(f"  state is SAFE: {safe}")
    print(f"  safe finishing sequence: {' -> '.join(procs[i] for i in seq)}")
    print("    (each process's remaining need fits the pool when its turn comes, then it releases)\n")

    print("  P1 requests one more A and two more C, i.e. [1,0,2]:")
    g, na, nav = request_resources(alloc, maxm, avail, 1, [1, 0, 2])
    print(f"    granted: {g}   (state stays safe; available now {nav})\n")

    print("  P4 requests [3,3,0] (would drain the pool into an unsafe state):")
    g2, _, _ = request_resources(alloc, maxm, avail, 4, [3, 3, 0])
    print(f"    granted: {g2}   (refused -- P4 must wait to avoid risking deadlock)\n")

    print("  deadlock detection: P0 holds A wants B, P1 holds B wants A, nothing free:")
    stuck = detect_deadlock([[1, 0], [0, 1]], [[0, 1], [1, 0]], [0, 0])
    print(f"    deadlocked processes: {stuck}   (a genuine circular wait)\n")

    print("  A state is safe if some order lets every process reach its MAX and finish, freeing its")
    print("  resources for the next. The banker grants a request only when the resulting state is still")
    print("  safe -- so the system can always drain to completion and never paints itself into deadlock.")

    _svg(os.path.join(outdir, "bankers.svg"), procs, res, alloc, maxm, avail, seq)
    print(f"\n  wrote {os.path.join(outdir, 'bankers.svg')}")


def _svg(path, procs, res, alloc, maxm, avail, seq, width=680, height=400):
    n = len(procs)
    m = len(res)
    need = _need(alloc, maxm)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="16">'
        f'Bankers algorithm: allocation, need, and a safe finishing order</text>',
    ]
    cell = 34
    # three tables: Allocation, Need
    def table(title, mat, ox, colour):
        out = [f'<text x="{ox}" y="70" fill="{colour}" font-size="12">{title}</text>']
        for j in range(m):
            out.append(f'<text x="{ox + 30 + j*cell:.0f}" y="90" fill="#8b949e" font-size="10" '
                       f'text-anchor="middle">{res[j]}</text>')
        for i in range(n):
            y = 100 + i * cell
            out.append(f'<text x="{ox:.0f}" y="{y+cell*0.6:.0f}" fill="#8b949e" font-size="11">'
                       f'{procs[i]}</text>')
            for j in range(m):
                x = ox + 15 + j * cell
                out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell-4:.1f}" height="{cell-4:.1f}" '
                           f'rx="2" fill="#161b22" stroke="{colour}" stroke-width="1"/>')
                out.append(f'<text x="{x+(cell-4)/2:.1f}" y="{y+cell*0.55:.0f}" fill="#e6edf3" '
                           f'font-size="11" text-anchor="middle">{mat[i][j]}</text>')
        return out

    parts += table("Allocation", alloc, 40, "#4dabf7")
    parts += table("Need", need, 220, "#ff922b")
    # available
    parts.append(f'<text x="400" y="70" fill="#06d6a0" font-size="12">Available</text>')
    for j in range(m):
        x = 415 + j * cell
        parts.append(f'<rect x="{x:.1f}" y="90" width="{cell-4:.1f}" height="{cell-4:.1f}" rx="2" '
                     f'fill="#161b22" stroke="#06d6a0" stroke-width="1"/>')
        parts.append(f'<text x="{x+(cell-4)/2:.1f}" y="107" fill="#e6edf3" font-size="11" '
                     f'text-anchor="middle">{avail[j]}</text>')

    # safe sequence
    parts.append(f'<text x="400" y="150" fill="#ffd43b" font-size="12">Safe sequence:</text>')
    for k, i in enumerate(seq):
        y = 170 + k * 30
        parts.append(f'<rect x="415" y="{y-16:.0f}" width="50" height="22" rx="4" fill="#161b22" '
                     f'stroke="#ffd43b" stroke-width="1.5"/>')
        parts.append(f'<text x="440" y="{y:.0f}" fill="#ffd43b" font-size="12" '
                     f'text-anchor="middle">{procs[i]}</text>')
        if k < len(seq) - 1:
            parts.append(f'<text x="475" y="{y:.0f}" fill="#8b949e" font-size="12">v</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
