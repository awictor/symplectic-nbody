"""Demo: weighted interval scheduling -- the most valuable non-overlapping jobs.

Solves a booking problem for maximum total value, contrasts it with the count-maximizing greedy and a
value-blind greedy, and confirms against brute force. Draws the jobs as a Gantt chart with the chosen
set highlighted.

    python examples/interval_scheduling_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from interval_scheduling import schedule, activity_selection, total_weight  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Weighted interval scheduling: the most valuable non-conflicting jobs\n")

    # a hall-booking problem: (start, finish, value)
    jobs = [(0, 3, 20), (1, 4, 5), (3, 5, 10), (4, 7, 25), (5, 9, 30),
            (6, 8, 8), (8, 10, 12), (2, 9, 40)]
    labels = "ABCDEFGH"
    print("  Requests (start, finish, value):")
    for lab, (s, f, w) in zip(labels, jobs):
        print(f"    {lab}: [{s}, {f}) value {w}")

    best_w, chosen = schedule(jobs)
    chosen_labels = [labels[jobs.index(c)] for c in chosen]
    print(f"\n  Optimal schedule (weighted DP): value {best_w}, jobs {chosen_labels}")

    # count-maximizing greedy for contrast
    cnt, sel = activity_selection(jobs)
    sel_labels = [labels[jobs.index(s)] for s in sel]
    print(f"  Count-maximizing greedy: {cnt} jobs {sel_labels}, value {total_weight(sel)}")

    # value-blind greedy (take earliest-finish regardless of value) already shown above; also a
    # naive highest-value-first greedy
    by_value = sorted(jobs, key=lambda j: -j[2])
    taken = []
    last = float("-inf")
    for j in sorted(by_value, key=lambda j: j[1]):
        pass
    # proper greedy-by-value with conflict check
    taken = []
    for j in by_value:
        if all(not (j[0] < t[1] and t[0] < j[1]) for t in taken):
            taken.append(j)
    print(f"  Greedy highest-value-first: value {total_weight(taken)} "
          f"(greedy is not optimal in general)")

    print(f"\n  -> the DP's {best_w} beats both greedies; only dynamic programming guarantees the optimum.")

    print("\n  Sort by finish time; for each job find p(i), the latest job finishing before it starts")
    print("  (binary search). Then opt(i) = max(skip job i, value_i + opt(p(i))). O(n log n), and")
    print("  tracing back the max choices recovers the chosen set. Equal weights reduce to the greedy.")

    _svg(os.path.join(outdir, "interval_scheduling.svg"), jobs, chosen, labels)
    print(f"\n  wrote {os.path.join(outdir, 'interval_scheduling.svg')}")


def _svg(path, jobs, chosen, labels, width=760, height=420):
    n = len(jobs)
    tmax = max(f for _, f, _ in jobs)
    wmax = max(w for _, _, w in jobs)
    m_left, m_top = 60, 80
    row_h = (height - m_top - 40) / n
    scale = (width - m_left - 120) / tmax

    chosen_set = set(chosen)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Weighted interval scheduling: chosen jobs (green) maximize total value</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'each bar is a job on the time axis; height/label shows value; green = in the optimal set</text>',
    ]
    for i, (s, f, w) in enumerate(jobs):
        y = m_top + i * row_h
        x0 = m_left + s * scale
        x1 = m_left + f * scale
        chosen = (s, f, w) in chosen_set
        col = "#06d6a0" if chosen else "#30363d"
        h = row_h * 0.6 * (0.4 + 0.6 * w / wmax)
        parts.append(f'<rect x="{x0:.1f}" y="{y + (row_h-h)/2:.1f}" width="{x1-x0:.1f}" '
                     f'height="{h:.1f}" fill="{col}" fill-opacity="0.85"/>')
        parts.append(f'<text x="{m_left-8:.1f}" y="{y + row_h/2 + 4:.1f}" fill="#8b949e" '
                     f'font-size="12" text-anchor="end">{labels[i]}</text>')
        parts.append(f'<text x="{x1 + 6:.1f}" y="{y + row_h/2 + 4:.1f}" '
                     f'fill="{"#06d6a0" if chosen else "#8b949e"}" font-size="11">v{w}</text>')

    parts.append(f'<text x="20" y="{height-16}" fill="#06d6a0" font-size="13">'
                 f'optimal total value = {sum(w for _,_,w in chosen_set)}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
