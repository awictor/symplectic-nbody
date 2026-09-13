"""Demo: real-time scheduling -- can periodic tasks meet their deadlines under RM and EDF?

Applies the Liu-Layland (RM) and utilisation (EDF) schedulability tests to a task set, confirms them
against a hyperperiod simulation, and shows a set that EDF can schedule but RM cannot. Draws the RM and
EDF timelines over the hyperperiod.

    python examples/rt_scheduling_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rt_scheduling import (Task, utilisation, liu_layland_bound, rm_schedulable_ll,  # noqa: E402
                           edf_schedulable, simulate, hyperperiod, rm_schedulable_exact,
                           edf_schedulable_exact)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Real-time scheduling: proving periodic tasks always meet their deadlines\n")

    tasks = [Task(1, 4, name="A"), Task(2, 6, name="B"), Task(1, 8, name="C")]
    n = len(tasks)
    U = utilisation(tasks)
    print(f"  task set (C=compute, T=period):")
    for t in tasks:
        print(f"    {t.name}: C={t.C}, T={t.T}  (uses {t.C/t.T:.3f} of the CPU)")
    print(f"  total utilisation U = {U:.4f}, hyperperiod = {hyperperiod(tasks)}\n")

    print(f"  RM Liu-Layland bound for {n} tasks: {liu_layland_bound(n):.4f}")
    print(f"    RM sufficient test (U <= bound): {rm_schedulable_ll(tasks)}")
    print(f"    RM exact (simulation):           {rm_schedulable_exact(tasks)}")
    print(f"  EDF exact test (U <= 1):           {edf_schedulable(tasks)}")
    print(f"    EDF exact (simulation):          {edf_schedulable_exact(tasks)}\n")

    # a set EDF schedules but RM (fixed priority) cannot
    hard = [Task(2, 5, name="X"), Task(4, 7, name="Y")]
    print(f"  a harder set X(C=2,T=5), Y(C=4,T=7): U = {utilisation(hard):.4f}")
    print(f"    RM schedulable:  {rm_schedulable_exact(hard)}")
    print(f"    EDF schedulable: {edf_schedulable_exact(hard)}")
    print("    (EDF's dynamic deadlines use the CPU more fully than RM's fixed priorities)\n")

    print("  Rate-monotonic gives shorter-period tasks higher fixed priority; Liu-Layland proved it")
    print("  optimal among fixed-priority policies, schedulable whenever U <= n(2^(1/n)-1). EDF always")
    print("  runs the job with the nearest deadline and is exactly schedulable iff U <= 1 -- the most")
    print("  any policy can promise. Both are checked against a full hyperperiod simulation here.")

    _svg(os.path.join(outdir, "rt_scheduling.svg"), tasks)
    print(f"\n  wrote {os.path.join(outdir, 'rt_scheduling.svg')}")


def _svg(path, tasks, width=760, height=360):
    n = len(tasks)
    H = min(hyperperiod(tasks), 48)
    # get the running task at each tick for RM and EDF by replaying the sim logic
    def timeline(policy):
        remaining = [0] * n
        abs_deadline = [0] * n
        released = [False] * n
        who = []
        for time in range(H):
            for i, t in enumerate(tasks):
                if remaining[i] > 0 and time >= abs_deadline[i]:
                    remaining[i] = 0
            for i, t in enumerate(tasks):
                if time % t.T == 0:
                    remaining[i] = t.C
                    abs_deadline[i] = time + t.D
                    released[i] = True
            ready = [i for i in range(n) if remaining[i] > 0]
            if not ready:
                who.append(None)
                continue
            if policy == "edf":
                c = min(ready, key=lambda i: (abs_deadline[i], i))
            else:
                c = min(ready, key=lambda i: (tasks[i].T, i))
            remaining[c] -= 1
            who.append(c)
        return who

    cols = ["#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#b197fc"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="16">'
        f'Which task runs each tick over one hyperperiod ({H} ticks)</text>',
    ]
    cell = (width - 100) / H
    for row, (policy, label) in enumerate((("rm", "RM"), ("edf", "EDF"))):
        y = 70 + row * 90
        who = timeline(policy)
        parts.append(f'<text x="20" y="{y+16:.0f}" fill="#8b949e" font-size="13">{label}</text>')
        for tick, c in enumerate(who):
            x = 70 + tick * cell
            fill = cols[c % len(cols)] if c is not None else "#161b22"
            parts.append(f'<rect x="{x:.1f}" y="{y:.0f}" width="{cell-1:.1f}" height="28" '
                         f'fill="{fill}" stroke="#0d1117" stroke-width="0.5"/>')
        # release ticks marked below
        for tick in range(H):
            for i, t in enumerate(tasks):
                if tick % t.T == 0:
                    x = 70 + tick * cell + cell / 2
                    parts.append(f'<text x="{x:.1f}" y="{y+40:.0f}" fill="{cols[i%len(cols)]}" '
                                 f'font-size="7" text-anchor="middle">^</text>')
    # legend
    for i, t in enumerate(tasks):
        parts.append(f'<rect x="{70 + i*90}" y="{height-30}" width="12" height="12" '
                     f'fill="{cols[i%len(cols)]}"/>')
        parts.append(f'<text x="{86 + i*90}" y="{height-20}" fill="#8b949e" font-size="11">'
                     f'{t.name} (T={t.T})</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
