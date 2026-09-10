"""Demo: radioactive decay and decay chains.

Prints radiometric ages for surviving fractions and the half-lives of common isotopes,
then draws a parent-daughter chain: the parent's exponential decay and the daughter's
Bateman rise-and-fall toward the parent curve (secular equilibrium).

    python examples/radioactive_decay_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from radioactive_decay import (remaining, activity, age_from_fraction,  # noqa: E402
                               n_half_lives, bateman_daughter,
                               secular_equilibrium_activity)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Radioactive decay: N = N0 2^(-t/t_half); age = t_half log2(N0/N)\n")
    print("  carbon-14 dating (t_half = 5730 yr):")
    print(f"  {'% remaining':>14}{'half-lives':>12}{'age (yr)':>12}")
    print("  " + "-" * 40)
    for frac in (0.90, 0.50, 0.25, 0.10, 0.01):
        print(f"  {frac*100:>13.0f}%{n_half_lives(frac):>12.2f}"
              f"{age_from_fraction(frac, 5730):>12.0f}")

    print("\n  Parent -> daughter chain (parent 8 d, daughter 0.6 d):")
    print(f"  {'time (d)':>10}{'parent':>10}{'daughter':>10}")
    for t in (0, 1, 2, 5, 10, 20):
        p = remaining(1000.0, float(t), 8.0)
        d = bateman_daughter(1000.0, float(t), 8.0, 0.6)
        print(f"  {t:>10}{p:>10.0f}{d:>10.0f}")

    print("\n  The daughter starts at zero, builds up as the parent feeds it, and once")
    print("  it is decaying as fast as it is produced it tracks the parent -- secular")
    print("  equilibrium, where daughter activity equals parent activity. That balance")
    print("  runs medical radioisotope generators and the radon from uranium in rock.")

    _svg(os.path.join(outdir, "radioactive_decay.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'radioactive_decay.svg')}")


def _svg(path, size=720, pad=72):
    t_p, t_d = 8.0, 0.6   # days
    ts = [0.05 * i for i in range(0, 601)]   # 0 .. 30 d
    parent = [remaining(1000.0, t, t_p) for t in ts]
    daughter = [bateman_daughter(1000.0, t, t_p, t_d) for t in ts]
    xmin, xmax = ts[0], ts[-1]
    ymax = 1050.0

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - y / ymax * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    pp = " ".join(f"{sx(ts[i]):.1f},{sy(parent[i]):.1f}" for i in range(len(ts)))
    pd = " ".join(f"{sx(ts[i]):.1f},{sy(daughter[i]):.1f}" for i in range(len(ts)))
    parts.append(f'<polyline points="{pp}" fill="none" stroke="#4dabf7" stroke-width="2.4"/>')
    parts.append(f'<polyline points="{pd}" fill="none" stroke="#ff922b" stroke-width="2.4"/>')

    # mark parent half-lives
    for k in range(1, 4):
        x = sx(k * t_p)
        parts.append(f'<line x1="{x:.1f}" y1="{sy(remaining(1000,k*t_p,t_p)):.1f}" '
                     f'x2="{x:.1f}" y2="{size-pad:.1f}" stroke="#30363d" stroke-width="0.8" stroke-dasharray="2 4"/>')

    parts.append(f'<text x="{pad+10}" y="{pad+22}" fill="#4dabf7" font-size="12">'
                 f'parent (t_half = 8 d)</text>')
    parts.append(f'<text x="{pad+10}" y="{pad+38}" fill="#ff922b" font-size="12">'
                 f'daughter (Bateman, t_half = 0.6 d)</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Parent decay and daughter build-up</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'daughter rises then tracks the parent -- secular equilibrium</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">time (days) -&gt;</text>')
    parts.append(f'<text x="{pad-18}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'number of nuclei</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
