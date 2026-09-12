"""Demo: Gale-Shapley stable matching -- pairing two sides so no couple wants to defect.

Matches applicants to schools by deferred acceptance, verifies stability, contrasts the applicant- and
school-optimal outcomes, and draws the matching with each side's preference rank shown.

    python examples/gale_shapley_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gale_shapley import (stable_matching, is_stable, blocking_pairs,  # noqa: E402
                          reviewer_optimal_matching, all_stable_matchings)

APPLICANTS = ["Ada", "Bo", "Cy", "Di"]
SCHOOLS = ["MIT", "Yale", "UCLA", "NYU"]

# applicant preferences over schools (best first), by index
APP_PREFS = [
    [0, 1, 2, 3],   # Ada: MIT > Yale > UCLA > NYU
    [1, 0, 2, 3],   # Bo:  Yale > MIT > UCLA > NYU
    [0, 1, 2, 3],   # Cy:  MIT > Yale > UCLA > NYU
    [1, 0, 3, 2],   # Di:  Yale > MIT > NYU > UCLA
]
# school preferences over applicants (best first)
SCH_PREFS = [
    [3, 1, 0, 2],   # MIT:  Di > Bo > Ada > Cy
    [0, 2, 1, 3],   # Yale: Ada > Cy > Bo > Di
    [1, 0, 3, 2],   # UCLA: Bo > Ada > Di > Cy
    [2, 3, 0, 1],   # NYU:  Cy > Di > Ada > Bo
]
N = 4


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    app_opt = stable_matching(N, APP_PREFS, SCH_PREFS)
    sch_opt = reviewer_optimal_matching(N, APP_PREFS, SCH_PREFS)

    print("Gale-Shapley stable matching: applicants <-> schools\n")
    print("  applicant-proposing (applicant-optimal) matching:")
    for a in range(N):
        s = app_opt[a]
        arank = APP_PREFS[a].index(s) + 1
        srank = SCH_PREFS[s].index(a) + 1
        print(f"    {APPLICANTS[a]:4s} -> {SCHOOLS[s]:5s}  "
              f"(applicant's #{arank} choice, school's #{srank} choice)")
    print(f"\n  stable (no blocking pair): {is_stable(N, APP_PREFS, SCH_PREFS, app_opt)}")

    all_stable = all_stable_matchings(N, APP_PREFS, SCH_PREFS)
    print(f"  number of stable matchings: {len(all_stable)}")

    if sch_opt != app_opt:
        print("\n  school-proposing (school-optimal) matching differs:")
        for a in range(N):
            print(f"    {APPLICANTS[a]:4s} -> {SCHOOLS[sch_opt[a]]}")
        print("  applicant-optimal is simultaneously school-PESSIMAL, and vice versa.")
    else:
        print("\n  the stable matching is unique: applicant- and school-optimal coincide.")

    print("\n  Deferred acceptance: each free applicant proposes to their next-favourite school; the")
    print("  school tentatively holds its best offer so far and rejects the rest. Since schools only")
    print("  ever trade up and applicants only ever move down their list, it ends in <= n^2 proposals")
    print("  with a provably stable result -- the mechanism behind real residency and school matches.")

    _svg(os.path.join(outdir, "gale_shapley.svg"), app_opt)
    print(f"\n  wrote {os.path.join(outdir, 'gale_shapley.svg')}")


def _svg(path, match_p, width=680, height=430):
    lx, rx = 190, 470
    top, gap = 100, 80

    def ly(i):
        return top + i * gap

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">'
        f'Stable matching: green links pair applicants to schools with no defector</text>',
        f'<text x="20" y="54" fill="#8b949e" font-size="12">'
        f'no applicant and school both prefer each other over their assigned partner</text>',
        f'<text x="{lx}" y="{top-24}" fill="#4dabf7" font-size="13" text-anchor="middle">'
        f'APPLICANTS</text>',
        f'<text x="{rx}" y="{top-24}" fill="#ffd43b" font-size="13" text-anchor="middle">'
        f'SCHOOLS</text>',
    ]

    for a in range(N):
        s = match_p[a]
        parts.append(f'<line x1="{lx}" y1="{ly(a)}" x2="{rx}" y2="{ly(s)}" '
                     f'stroke="#06d6a0" stroke-width="3"/>')

    for i in range(N):
        y = ly(i)
        parts.append(f'<circle cx="{lx}" cy="{y}" r="16" fill="#4dabf7"/>')
        parts.append(f'<text x="{lx-26}" y="{y+5}" fill="#e6edf3" font-size="13" '
                     f'text-anchor="end">{APPLICANTS[i]}</text>')
        parts.append(f'<circle cx="{rx}" cy="{y}" r="16" fill="#ffd43b"/>')
        parts.append(f'<text x="{rx+26}" y="{y+5}" fill="#e6edf3" font-size="13" '
                     f'text-anchor="start">{SCHOOLS[i]}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
