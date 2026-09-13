"""Demo: DFA minimization by Hopcroft's algorithm -- the unique smallest automaton.

Takes a redundant DFA for a simple language, minimizes it, shows the state count collapse, verifies
the language is unchanged, and detects that two differently-built DFAs are equivalent. Draws both
automata as state diagrams.

    python examples/dfa_minimization_demo.py [output_dir]
"""

import itertools
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dfa_minimization import DFA, minimize, equivalent  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("DFA minimization (Hopcroft): the unique smallest automaton for a language\n")

    # A deliberately redundant 6-state DFA for "binary strings with an even number of 1s".
    # States 0,2,4 = even; 1,3,5 = odd -- three copies of each parity, all mergeable.
    d = DFA(
        states=set(range(6)),
        alphabet=["0", "1"],
        transitions={
            (0, "0"): 2, (0, "1"): 1,
            (1, "0"): 3, (1, "1"): 2,
            (2, "0"): 4, (2, "1"): 3,
            (3, "0"): 5, (3, "1"): 4,
            (4, "0"): 0, (4, "1"): 5,
            (5, "0"): 1, (5, "1"): 0,
        },
        start=0,
        accepting={0, 2, 4},  # even number of 1s
    )

    print("  Language: binary strings with an even number of 1s.")
    print(f"  Original DFA: {len(d.states)} states (three redundant copies of each parity).")

    m = minimize(d)
    print(f"  Minimized DFA: {len(m.states)} states.\n")

    # verify language preserved
    print(f"  {'string':>8}  {'orig':>5}  {'min':>5}")
    for s in ["", "1", "11", "101", "1101", "10101"]:
        print(f"  {s or '(empty)':>8}  {str(d.accepts(s)):>5}  {str(m.accepts(s)):>5}")
    same = all(d.accepts("".join(t)) == m.accepts("".join(t))
               for L in range(8) for t in itertools.product("01", repeat=L))
    print(f"\n  Language identical over all strings up to length 7: {same}")

    # a second, independently built DFA for the same language
    d2 = DFA({0, 1}, ["0", "1"], {(0, "0"): 0, (0, "1"): 1, (1, "0"): 1, (1, "1"): 0}, 0, {0})
    print(f"  A separate 2-state DFA for the same language is equivalent: {equivalent(d, d2)}")

    print("\n  Myhill-Nerode: the minimal DFA is a canonical form, so language equivalence reduces")
    print("  to minimizing both machines and checking they are isomorphic.")

    _svg(os.path.join(outdir, "dfa_minimization.svg"), d, m)
    print(f"\n  wrote {os.path.join(outdir, 'dfa_minimization.svg')}")


def _svg(path, d, m, width=760, height=430):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="15">'
        f'Original {len(d.states)}-state DFA (left) collapses to {len(m.states)} states (right)</text>',
        '<defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">'
        '<path d="M0,0 L7,3 L0,6 Z" fill="#8b949e"/></marker></defs>',
    ]

    def draw(dfa, cx, cy, R, xoff, label):
        states = sorted(dfa.states, key=lambda s: str(s))
        n = len(states)
        pos = {}
        for i, s in enumerate(states):
            ang = -math.pi / 2 + 2 * math.pi * i / n
            pos[s] = (xoff + cx + R * math.cos(ang), cy + R * math.sin(ang))
        # edges
        for (s, c), t in dfa.transitions.items():
            if s not in pos or t not in pos:
                continue
            x1, y1 = pos[s]
            x2, y2 = pos[t]
            if s == t:
                parts.append(f'<circle cx="{x1:.0f}" cy="{y1-22:.0f}" r="10" fill="none" '
                             f'stroke="#30363d" stroke-width="1"/>')
                continue
            dx, dy = x2 - x1, y2 - y1
            dd = math.hypot(dx, dy) or 1
            ux, uy = dx / dd, dy / dd
            sx2, sy2 = x1 + ux * 16, y1 + uy * 16
            ex, ey = x2 - ux * 16, y2 - uy * 16
            px, py = -uy, ux
            mx, my = (sx2 + ex) / 2 + px * 12, (sy2 + ey) / 2 + py * 12
            parts.append(f'<path d="M{sx2:.0f},{sy2:.0f} Q{mx:.0f},{my:.0f} {ex:.0f},{ey:.0f}" '
                         f'fill="none" stroke="#30363d" stroke-width="1" marker-end="url(#a)"/>')
        # nodes
        for s in states:
            x, y = pos[s]
            acc = s in dfa.accepting
            col = "#06d6a0" if acc else "#4dabf7"
            if acc:
                parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="16" fill="none" '
                             f'stroke="{col}" stroke-width="1"/>')
            parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="13" fill="#161b22" '
                         f'stroke="{col}" stroke-width="2"/>')
            sid = str(s) if not isinstance(s, frozenset) else "".join(sorted(str(x) for x in s))
            parts.append(f'<text x="{x:.0f}" y="{y+3:.0f}" fill="{col}" font-size="8" '
                         f'text-anchor="middle">{sid[:4]}</text>')
        parts.append(f'<text x="{xoff+cx:.0f}" y="{cy+R+35:.0f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{label}</text>')

    draw(d, 150, 210, 110, 0, f"{len(d.states)} states")
    draw(m, 150, 210, 90, 400, f"{len(m.states)} states (minimal)")
    parts.append('<text x="380" y="410" fill="#8b949e" font-size="10" text-anchor="middle">'
                 'green ring = accepting</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
