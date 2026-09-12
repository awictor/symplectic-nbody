"""Demo: Thompson's NFA regex engine -- linear-time matching, no catastrophic backtracking.

Compiles a few patterns, matches sample strings, verifies agreement with Python's re, and times the
notorious catastrophic-backtracking pattern to show the linear-time guarantee. Draws the compiled NFA
state/transition counts and a timing curve.

    python examples/thompson_nfa_demo.py [output_dir]
"""

import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from thompson_nfa import Regex, compile_nfa, parse  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Thompson's NFA: regex matching in guaranteed linear time\n")

    examples = [
        ("(a|b)*c", ["abbac", "ababba", "c"]),
        ("colou?r", ["color", "colour", "colur"]),
        ("a.c", ["abc", "axc", "ac"]),
        ("(ab)+", ["ab", "abab", "aba"]),
    ]
    for pat, tests in examples:
        rx = Regex(pat)
        nfa = compile_nfa(parse(pat))
        print(f"  /{pat}/  ({len(nfa.states)} NFA states)")
        for t in tests:
            mine = rx.fullmatch(t)
            ref = re.fullmatch(pat, t) is not None
            flag = "OK" if mine == ref else "MISMATCH!"
            print(f"    fullmatch({t!r:12}) = {str(mine):5}  (re agrees: {mine == ref}) {flag}")
        print()

    # catastrophic backtracking demonstration
    print("  Catastrophic backtracking pattern /(a*)*b/ against strings of 'a' (no trailing b):")
    print("    a backtracking engine takes exponential time; Thompson's NFA stays linear.\n")
    rx = Regex("(a*)*b")
    timings = []
    for n in (50, 100, 200, 400, 800):
        t0 = time.time()
        res = rx.fullmatch("a" * n)
        dt = time.time() - t0
        timings.append((n, dt))
        print(f"    n={n:4d}: matched={res}  time={dt*1000:.2f} ms")

    print("\n  Never backtracks: the engine tracks the SET of all NFA states the machine could be in")
    print("  and advances the whole set one character at a time, so matching costs O(n*m) always --")
    print("  linear in the input length, whatever the pattern.")

    _svg(os.path.join(outdir, "thompson_nfa.svg"), timings)
    print(f"\n  wrote {os.path.join(outdir, 'thompson_nfa.svg')}")


def _svg(path, timings, width=760, height=380):
    ns = [n for n, _ in timings]
    ts = [t * 1000 for _, t in timings]   # ms
    nmax = max(ns)
    tmax = max(ts) if max(ts) > 0 else 1.0

    ox, oy = 60, 320
    pw, ph = width - 100, 250

    def px(n):
        return ox + n / nmax * pw

    def py(t):
        return oy - t / tmax * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'Match time of /(a*)*b/ vs input length -- linear, not exponential</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'a backtracking engine would blow up exponentially here; the NFA scales linearly</text>',
    ]
    # axes
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#30363d" stroke-width="1"/>')
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-ph}" stroke="#30363d" stroke-width="1"/>')
    # line + points
    pts = " ".join(f"{px(n):.1f},{py(t):.1f}" for n, t in zip(ns, ts))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="2"/>')
    for n, t in zip(ns, ts):
        parts.append(f'<circle cx="{px(n):.1f}" cy="{py(t):.1f}" r="4" fill="#ffd43b"/>')
        parts.append(f'<text x="{px(n):.0f}" y="{py(t)-10:.0f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{t:.1f}ms</text>')
    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+30}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">input length n (characters)</text>')
    parts.append(f'<text x="20" y="{oy-ph-2:.0f}" fill="#8b949e" font-size="11">match time (ms)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
