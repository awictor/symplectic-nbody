"""Demo: regret matching / CFR -- learning Nash equilibria of zero-sum games from self-play.

Solves Rock-Paper-Scissors and a skewed variant by self-play, prints the converged strategies and
game value, and plots exploitability falling toward zero over iterations.

    python examples/regret_matching_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from regret_matching import solve, brute_game_value  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Regret matching (flat CFR): Nash equilibria from self-play, no LP solver\n")

    rps = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    labels = ["Rock", "Paper", "Scissors"]
    res = solve(rps, iterations=20000)
    print("  Rock-Paper-Scissors:")
    for i, lab in enumerate(labels):
        print(f"    {lab:>8}: row {res['row'][i]:.3f}   col {res['col'][i]:.3f}")
    print(f"    game value {res['value']:+.4f}  exploitability {res['exploitability']:.5f}")
    print("    -> converges to uniform 1/3 each, value 0 (the known equilibrium).\n")

    # A skewed RPS where winning with Rock pays double.
    skew = [[0, -1, 2], [1, 0, -1], [-2, 1, 0]]
    res2 = solve(skew, iterations=40000)
    print("  Skewed RPS (a Rock win pays +2):")
    for i, lab in enumerate(labels):
        print(f"    {lab:>8}: row {res2['row'][i]:.3f}")
    print(f"    game value {res2['value']:+.4f}  (brute minimax "
          f"{brute_game_value(skew, grid=60):+.4f})")
    print("    -> equilibrium shifts AWAY from Rock, since the opponent avoids the costly matchup.\n")

    # exploitability curve on a slower-converging 4x4 game
    big = [[2, -1, 0, 3], [-1, 3, 1, -2], [0, 1, -2, 2], [3, -2, 2, -1]]
    iters = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
    curve = [(t, solve(big, iterations=t)["exploitability"]) for t in iters]
    print("  Exploitability vs iterations (4x4 game):")
    for t, e in curve:
        bar = "#" * int(e / curve[0][1] * 40) if curve[0][1] > 0 else ""
        print(f"    {t:>6}: {e:.5f} {bar}")
    print("    -> falls like O(1/sqrt(T)); average strategy approaches Nash.")

    _svg(os.path.join(outdir, "regret_matching.svg"), res["row"], labels, curve)
    print(f"\n  wrote {os.path.join(outdir, 'regret_matching.svg')}")


def _svg(path, rps_row, labels, curve, width=760, height=430):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="28" fill="#e6edf3" font-size="15">'
        'Left: converged RPS strategy. Right: exploitability -&gt; 0 (log-log)</text>',
    ]

    # ---- left: bar chart of the RPS strategy --------------------------------------------
    cols = ["#4dabf7", "#06d6a0", "#ffd43b"]
    bx, by, bw, bh = 60, 90, 70, 220
    parts.append(f'<line x1="{bx-10}" y1="{by+bh}" x2="{bx+3*bw+20}" y2="{by+bh}" '
                 f'stroke="#8b949e" stroke-width="1"/>')
    # 1/3 reference line
    yref = by + bh - bh * (1 / 3) / 0.5
    parts.append(f'<line x1="{bx-10}" y1="{yref:.0f}" x2="{bx+3*bw+20}" y2="{yref:.0f}" '
                 f'stroke="#ff6b6b" stroke-width="1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{bx+3*bw+24}" y="{yref+3:.0f}" fill="#ff6b6b" font-size="9">1/3</text>')
    for i, p in enumerate(rps_row):
        h = bh * p / 0.5
        x = bx + i * bw
        parts.append(f'<rect x="{x}" y="{by+bh-h:.0f}" width="{bw-14}" height="{h:.0f}" '
                     f'fill="{cols[i]}"/>')
        parts.append(f'<text x="{x+(bw-14)/2:.0f}" y="{by+bh+16:.0f}" fill="{cols[i]}" '
                     f'font-size="10" text-anchor="middle">{labels[i][:4]}</text>')
        parts.append(f'<text x="{x+(bw-14)/2:.0f}" y="{by+bh-h-5:.0f}" fill="#e6edf3" '
                     f'font-size="9" text-anchor="middle">{p:.2f}</text>')

    # ---- right: exploitability log-log curve --------------------------------------------
    ox, oy, ow, oh = 430, 90, 280, 220
    ts = [t for t, _ in curve]
    es = [max(e, 1e-6) for _, e in curve]
    lx = [math.log10(t) for t in ts]
    ly = [math.log10(e) for e in es]
    lxmin, lxmax = min(lx), max(lx)
    lymin, lymax = min(ly), max(ly)

    def px(v):
        return ox + ow * (v - lxmin) / (lxmax - lxmin or 1)

    def py(v):
        return oy + oh * (1 - (v - lymin) / (lymax - lymin or 1))

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" '
                 f'stroke="#30363d" stroke-width="1"/>')
    pts = " ".join(f"{px(lx[i]):.0f},{py(ly[i]):.0f}" for i in range(len(curve)))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    for i in range(len(curve)):
        parts.append(f'<circle cx="{px(lx[i]):.0f}" cy="{py(ly[i]):.0f}" r="3" fill="#ffd43b"/>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+22:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">log10 iterations</text>')
    parts.append(f'<text x="{ox-8}" y="{oy-6}" fill="#8b949e" font-size="10" '
                 f'text-anchor="end">log10 exploitability</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
