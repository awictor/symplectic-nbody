"""Deffuant demo: opinions merging into consensus or fragmenting into camps as the confidence threshold varies."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import deffuant_bounded as db


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    lines = []
    lines.append("Deffuant bounded-confidence -- when opinions merge, polarize, or fragment")
    lines.append("=" * 74)
    lines.append("")
    lines.append("Random pairs converge only if their opinions differ by less than a threshold d.")
    lines.append("")

    lines.append("Confidence threshold d decides consensus vs fragmentation:")
    lines.append("   d       clusters   opinions (cluster means)")
    lines.append("   " + "-" * 50)
    for d in (0.5, 0.3, 0.2, 0.15, 0.1):
        res = db.simulate(500, d=d, seed=2)
        cs = ", ".join(f"{c:.2f}" for c in res["clusters"])
        lines.append(f"   {d:4.2f}    {res['n_clusters']:4d}       {cs}")
    lines.append("")
    lines.append("Open-minded society (large d) -> one consensus; echo chambers (small d) -> many camps.")
    lines.append("The number of surviving opinion clusters is roughly 1/(2d).")
    lines.append("")

    # mean conservation
    res = db.simulate(500, d=0.3, seed=1)
    lines.append(f"The population mean opinion is conserved by every symmetric exchange:")
    lines.append(f"  final mean = {res['mean']:.4f} (uniform start mean 0.5), regardless of the outcome.")

    text = "\n".join(lines)
    print(text)

    svg = _svg()
    return text, svg


def _svg():
    W, H = 640, 440
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Opinion trajectories: consensus (left) vs fragmentation (right)</text>')

    # simulate with periodic snapshots to draw trajectories
    def run_traj(d, n=60, snaps=40, seed=3):
        rng = db._Rng(seed)
        ops = [rng.u() for _ in range(n)]
        history = [list(ops)]
        interactions_per_snap = 300
        for _ in range(snaps):
            for _ in range(interactions_per_snap):
                i = rng.randint(0, n)
                j = rng.randint(0, n - 1)
                if j >= i:
                    j += 1
                db.interact(ops, i, j, d, 0.5)
            history.append(list(ops))
        return history

    def draw_panel(d, x0, label):
        hist = run_traj(d)
        n = len(hist[0])
        snaps = len(hist)
        side_w, top, bot = 250, 60, 360

        def px(s):
            return x0 + s / (snaps - 1) * side_w

        def py(op):
            return bot - op * (bot - top)

        for agent in range(n):
            pts = " ".join(f"{px(s):.1f},{py(hist[s][agent]):.1f}" for s in range(snaps))
            # colour by final opinion band
            final = hist[-1][agent]
            col = P["blue"] if final < 0.4 else (P["green"] if final < 0.6 else P["red"])
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="0.7" opacity="0.6"/>')
        parts.append(f'<rect x="{x0}" y="{top}" width="{side_w}" height="{bot-top}" fill="none" '
                     f'stroke="{P["gray"]}" stroke-width="1"/>')
        parts.append(f'<text x="{x0}" y="{top - 4}" fill="{P["text"]}" font-size="11">{label}</text>')

    draw_panel(0.4, 40, "d=0.4 (open-minded)")
    draw_panel(0.12, 350, "d=0.12 (echo chambers)")

    parts.append(f'<text x="20" y="{H - 26}" fill="{P["gray"]}" font-size="11">'
                 f'each line is one person\'s opinion (y, in [0,1]) over time (x). Left: all lines merge '
                 f'to one consensus.</text>')
    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'Right: lines split into several stable camps that never reconcile -- fragmentation '
                 f'from a narrow confidence window.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
