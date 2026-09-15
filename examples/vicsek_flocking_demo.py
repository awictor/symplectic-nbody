"""Vicsek demo: the order-disorder flocking transition in noise, with velocity-field snapshots."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import vicsek_flocking as vf


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    lines = []
    lines.append("Vicsek model -- self-propelled particles flocking, an order-disorder transition")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Each particle aligns to neighbours within radius r, plus angular noise eta. No leader.")
    lines.append("")

    n, box, r = 300, 7.0, 1.0
    rho = vf.density(n, box)
    lines.append(f"N={n}, box={box}, radius r={r}, density rho={rho:.2f}.")
    lines.append("")
    lines.append("Order parameter phi vs noise eta (phi=1 aligned flock, phi=0 disorder):")
    lines.append("   eta     phi        regime")
    lines.append("   " + "-" * 34)
    for eta in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.28):
        o = vf.simulate(n, box, r, eta, steps=250, seed=1)["order"]
        regime = "flock" if o > 0.6 else ("critical" if o > 0.3 else "disorder")
        lines.append(f"   {eta:4.2f}    {o:.3f}      {regime}")
    lines.append("")
    lines.append("Below the critical noise the flock spontaneously aligns; above it, motion is random.")
    lines.append("A symmetry-breaking phase transition -- like a magnet, but in a driven, moving system.")
    lines.append("")

    # density dependence
    lines.append("Critical noise rises with density (more neighbours to align with):")
    lines.append("   N     density   phi at eta=3.0")
    lines.append("   " + "-" * 34)
    for nn in (80, 160, 320, 640):
        o = vf.simulate(nn, box, r, eta=3.0, steps=200, seed=1)["order"]
        lines.append(f"   {nn:4d}   {vf.density(nn, box):.2f}      {o:.3f}")

    text = "\n".join(lines)
    print(text)

    svg = _svg(n, box, r)
    return text, svg


def _svg(n, box, r):
    W, H = 640, 400
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Flock (low noise, left) vs disorder (high noise, right)</text>')

    def draw_flock(eta, x0, label):
        res = vf.simulate(120, box, r, eta, steps=200, seed=2)
        pos, theta = res["pos"], res["theta"]
        side = 200
        scale = side / box
        arrow = 7
        for i in range(len(pos)):
            px = x0 + pos[i][0] * scale
            py = 55 + pos[i][1] * scale
            dx = arrow * math.cos(theta[i])
            dy = arrow * math.sin(theta[i])
            # colour by heading direction
            hue = (theta[i] % (2 * math.pi)) / (2 * math.pi)
            col = P["green"] if res["order"] > 0.5 else P["red"]
            parts.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px+dx:.1f}" y2="{py+dy:.1f}" '
                         f'stroke="{col}" stroke-width="1" opacity="0.8"/>')
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.2" fill="{col}"/>')
        parts.append(f'<rect x="{x0}" y="{55}" width="{side}" height="{side}" fill="none" '
                     f'stroke="{P["gray"]}" stroke-width="1"/>')
        parts.append(f'<text x="{x0}" y="{49}" fill="{P["text"]}" font-size="11">{label}</text>')
        parts.append(f'<text x="{x0}" y="{55 + side + 14:.1f}" fill="{P["gray"]}" '
                     f'font-size="10">phi = {res["order"]:.2f}</text>')

    draw_flock(0.4, 40, "low noise (eta=0.4)")
    draw_flock(5.5, 380, "high noise (eta=5.5)")

    parts.append(f'<text x="20" y="{H - 26}" fill="{P["gray"]}" font-size="11">'
                 f'each arrow is a particle\'s velocity. Left: aligned green flock (order ~1). Right: '
                 f'random red headings (order ~0).</text>')
    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'the same local rule flips between collective motion and disorder as noise crosses the '
                 f'critical value.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
