"""TASEP demo: the open-boundary phase diagram and per-site density profiles in each phase."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import tasep


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    lines = []
    lines.append("TASEP -- driven lattice gas with an exactly-solved nonequilibrium phase diagram")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Particles hop right at rate 1; injected left at rate alpha, ejected right at rate beta.")
    lines.append("")

    L = 120
    steps = 4_000_000
    lines.append("Steady-state density and current vs the exact formulas:")
    lines.append("   alpha  beta   phase            rho (sim/exact)   J (sim/exact)")
    lines.append("   " + "-" * 62)
    for alpha, beta in [(0.25, 0.8), (0.8, 0.25), (0.8, 0.8), (0.4, 0.4)]:
        res = tasep.simulate(L, alpha, beta, steps, seed=1)
        pd = tasep.predicted_density(alpha, beta)
        pj = tasep.predicted_current(alpha, beta)
        lines.append(f"   {alpha:4.2f}  {beta:4.2f}   {res['phase']:15s}  {res['density']:.3f}/{pd:.3f}      "
                     f"{res['current']:.3f}/{pj:.3f}")
    lines.append("")
    lines.append("Three phases: low-density (rho=alpha), high-density (rho=1-beta), maximal-current (rho=1/2).")
    lines.append("The maximal current J=1/4 is a hard ceiling on single-file transport.")
    lines.append("")

    # phase diagram grid
    lines.append("Phase map over (alpha, beta) -- L=low-density, H=high-density, M=maximal-current:")
    lines.append("        beta ->")
    for a10 in range(9, 0, -1):
        alpha = a10 / 10
        row = f"   a={alpha:.1f}  "
        for b10 in range(1, 10):
            beta = b10 / 10
            p = tasep.phase(alpha, beta)
            row += {"low-density": "L", "high-density": "H", "maximal-current": "M"}[p]
        lines.append(row)
    lines.append("          " + "".join(str(b) for b in range(1, 10)))

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
                 f'TASEP phase diagram (left) and density profiles (right)</text>')

    # LEFT: phase diagram colored
    px0, py0, side = 50, 60, 250
    ncell = 50
    cw = side / ncell
    col_map = {"low-density": P["blue"], "high-density": P["red"], "maximal-current": P["green"]}
    for i in range(ncell):
        for j in range(ncell):
            alpha = (i + 0.5) / ncell
            beta = (j + 0.5) / ncell
            p = tasep.phase(alpha, beta)
            x = px0 + i * cw
            y = py0 + (ncell - 1 - j) * cw
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw+0.5:.1f}" height="{cw+0.5:.1f}" '
                         f'fill="{col_map[p]}" opacity="0.8"/>')
    parts.append(f'<rect x="{px0}" y="{py0}" width="{side}" height="{side}" fill="none" '
                 f'stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{px0}" y="{py0 + side + 16:.1f}" fill="{P["gray"]}" font-size="10">alpha -></text>')
    parts.append(f'<text x="{px0 - 30:.1f}" y="{py0 + 10:.1f}" fill="{P["gray"]}" font-size="10">beta</text>')
    parts.append(f'<text x="{px0 + 30:.1f}" y="{py0 + side - 30:.1f}" fill="{P["text"]}" font-size="11">LD</text>')
    parts.append(f'<text x="{px0 + side - 60:.1f}" y="{py0 + 40:.1f}" fill="{P["text"]}" font-size="11">HD</text>')
    parts.append(f'<text x="{px0 + side - 60:.1f}" y="{py0 + side - 30:.1f}" fill="{P["text"]}" font-size="11">MC</text>')

    # RIGHT: density profiles for the three phases
    rx0, rx1, ry0, ry1 = 360, 615, 60, 310
    L = 80
    cases = [(0.25, 0.8, P["blue"], "LD rho=0.25"), (0.8, 0.25, P["red"], "HD rho=0.75"),
             (0.8, 0.8, P["green"], "MC rho=0.5")]

    def rx(pos):
        return rx0 + pos / (L - 1) * (rx1 - rx0)

    def ry(d):
        return ry1 - d * (ry1 - ry0)

    parts.append(f'<line x1="{rx0}" y1="{ry(0.5):.1f}" x2="{rx1}" y2="{ry(0.5):.1f}" '
                 f'stroke="{P["gray"]}" stroke-width="0.5" stroke-dasharray="2,3"/>')
    for alpha, beta, col, label in cases:
        prof = tasep.steady_state_profile(L, alpha, beta, 1_500_000, seed=1)
        pts = " ".join(f"{rx(i):.1f},{ry(prof[i]):.1f}" for i in range(L))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.5"/>')
    parts.append(f'<rect x="{rx0}" y="{ry0}" width="{rx1-rx0}" height="{ry1-ry0}" fill="none" '
                 f'stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{rx0}" y="{ry0 - 4:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'density vs site (bulk flat, boundary layers at the ends)</text>')
    ly = ry1 + 16
    for alpha, beta, col, label in cases:
        parts.append(f'<text x="{rx0}" y="{ly:.1f}" fill="{col}" font-size="10">{label}</text>')
        ly += 14

    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'the (alpha,beta) plane splits into three exactly-solved phases; density profiles are '
                 f'flat in the bulk, bent at the driven boundaries.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
