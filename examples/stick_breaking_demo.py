"""Stick-breaking demo: a unit stick snapped into Dirichlet-process weights, and the concentration effect."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import stick_breaking as sb


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}
COLORS = ["#4dabf7", "#ffd43b", "#06d6a0", "#b197fc", "#ff6b6b", "#f783ac", "#63e6be", "#ffa94d",
          "#4dabf7", "#ffd43b", "#06d6a0", "#b197fc"]


def main():
    lines = []
    lines.append("Stick-breaking (GEM) -- explicit Dirichlet-process weights off a unit stick")
    lines.append("=" * 74)
    lines.append("")
    lines.append("Break Beta(1,alpha) fractions off a length-1 stick: pi_k = beta_k prod(1-beta_j).")
    lines.append("")

    lines.append("Expected weights decay geometrically, E[pi_k] = (1/(1+a))(a/(1+a))^(k-1):")
    lines.append("   k     alpha=1     alpha=5")
    lines.append("   " + "-" * 30)
    for k in range(1, 8):
        lines.append(f"   {k}     {sb.expected_weight(1.0, k):.4f}      {sb.expected_weight(5.0, k):.4f}")
    lines.append("")
    lines.append("Small alpha -> mass in a few big pieces; large alpha -> many thin, near-equal pieces.")
    lines.append("")

    lines.append("Weights needed to capture 95% of the mass (mean over 500 draws):")
    lines.append("   alpha    #weights for 95%")
    lines.append("   " + "-" * 30)
    for alpha in (0.3, 1.0, 3.0, 10.0):
        ks = [sb.weights_for_mass(alpha, 0.95, seed=(r + 1) * 7919) for r in range(500)]
        lines.append(f"   {alpha:5.1f}    {sum(ks)/len(ks):.1f}")
    lines.append("")

    # one realization
    w = sb.weights(1.0, 10, seed=7)
    lines.append(f"One realization (alpha=1, first 10 weights, sum={sum(w):.3f}):")
    lines.append("  " + "  ".join(f"{p:.3f}" for p in w))
    lines.append(f"  first weight took {w[0]/sum(w)*100:.0f}% of the (truncated) mass -- rich-get-richer.")

    text = "\n".join(lines)
    print(text)

    svg = _svg()
    return text, svg


def _svg():
    W, H = 640, 400
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Sticks broken for three concentrations alpha (each row is a unit stick)</text>')

    x0, x1 = 40, 610
    stick_w = x1 - x0
    rows = [(0.3, 70), (1.0, 160), (5.0, 250)]
    for alpha, y in rows:
        w = sb.weights(alpha, 40, seed=3)
        total = sum(w)
        # draw the pieces along the stick
        cx = x0
        for k, pi in enumerate(w):
            seg = pi / total * stick_w
            col = COLORS[k % len(COLORS)]
            parts.append(f'<rect x="{cx:.1f}" y="{y}" width="{max(seg, 0.3):.1f}" height="36" '
                         f'fill="{col}" opacity="0.85"/>')
            cx += seg
        parts.append(f'<rect x="{x0}" y="{y}" width="{stick_w}" height="36" fill="none" '
                     f'stroke="{P["gray"]}" stroke-width="1"/>')
        parts.append(f'<text x="{x0}" y="{y - 4}" fill="{P["text"]}" font-size="11">'
                     f'alpha = {alpha}</text>')

    parts.append(f'<text x="20" y="{H - 26}" fill="{P["gray"]}" font-size="11">'
                 f'alpha=0.3: one dominant piece grabs the stick. alpha=5: the stick shatters into '
                 f'many small pieces.</text>')
    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'these weights, with random atoms attached, ARE a draw from a Dirichlet process.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
