"""Lloyd-Max demo: optimal quantizer levels crowding where the probability mass is, beating uniform in SNR."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import lloyd_max as lm


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def gauss(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def main():
    lines = []
    lines.append("Lloyd-Max quantizer -- minimum-distortion levels for a known source")
    lines.append("=" * 68)
    lines.append("")

    lo, hi = -5.0, 5.0
    lines.append("Gaussian source N(0,1). Lloyd-Max places levels where the mass is;")
    lines.append("a uniform quantizer spaces them evenly and wastes levels in the tails.")
    lines.append("")

    lines.append("SNR vs number of levels (Lloyd-Max minus uniform):")
    lines.append("   levels   uniform SNR   Lloyd-Max SNR   gain")
    lines.append("   " + "-" * 46)
    for n in (2, 4, 8, 16, 32):
        lmq = lm.design(gauss, n, lo, hi)
        uq = lm.uniform_quantizer(n, lo, hi)
        s_lm = lm.snr_db(gauss, lmq, lo, hi)
        s_u = lm.snr_db(gauss, uq, lo, hi)
        lines.append(f"   {n:4d}     {s_u:8.2f}     {s_lm:10.2f}     {s_lm - s_u:+.2f} dB")
    lines.append("")

    # show the level positions for N=8
    n = 8
    lmq = lm.design(gauss, n, lo, hi)
    lines.append(f"Lloyd-Max levels (N={n}), note the crowding near 0:")
    lines.append("  " + "  ".join(f"{L:+.2f}" for L in lmq["levels"]))
    gaps = [lmq["levels"][k + 1] - lmq["levels"][k] for k in range(n - 1)]
    lines.append(f"  gaps: {'  '.join(f'{g:.2f}' for g in gaps)}")
    lines.append(f"  (smallest gap {min(gaps):.2f} near center, largest {max(gaps):.2f} in the tails)")
    lines.append("")
    lines.append(f"Converged in {lmq['iterations']} iterations, distortion {lmq['distortion']:.4f}.")

    text = "\n".join(lines)
    print(text)

    svg = _svg(lmq, lm.uniform_quantizer(n, lo, hi), lo, hi)
    return text, svg


def _svg(lmq, uq, lo, hi):
    W, H = 640, 430
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'Lloyd-Max vs uniform quantizer levels on a Gaussian source</text>')

    x0, x1 = 50, 610
    # PDF curve panel
    py0, py1 = 60, 250

    def px(x):
        return x0 + (x - lo) / (hi - lo) * (x1 - x0)

    pmax = gauss(0.0)

    def py(v):
        return py1 - (v / pmax) * (py1 - py0)

    # gaussian curve
    N = 300
    pts = " ".join(f"{px(lo + (hi - lo) * i / N):.1f},{py(gauss(lo + (hi - lo) * i / N)):.1f}"
                   for i in range(N + 1))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{P["blue"]}" stroke-width="2"/>')
    parts.append(f'<line x1="{x0}" y1="{py1}" x2="{x1}" y2="{py1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{x0}" y="{py0 - 6}" fill="{P["gray"]}" font-size="11">'
                 f'N(0,1) density -- levels should crowd under the peak</text>')

    # Lloyd-Max levels as yellow verticals dropping from the curve
    for L in lmq["levels"]:
        parts.append(f'<line x1="{px(L):.1f}" y1="{py(gauss(L)):.1f}" x2="{px(L):.1f}" y2="{py1}" '
                     f'stroke="{P["yellow"]}" stroke-width="1.5"/>')
        parts.append(f'<circle cx="{px(L):.1f}" cy="{py(gauss(L)):.1f}" r="3" fill="{P["yellow"]}"/>')

    # Uniform levels as red ticks below the axis
    uy = py1 + 20
    parts.append(f'<text x="{x0}" y="{uy - 4}" fill="{P["red"]}" font-size="10">uniform levels (evenly spaced):</text>')
    for L in uq["levels"]:
        parts.append(f'<line x1="{px(L):.1f}" y1="{uy}" x2="{px(L):.1f}" y2="{uy + 12}" '
                     f'stroke="{P["red"]}" stroke-width="1.5"/>')
    ly = uy + 40
    parts.append(f'<text x="{x0}" y="{ly - 4}" fill="{P["yellow"]}" font-size="10">Lloyd-Max levels (crowd near 0):</text>')
    for L in lmq["levels"]:
        parts.append(f'<line x1="{px(L):.1f}" y1="{ly}" x2="{px(L):.1f}" y2="{ly + 12}" '
                     f'stroke="{P["yellow"]}" stroke-width="1.5"/>')

    # x ticks
    for xx in (-4, -2, 0, 2, 4):
        parts.append(f'<text x="{px(xx):.1f}" y="{py1 + 12:.1f}" fill="{P["gray"]}" '
                     f'font-size="9" text-anchor="middle">{xx}</text>')

    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'Lloyd-Max packs levels where samples are common (the peak) and spreads them in the '
                 f'rare tails -- lower error per bit.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
