"""Hopfield demo: recall a corrupted memory by energy descent, and watch capacity collapse past 0.138 N."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import hopfield as hf


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}

# three 5x5 bitmap letters, drawn as +1 (on) / -1 (off)
GLYPHS = {
    "T": ["#####",
          "..#..",
          "..#..",
          "..#..",
          "..#.."],
    "L": ["#....",
          "#....",
          "#....",
          "#....",
          "#####"],
    "X": ["#...#",
          ".#.#.",
          "..#..",
          ".#.#.",
          "#...#"],
}


def glyph_to_pattern(rows):
    return [1 if c == "#" else -1 for r in rows for c in r]


def pattern_to_rows(p, w=5):
    out = []
    for i in range(0, len(p), w):
        out.append("".join("#" if v > 0 else "." for v in p[i:i + w]))
    return out


def main():
    lines = []
    lines.append("Hopfield network -- content-addressable memory by energy descent")
    lines.append("=" * 74)
    lines.append("")

    names = list(GLYPHS.keys())
    patterns = [glyph_to_pattern(GLYPHS[k]) for k in names]
    n = len(patterns[0])
    w = hf.hebbian_weights(patterns)
    lines.append(f"Stored {len(patterns)} letter patterns in an N={n} neuron net (5x5 bitmaps).")
    lines.append(f"Each stored pattern is a fixed point: "
                 f"{all(hf.is_fixed_point(w, p) for p in patterns)}. Energies:")
    for k, p in zip(names, patterns):
        lines.append(f"   '{k}'  E = {hf.energy(w, p):8.3f}")
    lines.append("")

    # corrupt 'T' and recall
    target = patterns[0]
    cue = hf.flip_bits(target, 3, seed=3)  # 3 of 25 pixels wrong
    rng = hf._Rng(2024)
    out = hf.recall(w, cue, rng, sweeps=30)
    lines.append("Show it a corrupted 'T' (3 of 25 pixels flipped) and let it settle:")
    lines.append("")
    cue_rows = pattern_to_rows(cue)
    out_rows = pattern_to_rows(out)
    tgt_rows = pattern_to_rows(target)
    lines.append("    corrupted cue     ->     recalled      (stored 'T')")
    for i in range(5):
        lines.append(f"      {cue_rows[i]}          {out_rows[i]}         {tgt_rows[i]}")
    lines.append("")
    lines.append(f"  overlap with stored 'T' = {hf.overlap(out, target):+.3f}  "
                 f"(1.0 = perfect recall).")
    lines.append("")

    # capacity curve
    lines.append("Capacity: recall accuracy vs load p/N (N=100, 10 flipped bits per cue):")
    lines.append("   load p/N     patterns     mean recall overlap")
    lines.append("   " + "-" * 46)
    N = 100
    for ratio in (0.02, 0.05, 0.10, 0.14, 0.20, 0.30):
        p = max(1, int(ratio * N))
        acc = hf.recall_accuracy(p, N, flips=10, seed=7)
        bar = "#" * int(acc * 30)
        lines.append(f"   {ratio:5.2f}        {p:4d}         {acc:.3f}  {bar}")
    lines.append("")
    lines.append(f"Recall stays near-perfect until the Amit-Gutfreund-Sompolinsky load "
                 f"p/N ~ {hf.CAPACITY_RATIO}, then collapses as spurious minima take over.")

    text = "\n".join(lines)
    print(text)

    svg = _svg(cue, out, target)
    return text, svg


def _svg(cue, out, target):
    W, H = 640, 440
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Associative recall: corrupted cue rolls downhill to the stored memory</text>')

    def draw_grid(state, x0, y0, cell, on_col, title):
        parts.append(f'<text x="{x0 + 2.5*cell:.0f}" y="{y0-8:.0f}" fill="{P["text"]}" '
                     f'font-size="12" text-anchor="middle">{title}</text>')
        for idx, v in enumerate(state):
            r, c = divmod(idx, 5)
            x = x0 + c * cell
            y = y0 + r * cell
            col = on_col if v > 0 else "#161b22"
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell-2}" height="{cell-2}" '
                         f'fill="{col}" stroke="{P["gray"]}" stroke-width="0.4"/>')

    cell = 34
    draw_grid(cue, 40, 80, cell, P["red"], "corrupted cue")
    draw_grid(out, 260, 80, cell, P["green"], "recalled")
    draw_grid(target, 470, 80, cell, P["blue"], "stored 'T'")

    # arrows between grids
    parts.append(f'<text x="228" y="{80+2.5*cell:.0f}" fill="{P["yellow"]}" font-size="24" '
                 f'text-anchor="middle">&#8594;</text>')
    parts.append(f'<text x="448" y="{80+2.5*cell:.0f}" fill="{P["gray"]}" font-size="16" '
                 f'text-anchor="middle">=</text>')

    # small capacity curve at the bottom
    N = 100
    ratios = [0.02, 0.05, 0.10, 0.14, 0.20, 0.30, 0.40]
    accs = [hf.recall_accuracy(max(1, int(rt * N)), N, flips=10, seed=7) for rt in ratios]
    gx0, gx1 = 80, W - 60
    gy0, gy1 = 300, 400
    def px(rt): return gx0 + rt / 0.40 * (gx1 - gx0)
    def py(a): return gy1 - a * (gy1 - gy0)
    parts.append(f'<line x1="{gx0}" y1="{gy1}" x2="{gx1}" y2="{gy1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<line x1="{gx0}" y1="{gy0}" x2="{gx0}" y2="{gy1}" stroke="{P["gray"]}" stroke-width="1"/>')
    # capacity line at 0.138
    xc = px(hf.CAPACITY_RATIO)
    parts.append(f'<line x1="{xc:.1f}" y1="{gy0}" x2="{xc:.1f}" y2="{gy1}" stroke="{P["yellow"]}" '
                 f'stroke-width="1" stroke-dasharray="4,3"/>')
    parts.append(f'<text x="{xc+4:.1f}" y="{gy0+12:.0f}" fill="{P["yellow"]}" font-size="10">p/N ~ 0.138</text>')
    pts = " ".join(f"{px(rt):.1f},{py(a):.1f}" for rt, a in zip(ratios, accs))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{P["purple"]}" stroke-width="2"/>')
    for rt, a in zip(ratios, accs):
        parts.append(f'<circle cx="{px(rt):.1f}" cy="{py(a):.1f}" r="3" fill="{P["purple"]}"/>')
    parts.append(f'<text x="{gx0}" y="{gy1+16:.0f}" fill="{P["gray"]}" font-size="10">load p/N</text>')
    parts.append(f'<text x="{gx1}" y="{gy1+16:.0f}" fill="{P["gray"]}" font-size="10" text-anchor="end">0.40</text>')
    parts.append(f'<text x="{gx0-6}" y="{py(1.0)+4:.0f}" fill="{P["gray"]}" font-size="9" text-anchor="end">1</text>')

    parts.append(f'<text x="20" y="{H-10}" fill="{P["gray"]}" font-size="11">'
                 f'Recall overlap (purple) holds near 1 until the load crosses ~0.138 N, then collapses.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
