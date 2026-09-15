"""Simpson's paradox demo: the kidney-stone reversal and how confounder adjustment restores the truth."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import simpsons_paradox as sp


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}

# Kidney-stone study (Charig 1986): A = open surgery, B = less-invasive procedure.
KIDNEY = [
    ("small stones", 81, 87, 234, 270),
    ("large stones", 192, 263, 55, 80),
]


def main():
    strata = [(a, at, b, bt) for (_, a, at, b, bt) in KIDNEY]

    lines = []
    lines.append("Simpson's paradox -- kidney-stone treatments (Charig 1986)")
    lines.append("=" * 74)
    lines.append("")
    lines.append("Treatment A (open surgery) vs B (less-invasive), success rates:")
    lines.append("")
    lines.append("   stratum         treatment A          treatment B         winner")
    lines.append("   " + "-" * 62)
    srates = sp.stratum_rates(strata)
    for (name, a, at, b, bt), (ra, rb) in zip(KIDNEY, srates):
        win = "A" if ra > rb else ("B" if rb > ra else "tie")
        lines.append(f"   {name:14s}  {a:3d}/{at:3d} = {ra*100:5.1f}%   {b:3d}/{bt:3d} = {rb*100:5.1f}%     {win}")
    pr_a, pr_b = sp.pooled_rates(strata)
    pa_s = sum(s[0] for s in strata); pa_t = sum(s[1] for s in strata)
    pb_s = sum(s[2] for s in strata); pb_t = sum(s[3] for s in strata)
    lines.append("   " + "-" * 62)
    win = "A" if pr_a > pr_b else "B"
    lines.append(f"   {'POOLED':14s}  {pa_s:3d}/{pa_t:3d} = {pr_a*100:5.1f}%   {pb_s:3d}/{pb_t:3d} = {pr_b*100:5.1f}%     {win}")
    lines.append("")
    lines.append(f"  A wins BOTH strata, yet B wins the pooled total. Reversal detected: {sp.is_reversal(strata)}.")
    lines.append("")

    # confounder
    alloc = sp.allocation(strata)
    lines.append("Why? The confounder (stone size) is allocated unevenly:")
    for (name, *_), (fa, fb) in zip(KIDNEY, alloc):
        lines.append(f"   {name:14s}  A sends {fa*100:5.1f}% of its cases here, B sends {fb*100:5.1f}%")
    lines.append("  A got the hard cases (large stones, worse for everyone), dragging its pooled rate down.")
    lines.append("")

    # adjustment
    p_or = sp.pooled_odds_ratio(strata)
    mh = sp.mantel_haenszel_or(strata)
    sa, sb = sp.standardized_rates(strata)
    lines.append("Which answer is real? The confounder-adjusted comparison:")
    lines.append(f"   naive pooled odds ratio  (A vs B) = {p_or:.3f}   (< 1: wrongly favours B)")
    lines.append(f"   Mantel-Haenszel odds ratio        = {mh:.3f}   (> 1: correctly favours A)")
    lines.append(f"   standardized rates:  A = {sa*100:.1f}%,  B = {sb*100:.1f}%   (A ahead, matching every stratum)")
    lines.append("")
    lines.append("  Holding the confounder fixed recovers the within-group truth: treatment A is better.")

    text = "\n".join(lines)
    print(text)

    svg = _svg(strata, srates, pr_a, pr_b, sa, sb)
    return text, svg


def _svg(strata, srates, pr_a, pr_b, sa, sb):
    W, H = 640, 440
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Simpson\'s paradox: success rate of treatment A (blue) vs B (yellow)</text>')

    # bar chart: 4 category groups -> small, large, POOLED, ADJUSTED
    groups = [
        ("small\nstones", srates[0][0], srates[0][1]),
        ("large\nstones", srates[1][0], srates[1][1]),
        ("POOLED", pr_a, pr_b),
        ("ADJUSTED\n(MH std.)", sa, sb),
    ]
    base_y = 360
    top_y = 70
    plot_h = base_y - top_y
    n = len(groups)
    group_w = (W - 100) / n
    bar_w = group_w * 0.32

    # y-axis gridlines 0..100%
    for pct in (0, 25, 50, 75, 100):
        y = base_y - pct / 100 * plot_h
        parts.append(f'<line x1="60" y1="{y:.1f}" x2="{W-20}" y2="{y:.1f}" '
                     f'stroke="{P["gray"]}" stroke-width="0.4" opacity="0.4"/>')
        parts.append(f'<text x="52" y="{y+4:.1f}" fill="{P["gray"]}" font-size="10" text-anchor="end">{pct}</text>')

    for gi, (label, ra, rb) in enumerate(groups):
        cx = 70 + gi * group_w + group_w / 2
        for k, (rate, col) in enumerate(((ra, P["blue"]), (rb, P["yellow"]))):
            bx = cx - bar_w + k * bar_w
            bh = rate * plot_h
            by = base_y - bh
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w-2:.1f}" height="{bh:.1f}" fill="{col}"/>')
            parts.append(f'<text x="{bx+bar_w/2-1:.1f}" y="{by-3:.1f}" fill="{P["text"]}" '
                         f'font-size="9" text-anchor="middle">{rate*100:.0f}</text>')
        # highlight the two decision groups
        if label in ("POOLED",):
            parts.append(f'<text x="{cx:.1f}" y="{base_y+30:.1f}" fill="{P["red"]}" font-size="10" '
                         f'text-anchor="middle">B wins (wrong)</text>')
        if label.startswith("ADJUSTED"):
            parts.append(f'<text x="{cx:.1f}" y="{base_y+30:.1f}" fill="{P["green"]}" font-size="10" '
                         f'text-anchor="middle">A wins (right)</text>')
        for li, ln in enumerate(label.split("\n")):
            parts.append(f'<text x="{cx:.1f}" y="{base_y+14+li*12:.1f}" fill="{P["text"]}" '
                         f'font-size="10" text-anchor="middle">{ln}</text>')

    parts.append(f'<text x="20" y="{H-26}" fill="{P["gray"]}" font-size="11">'
                 f'A beats B in both stone-size strata, yet loses the pooled bar -- A was given more of the</text>')
    parts.append(f'<text x="20" y="{H-10}" fill="{P["gray"]}" font-size="11">'
                 f'hard (large-stone) cases. Adjusting for that confounder flips the pooled verdict back to A.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
