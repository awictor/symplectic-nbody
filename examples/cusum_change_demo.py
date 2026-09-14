"""CUSUM demo: catch a small mean shift a control chart misses, with the accumulating sums crossing threshold."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import cusum_change as cc


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF
        self._spare = None

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self):
        if self._spare is not None:
            v = self._spare
            self._spare = None
            return v
        u1 = max(self.u(), 1e-12)
        u2 = self.u()
        r = math.sqrt(-2 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2)


def main():
    lines = []
    lines.append("CUSUM change detection -- catch a small sustained mean shift")
    lines.append("=" * 60)
    lines.append("")

    rng = _R(2)
    change_at = 120
    shift = 1.0    # a SMALL shift (1 sigma) -- invisible to a per-point control chart
    stream = [rng.normal() for _ in range(change_at)] + \
             [rng.normal() + shift for _ in range(120)]
    res = cc.cusum(stream, target=0.0, sigma=1.0, k=0.5, h=5.0)

    lines.append(f"Stream of {len(stream)} values, mean shifts by {shift} sigma at t={change_at}.")
    lines.append(f"A single value rarely leaves +/-3 sigma, so a Shewhart chart would miss it.")
    lines.append("")
    lines.append(f"CUSUM alarm at t = {res['alarm_index']} (side: {res['alarm_side']})")
    lines.append(f"  detection delay = {res['alarm_index'] - change_at} steps after the true change")
    lines.append(f"  decision interval H = {res['H']:.1f}")
    lines.append("")

    # how many raw points exceeded 3 sigma (Shewhart) before the CUSUM alarm?
    shewhart_hits = sum(1 for i in range(res["alarm_index"] + 1) if abs(stream[i]) > 3)
    lines.append(f"Shewhart 3-sigma rule: only {shewhart_hits} point(s) flagged before CUSUM alarmed.")
    lines.append("")

    # detection delay vs shift size
    lines.append("Detection delay vs shift size (sigma):")
    lines.append("   shift   delay (steps)")
    lines.append("   " + "-" * 24)
    for sh in (0.5, 1.0, 1.5, 2.0, 3.0):
        rr = _R(7)
        s = [rr.normal() for _ in range(100)] + [rr.normal() + sh for _ in range(300)]
        r = cc.cusum(s, 0.0, 1.0, k=0.5, h=5.0)
        delay = (r["alarm_index"] - 100) if r["alarm_index"] and r["alarm_index"] >= 100 else None
        lines.append(f"   {sh:5.1f}   {delay if delay is not None else 'no alarm'}")
    lines.append("")
    lines.append("Bigger shifts trip the accumulating sum faster -- the sequential-detection tradeoff.")

    text = "\n".join(lines)
    print(text)

    svg = _svg(stream, res, change_at)
    return text, svg


def _svg(stream, res, change_at):
    W, H = 640, 460
    P = PALETTE
    n = len(stream)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Raw stream (top) and accumulating CUSUM sums (bottom)</text>')

    x0, x1 = 50, 615

    def px(i):
        return x0 + i / (n - 1) * (x1 - x0)

    # TOP: raw stream
    ty0, ty1 = 50, 180
    vmin, vmax = min(stream), max(stream)

    def ty(v):
        return ty1 - (v - vmin) / (vmax - vmin) * (ty1 - ty0)

    # change-point marker
    parts.append(f'<line x1="{px(change_at):.1f}" y1="{ty0}" x2="{px(change_at):.1f}" y2="450" '
                 f'stroke="{P["gray"]}" stroke-width="1" stroke-dasharray="4,3"/>')
    parts.append(f'<text x="{px(change_at) + 4:.1f}" y="{ty0 + 10:.1f}" fill="{P["gray"]}" '
                 f'font-size="10">true change</text>')
    # zero line
    parts.append(f'<line x1="{x0}" y1="{ty(0):.1f}" x2="{x1}" y2="{ty(0):.1f}" stroke="#30363d" stroke-width="1"/>')
    pts = " ".join(f"{px(i):.1f},{ty(stream[i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{P["blue"]}" stroke-width="1" opacity="0.8"/>')
    parts.append(f'<text x="{x0}" y="{ty0 - 4:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'raw values -- the shift is hard to see by eye</text>')

    # BOTTOM: CUSUM sums
    by0, by1 = 230, 400
    smax = max(max(res["s_hi"]), max(res["s_lo"]), res["H"]) * 1.1

    def by(v):
        return by1 - v / smax * (by1 - by0)

    # threshold line
    parts.append(f'<line x1="{x0}" y1="{by(res["H"]):.1f}" x2="{x1}" y2="{by(res["H"]):.1f}" '
                 f'stroke="{P["red"]}" stroke-width="1.2" stroke-dasharray="5,4"/>')
    parts.append(f'<text x="{x0 + 5:.1f}" y="{by(res["H"]) - 4:.1f}" fill="{P["red"]}" '
                 f'font-size="10">decision interval H</text>')
    hi = " ".join(f"{px(i):.1f},{by(res['s_hi'][i]):.1f}" for i in range(n))
    lo = " ".join(f"{px(i):.1f},{by(res['s_lo'][i]):.1f}" for i in range(n))
    parts.append(f'<polyline points="{hi}" fill="none" stroke="{P["yellow"]}" stroke-width="1.8"/>')
    parts.append(f'<polyline points="{lo}" fill="none" stroke="{P["purple"]}" stroke-width="1.4" opacity="0.8"/>')
    parts.append(f'<line x1="{x0}" y1="{by1}" x2="{x1}" y2="{by1}" stroke="{P["gray"]}" stroke-width="1"/>')
    # alarm marker
    if res["alarm_index"] is not None:
        ai = res["alarm_index"]
        parts.append(f'<circle cx="{px(ai):.1f}" cy="{by(res["s_hi"][ai]):.1f}" r="5" fill="{P["red"]}"/>')
        parts.append(f'<text x="{px(ai) - 10:.1f}" y="{by(res["s_hi"][ai]) - 10:.1f}" fill="{P["red"]}" '
                     f'font-size="10">alarm t={ai}</text>')
    parts.append(f'<text x="{x0}" y="{by0 - 4:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'S_hi (yellow) accumulates the upward drift and crosses H shortly after the change</text>')

    parts.append(f'<text x="20" y="{H - 12}" fill="{P["gray"]}" font-size="11">'
                 f'the small persistent bias is invisible per-point but adds up in the cumulative sum, '
                 f'tripping the alarm fast.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
