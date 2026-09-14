"""Golomb coding demo: parametric codes for a geometric source, with the optimal-m sweep and Rice special case."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import golomb_coding as gc


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)


def geometric_sample(p, n, seed):
    rng = _R(seed)
    out = []
    for _ in range(n):
        u = rng.u()
        k = int(math.log(1 - u + 1e-18) / math.log(p))
        out.append(max(0, k))
    return out


def main():
    lines = []
    lines.append("Golomb coding -- optimal prefix code for geometric integers")
    lines.append("=" * 60)
    lines.append("")

    # show the codewords for a small m
    m = 5
    lines.append(f"Codewords for m={m} (unary quotient + truncated-binary remainder):")
    lines.append("   n    q  r   codeword")
    lines.append("   " + "-" * 28)
    for n in range(11):
        q, r = n // m, n % m
        lines.append(f"   {n:2d}   {q}  {r}   {gc.encode_int(n, m)}")
    lines.append("")

    # geometric source, optimal m near entropy
    p = 0.8
    H = gc.geometric_entropy(p)
    m_opt = gc.optimal_m(p)
    vals = geometric_sample(p, 8000, seed=1)
    lines.append(f"Geometric source P(n)=(1-p)p^n, p={p}. Entropy H={H:.4f} bits/symbol.")
    lines.append(f"Optimal m = ceil(-1/log2 p) = {m_opt}.")
    lines.append("")
    lines.append("Mean code length vs parameter m:")
    lines.append("   m     mean bits/symbol   gap to H")
    lines.append("   " + "-" * 38)
    best_m = None
    best_len = 1e9
    ms = list(range(1, 13))
    lens = []
    for mm in ms:
        L = gc.mean_code_length(vals, mm)
        lens.append(L)
        if L < best_len:
            best_len = L
            best_m = mm
        marker = "  <- optimal formula" if mm == m_opt else ""
        lines.append(f"   {mm:2d}    {L:.4f}          {L - H:+.4f}{marker}")
    lines.append("")
    lines.append(f"Empirically best m = {best_m}; formula predicts {m_opt}. "
                 f"Best rate {best_len:.4f} vs entropy {H:.4f} (within {best_len - H:.3f} bit).")
    lines.append("")

    # Rice special case
    lines.append("Rice coding (m = 2^k, plain binary remainder -- the fast codec special case):")
    for k in range(4):
        mm = gc.rice_m(k)
        L = gc.mean_code_length(vals, mm)
        lines.append(f"   k={k}  m={mm:2d}   mean {L:.4f} bits/symbol")

    text = "\n".join(lines)
    print(text)

    svg = _svg(ms, lens, H, m_opt)
    return text, svg


def _svg(ms, lens, H, m_opt):
    W, Hpx = 640, 400
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hpx}" '
             f'viewBox="0 0 {W} {Hpx}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{Hpx}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'Golomb mean code length vs parameter m</text>')

    x0, x1, y0, y1 = 55, 610, 60, 320
    mmin, mmax = min(ms), max(ms)
    lmax = max(lens)
    lmin = H * 0.98

    def px(m):
        return x0 + (m - mmin) / (mmax - mmin) * (x1 - x0)

    def py(L):
        return y1 - (L - lmin) / (lmax - lmin) * (y1 - y0)

    # entropy floor
    parts.append(f'<line x1="{x0}" y1="{py(H):.1f}" x2="{x1}" y2="{py(H):.1f}" '
                 f'stroke="{P["green"]}" stroke-width="1.5" stroke-dasharray="5,4"/>')
    parts.append(f'<text x="{x1 - 130}" y="{py(H) - 6:.1f}" fill="{P["green"]}" font-size="11">'
                 f'entropy H = {H:.3f}</text>')

    # curve
    pts = " ".join(f"{px(ms[i]):.1f},{py(lens[i]):.1f}" for i in range(len(ms)))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{P["yellow"]}" stroke-width="2"/>')
    for i in range(len(ms)):
        col = P["red"] if ms[i] == m_opt else P["blue"]
        rad = 5 if ms[i] == m_opt else 3
        parts.append(f'<circle cx="{px(ms[i]):.1f}" cy="{py(lens[i]):.1f}" r="{rad}" fill="{col}"/>')
    # mark optimal
    parts.append(f'<text x="{px(m_opt):.1f}" y="{py(lens[m_opt-1]) - 12:.1f}" fill="{P["red"]}" '
                 f'font-size="11" text-anchor="middle">optimal m={m_opt}</text>')

    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')
    for m in ms:
        parts.append(f'<text x="{px(m):.1f}" y="{y1 + 16:.1f}" fill="{P["gray"]}" '
                     f'font-size="10" text-anchor="middle">{m}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y1 + 34:.1f}" fill="{P["gray"]}" '
                 f'font-size="11" text-anchor="middle">parameter m</text>')
    parts.append(f'<text x="{x0 - 6}" y="{y0 - 8:.1f}" fill="{P["gray"]}" font-size="10">bits/symbol</text>')

    parts.append(f'<text x="20" y="{Hpx - 12}" fill="{P["gray"]}" font-size="11">'
                 f'the U-shaped curve bottoms at the formula-predicted m -- too small pays long unary '
                 f'prefixes, too large wastes remainder bits.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
