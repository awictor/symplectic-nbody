"""Tunstall coding demo: variable-to-fixed dictionary, rate approaching entropy, vs Huffman's fixed-to-variable."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import tunstall_coding as tc


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


def sample(probs, n, seed):
    rng = _R(seed)
    syms = sorted(probs)
    total = sum(probs.values())
    cdf = []
    acc = 0.0
    for s in syms:
        acc += probs[s] / total
        cdf.append((acc, s))
    out = []
    for _ in range(n):
        u = rng.u()
        for c, s in cdf:
            if u <= c:
                out.append(s)
                break
        else:
            out.append(syms[-1])
    return "".join(out)


def main():
    lines = []
    lines.append("Tunstall coding -- variable-length source strings to FIXED-length codes")
    lines.append("=" * 72)
    lines.append("")

    probs = {"a": 0.7, "b": 0.2, "c": 0.1}
    H = tc.source_entropy(probs)
    lines.append(f"Source: a=0.7, b=0.2, c=0.1.  Entropy H = {H:.4f} bits/symbol.")
    lines.append("(Huffman maps 1 symbol -> variable bits; Tunstall maps variable symbols -> fixed bits.)")
    lines.append("")

    # show the dictionary for a small k
    k = 4
    entries = tc.build_dictionary(probs, k)
    enc, dec = tc.assign_codewords(entries, k)
    lines.append(f"Tunstall dictionary for k={k} ({len(entries)} entries, each a {k}-bit codeword):")
    lines.append("   codeword   string   probability")
    lines.append("   " + "-" * 34)
    for string, prob in entries[:10]:
        lines.append(f"   {enc[string]}       {string:6s}   {prob:.4f}")
    if len(entries) > 10:
        lines.append(f"   ... ({len(entries) - 10} more)")
    lines.append("")

    # rate approaching entropy as k grows
    lines.append("Compression rate approaches the entropy as k grows:")
    lines.append("   k    dict size   bits/symbol   gap to H")
    lines.append("   " + "-" * 42)
    ks = []
    rates = []
    for kk in (2, 4, 6, 8, 10, 12, 14):
        e = tc.build_dictionary(probs, kk)
        bps = tc.bits_per_symbol(e, kk)
        ks.append(kk)
        rates.append(bps)
        lines.append(f"   {kk:2d}    {len(e):6d}      {bps:.4f}       {bps - H:+.4f}")
    lines.append("")

    # verify round-trip on a real message
    long_msg = sample(probs, 3000, seed=2)
    e12 = tc.build_dictionary(probs, 12)
    enc12, dec12 = tc.assign_codewords(e12, 12)
    bits = tc.encode(long_msg, enc12)
    back = tc.decode(bits, dec12, 12, length=len(long_msg))
    lines.append(f"Round-trip on 3000 symbols (k=12): {'OK' if back == long_msg else 'FAIL'}")
    lines.append(f"  encoded {len(long_msg)} symbols to {len(bits)} bits "
                 f"= {len(bits)/len(long_msg):.4f} bits/symbol (H={H:.4f}).")

    text = "\n".join(lines)
    print(text)

    svg = _svg(ks, rates, H)
    return text, svg


def _svg(ks, rates, H):
    W, H_px = 640, 400
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H_px}" '
             f'viewBox="0 0 {W} {H_px}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H_px}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'Tunstall rate converging to the source entropy</text>')

    x0, x1, y0, y1 = 60, 610, 60, 320
    kmin, kmax = min(ks), max(ks)
    rmax = max(rates)
    rmin = H * 0.98

    def px(k):
        return x0 + (k - kmin) / (kmax - kmin) * (x1 - x0)

    def py(r):
        return y1 - (r - rmin) / (rmax - rmin) * (y1 - y0)

    # entropy floor line
    parts.append(f'<line x1="{x0}" y1="{py(H):.1f}" x2="{x1}" y2="{py(H):.1f}" '
                 f'stroke="{P["green"]}" stroke-width="1.5" stroke-dasharray="5,4"/>')
    parts.append(f'<text x="{x1 - 150}" y="{py(H) - 6:.1f}" fill="{P["green"]}" font-size="11">'
                 f'entropy H = {H:.3f} (floor)</text>')

    # rate curve
    pts = " ".join(f"{px(ks[i]):.1f},{py(rates[i]):.1f}" for i in range(len(ks)))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{P["yellow"]}" stroke-width="2"/>')
    for i in range(len(ks)):
        parts.append(f'<circle cx="{px(ks[i]):.1f}" cy="{py(rates[i]):.1f}" r="3.5" fill="{P["blue"]}"/>')

    # axes
    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')
    for k in ks:
        parts.append(f'<text x="{px(k):.1f}" y="{y1 + 16:.1f}" fill="{P["gray"]}" '
                     f'font-size="10" text-anchor="middle">{k}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y1 + 34:.1f}" fill="{P["gray"]}" '
                 f'font-size="11" text-anchor="middle">codeword width k (bits)</text>')
    parts.append(f'<text x="{x0 - 6}" y="{y0 - 8:.1f}" fill="{P["gray"]}" font-size="10">bits/symbol</text>')

    parts.append(f'<text x="20" y="{H_px - 12}" fill="{P["gray"]}" font-size="11">'
                 f'larger dictionaries consume more source symbols per fixed codeword, driving the rate '
                 f'down toward the entropy.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
