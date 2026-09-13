"""Demo: FM-index backward search -- counting a pattern in O(|pattern|) via the BWT.

Builds an FM-index over a DNA-like text, traces the backward-search range shrinking one character at
a time as it counts a pattern, then draws the range narrowing across the BWT rows.

    python examples/fm_index_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fm_index import FMIndex, brute_count  # noqa: E402


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("FM-index: full-text search from the compressed Burrows-Wheeler transform\n")

    rng = _lcg(7)
    text = "".join("ACGT"[int(rng() * 4)] for _ in range(120))
    fm = FMIndex(text)

    print(f"  text: {len(text)} DNA bases, alphabet {fm.alphabet[1:]}")
    print(f"  BWT (first 60 chars): {fm.bwt[1:61]}\n")

    pattern = "TTA"
    # trace the backward search
    print(f"  Backward search for '{pattern}' -- range [lo, hi) shrinks right-to-left:")
    lo, hi = 0, fm.n
    print(f"    {'step':<6}{'char':<6}{'lo':>6}{'hi':>6}{'width':>8}")
    print(f"    {'init':<6}{'':<6}{lo:>6}{hi:>6}{hi - lo:>8}")
    trace = [(lo, hi)]
    for step, c in enumerate(reversed(pattern), 1):
        lo = fm.C[c] + fm._occ(c, lo)
        hi = fm.C[c] + fm._occ(c, hi)
        trace.append((lo, hi))
        print(f"    {step:<6}{c:<6}{lo:>6}{hi:>6}{hi - lo:>8}")

    cnt = fm.count(pattern)
    pos = fm.locate(pattern)
    print(f"\n  '{pattern}' occurs {cnt} times (brute force: {brute_count(text, pattern)})")
    print(f"  positions: {pos}")

    # a few more patterns
    print(f"\n  Counting more patterns in O(|pattern|), independent of text length:")
    for p in ["A", "GT", "ACGT", "TTTT", "CAG"]:
        print(f"    count('{p}') = {fm.count(p):>3}   (brute {brute_count(text, p)})")

    print(f"\n  recover_text() inverts the BWT correctly: {fm.recover_text() == text}")

    _svg(os.path.join(outdir, "fm_index.svg"), trace, fm.n, pattern)
    print(f"\n  wrote {os.path.join(outdir, 'fm_index.svg')}")


def _svg(path, trace, n, pattern, width=760, height=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Backward search for &quot;{pattern}&quot;: the BWT-row range narrows to the match count</text>',
    ]
    ox, oy, ow = 60, 60, width - 120
    rows_h = height - 120
    bar_h = rows_h / len(trace)

    def px(row):
        return ox + ow * row / n

    labels = ["init"] + list(reversed(pattern))
    colors = ["#8b949e", "#4dabf7", "#ffd43b", "#06d6a0", "#b197fc", "#ff922b"]
    for k, (lo, hi) in enumerate(trace):
        y = oy + k * bar_h
        color = colors[k % len(colors)]
        # full-width faint track
        parts.append(f'<rect x="{ox}" y="{y + 3:.1f}" width="{ow}" height="{bar_h - 6:.1f}" '
                     f'fill="#161b22" stroke="#30363d" stroke-width="0.5"/>')
        # active range
        parts.append(f'<rect x="{px(lo):.1f}" y="{y + 3:.1f}" width="{max(px(hi) - px(lo), 1.5):.1f}" '
                     f'height="{bar_h - 6:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{ox - 8}" y="{y + bar_h / 2 + 4:.1f}" fill="#e6edf3" '
                     f'font-size="11" text-anchor="end">{labels[k]}</text>')
        parts.append(f'<text x="{ox + ow + 8}" y="{y + bar_h / 2 + 4:.1f}" fill="{color}" '
                     f'font-size="10">width {hi - lo}</text>')
    parts.append(f'<text x="{ox + ow / 2:.0f}" y="{height - 20}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">sorted BWT rows 0 .. {n} (each row = one rotation of the text)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
