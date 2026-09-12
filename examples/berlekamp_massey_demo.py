"""Demo: Berlekamp-Massey -- reverse-engineering the recurrence hidden in a sequence.

Recovers the recurrence of famous integer sequences from their first terms, extrapolates them, and
shows how a short LFSR keystream is cracked from a truncated prefix -- the classic linear-complexity
attack. Draws the linear complexity of a stream growing with how much you observe.

    python examples/berlekamp_massey_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from berlekamp_massey import (berlekamp_massey, extend, berlekamp_massey_gf2,  # noqa: E402
                              lfsr_generate)


def fmt(coeffs):
    return "[" + ", ".join(str(c) for c in coeffs) + "]"


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Berlekamp-Massey: find the shortest linear recurrence behind a sequence\n")

    seqs = [
        ("Fibonacci", [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]),
        ("Tribonacci", [0, 0, 1, 1, 2, 4, 7, 13, 24, 44]),
        ("Pell", [0, 1, 2, 5, 12, 29, 70, 169]),
        ("powers of 2", [1, 2, 4, 8, 16, 32, 64]),
        ("Jacobsthal", [0, 1, 1, 3, 5, 11, 21, 43]),
    ]
    for name, seq in seqs:
        coeffs = berlekamp_massey(seq)
        nxt = [int(x) for x in extend(seq, 3)]
        terms = " + ".join(f"{coeffs[j]}*s[n-{j+1}]" for j in range(len(coeffs)))
        print(f"  {name:12s} complexity {len(coeffs)}:  s[n] = {terms}")
        print(f"               next 3 terms: {nxt}")

    print("\n  Linear-complexity attack on a stream cipher (LFSR keystream):")
    # a length-5 LFSR keystream -- fully determined by 2*5 = 10 output bits
    taps = [0, 0, 1, 0, 1]           # x^5 + x^2 + 1, a primitive (max-length) polynomial
    seed = [1, 0, 0, 1, 1]
    stream = lfsr_generate(taps, seed, 40)
    print(f"    true LFSR: length 5, taps {taps}")
    print(f"    observed keystream: {''.join(map(str, stream[:20]))}...")
    for prefix in (6, 8, 10, 12):
        L, coeffs = berlekamp_massey_gf2(stream[:prefix])
        recovered = lfsr_generate(coeffs, stream[:L], 40) if L > 0 else []
        cracked = recovered == stream
        print(f"    from {prefix:2d} bits: recovered complexity {L}, "
              f"predicts full stream: {cracked}")
    print("    Once you see 2L bits, Berlekamp-Massey recovers the whole register and predicts")
    print("    every future bit -- why a raw LFSR is cryptographically useless despite a long period.")

    _svg(os.path.join(outdir, "berlekamp_massey.svg"), stream, taps)
    print(f"\n  wrote {os.path.join(outdir, 'berlekamp_massey.svg')}")


def _svg(path, stream, taps, width=760, height=380):
    L_true = len(taps)
    # linear complexity recovered as a function of how many bits are observed
    xs = list(range(1, len(stream) + 1))
    comps = [berlekamp_massey_gf2(stream[:k])[0] for k in xs]

    ox, oy = 70, 300
    plot_w, plot_h = 640, 240
    maxc = max(comps) if comps else 1
    maxk = len(stream)

    def px(k):
        return ox + (k - 1) / max(1, maxk - 1) * plot_w

    def py(c):
        return oy - c / max(1, maxc) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'Linear complexity recovered vs bits observed (length-{L_true} LFSR)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'complexity climbs then locks at {L_true} -- after 2L bits the whole register is known</text>',
    ]

    # axes
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+plot_w}" y2="{oy}" stroke="#30363d"/>')
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-plot_h}" stroke="#30363d"/>')

    # the "locked" line at true complexity
    parts.append(f'<line x1="{ox}" y1="{py(L_true):.1f}" x2="{ox+plot_w}" y2="{py(L_true):.1f}" '
                 f'stroke="#ff6b6b" stroke-width="1" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{ox+plot_w-4:.0f}" y="{py(L_true)-6:.0f}" fill="#ff6b6b" '
                 f'font-size="11" text-anchor="end">complexity = {L_true} (cracked)</text>')

    # 2L marker
    parts.append(f'<line x1="{px(2*L_true):.1f}" y1="{oy}" x2="{px(2*L_true):.1f}" '
                 f'y2="{oy-plot_h}" stroke="#ffd43b" stroke-width="1" stroke-dasharray="2 3"/>')
    parts.append(f'<text x="{px(2*L_true):.0f}" y="{oy-plot_h-2:.0f}" fill="#ffd43b" '
                 f'font-size="10" text-anchor="middle">2L={2*L_true} bits</text>')

    pts = " ".join(f"{px(k):.1f},{py(c):.1f}" for k, c in zip(xs, comps))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="2"/>')
    parts.append(f'<text x="{ox+plot_w/2:.0f}" y="{oy+30}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">bits observed</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
