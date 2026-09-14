"""LT-code demo: recover a message from a fountain of symbols, and the overhead-vs-success curve (SVG)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import luby_transform as LT


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
YELLOW = "#ffd43b"
RED = "#ff6b6b"


def main(outdir=None):
    msg = b"LUBY TRANSFORM CODES: a fountain you drink from until you have enough. " * 6
    blen = 20
    blocks, orig = LT.chunk_message(msg, blen)
    k = len(blocks)

    lines = []
    lines.append("Luby Transform codes: rateless erasure coding")
    lines.append("=" * 52)
    lines.append(f"message: {orig} bytes -> {k} source blocks of {blen} bytes")
    lines.append("encoder emits an endless stream of XOR symbols (robust soliton degrees)")
    lines.append("")
    # show a few symbols' degrees
    enc = LT.LTEncoder(blocks, seed=7)
    sample = enc.generate(8)
    lines.append("first 8 encoded symbols (degree = #source blocks XORed):")
    for i, (nb, pl) in enumerate(sample):
        lines.append(f"  symbol {i}: degree {len(nb):>2}, touches blocks {list(nb)[:6]}"
                     f"{'...' if len(nb) > 6 else ''}")
    lines.append("")
    # decode at increasing overhead
    lines.append("decoding success vs symbols collected:")
    lines.append(f"{'symbols':>9}{'overhead':>10}{'decoded?':>10}")
    for oh in (1.0, 1.1, 1.2, 1.4, 1.6, 2.0):
        n = int(k * oh)
        syms = LT.LTEncoder(blocks, seed=7).generate(n)
        dec = LT.decode(syms, k, blen)
        ok = dec is not None and dec[:orig] == msg
        lines.append(f"{n:>9}{oh:>10.1f}{('yes' if ok else 'no'):>10}")
    lines.append("")
    lines.append("Any k(1+epsilon) symbols suffice, whichever arrive -- the channel can drop")
    lines.append("whatever it likes. Decoding is peeling: release a degree-1 symbol, XOR it out,")
    lines.append("repeat. This is the coding behind reliable multicast and distributed storage.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # success probability vs overhead, averaged over seeds
        overheads = [1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]
        probs = []
        for oh in overheads:
            n = int(k * oh)
            succ = 0
            trials = 40
            for s in range(trials):
                syms = LT.LTEncoder(blocks, seed=100 + s).generate(n)
                if LT.decode(syms, k, blen) is not None:
                    succ += 1
            probs.append(succ / trials)

        W, H = 680, 380
        ml, mt, w, h = 60, 55, 580, 260

        def sx(oh):
            return ml + (oh - overheads[0]) / (overheads[-1] - overheads[0]) * w

        def sy(p):
            return mt + h - p * h

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="30" fill="{TEXT}" font-size="15">'
                 f'LT decoding success vs overhead (k={k} blocks)</text>')
        s.append(f'<rect x="{ml}" y="{mt}" width="{w}" height="{h}" fill="none" stroke="{GRAY}" '
                 f'stroke-width="0.6"/>')
        # gridlines
        for p in (0.0, 0.25, 0.5, 0.75, 1.0):
            yy = sy(p)
            s.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+w}" y2="{yy:.1f}" stroke="#21262d"/>')
            s.append(f'<text x="{ml-6}" y="{yy+4:.1f}" fill="{GRAY}" font-size="9" '
                     f'text-anchor="end">{p:.2f}</text>')
        # curve
        pts = " ".join(f"{sx(overheads[i]):.1f},{sy(probs[i]):.1f}" for i in range(len(overheads)))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{GREEN}" stroke-width="2.5"/>')
        for i in range(len(overheads)):
            s.append(f'<circle cx="{sx(overheads[i]):.1f}" cy="{sy(probs[i]):.1f}" r="3" '
                     f'fill="{GREEN}"/>')
        # x ticks
        for oh in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0):
            s.append(f'<text x="{sx(oh):.1f}" y="{mt+h+16}" fill="{GRAY}" font-size="9" '
                     f'text-anchor="middle">{oh:.1f}k</text>')
        s.append(f'<text x="{ml+w/2:.0f}" y="{mt+h+34}" fill="{GRAY}" font-size="11" '
                 f'text-anchor="middle">symbols collected (multiples of k)</text>')
        s.append(f'<text x="{ml}" y="{H-12}" fill="{GRAY}" font-size="10">'
                 f'Success climbs sharply past k; a small constant overhead makes decoding '
                 f'near-certain, no matter which symbols the lossy channel delivered.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "luby_transform.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
