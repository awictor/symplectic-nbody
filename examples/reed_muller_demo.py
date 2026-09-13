"""Demo: first-order Reed-Muller code RM(1,5) -- the Mariner 9 Mars code.

Encodes a 6-bit message into a 32-bit codeword, corrupts it with random bit errors, and decodes by
the fast Walsh-Hadamard transform, showing it corrects up to 7 flips per word. Draws the correction
success rate versus error count.

    python examples/reed_muller_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reed_muller import encode, decode, code_parameters  # noqa: E402


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

    print("Reed-Muller RM(1,5): the code that protected Mariner 9's photos of Mars\n")

    m = 5
    n, k, d, t = code_parameters(m)
    print(f"  RM(1,{m}): {k} message bits -> {n}-bit codeword, min distance {d}, corrects {t} errors.\n")

    message = [1, 0, 1, 1, 0, 1]
    word = encode(message, m)
    print(f"  message  {message}")
    print(f"  codeword {''.join(map(str, word))}")

    # corrupt with 5 errors (within the radius of 7)
    rng = _lcg(2024)
    corrupted = list(word)
    err_pos = []
    while len(err_pos) < 5:
        p = int(rng() * n)
        if p not in err_pos:
            err_pos.append(p)
            corrupted[p] ^= 1
    print(f"  {5} bit errors flipped at positions {sorted(err_pos)}:")
    print(f"  received {''.join(map(str, corrupted))}")
    decoded, _ = decode(corrupted, m)
    print(f"  decoded  {decoded}  (correct: {decoded == message})\n")

    # success rate vs number of errors
    print(f"  Decoding success rate over 500 random words per error count:")
    print(f"    {'errors':>7}  {'success rate':>13}")
    curve = []
    for n_err in range(0, 12):
        successes = 0
        trials = 500
        for tr in range(trials):
            r2 = _lcg(1000 + tr)
            msg_int = int(r2() * (1 << (m + 1)))
            msg = [(msg_int >> i) & 1 for i in range(m + 1)]
            w = encode(msg, m)
            rec = list(w)
            flipped = set()
            while len(flipped) < n_err:
                flipped.add(int(r2() * n))
            for p in flipped:
                rec[p] ^= 1
            if decode(rec, m)[0] == msg:
                successes += 1
        rate = successes / trials
        curve.append((n_err, rate))
        mark = "  <- within correction radius" if n_err <= t else ""
        print(f"    {n_err:>7}  {rate:>13.3f}{mark}")

    print(f"\n  Up to {t} errors: always corrected. Beyond that, success degrades gracefully --")
    print("  exactly as the distance-16 code guarantees.")

    _svg(os.path.join(outdir, "reed_muller.svg"), curve, t)
    print(f"\n  wrote {os.path.join(outdir, 'reed_muller.svg')}")


def _svg(path, curve, t, width=760, height=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'RM(1,5) decoding success vs bit errors (100% up to t={t})</text>',
    ]
    ox, oy, ow, oh = 55, 55, width - 100, height - 100
    nmax = curve[-1][0]

    def px(ne):
        return ox + ow * ne / nmax

    def py(r):
        return oy + oh * (1 - r)

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    # correction-radius shading
    parts.append(f'<rect x="{ox}" y="{oy}" width="{px(t)-ox:.1f}" height="{oh}" '
                 f'fill="#06d6a0" fill-opacity="0.1"/>')
    parts.append(f'<line x1="{px(t):.1f}" y1="{oy}" x2="{px(t):.1f}" y2="{oy+oh}" '
                 f'stroke="#06d6a0" stroke-width="1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{px(t)+4:.0f}" y="{oy+14}" fill="#06d6a0" font-size="10">t={t}</text>')
    pts = " ".join(f"{px(ne):.1f},{py(r):.1f}" for ne, r in curve)
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="2"/>')
    for ne, r in curve:
        parts.append(f'<circle cx="{px(ne):.1f}" cy="{py(r):.1f}" r="3" fill="#ffd43b"/>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+22:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">number of bit errors per 32-bit word</text>')
    parts.append(f'<text x="{ox-6}" y="{oy-4}" fill="#8b949e" font-size="10" text-anchor="end">'
                 f'success rate</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
