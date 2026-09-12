"""Demo: Sprague-Grundy theory -- every impartial game is secretly a Nim heap.

Computes Grundy numbers for classic Nim, a subtraction game, and Kayles, shows the winning move in a
Nim position (make the Nim-sum zero), and draws the Grundy sequences as a coloured strip revealing
their periodicity.

    python examples/sprague_grundy_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sprague_grundy import (nim_grundy, nim_moves, nim_sum,  # noqa: E402
                            subtraction_grundy, kayles_grundy)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Sprague-Grundy: every impartial game is a Nim heap of size = its Grundy number\n")

    # Nim: winning move analysis
    heaps = (3, 4, 5)
    g = nim_grundy(heaps)
    print(f"  Nim heaps {heaps}: Grundy = {g} (= 3 XOR 4 XOR 5). Nonzero -> a WIN for the mover.")
    print("  Winning moves (leave a Grundy-0 position, i.e. Nim-sum 0):")
    for p in sorted(nim_moves(heaps)):
        if nim_grundy(p) == 0:
            print(f"    move to {p}")
    losing = (1, 2, 3)
    print(f"\n  Nim heaps {losing}: Grundy = {nim_grundy(losing)} -> a LOSS for the mover; "
          f"no move escapes.\n")

    # Subtraction game
    allowed = [1, 2, 3]
    sg = [subtraction_grundy(n, allowed) for n in range(16)]
    print(f"  Subtraction game, remove {allowed} stones: Grundy(n) for n=0..15:")
    print(f"    {sg}")
    print(f"    -> periodic with period 4 (Grundy = n mod 4); losing heaps are multiples of 4.\n")

    # Kayles
    kg = [kayles_grundy((n,)) for n in range(13)]
    print("  Kayles (knock down 1 or 2 adjacent pins from a row): Grundy(row of n) for n=0..12:")
    print(f"    {kg}")
    print("    -> the famous irregular Kayles nimbers, eventually periodic with period 12.\n")

    # composition
    print(f"  Composition: a position with a Nim heap of 5 and a Kayles row of 4 has Grundy")
    print(f"    {nim_grundy((5,))} XOR {kayles_grundy((4,))} = "
          f"{nim_sum(nim_grundy((5,)), kayles_grundy((4,)))}. Combine any games by XOR-ing nimbers.")

    _svg(os.path.join(outdir, "sprague_grundy.svg"), sg, kg)
    print(f"\n  wrote {os.path.join(outdir, 'sprague_grundy.svg')}")


# a small categorical palette indexed by Grundy value
_PAL = ["#161b22", "#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#ff6b6b", "#b197fc", "#e6edf3"]


def _svg(path, sub_seq, kayles_seq, width=760, height=340):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'Grundy numbers as colour: each cell n is coloured by its nimber</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'dark cells are Grundy 0 (losing positions); repeating colour blocks reveal periodicity'
        f'</text>',
    ]

    def strip(seq, y, label):
        cell = 44
        ox = 150
        parts.append(f'<text x="{ox-12}" y="{y+28}" fill="#8b949e" font-size="12" '
                     f'text-anchor="end">{label}</text>')
        for n, val in enumerate(seq):
            x = ox + n * cell
            col = _PAL[val % len(_PAL)]
            parts.append(f'<rect x="{x}" y="{y}" width="{cell-3}" height="40" fill="{col}"/>')
            txt = "#0d1117" if val in (3,) else "#e6edf3"
            parts.append(f'<text x="{x+(cell-3)/2:.0f}" y="{y+26}" fill="{txt}" font-size="14" '
                         f'text-anchor="middle">{val}</text>')
            parts.append(f'<text x="{x+(cell-3)/2:.0f}" y="{y-6}" fill="#8b949e" font-size="9" '
                         f'text-anchor="middle">{n}</text>')

    strip(sub_seq[:13], 100, "subtract{1,2,3}")
    strip(kayles_seq[:13], 210, "Kayles row")

    parts.append(f'<text x="150" y="300" fill="#8b949e" font-size="12">'
                 f'top: clean period-4 (Grundy = n mod 4). bottom: Kayles, irregular then '
                 f'period-12.</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
