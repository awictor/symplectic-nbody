"""Demo: Conway's Game of Life -- a universe from four rules.

Prints the fate of the classic patterns (block, blinker, glider) and an ASCII animation of a
glider crossing the grid, then draws the glider's four phases and a mixed board of a still
life, an oscillator, and a spaceship.

    python examples/game_of_life_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from game_of_life import (make_board, step, evolve, place, population,  # noqa: E402
                          center_of_mass, BLOCK, BLINKER, GLIDER)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Conway's Game of Life (B3/S23): live cell survives on 2-3, dead born on 3\n")
    print("  Classic patterns:")
    # block
    b = make_board(4, 4); place(b, BLOCK, 1, 1)
    print(f"    block   : still life (population {population(b)}, unchanged: {step(b) == b})")
    # blinker
    bl = make_board(5, 5); place(bl, BLINKER, 2, 1)
    print(f"    blinker : period-2 oscillator (returns in 2: {step(step(bl)) == bl})")
    # glider
    g = make_board(12, 12); place(g, GLIDER, 1, 1)
    c0 = center_of_mass(g); c4 = center_of_mass(evolve(g, 4)[-1])
    print(f"    glider  : spaceship, moves ({c4[0]-c0[0]:.0f},{c4[1]-c0[1]:.0f}) every 4 gens")

    print("\n  A glider crossing an 11x11 grid:")
    board = make_board(11, 11); place(board, GLIDER, 0, 0)
    for gen in (0, 4, 8, 12):
        b = evolve(board, gen)[-1]
        print(f"    generation {gen}:")
        for row in b:
            print("      " + "".join("#" if x else "." for x in row))
        print()

    print("  From B3/S23 come still lifes, blinking oscillators, gliders that fly, and -- via")
    print("  glider guns and collisions building logic gates -- a Turing-complete computer.")
    print("  Simple local rules, open-ended complexity.")

    _svg(os.path.join(outdir, "game_of_life.svg"))
    print(f"  wrote {os.path.join(outdir, 'game_of_life.svg')}")


def _draw_board(parts, board, x0, y0, cell, col):
    for r, row in enumerate(board):
        for c, v in enumerate(row):
            if v:
                parts.append(f'<rect x="{x0 + c*cell:.1f}" y="{y0 + r*cell:.1f}" '
                             f'width="{cell-0.6:.1f}" height="{cell-0.6:.1f}" fill="{col}"/>')
    # grid frame
    rows, cols = len(board), len(board[0])
    parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cols*cell:.1f}" height="{rows*cell:.1f}" '
                 f'fill="none" stroke="#21262d" stroke-width="1"/>')


def _svg(path, size=720, pad=50):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">Conway&#39;s Game of Life</text>',
        f'<text x="20" y="46" fill="#8b949e" font-size="12">'
        f'the glider&#39;s four phases (top), a mixed board of a still life, oscillator, and spaceship (bottom)</text>',
    ]

    # top: glider phases, each on a 6x6 board
    cell = 16
    g = make_board(6, 6); place(g, GLIDER, 1, 1)
    phases = evolve(g, 4)
    for k, b in enumerate(phases):
        x0 = pad + k * (6 * cell + 26)
        _draw_board(parts, b, x0, 62, cell, "#06d6a0")
        parts.append(f'<text x="{x0 + 3*cell:.1f}" y="{62 + 6*cell + 14:.1f}" fill="#8b949e" '
                     f'font-size="10" text-anchor="middle">t={k}</text>')
    parts.append(f'<text x="{pad:.1f}" y="{62 + 6*cell + 32:.1f}" fill="#8b949e" font-size="11">'
                 f'glider: same shape, shifted one cell down-right, every 4 generations</text>')

    # bottom: one board with block + blinker + glider
    big = make_board(18, 30)
    place(big, BLOCK, 2, 2)
    place(big, BLINKER, 8, 3)
    place(big, GLIDER, 3, 12)
    place(big, [[1,1,0],[1,0,1],[0,1,0]], 10, 20)   # a "boat"-ish still life
    by0 = 300
    bcell = min((size - 2 * pad) / 30, (size - by0 - 40) / 18)
    _draw_board(parts, big, pad, by0, bcell, "#4dabf7")
    parts.append(f'<text x="{pad:.1f}" y="{by0 + 18*bcell + 18:.1f}" fill="#8b949e" font-size="10">'
                 f'a mixed population: block (still), blinker (oscillator), glider (spaceship)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
