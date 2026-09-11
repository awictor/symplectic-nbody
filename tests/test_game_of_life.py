"""Tests for game_of_life: Conway's B3/S23 automaton."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import game_of_life as gol

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Neighbour count.
b = gol.make_board(3, 3)
gol.place(b, [[1, 1, 1]], 1, 0)   # middle row live
check("centre has 2 live neighbours", gol.live_neighbours(b, 1, 1) == 2)
check("corner counts in-bounds only", gol.live_neighbours(b, 0, 0) == 2)

# Empty board stays empty.
check("empty stays empty", gol.population(gol.step(gol.make_board(5, 5))) == 0)

# A lone cell dies (loneliness).
lone = gol.make_board(3, 3)
lone[1][1] = 1
check("lone cell dies", gol.population(gol.step(lone)) == 0)

# Block is a still life: unchanged forever.
board = gol.make_board(4, 4)
gol.place(board, gol.BLOCK, 1, 1)
after = gol.step(board)
check("block is a still life", after == board)
check("block survives many generations", gol.evolve(board, 10)[-1] == board)

# Blinker is a period-2 oscillator: flips horizontal<->vertical, returns after 2 steps.
bl = gol.make_board(5, 5)
gol.place(bl, gol.BLINKER, 2, 1)   # horizontal bar
one = gol.step(bl)
check("blinker changes after 1 step", one != bl)
check("blinker returns after 2 steps", gol.step(one) == bl)
check("blinker conserves 3 cells", gol.population(one) == 3)
# After 1 step it should be vertical (a column of 3).
check("blinker becomes vertical", one[1][2] == 1 and one[2][2] == 1 and one[3][2] == 1)

# Glider translates by (1,1) every 4 generations (diagonal spaceship).
g = gol.make_board(12, 12)
gol.place(g, gol.GLIDER, 1, 1)
com0 = gol.center_of_mass(g)
boards = gol.evolve(g, 4)
com4 = gol.center_of_mass(boards[-1])
check("glider keeps 5 cells", gol.population(boards[-1]) == 5)
check("glider moves down-right by ~1 in 4 steps",
      abs((com4[0] - com0[0]) - 1.0) < 1e-9 and abs((com4[1] - com0[1]) - 1.0) < 1e-9)

# Glider survives and stays a 5-cell pattern over several periods.
long_run = gol.evolve(g, 16)
check("glider persists over 16 gens", gol.population(long_run[-1]) == 5)
com16 = gol.center_of_mass(long_run[-1])
check("glider travelled ~4 cells diagonally in 16 gens",
      abs((com16[0] - com0[0]) - 4.0) < 1e-9)

# center_of_mass None for empty.
check("empty board has no centre", gol.center_of_mass(gol.make_board(3, 3)) is None)

# evolve length.
check("evolve length gens+1", len(gol.evolve(board, 7)) == 8)

# A dead cell with exactly 3 neighbours is born.
birth = gol.make_board(3, 3)
gol.place(birth, [[1, 1], [1, 0]], 0, 0)   # 3 live cells around (1,1)
check("cell born with 3 neighbours", gol.step(birth)[1][1] == 1)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all game_of_life tests passed")
