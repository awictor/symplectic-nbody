"""Conway's Game of Life: a universe from four rules.

John Conway's 1970 cellular automaton lives on a 2D grid of cells, each alive or dead, updated
all at once by counting the eight neighbours of each cell:

    a live cell with 2 or 3 live neighbours survives; otherwise it dies (loneliness/crowding),
    a dead cell with exactly 3 live neighbours is born.

That is the whole rule -- "B3/S23" -- yet it produces an astonishing menagerie. STILL LIFES
(block, beehive) never change. OSCILLATORS (the blinker, period 2) cycle forever. SPACESHIPS
(the glider) translate across the grid, moving one cell diagonally every four generations. And
because Life supports gliders, glider guns, and logic gates built from their collisions, it is
Turing-complete: you can build a computer inside it. It is the most famous demonstration that
simple local rules can generate open-ended complexity.

This module runs the dynamics on a finite grid (dead borders): counts neighbours, applies the
B3/S23 update, evolves the board, and ships the classic patterns (block, blinker, glider). It
reproduces the stable block, the period-2 blinker, and the glider translating diagonally by
(1,1) every 4 steps. Pure stdlib; the 2D-cellular-automaton companion to the elementary-CA and
sandpile notes.
"""

from __future__ import annotations


def make_board(rows: int, cols: int):
    """An all-dead rows x cols board (list of lists of 0/1)."""
    return [[0 for _ in range(cols)] for _ in range(rows)]


def live_neighbours(board, r, c) -> int:
    """Count the live cells among the 8 neighbours of (r,c) on a finite board (out-of-bounds
    cells count as dead)."""
    rows, cols = len(board), len(board[0])
    total = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                total += board[nr][nc]
    return total


def step(board):
    """One Game-of-Life generation (B3/S23), returning a new board. A live cell survives with
    2-3 neighbours; a dead cell is born with exactly 3."""
    rows, cols = len(board), len(board[0])
    nxt = make_board(rows, cols)
    for r in range(rows):
        for c in range(cols):
            n = live_neighbours(board, r, c)
            if board[r][c]:
                nxt[r][c] = 1 if n in (2, 3) else 0
            else:
                nxt[r][c] = 1 if n == 3 else 0
    return nxt


def evolve(board, generations: int):
    """Evolve a board `generations` steps; return the list of boards (length generations+1)."""
    boards = [[row[:] for row in board]]
    b = board
    for _ in range(generations):
        b = step(b)
        boards.append(b)
    return boards


def population(board) -> int:
    """Number of live cells on the board."""
    return sum(sum(row) for row in board)


def place(board, pattern, top: int, left: int):
    """Stamp a list-of-lists `pattern` of 0/1 onto `board` with its top-left at (top,left).
    Returns the board (mutated)."""
    for r, prow in enumerate(pattern):
        for c, v in enumerate(prow):
            if 0 <= top + r < len(board) and 0 <= left + c < len(board[0]):
                board[top + r][left + c] = v
    return board


# Classic patterns
BLOCK = [[1, 1], [1, 1]]                       # still life
BLINKER = [[1, 1, 1]]                          # period-2 oscillator
GLIDER = [[0, 1, 0], [0, 0, 1], [1, 1, 1]]     # diagonally-moving spaceship


def center_of_mass(board):
    """(row, col) centroid of the live cells, or None if empty. Used to track a glider's
    motion across the grid."""
    total = population(board)
    if total == 0:
        return None
    sr = sum(r * sum(row) for r, row in enumerate(board))
    sc = sum(c for row in board for c, v in enumerate(row) if v)
    return (sr / total, sc / total)
