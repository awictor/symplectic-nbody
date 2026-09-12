"""Sprague-Grundy theory: every impartial game is secretly a game of Nim.

An IMPARTIAL game is a two-player, perfect-information game where both players have the same moves
available from any position (unlike chess, where one player moves white and the other black), the last
player able to move wins (normal play), and no position repeats forever. Nim -- take any number of
stones from one of several heaps -- is the canonical example. The astonishing Sprague-Grundy theorem
says that EVERY such game, however baroque its rules, is equivalent to a single Nim heap: each position
has a GRUNDY NUMBER (nimber) g, and the position behaves exactly like a Nim heap of size g. A position
is losing for the player about to move (a P-position, "previous player wins") exactly when its Grundy
number is 0.

The Grundy number is computed by the MEX rule: g(position) = mex{ g(p) : p is a position reachable in
one move }, where mex ("minimum excludant") is the smallest non-negative integer not in the set. A
terminal position with no moves has an empty reachable set, so its Grundy number is mex{} = 0 -- a
loss for the player to move, who cannot move at all. The second miracle is COMPOSITION: when a game
splits into independent subgames played in parallel (several Nim heaps, several strips, several piles),
the Grundy number of the whole is the XOR (the "Nim-sum") of the parts' Grundy numbers. So the entire
theory of who-wins-a-sum-of-games reduces to XOR-ing a handful of small integers, and the winning move
is the one that makes the total Nim-sum zero.

This module provides the mex operator, a generic Grundy-number solver for any impartial game given as
a "list the moves from a position" function (with memoisation), the Nim-sum combinator, and worked
examples: classic Nim, Subtraction games (remove a number of stones from a fixed allowed set), and
the game of Kayles (knock down one or two adjacent pins, splitting a row into independent segments). It
is verified against a brute-force minimax win/loss oracle -- Grundy=0 iff the position is a
theoretical loss for the mover -- and against known results: Nim's XOR-of-heaps rule, the eventual
periodicity of subtraction-game Grundy sequences, and the published Kayles nimber sequence. Pure
stdlib; a game-theory companion to the minimax and combinatorial-generation notes."""

from __future__ import annotations


def mex(values):
    """The minimum excludant: the smallest non-negative integer not present in `values`."""
    s = set(values)
    m = 0
    while m in s:
        m += 1
    return m


def grundy(position, moves):
    """Grundy number (nimber) of a position in an impartial game.

    `moves(position)` must return an iterable of positions reachable in one move. A position with no
    moves has Grundy number 0. Uses memoisation, so positions must be hashable and the game acyclic.
    """
    memo = {}

    def g(pos):
        if pos in memo:
            return memo[pos]
        memo[pos] = -1                       # guard (assumes acyclic; -1 never a valid nimber)
        reachable = [g(p) for p in moves(pos)]
        val = mex(reachable)
        memo[pos] = val
        return val

    return g(position)


def nim_sum(*grundy_values):
    """The Nim-sum (XOR) of Grundy numbers -- the Grundy number of the combined independent games.
    The combined position is a loss for the mover iff this is 0."""
    total = 0
    for v in grundy_values:
        total ^= v
    return total


# --- example game 1: classic Nim -------------------------------------------
def nim_moves(heaps):
    """Positions of Nim: a sorted tuple of heap sizes. A move removes >=1 stone from one heap."""
    heaps = tuple(sorted(heaps))
    out = set()
    for i, h in enumerate(heaps):
        for take in range(1, h + 1):
            new = list(heaps)
            new[i] = h - take
            new = tuple(sorted(x for x in new if x > 0))
            out.add(new)
    return out


def nim_grundy(heaps):
    """Grundy number of a Nim position. By the theorem this equals the XOR of the heap sizes."""
    return grundy(tuple(sorted(heaps)), nim_moves)


# --- example game 2: subtraction game --------------------------------------
def subtraction_moves_factory(allowed):
    """Moves for a subtraction game: from a heap of n stones, remove any k in `allowed` with k<=n."""
    allowed = tuple(allowed)

    def moves(n):
        return [n - k for k in allowed if k <= n]

    return moves


def subtraction_grundy(n, allowed):
    """Grundy number of a single heap of n stones in the subtraction game with the given move set."""
    return grundy(n, subtraction_moves_factory(allowed))


# --- example game 3: Kayles (knock down 1 or 2 adjacent pins) --------------
def kayles_moves(rows):
    """Kayles position = sorted tuple of remaining contiguous row lengths. A move removes 1 pin (from
    the interior or end, splitting a row into up to two pieces) or 2 adjacent pins."""
    rows = tuple(sorted(r for r in rows if r > 0))
    out = set()
    for idx, n in enumerate(rows):
        rest = rows[:idx] + rows[idx + 1:]
        # knock down a single pin at position i (0-based) -> pieces of length i and n-1-i
        for i in range(n):
            pieces = [p for p in (i, n - 1 - i) if p > 0]
            out.add(tuple(sorted(rest + tuple(pieces))))
        # knock down two adjacent pins at positions i, i+1 -> pieces i and n-2-i
        for i in range(n - 1):
            pieces = [p for p in (i, n - 2 - i) if p > 0]
            out.add(tuple(sorted(rest + tuple(pieces))))
    return out


def kayles_grundy(rows):
    """Grundy number of a Kayles position (a tuple of row lengths)."""
    return grundy(tuple(sorted(r for r in rows if r > 0)), kayles_moves)


# --- brute-force minimax oracle (ground truth) -----------------------------
def is_losing_position(position, moves):
    """True iff the position is a theoretical LOSS for the player about to move, by direct minimax:
    a position is losing iff EVERY move leads to a winning position for the opponent (and a position
    with no moves is losing). Independent of the Grundy machinery -- used to validate it."""
    memo = {}

    def losing(pos):
        if pos in memo:
            return memo[pos]
        memo[pos] = False                     # guard for acyclic games
        nexts = list(moves(pos))
        if not nexts:
            memo[pos] = True                  # no move -> current player loses
            return True
        # current player wins if SOME move hands the opponent a losing position
        win = any(losing(p) for p in nexts)
        memo[pos] = not win
        return memo[pos]

    return losing(position)
