"""Elementary cellular automata: complexity from an eight-line rule.

A row of cells, each 0 or 1, updated in lock-step: a cell's next value depends only on itself
and its two neighbours. With three binary inputs there are 2^3 = 8 possible neighbourhoods, and
a "rule" just assigns an output bit to each -- so there are 2^8 = 256 elementary rules, numbered
0-255 by reading those eight output bits (Wolfram's convention). From that trivial definition
comes a zoo of behaviour that Wolfram sorted into four classes:

    Class 1: dies to a uniform state (e.g. rule 0)
    Class 2: settles into stripes or repeats (rule 90 -> the Sierpinski triangle)
    Class 3: chaotic, random-looking (rule 30 -- used as a random-number generator)
    Class 4: localized structures that interact -- complex, and rule 110 is Turing-complete.

That a one-dimensional automaton with the simplest possible rule can be a universal computer
(rule 110) is one of the most surprising results in computer science: computation needs almost
no ingredients. Rule 30's centre column is so unpredictable it was used in Mathematica's RNG.
Rule 90 draws the Sierpinski fractal from a single seed by XOR of its neighbours.

This module decodes a rule number into its 8-bit lookup table, applies one update step (with
periodic boundaries), evolves a row over many generations, and reports the rule's output for a
given neighbourhood, and reproduces rule 90's XOR structure, rule 30's aperiodicity, and rule
110's persistent structure. Pure stdlib; the discrete-dynamics companion to the sandpile and
logistic-map notes.
"""

from __future__ import annotations


def rule_table(rule_number: int):
    """The 8-entry output table for an elementary rule: table[k] is the new centre value for
    neighbourhood k (k = 4*left + 2*centre + right, 0..7), read from the rule number's bits."""
    return [(rule_number >> k) & 1 for k in range(8)]


def apply_rule(row, rule_number: int):
    """One synchronous update of a row (list of 0/1) under the given rule, periodic boundaries.
    Each cell's new value = table[4*left + 2*self + right]."""
    table = rule_table(rule_number)
    n = len(row)
    return [table[(row[(i - 1) % n] << 2) | (row[i] << 1) | row[(i + 1) % n]] for i in range(n)]


def evolve(row, rule_number: int, generations: int):
    """Evolve a starting row for `generations` steps; return the list of rows (length
    generations+1, including the initial row)."""
    rows = [list(row)]
    r = list(row)
    for _ in range(generations):
        r = apply_rule(r, rule_number)
        rows.append(r)
    return rows


def single_seed_row(width: int):
    """A row of given width, all 0 except a single 1 in the centre -- the classic seed."""
    row = [0] * width
    row[width // 2] = 1
    return row


def output_for_neighbourhood(rule_number: int, left: int, centre: int, right: int) -> int:
    """The rule's output bit for a specific (left, centre, right) neighbourhood."""
    return rule_table(rule_number)[(left << 2) | (centre << 1) | right]


def is_all_zero(row) -> bool:
    """True if the row has died out to all zeros."""
    return not any(row)


def population(row) -> int:
    """Number of live (1) cells in a row."""
    return sum(row)
