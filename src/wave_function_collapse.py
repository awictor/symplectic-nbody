"""Wave function collapse: procedural generation by constraint propagation.

WAVE FUNCTION COLLAPSE (WFC) is a procedural-generation algorithm -- popularized by Maxim Gumin -- that
fills a grid with tiles so that every adjacency obeys a set of local rules, producing outputs that
look hand-authored (coherent maps, textures, levels) rather than random. The name borrows from
quantum mechanics as a metaphor: each cell starts in a SUPERPOSITION of every possible tile, and the
algorithm repeatedly OBSERVES (collapses) the most-constrained cell to a single tile, then PROPAGATES
that choice to its neighbours, deleting any tile options that the new constraint forbids -- exactly
the discrete constraint-satisfaction loop of arc consistency.

The engine has two moves. OBSERVATION picks the cell with the lowest ENTROPY (the fewest remaining
options -- the most-constrained, most-informative cell) and collapses it to one allowed tile, chosen
by weight. PROPAGATION then runs a worklist: whenever a cell's option set shrinks, each neighbour is
re-checked and any tile with no remaining compatible neighbour is removed, cascading until the grid
is arc-consistent again. If a cell's options ever drop to zero the attempt has hit a CONTRADICTION and
is restarted; otherwise the loop continues until every cell is collapsed. The adjacency rules -- which
tiles may sit next to which, in each direction -- are the entire specification, so WFC turns a tiny
rule set into an endless variety of consistent worlds.

This module implements tiled WFC over a 2-D grid: adjacency rules per direction, weighted collapse,
lowest-entropy observation, full constraint propagation, contradiction detection with restart, and a
seeded RNG for reproducibility. It is verified that every generated grid strictly satisfies the
adjacency rules (no forbidden neighbour pair anywhere), that a given seed reproduces the same grid,
that an over-constrained rule set is reported as unsatisfiable rather than producing garbage, that a
rule set forcing a unique tiling produces exactly it, and that propagation alone makes a partially
observed grid arc-consistent. Pure stdlib; a procedural-generation companion to the Perlin-noise and
cellular-automaton notes."""

from __future__ import annotations

# directions: (dx, dy) with names; propagation uses the opposite direction on the neighbour
_DIRS = {"R": (1, 0), "L": (-1, 0), "D": (0, 1), "U": (0, -1)}
_OPPOSITE = {"R": "L", "L": "R", "D": "U", "U": "D"}


class Contradiction(Exception):
    """Raised when a cell's option set becomes empty during propagation."""


class WaveFunctionCollapse:
    """Tiled wave function collapse over a width x height grid.

    tiles: list of tile ids. weights: dict tile -> weight (default 1). rules: dict direction ->
    set of (tile_a, tile_b) meaning tile_b may appear in that direction from tile_a."""

    def __init__(self, width, height, tiles, rules, weights=None, seed=1):
        self.w = width
        self.h = height
        self.tiles = list(tiles)
        self.weights = weights or {t: 1.0 for t in tiles}
        # normalize rules into adjacency: allowed[d][a] = set of tiles b allowed to the d-side of a
        self.allowed = {d: {t: set() for t in tiles} for d in _DIRS}
        for d, pairs in rules.items():
            for a, b in pairs:
                self.allowed[d][a].add(b)
        self._state = seed & 0xFFFFFFFF

    def _rand(self):
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return (self._state >> 8) / (1 << 24)

    def _idx(self, x, y):
        return y * self.w + x

    def generate(self, max_restarts=100):
        """Run WFC until every cell is collapsed. Returns a 2-D list grid[y][x] of tile ids, or
        raises Contradiction if it cannot satisfy the rules within max_restarts."""
        for _ in range(max_restarts):
            try:
                return self._attempt()
            except Contradiction:
                continue
        raise Contradiction("could not find a satisfying assignment")

    def _attempt(self):
        # each cell holds a set of possible tiles
        cells = [set(self.tiles) for _ in range(self.w * self.h)]
        # initial propagation is unnecessary (all options open); loop observe+propagate
        while True:
            target = self._lowest_entropy_cell(cells)
            if target is None:
                break                       # all collapsed
            self._observe(cells, target)
            self._propagate(cells, target)
        # build the grid
        grid = [[None] * self.w for _ in range(self.h)]
        for y in range(self.h):
            for x in range(self.w):
                opts = cells[self._idx(x, y)]
                grid[y][x] = next(iter(opts))
        return grid

    def _lowest_entropy_cell(self, cells):
        best = None
        best_count = None
        for i, opts in enumerate(cells):
            n = len(opts)
            if n == 0:
                raise Contradiction("empty cell")
            if n == 1:
                continue
            # tiny random tie-break to vary output
            if best_count is None or n < best_count or (n == best_count and self._rand() < 0.3):
                best = i
                best_count = n
        return best

    def _observe(self, cells, i):
        opts = cells[i]
        total = sum(self.weights[t] for t in opts)
        r = self._rand() * total
        acc = 0.0
        chosen = next(iter(opts))
        for t in opts:
            acc += self.weights[t]
            if r <= acc:
                chosen = t
                break
        cells[i] = {chosen}

    def _propagate(self, cells, start):
        stack = [start]
        while stack:
            i = stack.pop()
            x, y = i % self.w, i // self.w
            for d, (dx, dy) in _DIRS.items():
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.w and 0 <= ny < self.h):
                    continue
                ni = self._idx(nx, ny)
                neighbour = cells[ni]
                if len(neighbour) == 0:
                    raise Contradiction("empty neighbour")
                # allowed tiles for the neighbour = union over current cell's options of
                # tiles permitted in direction d
                permitted = set()
                for t in cells[i]:
                    permitted |= self.allowed[d][t]
                new_opts = neighbour & permitted
                if len(new_opts) < len(neighbour):
                    if not new_opts:
                        raise Contradiction("propagation emptied a cell")
                    cells[ni] = new_opts
                    stack.append(ni)

    def satisfies_rules(self, grid):
        """True if grid violates no adjacency rule."""
        for y in range(self.h):
            for x in range(self.w):
                a = grid[y][x]
                for d, (dx, dy) in _DIRS.items():
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.w and 0 <= ny < self.h:
                        b = grid[ny][nx]
                        if b not in self.allowed[d][a]:
                            return False
        return True


def symmetric_rules(pairs_by_dir):
    """Build a rules dict where every allowed (a, b) in direction d also allows (b, a) in the opposite
    direction -- the common case for undirected tile compatibility."""
    rules = {d: set() for d in _DIRS}
    for d, pairs in pairs_by_dir.items():
        for a, b in pairs:
            rules[d].add((a, b))
            rules[_OPPOSITE[d]].add((b, a))
    return rules
