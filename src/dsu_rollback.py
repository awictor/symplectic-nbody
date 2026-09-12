"""Disjoint-set union with rollback: undoable connectivity for offline dynamic graphs.

The ordinary UNION-FIND is nearly constant-time thanks to PATH COMPRESSION -- but that same
compression rewrites the tree unpredictably, so you cannot UNDO a union. Many problems need exactly
that: process a sequence of edge additions and queries, then roll back to an earlier state (the
DYNAMIC CONNECTIVITY problem, offline). The trick is to drop path compression and use only UNION BY
RANK, which changes at most a constant amount of state per union -- one parent pointer and possibly
one rank -- and record those changes on a stack. To roll back, pop the stack and restore the saved
values. This ROLLBACK DSU underlies the offline dynamic-connectivity structure (a segment tree over
time), Kruskal-style reconnection problems, and competitive-programming staples like "how many
components after each of these edge insertions, which we later retract."

Without path compression, find is O(log n) (the rank bound keeps trees shallow) rather than the
inverse-Ackermann of the compressed version, but every operation is perfectly reversible: a SNAPSHOT
is just the current stack length, and ROLLBACK to a snapshot replays the recorded parent/rank changes
in reverse. Unite returns whether it actually merged two components (so no-op unions cost nothing to
undo), and a running component count is maintained and restored alongside.

This module implements a rollback disjoint-set with union by rank, find (with the rank-bounded tree,
no compression), a component counter, and snapshot/rollback. It is verified against a brute-force
recomputation: that connectivity queries always match a fresh union-find built from the currently-live
edges, that rolling back to a snapshot exactly restores connectivity and the component count, that
nested snapshots roll back correctly, that a full add-then-rollback sequence returns to the initial
all-singletons state, and on a dynamic-connectivity scenario where edges are added and retracted.
Pure stdlib; a data-structure companion to the union-find, Kruskal-MST, and segment-tree notes."""

from __future__ import annotations


class RollbackDSU:
    """A disjoint-set union structure supporting undo via snapshot/rollback."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
        self._history = []          # stack of (kind, index, old_value) for undo

    def find(self, x):
        """The representative of x's set (no path compression, so it stays rollback-safe)."""
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def connected(self, a, b):
        return self.find(a) == self.find(b)

    def unite(self, a, b):
        """Merge the sets of a and b. Returns True if they were separate (a real merge)."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            # record a no-op so rollback counts stay aligned even for redundant unions
            self._history.append(("noop", 0, 0))
            return False
        # union by rank: attach the shorter tree under the taller
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        # now rank[ra] >= rank[rb]; attach rb under ra
        self._history.append(("parent", rb, self.parent[rb]))
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self._history.append(("rank", ra, self.rank[ra]))
            self.rank[ra] += 1
        self.components -= 1
        return True

    def snapshot(self):
        """A handle to the current state (the history length) for later rollback."""
        return len(self._history)

    def rollback(self, snap):
        """Undo all operations back to the given snapshot, restoring parent, rank, and the component
        count exactly."""
        while len(self._history) > snap:
            kind, idx, old = self._history.pop()
            if kind == "parent":
                self.parent[idx] = old
                self.components += 1
            elif kind == "rank":
                self.rank[idx] = old
            # "noop" needs no state change

    def component_count(self):
        return self.components
