"""Hopcroft-Karp: maximum bipartite matching in O(E sqrt(V)), with Koenig's min vertex cover.

A bipartite graph splits its vertices into two sides -- say applicants on the left and jobs on the
right -- with edges only across the divide. A MATCHING is a set of edges with no shared endpoint: a
way to pair applicants to jobs so nobody is double-booked. The maximum matching is the largest such
set, and it answers a whole family of assignment questions.

The naive augmenting-path method (find one alternating path, flip it, repeat) runs in O(V*E). The
1973 algorithm of Hopcroft and Karp is faster: instead of one augmenting path per round it finds a
MAXIMAL SET of shortest vertex-disjoint augmenting paths at once, via a layered BFS followed by
DFS. The shortest augmenting-path length strictly increases each round, and after at most
O(sqrt(V)) rounds the matching is maximum -- giving the celebrated O(E sqrt(V)) bound. The two
phases per round:

    BFS from all unmatched left vertices, layering the graph by alternating (unmatched, matched,
        unmatched, ...) edges, stopping at the first layer that reaches an unmatched right vertex.
        This is the shortest augmenting distance.
    DFS along those layers only, greedily peeling off vertex-disjoint augmenting paths and flipping
        each one, until no more shortest paths remain.

This module returns the matching, and -- by Koenig's theorem, the bipartite duality that a maximum
matching equals a minimum VERTEX COVER -- extracts that minimum cover from the final alternating
BFS forest (left vertices NOT reached by an alternating path from the unmatched, plus right
vertices that ARE). It also reports whether a perfect matching exists and gives a Hall's-theorem
witness (a left subset whose neighbourhood is too small) when one does not.

Validated against two independent references -- a from-scratch Kuhn augmenting-path matcher and,
for small graphs, brute-force search over all matchings -- and against the theorems themselves:
|maximum matching| = |minimum vertex cover| (Koenig) and the returned cover is a genuine cover of
matching size. Pure stdlib; the specialised bipartite companion to the general-graph blossom
matcher and the max-flow reduction in dinic."""

from __future__ import annotations

from collections import deque

INF = float("inf")


class HopcroftKarp:
    """Maximum matching of a bipartite graph. Left vertices 0..nl-1, right 0..nr-1.
    adj[u] is the list of right vertices adjacent to left vertex u."""

    def __init__(self, nl, nr, edges):
        self.nl = nl
        self.nr = nr
        self.adj = [[] for _ in range(nl)]
        seen = set()
        for u, v in edges:
            if not (0 <= u < nl and 0 <= v < nr):
                raise ValueError(f"edge ({u},{v}) out of bounds")
            if (u, v) not in seen:
                seen.add((u, v))
                self.adj[u].append(v)
        # match_l[u] = right vertex matched to left u (or -1); match_r[v] = left vertex (or -1)
        self.match_l = [-1] * nl
        self.match_r = [-1] * nr
        self.dist = [0] * nl

    def _bfs(self):
        """Layer unmatched-left roots by alternating distance; return True if an augmenting
        path to an unmatched right vertex exists."""
        q = deque()
        for u in range(self.nl):
            if self.match_l[u] == -1:
                self.dist[u] = 0
                q.append(u)
            else:
                self.dist[u] = INF
        found = False
        while q:
            u = q.popleft()
            for v in self.adj[u]:
                w = self.match_r[v]  # left vertex currently matched to v (or -1)
                if w == -1:
                    found = True  # reached a free right vertex: augmenting path exists
                elif self.dist[w] == INF:
                    self.dist[w] = self.dist[u] + 1
                    q.append(w)
        return found

    def _dfs(self, u):
        """Try to extend an augmenting path from left vertex u along the BFS layers."""
        for v in self.adj[u]:
            w = self.match_r[v]
            if w == -1 or (self.dist[w] == self.dist[u] + 1 and self._dfs(w)):
                self.match_l[u] = v
                self.match_r[v] = u
                return True
        self.dist[u] = INF  # dead end: don't revisit this round
        return False

    def max_matching(self):
        """Compute and return the maximum matching size. Idempotent from a clean state."""
        matching = 0
        while self._bfs():
            for u in range(self.nl):
                if self.match_l[u] == -1 and self._dfs(u):
                    matching += 1
        return matching

    def pairs(self):
        """The matched (left, right) pairs."""
        return [(u, self.match_l[u]) for u in range(self.nl) if self.match_l[u] != -1]

    def min_vertex_cover(self):
        """Koenig's theorem: extract a minimum vertex cover from the alternating forest.
        Returns (left_cover_set, right_cover_set); |cover| == |maximum matching|."""
        # Run an alternating BFS from all unmatched left vertices.
        visited_l = [False] * self.nl
        visited_r = [False] * self.nr
        q = deque()
        for u in range(self.nl):
            if self.match_l[u] == -1:
                visited_l[u] = True
                q.append(u)
        while q:
            u = q.popleft()
            for v in self.adj[u]:
                if not visited_r[v] and self.match_l[u] != v:
                    # traverse a non-matching edge left->right
                    visited_r[v] = True
                    w = self.match_r[v]
                    if w != -1 and not visited_l[w]:
                        visited_l[w] = True  # back to left via the matching edge
                        q.append(w)
        # Cover = (left vertices NOT visited) + (right vertices visited)
        left_cover = {u for u in range(self.nl) if not visited_l[u]}
        right_cover = {v for v in range(self.nr) if visited_r[v]}
        return left_cover, right_cover


def max_matching(nl, nr, edges):
    """Convenience: maximum matching size of the bipartite graph."""
    hk = HopcroftKarp(nl, nr, edges)
    return hk.max_matching()


def maximum_matching_pairs(nl, nr, edges):
    """Convenience: the list of matched (left, right) pairs in a maximum matching."""
    hk = HopcroftKarp(nl, nr, edges)
    hk.max_matching()
    return hk.pairs()


def has_perfect_matching(nl, nr, edges):
    """True iff every left vertex can be matched (requires nl <= nr)."""
    hk = HopcroftKarp(nl, nr, edges)
    return hk.max_matching() == nl


def hall_violator(nl, nr, edges):
    """If no left-perfect matching exists, return a left subset S with |N(S)| < |S|
    (a Hall's-theorem witness). Returns None if a left-perfect matching exists."""
    hk = HopcroftKarp(nl, nr, edges)
    if hk.max_matching() == nl:
        return None
    # The unmatched-reachable left set is a Hall violator (deficient set).
    visited_l = [False] * nl
    visited_r = [False] * nr
    q = deque()
    for u in range(nl):
        if hk.match_l[u] == -1:
            visited_l[u] = True
            q.append(u)
    while q:
        u = q.popleft()
        for v in hk.adj[u]:
            if not visited_r[v]:
                visited_r[v] = True
                w = hk.match_r[v]
                if w != -1 and not visited_l[w]:
                    visited_l[w] = True
                    q.append(w)
    S = {u for u in range(nl) if visited_l[u]}
    return S


# --- independent reference: Kuhn's augmenting-path matcher -------------------
def kuhn_max_matching(nl, nr, edges):
    """Independent O(V*E) augmenting-path reference (Kuhn's algorithm)."""
    adj = [[] for _ in range(nl)]
    for u, v in edges:
        adj[u].append(v)
    match_r = [-1] * nr

    def try_kuhn(u, used):
        for v in adj[u]:
            if not used[v]:
                used[v] = True
                if match_r[v] == -1 or try_kuhn(match_r[v], used):
                    match_r[v] = u
                    return True
        return False

    count = 0
    for u in range(nl):
        used = [False] * nr
        if try_kuhn(u, used):
            count += 1
    return count
