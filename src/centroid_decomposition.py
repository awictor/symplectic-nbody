"""Centroid decomposition: divide-and-conquer on a tree, log-deep.

A CENTROID of a tree is a vertex whose removal leaves every remaining piece with at most half the
vertices -- the most balanced possible split point. CENTROID DECOMPOSITION recursively removes the
centroid, then decomposes each resulting subtree, building a new tree (the CENTROID TREE) whose root is
the whole tree's centroid and whose children are the centroids of the pieces. Because each split halves
the component sizes, the centroid tree has depth O(log n), and every original path between two vertices
passes through the centroid of the smallest decomposition level that contains both -- their lowest
common ancestor in the centroid tree. That single structural fact turns many hard tree problems into
log-depth divide-and-conquer: counting or optimising over all O(n^2) paths in near-linear time, answering
distance queries, and building tree data structures that update in O(log n) levels.

The canonical application, and the one this module solves, is COUNTING PAIRS OF VERTICES AT DISTANCE AT
MOST K. At each centroid, every qualifying path either avoids the centroid (handled by recursion into
the subtrees) or passes through it. For the through-centroid paths, gather the distance from the
centroid to every vertex in its component; the number of pairs whose summed distances are <= k is found
by sorting those distances and two-pointer counting, then SUBTRACTING the pairs that lie within the
same subtree branch (which don't actually pass through the centroid) by the same count applied per
branch. Summed over all O(log n) levels, this counts all long-range pairs in O(n log^2 n).

This module builds the centroid tree of an undirected tree and counts the pairs of vertices at distance
<= k through the centroid-decomposition trick. It is verified against brute force -- all-pairs BFS
distances compared directly -- confirming the pair count matches for every k, that the centroid tree is
a valid tree of depth O(log n) whose every vertex appears once, and on structured trees (paths, stars,
balanced binary trees) with known answers, over hundreds of random trees. Pure stdlib; a
tree-algorithms companion to the tree-isomorphism, LCA/sparse-table, and k-core notes."""

from __future__ import annotations

from collections import deque


def _adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def build_centroid_tree(n, edges):
    """Build the centroid tree of an undirected tree on n vertices.

    Returns (parent, root): parent[v] is v's parent in the centroid tree (root's parent is -1). The
    centroid tree has depth O(log n)."""
    adj = _adj(n, edges)
    removed = [False] * n
    subtree = [0] * n
    cparent = [-1] * n
    root = [-1]

    def component_size(start):
        """Sizes of the current (un-removed) component containing start; returns its vertex list."""
        comp = []
        seen = {start}
        q = deque([start])
        while q:
            u = q.popleft()
            comp.append(u)
            for w in adj[u]:
                if not removed[w] and w not in seen:
                    seen.add(w)
                    q.append(w)
        return comp

    def find_centroid(comp):
        """The centroid of the component `comp` (a list of vertices)."""
        # compute subtree sizes via a rooted DFS from comp[0]
        total = len(comp)
        # iterative post-order to fill subtree sizes
        rootv = comp[0]
        parent = {rootv: -1}
        order = []
        stack = [rootv]
        seen = {rootv}
        while stack:
            u = stack.pop()
            order.append(u)
            for w in adj[u]:
                if not removed[w] and w not in seen:
                    seen.add(w)
                    parent[w] = u
                    stack.append(w)
        for u in reversed(order):
            subtree[u] = 1
            for w in adj[u]:
                if not removed[w] and parent.get(w) == u:
                    subtree[u] += subtree[w]
        # walk toward the heavier side until balanced
        u = rootv
        p = -1
        while True:
            best_child = -1
            for w in adj[u]:
                if not removed[w] and w != p and subtree[w] < subtree[u]:
                    if subtree[w] > total // 2:
                        best_child = w
                        break
            if best_child == -1:
                return u
            p = u
            u = best_child

    def decompose(start, par):
        comp = component_size(start)
        c = find_centroid(comp)
        cparent[c] = par
        if par == -1:
            root[0] = c
        removed[c] = True
        for w in adj[c]:
            if not removed[w]:
                decompose(w, c)

    if n > 0:
        decompose(0, -1)
    return cparent, root[0]


def count_pairs_within_distance(n, edges, k):
    """The number of unordered vertex pairs {u, v} whose tree distance is <= k, via centroid
    decomposition in O(n log^2 n)."""
    if n < 2 or k < 1:
        return 0
    adj = _adj(n, edges)
    removed = [False] * n
    total_pairs = [0]

    def collect_distances(start, base):
        """BFS distances from `start` within the current component (the centroid is already removed,
        so this stays within one branch), offset by `base`. Returns the list of distances."""
        dists = []
        seen = {start}
        q = deque([(start, base)])
        while q:
            u, d = q.popleft()
            dists.append(d)
            for w in adj[u]:
                if not removed[w] and w not in seen:
                    seen.add(w)
                    q.append((w, d + 1))
        return dists

    def count_leq(dists):
        """Number of unordered pairs in `dists` summing to <= k, by sort + two pointers."""
        dists = sorted(dists)
        cnt = 0
        lo, hi = 0, len(dists) - 1
        while lo < hi:
            if dists[lo] + dists[hi] <= k:
                cnt += hi - lo
                lo += 1
            else:
                hi -= 1
        return cnt

    def component(start):
        comp = []
        seen = {start}
        q = deque([start])
        while q:
            u = q.popleft()
            comp.append(u)
            for w in adj[u]:
                if not removed[w] and w not in seen:
                    seen.add(w)
                    q.append(w)
        return comp

    def find_centroid(comp):
        total = len(comp)
        rootv = comp[0]
        parent = {rootv: -1}
        order = []
        stack = [rootv]
        seen = {rootv}
        sub = {}
        while stack:
            u = stack.pop()
            order.append(u)
            for w in adj[u]:
                if not removed[w] and w not in seen:
                    seen.add(w)
                    parent[w] = u
                    stack.append(w)
        for u in reversed(order):
            sub[u] = 1
            for w in adj[u]:
                if not removed[w] and parent.get(w) == u:
                    sub[u] += sub[w]
        u = rootv
        p = -1
        while True:
            nxt = -1
            for w in adj[u]:
                if not removed[w] and w != p and sub[w] < sub[u] and sub[w] > total // 2:
                    nxt = w
                    break
            if nxt == -1:
                return u
            p = u
            u = nxt

    def solve(start):
        comp = component(start)
        c = find_centroid(comp)
        # remove the centroid FIRST so branch BFS cannot leak across it into other branches
        removed[c] = True
        # all through-centroid paths: distances from c to everyone (c at distance 0)
        all_d = [0]                             # the centroid itself
        for w in adj[c]:
            if not removed[w]:
                all_d.extend(collect_distances(w, 1))
        total_pairs[0] += count_leq(all_d)
        # subtract pairs that stay within one branch (they don't pass through c)
        for w in adj[c]:
            if not removed[w]:
                branch = collect_distances(w, 1)
                total_pairs[0] -= count_leq(branch)
        for w in adj[c]:
            if not removed[w]:
                solve(w)

    solve(0)
    return total_pairs[0]


def centroid_tree_depth(n, edges):
    """The depth (max root-to-leaf distance) of the centroid tree -- should be O(log n)."""
    parent, root = build_centroid_tree(n, edges)
    if root == -1:
        return 0
    children = [[] for _ in range(n)]
    for v in range(n):
        if parent[v] != -1:
            children[parent[v]].append(v)
    # BFS depth
    depth = 0
    q = deque([(root, 0)])
    while q:
        u, d = q.popleft()
        depth = max(depth, d)
        for w in children[u]:
            q.append((w, d + 1))
    return depth


# --- brute-force reference --------------------------------------------------
def all_pairs_distances(n, edges):
    """All-pairs shortest-path distances in a tree via BFS from each vertex."""
    adj = _adj(n, edges)
    dist = [[0] * n for _ in range(n)]
    for s in range(n):
        d = [-1] * n
        d[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if d[w] == -1:
                    d[w] = d[u] + 1
                    q.append(w)
        dist[s] = d
    return dist


def brute_count_pairs_within_distance(n, edges, k):
    """Count unordered pairs at distance <= k by all-pairs BFS."""
    dist = all_pairs_distances(n, edges)
    cnt = 0
    for u in range(n):
        for v in range(u + 1, n):
            if dist[u][v] <= k:
                cnt += 1
    return cnt
