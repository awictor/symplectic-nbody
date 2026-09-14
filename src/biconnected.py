"""Biconnected components: carving a graph into blocks that survive any single vertex failure.

A graph is BICONNECTED if it stays connected after removing any one vertex -- there are at least two
vertex-disjoint paths between every pair, so no single point of failure. Most graphs are not biconnected
as a whole, but they decompose uniquely into maximal biconnected pieces called BLOCKS, glued together
at the ARTICULATION POINTS (cut vertices). This block decomposition is the structural map of a network's
robustness: within a block you can lose any node and stay connected; the articulation points are the
fragile joints where the whole thing can split.

Hopcroft and Tarjan's algorithm finds the blocks in a single DFS. As it explores, it pushes each visited
EDGE onto a stack and tracks disc[v] (discovery time) and low[v] (the earliest ancestor reachable by a
back-edge from v's subtree). When it retreats over a tree edge (u, v) with low[v] >= disc[u], the vertex
u is an articulation point and everything pushed on the edge stack since (u, v) forms one complete block
-- so it pops those edges off as a biconnected component. Because the partition is by EDGES, an
articulation point belongs to several blocks at once (that is exactly what makes it a cut vertex), while
every non-cut vertex lives in a single block.

This module builds the biconnected (edge) decomposition and the articulation points by one iterative
DFS, and constructs the BLOCK-CUT TREE. It is validated: the blocks partition the graph's edges exactly
(every edge in exactly one block); a vertex is an articulation point if and only if it appears in two or
more blocks (cross-checked against a brute-force "does deleting it disconnect the graph" test); a single
cycle is one block with no articulation points; a tree has every edge as its own block and every
internal vertex as a cut vertex; two cycles sharing one vertex give two blocks meeting at that cut
vertex; and the block-cut tree is acyclic with the right node count. Pure stdlib; the connectivity
companion to the bridges, SCC, and union-find tools."""

from __future__ import annotations


def biconnected_components(n, edges):
    """Biconnected components of an undirected graph as lists of edges. Returns (blocks, articulation).

    n vertices 0..n-1; edges a list of (u, v). Parallel edges/self-loops are ignored for simplicity.
    blocks: list of edge-lists; articulation: set of cut-vertex indices."""
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        if u == v:
            continue
        adj[u].append(v)
        adj[v].append(u)

    disc = [-1] * n
    low = [0] * n
    timer = [0]
    blocks = []
    articulation = set()
    edge_stack = []

    def dfs(root):
        # iterative DFS; stack holds (vertex, parent, iterator index, children count)
        stack = [(root, -1, 0, 0)]
        disc[root] = low[root] = timer[0]
        timer[0] += 1
        root_children = 0
        while stack:
            u, parent, idx, _ = stack[-1]
            if idx < len(adj[u]):
                stack[-1] = (u, parent, idx + 1, 0)
                w = adj[u][idx]
                if disc[w] == -1:
                    edge_stack.append((u, w))
                    disc[w] = low[w] = timer[0]
                    timer[0] += 1
                    if u == root:
                        root_children += 1
                    stack.append((w, u, 0, 0))
                elif w != parent and disc[w] < disc[u]:
                    # back edge
                    edge_stack.append((u, w))
                    low[u] = min(low[u], disc[w])
            else:
                stack.pop()
                if stack:
                    pu = stack[-1][0]
                    pu_parent = stack[-1][1]
                    low[pu] = min(low[pu], low[u])
                    # low[u] >= disc[pu] closes a block at (pu, u); pu is a cut vertex unless it is
                    # the root (the root's articulation status is decided by its child count below).
                    if low[u] >= disc[pu]:
                        if pu_parent != -1:
                            articulation.add(pu)
                        _pop_block(edge_stack, pu, u, blocks)
        # root is an articulation point iff it has >= 2 DFS children
        if root_children >= 2:
            articulation.add(root)

    def _pop_block(estack, u, v, blocks_out):
        comp = []
        while estack:
            e = estack.pop()
            comp.append(e)
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                break
        if comp:
            blocks_out.append(comp)

    for s in range(n):
        if disc[s] == -1:
            dfs(s)
            # any edges left on the stack after this DFS tree form the last block(s) at the root
            if edge_stack:
                comp = []
                while edge_stack:
                    comp.append(edge_stack.pop())
                blocks.append(comp)
    return blocks, articulation


def block_cut_tree(n, edges):
    """The block-cut tree: nodes are blocks and articulation points; edges join a cut vertex to each
    block containing it. Returns (block_nodes, cut_nodes, tree_edges)."""
    blocks, articulation = biconnected_components(n, edges)
    block_nodes = list(range(len(blocks)))          # block i -> node ('B', i)
    cut_nodes = sorted(articulation)
    tree_edges = []
    for bi, block in enumerate(blocks):
        verts = set()
        for (u, v) in block:
            verts.add(u)
            verts.add(v)
        for a in verts:
            if a in articulation:
                tree_edges.append((("B", bi), ("C", a)))
    return block_nodes, cut_nodes, tree_edges


def _brute_articulation(n, edges):
    """Reference: a vertex is an articulation point if deleting it increases the component count."""
    def components(removed):
        adj = [[] for _ in range(n)]
        for (u, v) in edges:
            if u == removed or v == removed or u == v:
                continue
            adj[u].append(v)
            adj[v].append(u)
        seen = [False] * n
        count = 0
        for s in range(n):
            if s == removed or seen[s]:
                continue
            count += 1
            stack = [s]
            seen[s] = True
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if not seen[y]:
                        seen[y] = True
                        stack.append(y)
        return count

    base = components(-1)
    arts = set()
    for r in range(n):
        # deleting r: base counted r as its own component if isolated; compare on the rest
        # components ignoring r among the other n-1 vertices
        if components(r) - (0 if _isolated(n, edges, r) else 0) > base:
            arts.add(r)
    return arts


def _isolated(n, edges, r):
    return all(u != r and v != r for (u, v) in edges)
