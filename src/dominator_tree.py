"""Dominator tree of a control-flow graph: who must you pass through to reach a node.

In a directed graph with a designated ENTRY, a node d DOMINATES a node n if every path from the
entry to n goes through d. Domination is the backbone of compiler analysis: it defines where a
variable's definition is guaranteed live, where to place phi-functions when building SSA form, which
loops are natural (a back edge to a header that dominates its tail), and which code motions are
safe. Every node except the entry has a unique IMMEDIATE DOMINATOR -- the closest strict dominator
on every entry path -- and those immediate-dominator edges form a tree rooted at the entry: the
DOMINATOR TREE. Node d dominates n exactly when d is an ancestor of n in that tree.

The definitional test is expensive: d dominates n iff deleting d disconnects n from the entry, which
is a reachability computation per node pair. This module instead builds the whole tree with the
iterative data-flow algorithm of Cooper, Harvey, and Kennedy (2001) -- simple to state, and fast in
practice on real control-flow graphs:

    number the nodes in reverse postorder (so a node's predecessors tend to come first),
    idom[entry] = entry, all others undefined,
    repeat until nothing changes:
        for each node n != entry in reverse postorder:
            new_idom = the first already-processed predecessor
            for every other processed predecessor p:
                new_idom = INTERSECT(p, new_idom)
            idom[n] = new_idom

where INTERSECT walks the two nodes up the partially-built tree by reverse-postorder number until
they meet -- the running nearest-common-ancestor. Unreachable nodes (never numbered) are dropped:
they have no dominator relationship to the entry.

This module returns the immediate-dominator map and the dominator tree, and answers dominates(a, b)
by walking up the idom chain. Validated against the definition itself: for every reachable node n
and every candidate d, the tree's dominance verdict matches a brute-force check that removing d makes
n unreachable from the entry. Pure stdlib; the control-flow companion to the SCC and topological-sort
graph tools."""

from __future__ import annotations


def _reverse_postorder(n, succ, entry):
    """DFS reverse postorder from the entry. Returns (order, rpo_number, visited-set)."""
    order = []
    visited = [False] * n
    # iterative DFS emitting postorder
    stack = [(entry, 0)]
    visited[entry] = True
    while stack:
        u, i = stack.pop()
        if i < len(succ[u]):
            stack.append((u, i + 1))
            w = succ[u][i]
            if not visited[w]:
                visited[w] = True
                stack.append((w, 0))
        else:
            order.append(u)  # postorder: emit after children
    order.reverse()  # reverse postorder
    rpo = {node: idx for idx, node in enumerate(order)}
    return order, rpo, visited


class DominatorTree:
    """Dominator tree of a directed graph on nodes 0..n-1 with a given entry."""

    def __init__(self, n, edges, entry=0):
        if n <= 0:
            raise ValueError("n must be positive")
        if not (0 <= entry < n):
            raise ValueError("entry out of range")
        self.n = n
        self.entry = entry
        self.succ = [[] for _ in range(n)]
        self.pred = [[] for _ in range(n)]
        for u, v in edges:
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError(f"edge ({u},{v}) out of bounds")
            self.succ[u].append(v)
            self.pred[v].append(u)
        self.idom = self._compute_idom()

    def _compute_idom(self):
        order, rpo, visited = self._rpo()
        self.rpo = rpo
        self.reachable = visited
        idom = {self.entry: self.entry}

        def intersect(a, b):
            while a != b:
                while rpo[a] > rpo[b]:
                    a = idom[a]
                while rpo[b] > rpo[a]:
                    b = idom[b]
            return a

        changed = True
        while changed:
            changed = False
            for node in order:
                if node == self.entry:
                    continue
                new_idom = None
                for p in self.pred[node]:
                    if p not in rpo:
                        continue  # unreachable predecessor
                    if p in idom:
                        if new_idom is None:
                            new_idom = p
                        else:
                            new_idom = intersect(p, new_idom)
                if new_idom is not None and idom.get(node) != new_idom:
                    idom[node] = new_idom
                    changed = True
        # the entry's self-loop idom is a convention; keep it for chain walking
        return idom

    def _rpo(self):
        return _reverse_postorder(self.n, self.succ, self.entry)

    def immediate_dominators(self):
        """Map node -> immediate dominator (entry maps to itself). Only reachable nodes appear."""
        return dict(self.idom)

    def tree_edges(self):
        """The dominator tree as (idom, node) edges, excluding the entry self-edge."""
        return [(self.idom[v], v) for v in self.idom if v != self.entry]

    def dominates(self, a, b):
        """True if a dominates b (a lies on every entry->b path). a == b counts (reflexive)."""
        if b not in self.idom or a not in self.idom:
            return False
        x = b
        while True:
            if x == a:
                return True
            if x == self.entry:
                return a == self.entry
            x = self.idom[x]

    def dominators_of(self, b):
        """The full set of nodes that dominate b (including b and the entry)."""
        if b not in self.idom:
            return set()
        doms = set()
        x = b
        while True:
            doms.add(x)
            if x == self.entry:
                break
            x = self.idom[x]
        return doms


# --- brute-force reference ---------------------------------------------------
def _reachable_without(n, succ, entry, blocked):
    """Set of nodes reachable from entry without passing through `blocked` (blocked != entry)."""
    if blocked == entry:
        return set()
    seen = [False] * n
    seen[entry] = True
    stack = [entry]
    while stack:
        u = stack.pop()
        for w in succ[u]:
            if w != blocked and not seen[w]:
                seen[w] = True
                stack.append(w)
    return {i for i in range(n) if seen[i]}


def brute_dominates(n, edges, entry, d, target):
    """Definitional check: d dominates target iff removing d makes target unreachable from entry
    (and target is reachable in the first place). d == target dominates reflexively."""
    succ = [[] for _ in range(n)]
    for u, v in edges:
        succ[u].append(v)
    reach = _reachable_without(n, succ, entry, blocked=-1)
    if target not in reach:
        return None  # target unreachable: dominance undefined
    if d == target:
        return True
    if d == entry:
        return True  # entry dominates everything reachable
    without = _reachable_without(n, succ, entry, blocked=d)
    return target not in without
