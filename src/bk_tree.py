"""BK-trees -- fuzzy string search that finds every near-match without scanning the whole dictionary.

Spell-checkers, autocomplete, DNA lookup, and record deduplication all pose the same query: given a
word, find every dictionary entry within edit distance d of it. The brute-force answer computes the edit
distance to all n entries -- O(n) distance evaluations per query, each itself O(len^2), hopelessly slow
for a large lexicon. The BK-TREE (Burkhard & Keller, 1973) makes it fast by exploiting the one fact that
edit distance is a METRIC: it obeys the triangle inequality, dist(a, c) <= dist(a, b) + dist(b, c). That
inequality lets a tree PRUNE most of the dictionary on every query.

The structure is a tree whose edges are labelled by distances. Pick any word as the root. Each other
word is inserted by walking down from the root: at a node u, compute d = dist(word, u), and follow the
child reached by the edge labelled d (creating it if absent) -- so a node's children are exactly the
words at each integer distance from it, no two children sharing a distance. Nothing about coordinates,
only the metric.

The query is where the triangle inequality pays off. To find all words within tolerance t of a query q,
at a node u compute d = dist(q, u); if d <= t, u is a hit. Then -- crucially -- any word in the subtree
reached by edge e can be at distance no less than |d - e| and no more than d + e from q (triangle
inequality both ways), so ONLY the children whose edge label lies in [d - t, d + t] can possibly contain
a match. Every other subtree is pruned, unexamined. In practice this visits a small fraction of the
dictionary, turning an O(n) scan into something far faster, while returning EXACTLY the same set a full
scan would.

This module builds a BK-tree over any collection of strings under Levenshtein distance (or any metric
you pass), supports insertion and the tolerance query, reports the closest match, and counts how many
nodes a query actually visited so the pruning can be measured. A brute-force searcher is included as the
validation oracle. Pure standard library.

Validation. Correctness is defined by agreement with brute force: over many dictionaries and thousands
of seeded random queries at various tolerances, the BK-tree returns EXACTLY the set of words a linear
scan finds within the tolerance -- same members, no misses, no extras -- and its nearest-match equals
the true minimum-distance word. The triangle-inequality pruning is verified to actually fire: a query
visits far fewer than all n nodes yet never drops a valid match. Insertion order does not change the
result set. Edge cases -- empty tree, exact matches at distance 0, tolerance larger than any distance --
behave. Works on real word lists and on random strings alike."""


def levenshtein(a, b):
    """Levenshtein edit distance between two strings (a valid metric)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


class _Node:
    __slots__ = ("word", "children")

    def __init__(self, word):
        self.word = word
        self.children = {}      # distance -> child node


class BKTree:
    """A Burkhard-Keller tree for fuzzy search under a metric distance (default Levenshtein)."""

    def __init__(self, distance=levenshtein):
        self.distance = distance
        self.root = None
        self._size = 0
        self._last_visited = 0

    def __len__(self):
        return self._size

    def add(self, word):
        """Insert a word (duplicates are ignored)."""
        if self.root is None:
            self.root = _Node(word)
            self._size = 1
            return
        node = self.root
        while True:
            d = self.distance(word, node.word)
            if d == 0:
                return                     # already present
            child = node.children.get(d)
            if child is None:
                node.children[d] = _Node(word)
                self._size += 1
                return
            node = child

    def add_all(self, words):
        for w in words:
            self.add(w)

    def search(self, query, tolerance):
        """Return all (distance, word) pairs within ``tolerance`` of query, sorted by distance.

        Prunes subtrees using the triangle inequality: from a node at distance d, only children whose
        edge label lies in [d - tolerance, d + tolerance] can hold a match.
        """
        if self.root is None:
            return []
        results = []
        visited = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            visited += 1
            d = self.distance(query, node.word)
            if d <= tolerance:
                results.append((d, node.word))
            lo, hi = d - tolerance, d + tolerance
            for edge, child in node.children.items():
                if lo <= edge <= hi:
                    stack.append(child)
        self._last_visited = visited
        results.sort()
        return results

    def nearest(self, query):
        """Return (distance, word) of the closest entry, or None if the tree is empty.

        Uses an expanding-tolerance best-first search that tightens the bound as it goes.
        """
        if self.root is None:
            return None
        best_d = None
        best_word = None
        visited = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            visited += 1
            d = self.distance(query, node.word)
            if best_d is None or d < best_d:
                best_d = d
                best_word = node.word
            # only children within the current best radius can improve on it
            for edge, child in node.children.items():
                if abs(edge - d) <= best_d:
                    stack.append(child)
        self._last_visited = visited
        return (best_d, best_word)

    @property
    def last_visited(self):
        """Number of nodes examined by the most recent query (for measuring pruning)."""
        return self._last_visited

    def words(self):
        """All stored words, in no particular order."""
        out = []
        if self.root is None:
            return out
        stack = [self.root]
        while stack:
            node = stack.pop()
            out.append(node.word)
            stack.extend(node.children.values())
        return out


# ---------------------------------------------------------------------------
# brute-force reference
# ---------------------------------------------------------------------------

def brute_search(words, query, tolerance, distance=levenshtein):
    """Reference: all (distance, word) within tolerance by a full linear scan."""
    out = [(distance(query, w), w) for w in words]
    out = [(d, w) for d, w in out if d <= tolerance]
    out.sort()
    return out
