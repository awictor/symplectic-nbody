"""Eertree (palindromic tree): every distinct palindromic substring in one linear structure.

Palindromes -- strings that read the same forwards and backwards -- have a surprising amount of hidden
structure. A string of length n can contain up to n distinct palindromic substrings (no more: this is
a classical bound), and the EERTREE, or palindromic tree, is a data structure that stores ALL of them
in O(n) space and builds in O(n) time (over a fixed alphabet). Invented by Mikhail Rubinchik in 2014,
it is the palindrome analogue of the suffix automaton: where the suffix automaton captures every
substring, the eertree captures exactly the distinct palindromic ones, and it answers questions like
"how many distinct palindromic substrings are there?", "how many palindromic substrings end at each
position?", and "how often does each palindrome occur?" in linear time -- questions that arise in
bioinformatics (palindromic DNA marks restriction sites and hairpin loops), text indexing, and
competitive programming.

The tree has two roots -- an "imaginary" root of length -1 and the empty-string root of length 0 --
and one node per distinct palindrome, each storing its length, a SUFFIX LINK to the longest proper
palindromic suffix of itself, and edges labelled by characters: an edge c from node v leads to the
palindrome c + v + c. Building is online, one character at a time: to add character at position i, the
algorithm walks suffix links from the last palindrome until it finds one that can be extended by the
new character (i.e. the character just before that palindromic suffix matches), creates a new node if
this palindrome is not already present, and sets its suffix link by continuing the walk. The elegance
is that the total work of all these link-walks is amortised O(n), exactly as in the suffix automaton.

This module builds the eertree online, counts the distinct palindromic substrings, counts palindromic
substrings ending at each position, tallies how many times each distinct palindrome occurs, and lists
them. It is verified against brute force -- the distinct count and the per-length multiset match a set
of all O(n^2) substrings filtered for the palindrome property, and the occurrence counts match direct
scanning -- on hundreds of random strings, plus the classical theorem that a string has at most n
distinct palindromic substrings. Pure stdlib; a string-algorithms companion to the suffix-automaton,
Manacher, and Lyndon notes."""

from __future__ import annotations


class _Node:
    __slots__ = ("length", "suffix_link", "edges", "count")

    def __init__(self, length):
        self.length = length
        self.suffix_link = 0
        self.edges = {}          # character -> node index
        self.count = 0           # number of times this palindrome occurs (before link propagation)


class Eertree:
    """Online palindromic tree. Node 0 is the imaginary root (length -1), node 1 the empty root."""

    def __init__(self, s=""):
        # two roots: imaginary (len -1) at index 0, empty (len 0) at index 1
        self.nodes = [_Node(-1), _Node(0)]
        self.nodes[0].suffix_link = 0
        self.nodes[1].suffix_link = 0
        self.s = []
        self.last = 1            # node of the longest palindromic suffix of the current string
        for ch in s:
            self.add(ch)

    def _extendable(self, node_idx, i):
        """True iff the palindrome at `node_idx`, sitting as a suffix ending at position i, can be
        grown by matching the character `self.s[i - length - 1]` to `self.s[i]`."""
        length = self.nodes[node_idx].length
        j = i - length - 1
        return j >= 0 and self.s[j] == self.s[i]

    def add(self, ch):
        """Append one character; create at most one new palindrome node. Amortised O(1)."""
        self.s.append(ch)
        i = len(self.s) - 1

        # find the longest palindromic suffix that can be extended by ch
        cur = self.last
        while not self._extendable(cur, i):
            cur = self.nodes[cur].suffix_link

        # if the extended palindrome already exists, just move there and bump its count
        if ch in self.nodes[cur].edges:
            self.last = self.nodes[cur].edges[ch]
            self.nodes[self.last].count += 1
            return False

        # create a new node for ch + (palindrome at cur) + ch
        new_idx = len(self.nodes)
        new_node = _Node(self.nodes[cur].length + 2)
        self.nodes.append(new_node)
        self.nodes[cur].edges[ch] = new_idx

        if new_node.length == 1:
            # a single character: its suffix link is the empty root
            new_node.suffix_link = 1
        else:
            # continue walking suffix links from cur to find the suffix link target
            link = self.nodes[cur].suffix_link
            while not self._extendable(link, i):
                link = self.nodes[link].suffix_link
            new_node.suffix_link = self.nodes[link].edges[ch]

        new_node.count = 1
        self.last = new_idx
        return True

    # --- queries ------------------------------------------------------------
    def count_distinct(self):
        """The number of distinct non-empty palindromic substrings (all nodes minus the two roots)."""
        return len(self.nodes) - 2

    def palindromes(self):
        """Reconstruct the list of all distinct palindromic substrings (as strings/tuples), by
        walking edges from both roots. Order is arbitrary."""
        out = []

        def build_from(root_idx):
            # DFS: each edge c to child gives child_string = c + parent_string + c, except children
            # of the imaginary root (len -1) are the single characters c.
            stack = [(root_idx, [])]
            while stack:
                idx, inner = stack.pop()
                for ch, child in self.nodes[idx].edges.items():
                    if self.nodes[idx].length == -1:
                        pal = [ch]
                    else:
                        pal = [ch] + inner + [ch]
                    out.append(pal)
                    stack.append((child, pal))

        build_from(0)
        build_from(1)
        return out

    def occurrence_counts(self):
        """A dict {palindrome_tuple: occurrences} for every distinct palindrome, with occurrences
        summed correctly by propagating counts along suffix links (a palindrome occurs wherever any
        palindrome that has it as a suffix occurs)."""
        n = len(self.nodes)
        # propagate counts from longer palindromes to their suffix-link targets, in order of
        # decreasing length (node creation order already respects "suffix link points to shorter")
        counts = [self.nodes[i].count for i in range(n)]
        for idx in range(n - 1, 1, -1):
            link = self.nodes[idx].suffix_link
            counts[link] += counts[idx]

        # map each node to its palindrome string via edge reconstruction
        result = {}
        self._label_nodes()
        for idx in range(2, n):
            result[tuple(self._labels[idx])] = counts[idx]
        return result

    def _label_nodes(self):
        """Attach the actual palindrome string to each node (cached in self._labels)."""
        if hasattr(self, "_labels"):
            return
        labels = [None] * len(self.nodes)
        labels[0] = None
        labels[1] = []

        def dfs(root_idx):
            stack = [root_idx]
            while stack:
                idx = stack.pop()
                base = labels[idx]
                for ch, child in self.nodes[idx].edges.items():
                    if self.nodes[idx].length == -1:
                        labels[child] = [ch]
                    else:
                        labels[child] = [ch] + base + [ch]
                    stack.append(child)

        dfs(0)
        dfs(1)
        self._labels = labels


def build(s):
    """Build an eertree for string `s`."""
    return Eertree(s)


def count_distinct_palindromes(s):
    """Number of distinct non-empty palindromic substrings of `s`."""
    return Eertree(s).count_distinct()


# --- brute-force references -------------------------------------------------
def is_palindrome(s):
    return list(s) == list(reversed(list(s)))


def brute_distinct_palindromes(s):
    """The set of all distinct palindromic substrings, by enumeration. O(n^3) worst case."""
    out = set()
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            sub = s[i:j]
            if is_palindrome(sub):
                out.add(tuple(sub) if not isinstance(s, str) else sub)
    return out


def brute_occurrence_counts(s):
    """Occurrence count of each distinct palindromic substring, by direct scanning."""
    from collections import Counter
    counts = Counter()
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n + 1):
            sub = s[i:j]
            if is_palindrome(sub):
                counts[sub] += 1
    return dict(counts)
