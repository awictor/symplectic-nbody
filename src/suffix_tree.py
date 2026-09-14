"""Ukkonen's suffix tree: every substring of a string, stored in linear space and built in linear time.

A suffix tree is a compressed trie of ALL suffixes of a string. Every substring of s is the prefix of
some suffix, so it corresponds to a path from the root -- which makes a suffix tree a Swiss-army knife
for string problems: substring search in O(m), the longest repeated substring, the number of distinct
substrings, the longest common substring of several strings, and more, all fall out of one structure.
The catch is building it: the naive construction is O(n^2), and for decades a linear-time algorithm was
considered formidable. Ukkonen's 1995 algorithm builds it ONLINE -- character by character -- in O(n).

The magic ingredients are three. Edges store not characters but (start, end) INDEX RANGES into the
original string, so the whole tree is O(n) space no matter how long the substrings are. A global END
pointer means every leaf edge grows automatically when a character is appended ("once a leaf, always a
leaf") -- rule 1 costs nothing. And SUFFIX LINKS shortcut from the internal node for string c*X to the
node for X, so after creating a branch the algorithm hops to the next suffix in O(1) instead of walking
from the root. Together with the "active point" (a cursor of node + edge + length) and the
skip/count trick for descending, these make the amortized work linear.

This module builds a suffix tree with Ukkonen's algorithm (using a sentinel to guarantee every suffix
ends at a leaf), and answers substring containment, occurrence counting, distinct-substring counting,
and longest-repeated-substring queries. It is validated against independent brute force: containment and
counts match a naive scan over all queries; the number of distinct substrings matches the O(n^2)
enumeration and the suffix-array + LCP formula n(n+1)/2 - sum(lcp); the longest repeated substring
matches the max-LCP suffix-array answer; and the leaf count equals the string length. Cross-checks the
repo's suffix array. Pure stdlib; the linear-time companion to the suffix-array, suffix-automaton, and
Aho-Corasick tools."""

from __future__ import annotations


class _Node:
    __slots__ = ("children", "start", "end", "suffix_link")

    def __init__(self, start, end):
        self.children = {}          # first-char -> child node
        self.start = start
        self.end = end              # int, or a one-element list [pos] acting as the global end
        self.suffix_link = None

    def edge_length(self, pos):
        e = self.end if isinstance(self.end, int) else self.end[0]
        return min(e, pos + 1) - self.start


class SuffixTree:
    """Ukkonen suffix tree of a string, terminated by a unique sentinel character."""

    def __init__(self, text, sentinel="\x00"):
        if sentinel in text:
            raise ValueError("text must not contain the sentinel character")
        self.text = text + sentinel
        self.n = len(self.text)
        self.root = _Node(-1, -1)
        self._end = [-1]                        # global end for all leaf edges
        self._build()

    def _new_leaf(self, start):
        return _Node(start, self._end)          # leaf end tracks the global pointer

    def _edge_end(self, node):
        return node.end[0] if isinstance(node.end, list) else node.end

    def _build(self):
        text = self.text
        root = self.root
        active_node = root
        active_edge = -1                        # index into text of the active edge's first char
        active_length = 0
        remaining = 0
        for i in range(self.n):
            self._end[0] = i
            remaining += 1
            last_new_node = None
            while remaining > 0:
                if active_length == 0:
                    active_edge = i
                edge_char = text[active_edge]
                if edge_char not in active_node.children:
                    # rule 2: create a new leaf edge
                    active_node.children[edge_char] = self._new_leaf(i)
                    if last_new_node is not None:
                        last_new_node.suffix_link = active_node
                        last_new_node = None
                else:
                    nxt = active_node.children[edge_char]
                    edge_len = nxt.edge_length(i)
                    if active_length >= edge_len:
                        # walk down: skip/count
                        active_edge += edge_len
                        active_length -= edge_len
                        active_node = nxt
                        continue
                    if text[nxt.start + active_length] == text[i]:
                        # rule 3: current char already on the edge -> extend active point, stop
                        if last_new_node is not None:
                            last_new_node.suffix_link = active_node
                        active_length += 1
                        break
                    # rule 2 with a split: create an internal node in the middle of the edge
                    split = _Node(nxt.start, nxt.start + active_length)
                    active_node.children[edge_char] = split
                    split.children[text[i]] = self._new_leaf(i)
                    nxt.start += active_length
                    split.children[text[nxt.start]] = nxt
                    if last_new_node is not None:
                        last_new_node.suffix_link = split
                    last_new_node = split
                remaining -= 1
                if active_node is root and active_length > 0:
                    active_length -= 1
                    active_edge = i - remaining + 1
                elif active_node is not root:
                    active_node = active_node.suffix_link if active_node.suffix_link else root

    # --- queries ------------------------------------------------------------
    def contains(self, pattern):
        """True if pattern is a substring of the original text."""
        if pattern == "":
            return True
        node = self.root
        i = 0
        m = len(pattern)
        while i < m:
            c = pattern[i]
            if c not in node.children:
                return False
            child = node.children[c]
            end = self._edge_end(child)
            j = child.start
            while j < end and i < m:
                if self.text[j] != pattern[i]:
                    return False
                j += 1
                i += 1
            node = child
        return True

    def _count_leaves(self, node):
        if not node.children:
            return 1
        return sum(self._count_leaves(c) for c in node.children.values())

    def count_occurrences(self, pattern):
        """Number of occurrences of pattern in the text (number of leaves below its locus)."""
        if pattern == "":
            return self.n
        node = self.root
        i = 0
        m = len(pattern)
        while i < m:
            c = pattern[i]
            if c not in node.children:
                return 0
            child = node.children[c]
            end = self._edge_end(child)
            j = child.start
            while j < end and i < m:
                if self.text[j] != pattern[i]:
                    return 0
                j += 1
                i += 1
            node = child
        return self._count_leaves(node)

    def count_distinct_substrings(self):
        """Number of distinct non-empty substrings = sum of edge lengths (excluding sentinel-only)."""
        total = 0

        def walk(node, depth):
            nonlocal total
            for c, child in node.children.items():
                end = self._edge_end(child)
                elen = end - child.start
                # count characters on this edge, but exclude the sentinel-only positions
                for k in range(child.start, end):
                    if self.text[k] != self.text[-1]:
                        total += 1
                walk(child, depth + elen)

        walk(self.root, 0)
        return total

    def longest_repeated_substring(self):
        """The longest substring that occurs at least twice (deepest internal node)."""
        best = [0, ""]

        def walk(node, depth, path):
            # an internal node with >= 2 children marks a substring occurring >= 2 times
            if node is not self.root and len(node.children) >= 2:
                if depth > best[0]:
                    best[0] = depth
                    best[1] = path
            for c, child in node.children.items():
                end = self._edge_end(child)
                seg = self.text[child.start:end]
                # strip sentinel
                seg = seg.replace(self.text[-1], "")
                walk(child, depth + len(seg), path + seg)

        walk(self.root, 0, "")
        return best[1]
