"""Aho-Corasick: finding many patterns in one pass.

Boyer-Moore finds one pattern fast; but a spam filter, virus scanner, or DNA motif search needs
to find HUNDREDS of patterns at once. Running a single-pattern search once per pattern costs
O(n * number_of_patterns). The Aho-Corasick automaton (1975) finds every occurrence of every
pattern in a single left-to-right scan, in O(n + total_pattern_length + number_of_matches) time
-- independent of how many patterns you search for.

It builds on a trie (prefix tree) of the patterns, then adds two kinds of links that turn the
trie into a finite-state machine:

  * FAILURE links: from each node, a pointer to the longest proper suffix of the path-so-far that
    is also a prefix of some pattern -- where to fall back when the next character does not extend
    the current match, so no input character is ever re-examined (the same idea as KMP, generalized
    to many patterns);

  * OUTPUT links: a chain that reports every pattern ending at the current state, so overlapping
    and nested matches (e.g. "he", "she", "his", "hers" all in "ushers") are all caught.

The automaton is built once by breadth-first traversal, then a single pass over the text reports
all matches. This module builds the trie, computes the failure and output links by BFS, and finds
all matches (with positions), and checks the results against a brute-force per-pattern search.
Pure stdlib; the multi-pattern-search companion to the Boyer-Moore note.
"""

from __future__ import annotations

from collections import deque


class _Node:
    __slots__ = ("children", "fail", "outputs", "depth")

    def __init__(self, depth=0):
        self.children = {}          # char -> _Node
        self.fail = None            # failure link
        self.outputs = []           # pattern indices ending at this node
        self.depth = depth


class AhoCorasick:
    """A multi-pattern string-matching automaton."""

    def __init__(self, patterns):
        self.patterns = [p for p in patterns]
        self.root = _Node()
        self._build_trie()
        self._build_links()

    def _build_trie(self):
        for idx, pat in enumerate(self.patterns):
            node = self.root
            for ch in pat:
                if ch not in node.children:
                    node.children[ch] = _Node(node.depth + 1)
                node = node.children[ch]
            if pat:                 # non-empty patterns terminate at a node
                node.outputs.append(idx)

    def _build_links(self):
        """BFS to set failure and output links (Aho-Corasick construction)."""
        q = deque()
        # depth-1 nodes fail to the root
        for child in self.root.children.values():
            child.fail = self.root
            q.append(child)
        while q:
            node = q.popleft()
            for ch, child in node.children.items():
                # follow failure links to find where `ch` continues a suffix
                f = node.fail
                while f is not None and ch not in f.children:
                    f = f.fail
                child.fail = f.children[ch] if (f and ch in f.children) else self.root
                # inherit the outputs reachable through the failure link
                child.outputs = child.outputs + child.fail.outputs
                q.append(child)

    def find_all(self, text):
        """Yield (end_index, pattern_index) for every occurrence of every pattern in text, where
        end_index is the position of the pattern's last character. Returns a list."""
        results = []
        node = self.root
        for i, ch in enumerate(text):
            while node is not None and ch not in node.children:
                node = node.fail
            node = node.children[ch] if (node and ch in node.children) else self.root
            for idx in node.outputs:
                results.append((i, idx))
        return results

    def search(self, text):
        """Return a dict {pattern: [start indices]} listing every match of each pattern in text.
        Overlapping and nested matches are all included."""
        out = {p: [] for p in self.patterns}
        for end_i, idx in self.find_all(text):
            pat = self.patterns[idx]
            start = end_i - len(pat) + 1
            out[pat].append(start)
        # dedupe/sort (a duplicate pattern in the input maps to the same key)
        for p in out:
            out[p] = sorted(set(out[p]))
        return out

    def count_matches(self, text) -> int:
        """Total number of pattern occurrences in text (counting overlaps)."""
        return len(self.find_all(text))

    def contains_any(self, text) -> bool:
        """True if any pattern occurs in text (a fast blocklist check)."""
        node = self.root
        for ch in text:
            while node is not None and ch not in node.children:
                node = node.fail
            node = node.children[ch] if (node and ch in node.children) else self.root
            if node.outputs:
                return True
        return False


# --- brute-force reference (for validation) --------------------------------

def brute_search(patterns, text):
    """Per-pattern naive search; returns {pattern: [start indices]} including overlaps."""
    out = {}
    for p in patterns:
        hits = []
        if p:
            start = 0
            while True:
                i = text.find(p, start)
                if i == -1:
                    break
                hits.append(i)
                start = i + 1       # allow overlapping matches
        out[p] = sorted(set(hits))
    return out
