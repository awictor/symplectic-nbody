"""Tries: the prefix tree behind autocomplete, spell-check, and IP routing.

A TRIE (from re-TRIE-val) stores a set of strings as a tree where each edge is a character and each
path from the root spells a prefix. Words sharing a prefix share the path for it, so the structure
is naturally compressed by common beginnings, and every operation runs in O(length of the key) --
INDEPENDENT of how many keys are stored, unlike a hash set whose collisions degrade or a balanced
tree's O(log n) comparisons of whole strings. That per-character walk is exactly what powers
autocomplete (find every word under a prefix), longest-prefix matching (IP routers), and dictionary
spell-check.

Each node holds a map from character to child plus a flag marking the end of a stored word. Insert
and lookup walk the characters; prefix search walks to the prefix node then collects every word in
its subtree; deletion unmarks the word and prunes now-childless, non-terminal nodes on the way back
up. Building a trie over all SUFFIXES of a text turns it into a substring index: any substring is a
prefix of some suffix, so substring search becomes a prefix walk.

This module implements a trie with insert, search, prefix membership, autocomplete (words by
prefix, optionally ranked by insertion frequency), deletion with pruning, and a suffix-trie
substring index -- verified that it stores and retrieves words, distinguishes a stored word from a
mere prefix, autocompletes exactly the words under a prefix, deletes without disturbing others,
counts distinct words, and detects substrings and their positions. Pure stdlib; a string
data-structure companion to the Aho-Corasick and Boyer-Moore notes."""

from __future__ import annotations


class _Node:
    __slots__ = ("children", "is_word", "count")

    def __init__(self):
        self.children = {}
        self.is_word = False
        self.count = 0          # how many times this exact word was inserted


class Trie:
    """A prefix tree of strings with O(len) insert, search, and prefix operations."""

    def __init__(self):
        self.root = _Node()
        self._size = 0          # number of distinct words

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, _Node())
        if not node.is_word:
            self._size += 1
        node.is_word = True
        node.count += 1

    def _find(self, prefix):
        node = self.root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def search(self, word):
        """True only if the exact word was inserted (not merely a prefix of one)."""
        node = self._find(word)
        return node is not None and node.is_word

    def starts_with(self, prefix):
        """True if any stored word begins with `prefix`."""
        return self._find(prefix) is not None

    def count(self, word):
        """How many times the exact word was inserted."""
        node = self._find(word)
        return node.count if node and node.is_word else 0

    def _collect(self, node, prefix, out):
        if node.is_word:
            out.append((prefix, node.count))
        for ch, child in sorted(node.children.items()):
            self._collect(child, prefix + ch, out)

    def autocomplete(self, prefix, limit=None, by_frequency=False):
        """Every stored word beginning with `prefix`. Sorted alphabetically, or by insertion
        frequency (then alphabetically) if by_frequency; capped at `limit` if given."""
        node = self._find(prefix)
        if node is None:
            return []
        out = []
        self._collect(node, prefix, out)
        if by_frequency:
            out.sort(key=lambda wc: (-wc[1], wc[0]))
        else:
            out.sort(key=lambda wc: wc[0])
        words = [w for w, _ in out]
        return words[:limit] if limit is not None else words

    def words(self):
        """All stored words, alphabetically."""
        return self.autocomplete("")

    def __len__(self):
        return self._size

    def __contains__(self, word):
        return self.search(word)

    def delete(self, word):
        """Remove a word; prune nodes that become childless and non-terminal. Returns True if the
        word was present."""
        path = [self.root]
        node = self.root
        for ch in word:
            node = node.children.get(ch)
            if node is None:
                return False
            path.append(node)
        if not node.is_word:
            return False
        node.is_word = False
        node.count = 0
        self._size -= 1
        # prune upward while nodes are non-terminal and childless
        for i in range(len(word) - 1, -1, -1):
            child = path[i + 1]
            if child.children or child.is_word:
                break
            del path[i].children[word[i]]
        return True

    def longest_prefix_of(self, text):
        """The longest stored word that is a prefix of `text` (IP-routing style). '' if none."""
        node = self.root
        best = ""
        cur = ""
        for ch in text:
            node = node.children.get(ch)
            if node is None:
                break
            cur += ch
            if node.is_word:
                best = cur
        return best


class SuffixTrie:
    """A substring index: a trie built over all suffixes of a text, so substring search is a prefix
    walk. Built in O(n^2) for a length-n text (fine for modest texts; suffix automata do it in
    O(n), a future refinement)."""

    def __init__(self, text):
        self.text = text
        self.trie = Trie()
        # insert every suffix; also record start positions per suffix for locate()
        self._starts = {}
        for i in range(len(text)):
            self.trie.insert(text[i:])
            self._starts.setdefault(text[i:], i)

    def contains(self, pattern):
        """True if `pattern` occurs anywhere in the text (it is a prefix of some suffix)."""
        if pattern == "":
            return True
        return self.trie.starts_with(pattern)

    def occurrences(self, pattern):
        """All start indices where `pattern` occurs, ascending (brute scan over the text)."""
        if pattern == "":
            return list(range(len(self.text) + 1))
        n, m = len(self.text), len(pattern)
        return [i for i in range(n - m + 1) if self.text[i:i + m] == pattern]
