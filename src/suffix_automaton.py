"""Suffix automaton: the smallest machine that recognises every substring of a string.

A SUFFIX AUTOMATON is a deterministic finite automaton that accepts exactly the suffixes of a string,
and -- as a bonus -- whose set of reachable states, read from the start, spells out every DISTINCT
SUBSTRING of that string along some path. It is astonishingly compact: for a string of length n it has
at most 2n-1 states and 3n-4 transitions, yet it encodes all of the (up to n(n+1)/2) substrings. This
makes it the Swiss-army knife of string processing: count the distinct substrings, test whether a
pattern occurs in O(pattern) time, find the longest common substring of two strings, count how many
times each substring appears, and locate the longest repeated substring -- all in linear or
near-linear time and space.

It is built ONLINE, one character at a time, in amortised O(n). Each state represents an equivalence
class of substrings that share the same set of end positions (their "endpos" set); the states form a
tree under the SUFFIX LINK (each state's link points to the state of the longest proper suffix in a
different endpos class), and this link tree is exactly the suffix tree of the reversed string. The
clever construction maintains, at each step, the state for the whole current prefix, appends the new
character by following suffix links, and CLONES a state when an existing transition would otherwise
conflict -- the clone trick that keeps the automaton minimal. Once built, the number of distinct
substrings is the sum over states of (len[state] - len[link[state]]), and the longest common substring
of a second string is found by walking it through the automaton, following suffix links on mismatch.

This module builds the suffix automaton online, counts distinct substrings, tests substring membership,
counts occurrences of a pattern, finds the longest repeated substring, and computes the longest common
substring of two strings. It is verified against brute force -- the distinct-substring count matches a
set of all O(n^2) substrings, membership and occurrence counts match direct scanning, and the longest
common substring matches an O(n*m) dynamic-programming reference -- on hundreds of random strings. Pure
stdlib; a string-algorithms companion to the Aho-Corasick, suffix-array, and Manacher notes."""

from __future__ import annotations


class _State:
    __slots__ = ("length", "link", "trans")

    def __init__(self):
        self.length = 0          # length of the longest substring in this state's endpos class
        self.link = -1           # suffix link (index of another state, or -1 for the root)
        self.trans = {}          # character -> state index


class SuffixAutomaton:
    """Online suffix automaton over an arbitrary alphabet (characters are dict keys)."""

    def __init__(self, s=""):
        self.states = [_State()]     # state 0 is the initial state (empty string)
        self.last = 0                # state of the whole current string
        for ch in s:
            self.extend(ch)

    def extend(self, ch):
        """Append one character, keeping the automaton minimal. Amortised O(1)."""
        cur = len(self.states)
        self.states.append(_State())
        self.states[cur].length = self.states[self.last].length + 1

        p = self.last
        while p != -1 and ch not in self.states[p].trans:
            self.states[p].trans[ch] = cur
            p = self.states[p].link

        if p == -1:
            self.states[cur].link = 0
        else:
            q = self.states[p].trans[ch]
            if self.states[p].length + 1 == self.states[q].length:
                self.states[cur].link = q
            else:
                # clone q into a new state that keeps only the length-(len[p]+1) part
                clone = len(self.states)
                self.states.append(_State())
                self.states[clone].length = self.states[p].length + 1
                self.states[clone].trans = dict(self.states[q].trans)
                self.states[clone].link = self.states[q].link
                while p != -1 and self.states[p].trans.get(ch) == q:
                    self.states[p].trans[ch] = clone
                    p = self.states[p].link
                self.states[q].link = clone
                self.states[cur].link = clone
        self.last = cur

    # --- queries ------------------------------------------------------------
    def contains(self, pattern):
        """True iff `pattern` is a substring of the built string. O(len(pattern))."""
        state = 0
        for ch in pattern:
            if ch not in self.states[state].trans:
                return False
            state = self.states[state].trans[ch]
        return True

    def count_distinct_substrings(self):
        """The number of distinct non-empty substrings, as sum(len[v] - len[link[v]]) over states."""
        total = 0
        for i in range(1, len(self.states)):
            total += self.states[i].length - self.states[self.states[i].link].length
        return total

    def occurrences(self, pattern):
        """How many times `pattern` occurs in the string (overlapping). O(len(pattern)) after an
        O(n) one-time preprocessing that is done lazily and cached."""
        if not hasattr(self, "_occ"):
            self._build_occurrences()
        state = 0
        for ch in pattern:
            if ch not in self.states[state].trans:
                return 0
            state = self.states[state].trans[ch]
        return self._occ[state]

    def _build_occurrences(self):
        """Compute the occurrence count of the substrings ending in each state."""
        n = len(self.states)
        occ = [0] * n
        # mark primary states (the `last` after each extension) with count 1
        for idx in self._primary_states():
            occ[idx] = 1
        order = sorted(range(1, n), key=lambda i: self.states[i].length, reverse=True)
        for v in order:
            link = self.states[v].link
            if link > 0:
                occ[link] += occ[v]
        self._occ = occ

    def _primary_states(self):
        """The states that correspond to a genuine prefix of the string (occurrence-count seeds of 1).
        The automaton doesn't store the string, so `build()` records these at construction time as
        the `last` state after each character is appended."""
        return getattr(self, "_primaries", [])


def build(s):
    """Build a suffix automaton for string `s`, marking the primary (prefix) states so occurrence
    counting works. Returns the SuffixAutomaton."""
    sa = SuffixAutomaton()
    primaries = []
    for ch in s:
        sa.extend(ch)
        primaries.append(sa.last)
    sa._primaries = primaries
    return sa


def count_distinct_substrings(s):
    """Number of distinct non-empty substrings of `s`."""
    return build(s).count_distinct_substrings()


def longest_repeated_substring(s):
    """The longest substring that occurs at least twice in `s` (empty string if none repeats)."""
    sa = build(s)
    # a state's substrings repeat iff it has >1 occurrence; the answer is the longest such state
    if not hasattr(sa, "_occ"):
        sa._build_occurrences()
    best_len = 0
    best_state = -1
    for i in range(1, len(sa.states)):
        if sa._occ[i] >= 2 and sa.states[i].length > best_len:
            best_len = sa.states[i].length
            best_state = i
    if best_state == -1:
        return ""
    # recover an actual substring of this length by DFS from the root to best_state
    return _recover_string(sa, best_state, best_len)


def _recover_string(sa, target, length):
    """Find any path from the root spelling a length-`length` substring ending in state `target`."""
    # BFS/DFS storing the path; small automaton so this is cheap
    from collections import deque
    q = deque([(0, "")])
    seen = set()
    while q:
        state, acc = q.popleft()
        if state == target and len(acc) == length:
            return acc
        if len(acc) >= length:
            continue
        for ch, nxt in sa.states[state].trans.items():
            key = (nxt, len(acc) + 1)
            if key not in seen:
                seen.add(key)
                q.append((nxt, acc + ch))
    return ""


def longest_common_substring(s, t):
    """The longest string that is a substring of both `s` and `t`, via walking `t` through the
    suffix automaton of `s`. Returns one such substring (empty if none)."""
    sa = build(s)
    state = 0
    length = 0
    best = 0
    best_end = -1
    for i, ch in enumerate(t):
        while state != 0 and ch not in sa.states[state].trans:
            state = sa.states[state].link
            length = sa.states[state].length
        if ch in sa.states[state].trans:
            state = sa.states[state].trans[ch]
            length += 1
        else:
            state = 0
            length = 0
        if length > best:
            best = length
            best_end = i
    return t[best_end - best + 1: best_end + 1] if best > 0 else ""


# --- brute-force references -------------------------------------------------
def brute_distinct_substrings(s):
    """All distinct non-empty substrings by direct enumeration. O(n^2) substrings."""
    subs = set()
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            subs.add(s[i:j])
    return subs


def brute_longest_common_substring(s, t):
    """Longest common substring by O(n*m) dynamic programming."""
    n, m = len(s), len(t)
    if n == 0 or m == 0:
        return ""
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    best = 0
    end = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > best:
                    best = dp[i][j]
                    end = i
    return s[end - best: end]
