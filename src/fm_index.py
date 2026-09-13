"""FM-index: full-text substring search straight out of the compressed Burrows-Wheeler transform.

How do you count every occurrence of a pattern in a huge text in time proportional to the PATTERN,
not the text -- while storing essentially just a compressible permutation of the text? The FM-index
(Ferragina & Manzini, 2000) is the answer, and it powers the read aligners (Bowtie, BWA) that map
billions of DNA fragments to the human genome. It is built entirely on the Burrows-Wheeler transform
and two tiny primitives.

Take the BWT L of the text (the last column of the sorted rotation matrix). Two structures over L:

  C[c]  = the number of characters in the whole text strictly smaller than c. Because the first
          column F of the sorted rotations is just the sorted characters, C[c] is exactly where
          character c's block begins in F.
  Occ(c, i) = the number of occurrences of c in the prefix L[0:i] -- a RANK query on the BWT.

From these comes the LF-MAPPING, the heart of it all: the row whose last character is the k-th c in L
maps to row C[c] + (that rank) in the sorted order. LF walks the text BACKWARD one character per step.

BACKWARD SEARCH counts a pattern P without ever decompressing. Maintain a range [lo, hi) of BWT rows
whose sorted rotations start with the current suffix of P. Process P right-to-left; for each character
c, shrink the range by:

    lo <- C[c] + Occ(c, lo)
    hi <- C[c] + Occ(c, hi)

After consuming all of P, hi - lo is the number of occurrences -- computed in O(|P|) rank queries,
independent of the text length. To turn those rows into text POSITIONS, a sampled suffix array walks
each row back via LF until it hits a sampled row, adding the number of steps.

This module builds the index (BWT via the suffix array, the C table, and a checkpointed Occ so rank
is O(1) amortized), and implements count(pattern) and locate(pattern). It is validated against a
brute-force scan on random and structured (DNA, repetitive) texts: count matches the true number of
occurrences for every substring tested, locate returns exactly the correct sorted positions,
LF-mapping inverts the BWT back to the original text, and empty / absent / whole-text patterns behave.
Pure stdlib; the compressed-index companion to the suffix array, the Burrows-Wheeler transform, and
the wavelet tree's rank/select."""

from __future__ import annotations

from suffix_array import build_suffix_array

# sentinel smaller than any real character
_SENTINEL = "\x00"


class FMIndex:
    """An FM-index over a text, supporting count and locate in pattern-proportional time."""

    def __init__(self, text, sa_sample=4, occ_step=8):
        if _SENTINEL in text:
            raise ValueError("text must not contain the null sentinel byte")
        self.text = text
        self.n = len(text) + 1  # including the sentinel
        s = text + _SENTINEL

        # suffix array of the sentinel-terminated text
        sa = build_suffix_array(s)
        # build_suffix_array over s of length n returns a permutation of 0..n-1
        self.sa = sa

        # BWT: L[i] = s[sa[i] - 1]  (character preceding each suffix, wrapping)
        self.bwt = "".join(s[(sa[i] - 1) % self.n] for i in range(self.n))

        # C table: count of characters strictly less than c across the text
        self.alphabet = sorted(set(s))
        counts = {c: 0 for c in self.alphabet}
        for ch in s:
            counts[ch] += 1
        self.C = {}
        total = 0
        for c in self.alphabet:
            self.C[c] = total
            total += counts[c]

        # checkpointed Occ: occ_check[b][c] = count of c in bwt[0 : b*occ_step]
        self.occ_step = occ_step
        self._build_occ()

        # sampled suffix array: store sa value only where sa[i] % sa_sample == 0
        self.sa_sample = sa_sample
        self.sa_sampled = {}
        for i in range(self.n):
            if sa[i] % sa_sample == 0:
                self.sa_sampled[i] = sa[i]

    def _build_occ(self):
        step = self.occ_step
        n = self.n
        n_check = n // step + 1
        # checkpoints of cumulative counts
        self.occ_check = [dict.fromkeys(self.alphabet, 0)]
        running = dict.fromkeys(self.alphabet, 0)
        for i in range(n):
            running[self.bwt[i]] += 1
            if (i + 1) % step == 0:
                self.occ_check.append(dict(running))
        # ensure we have a final checkpoint reference count available
        self._n_check = len(self.occ_check)

    def _occ(self, c, i):
        """Number of occurrences of c in bwt[0:i]."""
        if c not in self.C:
            return 0
        if i <= 0:
            return 0
        step = self.occ_step
        chk = i // step
        base = self.occ_check[chk][c]
        # scan the remainder from the checkpoint to position i
        start = chk * step
        extra = 0
        for j in range(start, i):
            if self.bwt[j] == c:
                extra += 1
        return base + extra

    def count(self, pattern):
        """Number of occurrences of `pattern` in the text (O(|pattern|) rank queries)."""
        if pattern == "":
            return len(self.text)
        lo, hi = 0, self.n
        for c in reversed(pattern):
            if c not in self.C:
                return 0
            lo = self.C[c] + self._occ(c, lo)
            hi = self.C[c] + self._occ(c, hi)
            if lo >= hi:
                return 0
        return hi - lo

    def _range(self, pattern):
        lo, hi = 0, self.n
        for c in reversed(pattern):
            if c not in self.C:
                return (0, 0)
            lo = self.C[c] + self._occ(c, lo)
            hi = self.C[c] + self._occ(c, hi)
            if lo >= hi:
                return (0, 0)
        return (lo, hi)

    def _lf(self, i):
        """LF-mapping: from row i to the row of the preceding character."""
        c = self.bwt[i]
        return self.C[c] + self._occ(c, i)

    def _locate_row(self, i):
        """Text position of BWT row i, walking LF until a sampled row is hit."""
        steps = 0
        row = i
        while row not in self.sa_sampled:
            row = self._lf(row)
            steps += 1
        return (self.sa_sampled[row] + steps) % self.n

    def locate(self, pattern):
        """Sorted list of start positions where `pattern` occurs in the text."""
        if pattern == "":
            return list(range(len(self.text)))
        lo, hi = self._range(pattern)
        positions = [self._locate_row(i) for i in range(lo, hi)]
        return sorted(positions)

    def recover_text(self):
        """Reconstruct the original text from the BWT via repeated LF-mapping (inverse BWT)."""
        # start at row 0 (the sentinel-terminated suffix) and walk LF, collecting characters
        row = 0
        chars = []
        for _ in range(self.n):
            chars.append(self.bwt[row])
            row = self._lf(row)
        # chars now spells s reversed (with sentinel); reverse and strip sentinel
        rev = "".join(reversed(chars))
        return rev.replace(_SENTINEL, "")


def brute_count(text, pattern):
    """Reference: count occurrences (overlapping) of pattern in text by scanning."""
    if pattern == "":
        return len(text)
    n, m = len(text), len(pattern)
    return sum(1 for i in range(n - m + 1) if text[i:i + m] == pattern)


def brute_locate(text, pattern):
    """Reference: sorted start positions of pattern in text."""
    if pattern == "":
        return list(range(len(text)))
    n, m = len(text), len(pattern)
    return [i for i in range(n - m + 1) if text[i:i + m] == pattern]
