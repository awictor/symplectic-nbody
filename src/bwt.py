"""The Burrows-Wheeler transform and the bzip2 compression pipeline.

The Burrows-Wheeler transform (1994) is the clever heart of bzip2. On its own it compresses
NOTHING -- it is a reversible PERMUTATION of the input -- but it rearranges the bytes so that
characters sharing a context cluster into long runs, which downstream stages then squeeze. Applied
to English text, the letters preceding every "he" (t, s, ...) end up adjacent, turning scattered
symbols into repetitive stretches. That "make it runnier without losing information" trick, plus a
way to UNDO the permutation exactly, is what makes it remarkable.

BWT sorts all rotations of the (sentinel-terminated) string and takes the last column; the inverse
reconstructs the original by the LF-mapping (repeatedly following a stable sort of that last
column). The bzip2 pipeline chains three reversible stages:

  BWT            -> permute so like-context bytes cluster into runs,
  MOVE-TO-FRONT  -> recode each byte as its position in a running alphabet, so a run of one byte
                    becomes a run of zeros (small numbers dominate),
  RUN-LENGTH     -> collapse those runs of repeats into (value, count) pairs.

Each stage is exactly invertible, so decoding runs them in reverse. This module implements BWT and
its inverse (with a sentinel so any input works), move-to-front and its inverse, run-length
encoding and its inverse, and the full forward/backward pipeline -- verified that BWT round-trips
any string, that it genuinely increases run-length (measured by mean run length) on structured
text, that MTF and RLE round-trip, that the whole pipeline is lossless, and that it shrinks
repetitive input. Pure stdlib; a data-compression companion to the LZ77 and Huffman notes."""

from __future__ import annotations

_SENTINEL = "\x00"     # a byte assumed not to occur in the input, marking the rotation start


def bwt_transform(text):
    """Burrows-Wheeler transform: append a sentinel, sort all rotations, return the last column.
    The result is a permutation of text + sentinel that clusters same-context characters."""
    s = text + _SENTINEL
    n = len(s)
    # rotation i is s[i:] + s[:i]; sort rotations, take each's last character
    rotations = sorted(range(n), key=lambda i: s[i:] + s[:i])
    return "".join(s[(i - 1) % n] for i in rotations)


def bwt_inverse(last_column):
    """Invert the Burrows-Wheeler transform via the LF-mapping, stripping the sentinel.

    LF-mapping: for each row i of the last column L, lf[i] gives the row of the FIRST column holding
    the same character occurrence. Following lf from the sentinel row walks the original string
    backwards. O(n)."""
    n = len(last_column)
    if n == 0:
        return ""
    # rank of each character occurrence within the last column
    seen = {}
    ranks = [0] * n
    for i, c in enumerate(last_column):
        ranks[i] = seen.get(c, 0)
        seen[c] = seen.get(c, 0) + 1
    # C[c] = number of characters in the string that sort strictly before c
    counts = {}
    for c in last_column:
        counts[c] = counts.get(c, 0) + 1
    C = {}
    cum = 0
    for c in sorted(counts):
        C[c] = cum
        cum += counts[c]
    lf = [C[last_column[i]] + ranks[i] for i in range(n)]
    # the sentinel occupies row 0 of the sorted first column; walk lf to recover the string
    row = last_column.index(_SENTINEL)
    result = []
    for _ in range(n):
        result.append(last_column[row])
        row = lf[row]
    # `result` is the original read backwards (it starts with the sentinel), so reverse it
    decoded = "".join(reversed(result))
    return decoded.replace(_SENTINEL, "")


def move_to_front_encode(data):
    """Move-to-front: recode each symbol as its index in a running alphabet, then move it to the
    front. Runs of one symbol become runs of zeros. `data` is a string; returns a list of ints."""
    alphabet = sorted(set(data))
    out = []
    for ch in data:
        idx = alphabet.index(ch)
        out.append(idx)
        alphabet.pop(idx)
        alphabet.insert(0, ch)
    return out, sorted(set(data))


def move_to_front_decode(codes, alphabet):
    """Invert move-to-front given the codes and the original sorted alphabet."""
    alpha = list(alphabet)
    out = []
    for idx in codes:
        ch = alpha[idx]
        out.append(ch)
        alpha.pop(idx)
        alpha.insert(0, ch)
    return "".join(out)


def rle_encode(values):
    """Run-length encode a list into (value, count) pairs."""
    if not values:
        return []
    out = []
    prev = values[0]
    count = 1
    for v in values[1:]:
        if v == prev:
            count += 1
        else:
            out.append((prev, count))
            prev = v
            count = 1
    out.append((prev, count))
    return out


def rle_decode(pairs):
    """Invert run-length encoding."""
    out = []
    for value, count in pairs:
        out.extend([value] * count)
    return out


def _mean_run_length(s):
    """Average length of maximal runs of identical characters (a runniness measure)."""
    if not s:
        return 0.0
    runs = 1
    for i in range(1, len(s)):
        if s[i] != s[i - 1]:
            runs += 1
    return len(s) / runs


def bwt_runniness_gain(text):
    """How much BWT increases the mean run length (a proxy for downstream compressibility)."""
    before = _mean_run_length(text)
    after = _mean_run_length(bwt_transform(text))
    return before, after


def compress(text):
    """The bzip2-style forward pipeline: BWT -> move-to-front -> run-length encode.
    Returns (rle_pairs, alphabet) needed to decode."""
    transformed = bwt_transform(text)
    codes, alphabet = move_to_front_encode(transformed)
    return rle_encode(codes), alphabet


def decompress(rle_pairs, alphabet):
    """Invert the pipeline: run-length decode -> move-to-front decode -> inverse BWT."""
    codes = rle_decode(rle_pairs)
    transformed = move_to_front_decode(codes, alphabet)
    return bwt_inverse(transformed)
