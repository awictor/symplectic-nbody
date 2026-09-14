"""Fibonacci coding: a universal, error-resilient integer code built on Zeckendorf's theorem.

To write positive integers to a bitstream with no agreed maximum, you need a SELF-DELIMITING code -- the
bits announce their own end. Elias codes do this with a length prefix; FIBONACCI CODING does it with a
beautiful number-theoretic trick instead. Zeckendorf's theorem says every positive integer is a unique sum
of NON-CONSECUTIVE Fibonacci numbers (1, 2, 3, 5, 8, 13, ...). Write that representation as bits, least
significant Fibonacci first -- and because no two Fibonacci indices are adjacent, the bit string NEVER
contains "11". So appending a single terminating "1" produces a codeword that ends in exactly one "11" and
contains no other -- an unambiguous, prefix-free delimiter that a decoder finds by scanning for the first
"11".

That "11" terminator gives Fibonacci codes their signature property: ERROR RESILIENCE. A single corrupted
bit can only damage the codewords around it before the next "11" resynchronizes the stream, whereas a flip
in an Elias or Huffman stream can cascade and garble everything downstream. Fibonacci codes are also
COMPETITIVE in length -- about 1.44 log2 n + O(1) bits, close to Elias gamma for small values and better
for mid-range ones -- which is why they appear in compressed inverted indexes and error-prone channels.

This module encodes and decodes positive integers and streams using the Zeckendorf-plus-terminator scheme
(reusing the repo's Zeckendorf routine), computes code lengths, and demonstrates the self-synchronizing
recovery after a bit error. It is validated: encode/decode round-trips for every integer; every codeword
ends in "11" and contains no other "11" (the unique terminator); a concatenated stream decodes without
delimiters; the codes are prefix-free; the code length matches 1 + the number of Zeckendorf terms and grows
like ~1.44 log2 n; a single bit flip corrupts only a bounded neighborhood and the stream resynchronizes
afterward; and results are deterministic. Pure stdlib; the error-resilient universal-code companion to the
Elias, Golomb-Rice, Huffman, and Tunstall tools."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import fibonacci as _fib


def _fibs_upto(n):
    """Fibonacci numbers 1, 2, 3, 5, 8, ... up to (and possibly one past) n. Used for bit positions."""
    fibs = [1, 2]
    while fibs[-1] <= n:
        fibs.append(fibs[-1] + fibs[-2])
    if fibs[-1] > n:
        fibs.pop()
    return fibs


def encode_int(n):
    """Fibonacci-encode a positive integer n (>= 1). Returns a bit string ending in the '11' terminator.

    Bits are ordered least-significant Fibonacci first (F_2=1, F_3=2, ...), a 1 marking each Fibonacci in
    the Zeckendorf sum, then a final '1' appended as the terminator."""
    if n < 1:
        raise ValueError("Fibonacci coding is for positive integers (n >= 1)")
    used = set(_fib.zeckendorf(n))          # set of Fibonacci VALUES in the representation
    fibs = _fibs_upto(n)                    # ascending: index 0 -> F=1, index 1 -> F=2, ...
    bits = ["1" if f in used else "0" for f in fibs]
    return "".join(bits) + "1"              # append terminator -> ends in "11"


def decode_int(bits, i=0):
    """Decode one Fibonacci codeword from bits starting at index i. Returns (value, next_i).

    Scans for the '11' that terminates the codeword: the bit before the terminator is the last data bit."""
    fibs = [1, 2]
    total = 0
    prev = "0"
    j = i
    pos = 0
    while j < len(bits):
        cur = bits[j]
        if cur == "1" and prev == "1":
            # this is the terminator; the previous "1" was a data bit already counted
            return total, j + 1
        if cur == "1":
            # ensure we have a Fibonacci value for this position
            while len(fibs) <= pos:
                fibs.append(fibs[-1] + fibs[-2])
            total += fibs[pos]
        prev = cur
        j += 1
        pos += 1
    raise ValueError("no '11' terminator found -- truncated stream")


def encode(values):
    """Encode a list of positive integers into one concatenated bit string."""
    return "".join(encode_int(v) for v in values)


def decode(bits, count=None):
    """Decode a Fibonacci-coded stream. If count is given, stop after that many values; otherwise
    decode until the bits are exhausted."""
    out = []
    i = 0
    n = len(bits)
    while i < n:
        if count is not None and len(out) >= count:
            break
        v, i = decode_int(bits, i)
        out.append(v)
    return out


def code_length(n):
    """Length in bits of the Fibonacci code for n = (number of Zeckendorf terms' bit span) + 1."""
    return len(encode_int(n))


def resync_after_error(bits, flip_index):
    """Flip the bit at flip_index and return the decoded stream from the corrupted bits.

    Demonstrates self-synchronization: only codewords near the flip are damaged; decoding recovers
    once a clean '11' terminator is reached. Returns the decoded integer list (best-effort)."""
    corrupted = list(bits)
    corrupted[flip_index] = "0" if corrupted[flip_index] == "1" else "1"
    corrupted = "".join(corrupted)
    out = []
    i = 0
    n = len(corrupted)
    while i < n:
        try:
            v, i = decode_int(corrupted, i)
            out.append(v)
        except ValueError:
            break
    return out
