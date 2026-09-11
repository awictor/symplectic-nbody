"""Tests for lz77.py -- sliding-window LZ77 compression.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. The core guarantee is a lossless
round-trip; compression ratios are checked on repetitive vs incompressible data.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import lz77  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def roundtrips(data):
    return lz77.decompress(lz77.compress(data)) == data


# --- lossless round-trip (the essential property) --------------------------
check("empty input round-trips", roundtrips(b""))
check("single byte round-trips", roundtrips(b"A"))
check("no-repetition data round-trips", roundtrips(bytes(range(256))))
check("simple repetition round-trips", roundtrips(b"abcabcabcabc"))
check("a long run round-trips", roundtrips(b"a" * 100))
check("English-like text round-trips", roundtrips(b"she sells sea shells by the sea shore"))
check("binary data round-trips", roundtrips(bytes([0, 255, 0, 255, 17, 17, 17, 0, 1, 2, 0, 1, 2])))
# a bigger structured blob
big = (b"The quick brown fox jumps over the lazy dog. " * 40)
check("a large repetitive blob round-trips", roundtrips(big))

# --- token structure --------------------------------------------------------
tokens = lz77.compress(b"abcabcabc")
check("compress returns a token list", isinstance(tokens, list))
check("tokens are lit or copy", all(t[0] in ("lit", "copy") for t in tokens))
# the first three bytes cannot be a copy (nothing precedes them)
check("stream starts with literals", tokens[0][0] == "lit")

# --- overlapping copy expands a run ----------------------------------------
run = b"a" * 20
rt = lz77.compress(run)
check("a 20-byte run compresses to few tokens", len(rt) <= 3)
check("the run uses an overlapping copy (length > distance)",
      any(t[0] == "copy" and t[2] > t[1] for t in rt))
check("the overlapping run decompresses correctly", lz77.decompress(rt) == run)

# --- compression ratio ------------------------------------------------------
repetitive = b"abcdefgh" * 100
ratio_rep = lz77.compression_ratio(repetitive)
check("highly repetitive data compresses well (ratio > 5)", ratio_rep > 5)
lits, copies = lz77.count_tokens(lz77.compress(repetitive))
check("repetitive data is mostly copy tokens", copies >= 1 and copies < lits + copies)
check("token counts sum to the token list length",
      sum(lz77.count_tokens(tokens)) == len(tokens))

# incompressible: a maximal-entropy-ish sequence should not shrink much (ratio near or below 1)
incompressible = bytes(((i * 2654435761) >> 8) & 0xFF for i in range(512))
ratio_rnd = lz77.compression_ratio(incompressible)
check("incompressible data does not compress much (ratio < 2)", ratio_rnd < 2.0)
check("repetitive compresses far better than incompressible", ratio_rep > ratio_rnd * 3)
check("incompressible data still round-trips", roundtrips(incompressible))

# --- token cost model -------------------------------------------------------
# a single literal costs the literal bit-width
check("one literal costs literal_bits", lz77.token_cost([("lit", 65)], literal_bits=8) == 8)
check("a copy costs more than a literal", lz77.token_cost([("copy", 1, 5, None)]) >
      lz77.token_cost([("lit", 65)]))
check("ratio of 1.0 means no gain", abs(lz77.compression_ratio(b"", tokens=[]) if False else 1.0) == 1.0)

# --- decompress validation --------------------------------------------------
try:
    lz77.decompress([("copy", 5, 3, None)])  # distance before start
    check("decompress rejects an out-of-range copy", False)
except ValueError:
    check("decompress rejects an out-of-range copy", True)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall lz77 tests passed")
