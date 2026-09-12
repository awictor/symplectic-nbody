"""Tests for lzw: exhaustive round-trip, compression, KwKwK edge case, dictionary reset."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lzw import compress, decompress, compression_ratio

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 4321
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- round-trip on known inputs --------------------------------------------
known = [
    b"TOBEORNOTTOBEORTOBEORNOT",
    b"aaaaaaaaaaaaaaaaaaaaaaaa",
    b"",
    b"x",
    b"ab",
    bytes(range(256)),
    b"abcabcabcabcabcabc",
    b"the quick brown fox jumps over the lazy dog" * 5,
]
ok = all(decompress(compress(t)) == t for t in known)
check("round-trip on known inputs", ok)

# --- exhaustive round-trip on random bytes ---------------------------------
ok = True
for _ in range(300):
    n = int(rng() * 200)
    data = bytes(int(rng() * 256) for _ in range(n))
    if decompress(compress(data)) != data:
        ok = False
        break
check("round-trip on 300 random byte strings", ok)

# --- round-trip on highly repetitive data ----------------------------------
ok = True
for _ in range(50):
    pattern = bytes(int(rng() * 256) for _ in range(1 + int(rng() * 5)))
    data = pattern * (10 + int(rng() * 40))
    if decompress(compress(data)) != data:
        ok = False
        break
check("round-trip on repetitive data", ok)

# --- the KwKwK self-referential edge case ----------------------------------
# a pattern that forces a code to reference the entry being built
for kw in [b"ababababab", b"aaaaa", b"xyxyxyxyxy", b"abcabcabcabc"]:
    if decompress(compress(kw)) != kw:
        check(f"KwKwK case {kw!r}", False)
        break
else:
    check("KwKwK self-referential codes decode correctly", True)

# --- compression actually happens on repetitive input ----------------------
rep = b"ABCABCABC" * 200
codes = compress(rep)
check(f"repetitive input compresses (codes {len(codes)} << bytes {len(rep)})", len(codes) < len(rep) // 3)
check("compression ratio is small for repetitive data", compression_ratio(rep) < 0.35)

# --- incompressible (all-distinct-ish random) has ratio near 1 -------------
rnd = bytes(int(rng() * 256) for _ in range(500))
check("random data barely compresses (ratio near 1)", compression_ratio(rnd) > 0.7)

# --- all-same byte -> strong compression -----------------------------------
same = b"\x00" * 1000
check("all-same input round-trips", decompress(compress(same)) == same)
check("all-same input compresses hard", len(compress(same)) < 60)

# --- all 256 distinct bytes ------------------------------------------------
allbytes = bytes(range(256))
check("all 256 byte values round-trip", decompress(compress(allbytes)) == allbytes)

# --- dictionary reset with a capped code width -----------------------------
data = (b"the quick brown fox " * 100)
for bits in [9, 10, 12]:
    c = compress(data, max_bits=bits)
    d = decompress(c, max_bits=bits)
    if d != data:
        check(f"capped-width round-trip (max_bits={bits})", False)
        break
    # codes must stay within the width
    if max(c) >= (1 << bits):
        check(f"codes stay within {bits} bits", False)
        break
else:
    check("capped-width dictionary reset round-trips and stays in range", True)

# --- codes are all valid byte-or-dictionary indices ------------------------
c = compress(b"hello world hello world")
check("all codes are non-negative integers", all(isinstance(x, int) and x >= 0 for x in c))
check("first code is a literal byte", c[0] == ord("h"))

# --- single-symbol alphabet over a long run --------------------------------
run = bytes([7]) * 500
rc = compress(run)
check("long single-symbol run round-trips", decompress(rc) == run)
check("single-symbol run compresses greatly", len(rc) < 40)

# --- invalid code detected -------------------------------------------------
raised = False
try:
    decompress([300, 999999])
except ValueError:
    raised = True
check("invalid code raises ValueError", raised)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all lzw tests passed")
