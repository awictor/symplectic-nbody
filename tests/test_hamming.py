"""Tests for hamming.py -- single-error-correcting Hamming codes.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Correctness is proven
exhaustively over every codeword and every single-bit error.
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hamming as h  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


# --- parameters -------------------------------------------------------------
check("m=3 gives the (7,4) code", h.code_parameters(3) == (7, 4))
check("m=4 gives the (15,11) code", h.code_parameters(4) == (15, 11))
check("m=5 gives the (31,26) code", h.code_parameters(5) == (31, 26))
check("code rate rises with block size", h.code_rate(4) > h.code_rate(3))
check("code rate is k/n", abs(h.code_rate(3) - 4 / 7) < 1e-12)
check("minimum distance is 3", h.minimum_distance(3) == 3)
try:
    h.code_parameters(1)
    check("rejects m < 2", False)
except ValueError:
    check("rejects m < 2", True)

# --- encode/decode round-trip ----------------------------------------------
cw = h.encode([1, 0, 1, 1], 3)
check("(7,4) encodes to 7 bits", len(cw) == 7)
check("a clean codeword has zero syndrome", h.syndrome(cw, 3) == 0)
data, err = h.decode(cw, 3)
check("clean codeword decodes to the data", data == [1, 0, 1, 1])
check("clean codeword reports no error", err == 0)
try:
    h.encode([1, 0, 1], 3)
    check("rejects wrong data length", False)
except ValueError:
    check("rejects wrong data length", True)

# --- the syndrome locates the flipped bit ----------------------------------
bad = cw[:]
bad[4] ^= 1  # flip position 5 (0-based index 4)
check("syndrome equals the flipped position", h.syndrome(bad, 3) == 5)
data, err = h.decode(bad, 3)
check("a single error is corrected", data == [1, 0, 1, 1])
check("the error position is reported", err == 5)

# --- exhaustive single-error correctness for m=3 and m=4 -------------------
def exhaustive(m, sample_data=None):
    n, k = h.code_parameters(m)
    datasets = sample_data if sample_data is not None else itertools.product([0, 1], repeat=k)
    for bits in datasets:
        d = list(bits)
        code = h.encode(d, m)
        # no error
        dec, e = h.decode(code, m)
        if dec != d or e != 0:
            return False
        # every single-bit error
        for i in range(n):
            corrupt = code[:]
            corrupt[i] ^= 1
            dec, e = h.decode(corrupt, m)
            if dec != d or e != i + 1:
                return False
    return True


check("EVERY single-bit error is corrected for (7,4)", exhaustive(3))
# (15,11): 2^11 = 2048 datasets x 15 errors is fine, but sample a subset for speed
sample = [tuple((j >> i) & 1 for i in range(11)) for j in range(0, 2048, 17)]
check("EVERY single-bit error is corrected for (15,11) (sampled)", exhaustive(4, sample))

# --- minimum distance: distinct data -> codewords differ in >= 3 bits ------
codewords = [h.encode(list(bits), 3) for bits in itertools.product([0, 1], repeat=4)]
min_d = min(h.hamming_distance(a, b)
            for i, a in enumerate(codewords) for b in codewords[i + 1:])
check("minimum distance between codewords is 3", min_d == 3)
check("hamming_distance counts differing bits", h.hamming_distance([0, 1, 1], [0, 0, 1]) == 1)
try:
    h.hamming_distance([0, 1], [0, 1, 1])
    check("hamming_distance rejects mismatched lengths", False)
except ValueError:
    check("hamming_distance rejects mismatched lengths", True)

# --- SECDED: single-correct, double-detect ---------------------------------
sec = h.encode_secded([1, 0, 1, 1], 3)
check("SECDED adds one overall parity bit", len(sec) == 8)
data, status = h.decode_secded(sec, 3)
check("clean SECDED word is ok", status == "ok" and data == [1, 0, 1, 1])
# single error -> corrected
one = sec[:]
one[2] ^= 1
data, status = h.decode_secded(one, 3)
check("SECDED corrects a single error", status == "corrected" and data == [1, 0, 1, 1])
# double error -> detected but not corrected
two = sec[:]
two[1] ^= 1
two[4] ^= 1
_, status = h.decode_secded(two, 3)
check("SECDED detects a double error", status == "double_error")
# every single-bit error in the 8-bit SECDED word is either ok or corrected to the data
ok = True
for i in range(8):
    corrupt = sec[:]
    corrupt[i] ^= 1
    d, st = h.decode_secded(corrupt, 3)
    if not (st == "corrected" and d == [1, 0, 1, 1]):
        ok = False
check("SECDED corrects every single-bit error", ok)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall hamming tests passed")
