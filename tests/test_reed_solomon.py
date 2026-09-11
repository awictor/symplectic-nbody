"""Tests for reed_solomon: GF(256) laws, encode/decode, t-error correction, uncorrectable flag."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import reed_solomon as rs

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- GF(256) field laws ----------------------------------------------------
check("addition is XOR", rs.gf_add(0x53, 0xCA) == (0x53 ^ 0xCA))
check("multiply by zero", rs.gf_mul(0, 123) == 0 and rs.gf_mul(123, 0) == 0)
check("multiply identity", rs.gf_mul(1, 200) == 200)
check("mul commutes", all(rs.gf_mul(a, b) == rs.gf_mul(b, a) for a in (3, 17, 200) for b in (5, 99, 255)))
check("mul associates",
      rs.gf_mul(rs.gf_mul(3, 7), 5) == rs.gf_mul(3, rs.gf_mul(7, 5)))
check("inverse works", all(rs.gf_mul(a, rs.gf_inv(a)) == 1 for a in range(1, 256)))
check("division inverts multiplication", all(rs.gf_div(rs.gf_mul(a, b), b) == a
                                             for a in (1, 50, 255) for b in (2, 99, 200)))
check("distributive law",
      all(rs.gf_mul(a, b ^ c) == (rs.gf_mul(a, b) ^ rs.gf_mul(a, c))
          for a in (7, 88) for b in (3, 200) for c in (5, 111)))

# --- polynomial helpers ----------------------------------------------------
check("poly eval Horner", rs.poly_eval([1, 0, 0], 5) == rs.gf_mul(5, 5))  # x^2 at 5
gen = rs.generator_poly(4)
check("generator degree", len(gen) - 1 == 4)
check("generator monic", gen[0] == 1)
# every root 2^i is a zero of the generator
check("generator has the code roots",
      all(rs.poly_eval(gen, rs._EXP[i]) == 0 for i in range(4)))

# --- encode: codeword length and divisibility ------------------------------
msg = [0x40, 0x1b, 0xa4, 0x62, 0x33, 0x11]
nsym = 6
cw = rs.rs_encode(msg, nsym)
check("codeword length = message + parity", len(cw) == len(msg) + nsym)
check("message prefix preserved", cw[:len(msg)] == msg)
# a valid codeword has all-zero syndromes
check("clean codeword has zero syndromes", max(rs._syndromes(cw, nsym)) == 0)
check("message recovered from codeword", rs.message_from_codeword(cw, nsym) == msg)

# --- clean decode is the identity ------------------------------------------
dec, ne = rs.rs_decode(cw, nsym)
check("clean decode returns the codeword", dec == cw and ne == 0)


# --- corrupting up to t = nsym//2 bytes anywhere is corrected ---------------
def lcg(seed):
    s = [seed & 0xFFFFFFFF]

    def nxt():
        s[0] = (1664525 * s[0] + 1013904223) & 0xFFFFFFFF
        return s[0] >> 16
    return nxt


all_ok = True
trials = 0
for n_err in (1, 2, 3):
    for t in range(25):
        rng = lcg(t * 13 + n_err * 101 + 1)
        c = cw[:]
        positions = []
        while len(positions) < n_err:
            p = rng() % len(cw)
            if p not in positions:
                positions.append(p)
        for p in positions:
            err = (rng() % 255) + 1              # nonzero error
            c[p] ^= err
        try:
            corrected, fixed = rs.rs_decode(c, nsym)
            trials += 1
            if corrected != cw or fixed != n_err:
                all_ok = False
        except ValueError:
            all_ok = False
check("corrects 1..t errors in every trial", all_ok)
check("ran the full corruption sweep", trials == 75)

# --- a single error is corrected and reported ------------------------------
c1 = cw[:]
c1[3] ^= 0x9d
d1, f1 = rs.rs_decode(c1, nsym)
check("single-error decode restores data", d1 == cw)
check("single-error count is 1", f1 == 1)

# --- more than t errors is flagged, not silently mis-corrected -------------
c_bad = cw[:]
for p in (0, 3, 6, 9):        # 4 > t=3 errors
    c_bad[p] ^= 0x7
raised = False
try:
    rs.rs_decode(c_bad, nsym)
except ValueError:
    raised = True
check("uncorrectable block raises ValueError", raised)

# --- a longer message with 8 parity bytes corrects 4 errors ----------------
msg2 = list(range(20))
nsym2 = 8
cw2 = rs.rs_encode(msg2, nsym2)
c2 = cw2[:]
for p, e in [(1, 0x11), (7, 0xff), (13, 0x05), (19, 0xa0)]:
    c2[p] ^= e
d2, f2 = rs.rs_decode(c2, nsym2)
check("nsym=8 corrects 4 errors", d2 == cw2 and f2 == 4)
check("longer message recovered", rs.message_from_codeword(d2, nsym2) == msg2)

# --- errors landing in the parity bytes are also fixed ---------------------
c3 = cw[:]
c3[len(msg)] ^= 0x33          # corrupt a parity byte
c3[len(msg) + 2] ^= 0x44
d3, f3 = rs.rs_decode(c3, nsym)
check("errors in parity region are corrected", d3 == cw)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all reed_solomon tests passed")
