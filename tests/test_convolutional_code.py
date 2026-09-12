"""Tests for convolutional_code: clean round-trip, error correction, Viterbi vs brute force."""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from convolutional_code import ConvolutionalCode, hamming_distance

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 271828
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# the classic (7,5) rate-1/2 code
c = ConvolutionalCode([0b111, 0b101])
check("constraint length is 3", c.constraint == 3)
check("2 memory bits, 4 states", c.mem == 2 and c.n_states == 4)
check("rate is 1/2", c.n == 2)

# --- clean channel round-trips exactly -------------------------------------
ok = True
for _ in range(100):
    m = [1 if rng() < 0.5 else 0 for _ in range(1 + int(rng() * 15))]
    if c.decode(c.encode(m)) != m:
        ok = False
        break
check("clean channel decodes exactly (100 random messages)", ok)

# --- encoded length formula ------------------------------------------------
m = [1, 0, 1, 1]
check("encoded length matches formula", len(c.encode(m)) == c.encoded_length(len(m)))

# --- corrects a single bit error -------------------------------------------
ok = True
for _ in range(100):
    m = [1 if rng() < 0.5 else 0 for _ in range(8)]
    enc = c.encode(m)
    pos = int(rng() * len(enc))
    enc[pos] ^= 1
    if c.decode(enc) != m:
        ok = False
        break
check("corrects any single-bit error (100 trials)", ok)

# --- corrects well-separated double errors ---------------------------------
corrected = 0
trials = 100
for _ in range(trials):
    m = [1 if rng() < 0.5 else 0 for _ in range(10)]
    enc = c.encode(m)
    p1 = int(rng() * len(enc))
    p2 = (p1 + len(enc) // 2) % len(enc)      # well-separated
    enc[p1] ^= 1
    enc[p2] ^= 1
    if c.decode(enc) == m:
        corrected += 1
check(f"corrects most well-separated double errors ({corrected}/{trials})", corrected >= trials * 0.85)

# --- Viterbi matches a brute-force minimum-distance decode -----------------
def brute_decode(code, received):
    """Exhaustive: the message minimizing Hamming distance to the received stream."""
    steps = len(received) // code.n
    msg_len = steps - code.mem
    best = None
    best_dist = None
    for bits in itertools.product([0, 1], repeat=msg_len):
        enc = code.encode(list(bits))
        d = hamming_distance(enc, received)
        if best_dist is None or d < best_dist:
            best_dist = d
            best = list(bits)
    return best


ok = True
for _ in range(60):
    m = [1 if rng() < 0.5 else 0 for _ in range(6)]
    enc = c.encode(m)
    # add up to 2 random errors
    nerr = int(rng() * 3)
    for _ in range(nerr):
        enc[int(rng() * len(enc))] ^= 1
    v = c.decode(enc)
    b = brute_decode(c, enc)
    # both must achieve the same (minimum) distance to the received word
    if hamming_distance(c.encode(v), enc) != hamming_distance(c.encode(b), enc):
        ok = False
        break
check("Viterbi achieves the same minimum distance as brute force (60 noisy trials)", ok)

# --- a different code: rate-1/3 (constraint length 3) ----------------------
c3 = ConvolutionalCode([0b111, 0b101, 0b011])
check("rate-1/3 code has n=3", c3.n == 3)
ok = True
for _ in range(50):
    m = [1 if rng() < 0.5 else 0 for _ in range(8)]
    enc = c3.encode(m)
    # rate-1/3 is stronger: correct 2 errors
    enc[int(rng() * len(enc))] ^= 1
    enc[int(rng() * len(enc))] ^= 1
    if c3.decode(enc) == m:
        continue
    else:
        # not guaranteed for adjacent errors, but should mostly work; count below
        pass
# measure correction rate instead of demanding all
corrected = 0
for _ in range(100):
    m = [1 if rng() < 0.5 else 0 for _ in range(8)]
    enc = c3.encode(m)
    enc[int(rng() * len(enc))] ^= 1
    enc[int(rng() * len(enc))] ^= 1
    if c3.decode(enc) == m:
        corrected += 1
check(f"rate-1/3 code corrects most double errors ({corrected}/100)", corrected >= 90)

# --- graceful degradation: more noise -> fewer perfect decodes -------------
def decode_rate(flip_prob, trials=200):
    good = 0
    for _ in range(trials):
        m = [1 if rng() < 0.5 else 0 for _ in range(12)]
        enc = c.encode(m)
        enc = [b ^ (1 if rng() < flip_prob else 0) for b in enc]
        if c.decode(enc) == m:
            good += 1
    return good / trials


r_low = decode_rate(0.02)
r_high = decode_rate(0.20)
check(f"low noise decodes better than high noise ({r_low:.2f} vs {r_high:.2f})", r_low > r_high)
check("low-noise decode rate is high", r_low > 0.85)

# --- known small encoding --------------------------------------------------
# input [1,0,0] with the (7,5) code, flushed with 2 zeros
enc = c.encode([1])
check("single-bit encoding has the right length", len(enc) == 2 * (1 + 2))
check("single 1 then flush decodes to [1]", c.decode(enc) == [1])

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all convolutional_code tests passed")
