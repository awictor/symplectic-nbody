"""Tests for shamir: any-k reconstruction, subset invariance, below-threshold failure, bytes."""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shamir import split, reconstruct, split_bytes, reconstruct_bytes, _PRIME

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- any k of n shares reconstruct the secret ------------------------------
secret = 987654321987654321
shares = split(secret, 3, 6, seed=7)
check("6 shares produced", len(shares) == 6)
check("all 6 shares reconstruct", reconstruct(shares) == secret)
check("first 3 reconstruct", reconstruct(shares[:3]) == secret)
check("last 3 reconstruct", reconstruct(shares[3:]) == secret)

# --- EVERY subset of exactly k shares gives the same secret ----------------
ok = all(reconstruct(list(combo)) == secret for combo in itertools.combinations(shares, 3))
check("every 3-share subset reconstructs the same secret", ok)

# --- k-1 shares do NOT reconstruct (reveal nothing) ------------------------
below = [reconstruct(list(combo)) for combo in itertools.combinations(shares, 2)]
check("no 2-share subset accidentally recovers the secret", all(v != secret for v in below))

# --- different secrets give different shares -------------------------------
sh_a = split(111, 2, 3, seed=1)
sh_b = split(222, 2, 3, seed=1)
check("different secrets produce different shares", sh_a != sh_b)

# --- randomized: many secrets and (k, n) --------------------------------
state = 42
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return state


ok = True
for _ in range(50):
    s = rng() % (2 ** 128)
    n = 3 + rng() % 5
    k = 2 + rng() % (n - 1)
    shares = split(s, k, n, seed=rng())
    # pick k random distinct shares
    idx = sorted(range(n), key=lambda i: (rng(), i))[:k]
    subset = [shares[i] for i in idx]
    if reconstruct(subset) != s:
        ok = False
        break
    # k-1 shares should (almost surely) not equal the secret
    if k >= 2 and reconstruct(subset[:k - 1]) == s:
        ok = False
        break
check("randomized (k,n) sharing reconstructs from k and fails below k", ok)

# --- k = 1: every share IS the secret --------------------------------------
sh1 = split(555, 1, 4, seed=2)
check("k=1: each single share reconstructs the secret", all(reconstruct([sh]) == 555 for sh in sh1))

# --- k = n: all shares required --------------------------------------------
shn = split(777, 4, 4, seed=3)
check("k=n: all 4 shares reconstruct", reconstruct(shn) == 777)
check("k=n: any 3 do not reconstruct", reconstruct(shn[:3]) != 777)

# --- byte-string secret round-trips ----------------------------------------
msg = b"the launch code is 42"
length, byte_shares = split_bytes(msg, 3, 5, seed=9)
check("byte secret reconstructs from k shares", reconstruct_bytes(length, byte_shares[:3]) == msg)
check("byte secret reconstructs from a different k subset",
      reconstruct_bytes(length, byte_shares[2:5]) == msg)

# --- shares look random (spread across the field) --------------------------
shares = split(0, 5, 10, seed=11)     # even secret 0 -> non-trivial shares
ys = [y for _, y in shares]
check("shares of secret 0 are not all zero (masked by the polynomial)", any(y != 0 for y in ys))
check("share values are within the field", all(0 <= y < _PRIME for _, y in shares))

# --- reconstructing secret 0 works -----------------------------------------
check("secret 0 reconstructs to 0", reconstruct(shares[:5]) == 0)

# --- validation ------------------------------------------------------------
raised = 0
for bad in [lambda: split(5, 4, 3), lambda: split(_PRIME, 2, 3), lambda: split(-1, 2, 3)]:
    try:
        bad()
    except ValueError:
        raised += 1
check("invalid parameters raise ValueError", raised == 3)

# --- a large 256-bit secret ------------------------------------------------
big = 2 ** 255 + 12345
bshares = split(big, 3, 5, seed=13)
check("256-bit secret reconstructs", reconstruct(bshares[:3]) == big)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all shamir tests passed")
