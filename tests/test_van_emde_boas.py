"""Tests for van_emde_boas: membership, min/max, successor/predecessor vs brute force."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from van_emde_boas import VEBTree

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 91
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def brute_successor(sorted_keys, x):
    for k in sorted_keys:
        if k > x:
            return k
    return None


def brute_predecessor(sorted_keys, x):
    best = None
    for k in sorted_keys:
        if k < x:
            best = k
        else:
            break
    return best


# --- basic operations ------------------------------------------------------
v = VEBTree(16)
for x in [3, 1, 9, 14, 6, 7]:
    v.insert(x)
check("sorted list is correct", v.to_sorted_list() == [1, 3, 6, 7, 9, 14])
check("minimum", v.minimum() == 1)
check("maximum", v.maximum() == 14)
check("membership present", all(k in v for k in [1, 3, 6, 7, 9, 14]))
check("membership absent", all(k not in v for k in [0, 2, 5, 8, 15]))

# --- successor / predecessor at every point --------------------------------
keys = sorted([1, 3, 6, 7, 9, 14])
succ_ok = pred_ok = True
for x in range(16):
    if v.successor(x) != brute_successor(keys, x):
        succ_ok = False
    if v.predecessor(x) != brute_predecessor(keys, x):
        pred_ok = False
check("successor matches brute force at every point", succ_ok)
check("predecessor matches brute force at every point", pred_ok)

# --- duplicate insert is idempotent ----------------------------------------
v.insert(9)
check("duplicate insert doesn't change the set", v.to_sorted_list() == keys)

# --- delete ----------------------------------------------------------------
v.delete(9)
v.delete(1)
check("delete removes keys", v.to_sorted_list() == [3, 6, 7, 14])
check("min updates after deleting the old min", v.minimum() == 3)
check("delete of absent key is a no-op", (v.delete(100) or True) and v.to_sorted_list() == [3, 6, 7, 14])

# --- power-of-two rounding -------------------------------------------------
v2 = VEBTree(1000)      # rounds up to 1024
check("universe rounds up to a power of two", v2.u == 1024)
v2.insert(999)
check("can insert near the rounded universe top", 999 in v2)

# --- big randomized stream vs a reference set ------------------------------
U = 1024
veb = VEBTree(U)
ref = set()
ok = True
for _ in range(4000):
    x = int(rng() * U)
    if rng() < 0.6:
        veb.insert(x)
        ref.add(x)
    else:
        veb.delete(x)
        ref.discard(x)
    if rng() < 0.02:
        if veb.to_sorted_list() != sorted(ref):
            ok = False
            break
check("random insert/delete stream matches a reference set", ok and veb.to_sorted_list() == sorted(ref))

# --- full successor/predecessor sweep over the final set -------------------
final = sorted(ref)
succ_ok = pred_ok = True
for x in range(0, U, 3):
    if veb.successor(x) != brute_successor(final, x):
        succ_ok = False
        break
    if veb.predecessor(x) != brute_predecessor(final, x):
        pred_ok = False
        break
check("successor sweep matches brute force over the streamed set", succ_ok)
check("predecessor sweep matches brute force over the streamed set", pred_ok)

# --- min/max correct on the streamed set -----------------------------------
check("min matches the reference set", veb.minimum() == (min(ref) if ref else None))
check("max matches the reference set", veb.maximum() == (max(ref) if ref else None))

# --- walking successors from the min reproduces the sorted list ------------
walk = []
x = veb.minimum()
while x is not None:
    walk.append(x)
    x = veb.successor(x)
check("successor walk reproduces the sorted list", walk == sorted(ref))

# --- larger universe -------------------------------------------------------
big = VEBTree(65536)
vals = sorted(set(int(rng() * 65536) for _ in range(500)))
for x in vals:
    big.insert(x)
check("large-universe sorted list correct", big.to_sorted_list() == vals)
check("large-universe successor", all(big.successor(vals[i]) == vals[i + 1] for i in range(0, len(vals) - 1, 25)))
check("large-universe membership", all(x in big for x in vals[::20]))

# --- tiny universes --------------------------------------------------------
t2 = VEBTree(2)
t2.insert(0); t2.insert(1)
check("u=2 sorted", t2.to_sorted_list() == [0, 1])
check("u=2 successor", t2.successor(0) == 1 and t2.successor(1) is None)
check("u=2 predecessor", t2.predecessor(1) == 0 and t2.predecessor(0) is None)
t2.delete(0)
check("u=2 delete", t2.to_sorted_list() == [1])

# --- single element --------------------------------------------------------
s = VEBTree(64)
s.insert(42)
check("single element min==max", s.minimum() == 42 and s.maximum() == 42)
check("single element no successor", s.successor(42) is None)
check("single element predecessor", s.predecessor(42) is None)
s.delete(42)
check("empty after deleting the only element", s.minimum() is None)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all van_emde_boas tests passed")
