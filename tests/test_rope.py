"""Tests for rope: mirror a Python string over thousands of random edits, stay balanced."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rope import Rope  # noqa: E402


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def nxt(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s

    def randint(self, lo, hi):
        if hi < lo:
            return lo
        return lo + (self.nxt() >> 8) % (hi - lo + 1)

    def choice(self, seq):
        return seq[(self.nxt() >> 8) % len(seq)]


def main():
    rng = LCG(2024)
    alpha = "abcdefgh"

    # ---- 1. construction / to_string round trip --------------------------------------
    for s in ("", "a", "hello", "abcdefghijklmnopqrstuvwxyz" * 3):
        check(f"to_string round trip (len {len(s)})", Rope(s).to_string() == s)

    # ---- 2. length and indexing match the string -------------------------------------
    s = "".join(rng.choice(alpha) for _ in range(200))
    r = Rope(s)
    check("length matches", len(r) == len(s))
    idx_ok = all(r[i] == s[i] for i in range(len(s)))
    check("every index matches the string", idx_ok)
    try:
        r[len(s)]
        check("out-of-range index raises", False)
    except IndexError:
        check("out-of-range index raises", True)

    # ---- 3. concatenation ------------------------------------------------------------
    a = Rope("hello ")
    b = Rope("world")
    check("concat contents", (a + b).to_string() == "hello world")
    check("concat length additive", len(a + b) == len(a) + len(b))
    check("concat with empty", (Rope("x") + Rope("")).to_string() == "x"
          and (Rope("") + Rope("y")).to_string() == "y")

    # ---- 4. split then concat is identity --------------------------------------------
    s = "".join(rng.choice(alpha) for _ in range(300))
    r = Rope(s)
    split_bad = 0
    for _ in range(100):
        pos = rng.randint(0, len(s))
        left, right = r.split(pos)
        if left.to_string() != s[:pos] or right.to_string() != s[pos:]:
            split_bad += 1
        if left.concat(right).to_string() != s:
            split_bad += 1
    check("split matches string slicing, split+concat is identity", split_bad == 0,
          f"{split_bad} failures")

    # ---- 5. substring ----------------------------------------------------------------
    sub_bad = 0
    for _ in range(100):
        i = rng.randint(0, len(s))
        j = rng.randint(i, len(s))
        if r.substring(i, j).to_string() != s[i:j]:
            sub_bad += 1
    check("substring matches string slicing", sub_bad == 0, f"{sub_bad} failures")

    # ---- 6. the big one: thousands of random edits mirror a Python string -------------
    ref = "".join(rng.choice(alpha) for _ in range(50))
    rope = Rope(ref)
    mismatch = False
    for step in range(4000):
        op = rng.randint(0, 3)
        if op == 0 and len(ref) < 5000:            # insert
            pos = rng.randint(0, len(ref))
            ins = "".join(rng.choice(alpha) for _ in range(rng.randint(1, 6)))
            ref = ref[:pos] + ins + ref[pos:]
            rope = rope.insert(pos, ins)
        elif op == 1 and len(ref) > 0:             # delete
            start = rng.randint(0, len(ref) - 1)
            end = rng.randint(start, min(len(ref), start + 8))
            ref = ref[:start] + ref[end:]
            rope = rope.delete(start, end)
        elif op == 2:                              # concat a chunk
            chunk = "".join(rng.choice(alpha) for _ in range(rng.randint(1, 6)))
            ref = ref + chunk
            rope = rope + Rope(chunk)
        else:                                       # split and swap halves
            pos = rng.randint(0, len(ref))
            ref = ref[pos:] + ref[:pos]
            left, right = rope.split(pos)
            rope = right.concat(left)
        if rope.to_string() != ref:
            mismatch = True
            print(f"    MISMATCH at step {step}, op {op}")
            break
    check("rope mirrors the string over 4000 random edits", not mismatch)
    check("final length matches", len(rope) == len(ref))

    # ---- 7. stays balanced after all those edits -------------------------------------
    check("tree stays balanced (height ~ log n)", rope.is_balanced(),
          f"height {rope.height} for length {len(rope)} (log2={math.log2(max(2,len(rope))):.1f})")

    # a pathological build: 2000 single-character concats would degenerate a naive rope
    patho = Rope("")
    for _ in range(2000):
        patho = patho + Rope("x")
    check("pathological repeated concat stays balanced", patho.is_balanced(),
          f"height {patho.height} for length {len(patho)}")
    check("pathological concat correct length", len(patho) == 2000)

    # ---- 8. edit error handling ------------------------------------------------------
    r = Rope("abcde")
    try:
        r.delete(3, 2)
        check("bad delete range raises", False)
    except IndexError:
        check("bad delete range raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
