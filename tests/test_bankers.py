"""Tests for bankers: safety vs brute force, safe-sequence validity, request grants, deadlock detection."""

import os
import sys
from itertools import permutations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bankers import (is_safe, request_resources, release_resources, detect_deadlock,  # noqa: E402
                     brute_is_safe, _need, _le)


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
        return lo + (self.nxt() >> 8) % (hi - lo + 1)


def valid_safe_sequence(alloc, maximum, available, seq):
    """Verify a claimed safe sequence: each process's need fits the running pool when its turn comes."""
    n = len(alloc)
    m = len(available)
    if sorted(seq) != list(range(n)):
        return False
    need = _need(alloc, maximum)
    work = list(available)
    for i in seq:
        if not _le(need[i], work):
            return False
        for j in range(m):
            work[j] += alloc[i][j]
    return True


def main():
    # ---- 1. classic Silberschatz textbook instance ------------------------------------
    alloc = [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]
    maxm = [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]
    avail = [3, 3, 2]
    safe, seq = is_safe(alloc, maxm, avail)
    check("textbook state is safe", safe)
    check("safe sequence is valid", valid_safe_sequence(alloc, maxm, avail, seq))
    check("agrees with brute force", safe == brute_is_safe(alloc, maxm, avail))

    # ---- 2. safety verdict matches brute force over random states ---------------------
    rng = LCG(2024)
    mism = 0
    seq_bad = 0
    for _ in range(300):
        n = rng.randint(2, 5)
        m = rng.randint(1, 3)
        maximum = [[rng.randint(0, 5) for _ in range(m)] for _ in range(n)]
        alloc = [[rng.randint(0, maximum[i][j]) for j in range(m)] for i in range(n)]
        available = [rng.randint(0, 6) for _ in range(m)]
        safe, seq = is_safe(alloc, maximum, available)
        if safe != brute_is_safe(alloc, maximum, available):
            mism += 1
        if safe and not valid_safe_sequence(alloc, maximum, available, seq):
            seq_bad += 1
    check("safety verdict matches brute force (300 states)", mism == 0, f"{mism}")
    check("every reported safe sequence is valid", seq_bad == 0, f"{seq_bad}")

    # ---- 3. granting a request never yields an unsafe state ---------------------------
    unsafe_grant = 0
    for _ in range(300):
        n = rng.randint(2, 5)
        m = rng.randint(1, 3)
        maximum = [[rng.randint(0, 5) for _ in range(m)] for _ in range(n)]
        alloc = [[rng.randint(0, maximum[i][j]) for j in range(m)] for i in range(n)]
        available = [rng.randint(0, 6) for _ in range(m)]
        if not is_safe(alloc, maximum, available)[0]:
            continue                          # only grant from a safe state
        p = rng.randint(0, n - 1)
        need_p = _need(alloc, maximum)[p]
        req = [rng.randint(0, need_p[j]) for j in range(m)]
        granted, na, nav = request_resources(alloc, maximum, available, p, req)
        if granted and not is_safe(na, maximum, nav)[0]:
            unsafe_grant += 1
    check("granted requests keep the state safe", unsafe_grant == 0, f"{unsafe_grant}")

    # ---- 4. requests exceeding NEED or AVAILABLE are refused --------------------------
    alloc = [[1, 0], [0, 1]]
    maxm = [[2, 2], [2, 2]]
    avail = [1, 1]
    # request beyond need
    g, _, _ = request_resources(alloc, maxm, avail, 0, [5, 0])
    check("request beyond declared need refused", not g)
    # request beyond available
    g, _, _ = request_resources(alloc, maxm, avail, 0, [1, 1])   # need is [1,2], avail [1,1] -> ok actually
    # make one clearly over available
    g2, _, _ = request_resources(alloc, maxm, [0, 0], 0, [1, 0])
    check("request beyond available refused", not g2)

    # ---- 5. release keeps the state safe ----------------------------------------------
    alloc = [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]
    maxm = [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]
    avail = [3, 3, 2]
    na, nav = release_resources(alloc, avail, 2, [1, 0, 1])
    check("release returns resources to the pool", nav == [4, 3, 3] and na[2] == [2, 0, 1])
    check("state after release is still safe", is_safe(na, maxm, nav)[0])

    # ---- 6. deadlock detection --------------------------------------------------------
    # a clear circular wait: P0 holds R0 wants R1; P1 holds R1 wants R0; nothing free
    alloc = [[1, 0], [0, 1]]
    request = [[0, 1], [1, 0]]
    available = [0, 0]
    stuck = detect_deadlock(alloc, request, available)
    check("circular wait detected as deadlock", set(stuck) == {0, 1}, f"{stuck}")
    # if a unit is free, it can break
    stuck2 = detect_deadlock(alloc, request, [0, 1])
    check("free resource breaks the deadlock", stuck2 == [], f"{stuck2}")
    # no requests -> no deadlock
    check("idle processes are never deadlocked",
          detect_deadlock([[1, 1], [0, 0]], [[0, 0], [0, 0]], [0, 0]) == [])

    # ---- 7. all-zero / trivial states -------------------------------------------------
    check("empty allocation is safe", is_safe([[0]], [[0]], [0])[0])
    check("process needing nothing is safe", is_safe([[2, 1]], [[2, 1]], [0, 0])[0])
    # a process whose need exceeds all resources ever available -> unsafe
    check("impossible need is unsafe", not is_safe([[0]], [[5]], [2])[0])

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
