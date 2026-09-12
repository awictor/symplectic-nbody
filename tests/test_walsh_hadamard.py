"""Tests for walsh_hadamard: FWHT and XOR/OR/AND convolutions vs the brute-force O(n^2) definition."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from walsh_hadamard import (fwht, xor_convolution, or_convolution, and_convolution,
                            zeta_transform, mobius_transform, superset_zeta, superset_mobius,
                            brute_convolution)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def rand(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s >> 16

    def randint(self, lo, hi):
        return lo + self.rand() % (hi - lo + 1)


XOR = lambda i, j: i ^ j     # noqa: E731
OR = lambda i, j: i | j      # noqa: E731
AND = lambda i, j: i & j     # noqa: E731


# --- hand cases -------------------------------------------------------------
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
check("XOR convolution matches brute (hand case)",
      xor_convolution(a, b) == brute_convolution(a, b, XOR))
check("OR convolution matches brute (hand case)",
      or_convolution(a, b) == brute_convolution(a, b, OR))
check("AND convolution matches brute (hand case)",
      and_convolution(a, b) == brute_convolution(a, b, AND))

# FWHT of a delta is all ones; a known 2-point transform
check("FWHT of [1,0] is [1,1]", fwht([1, 0]) == [1, 1])
check("FWHT of [1,1] is [2,0]", fwht([1, 1]) == [2, 0])
check("FWHT of [1,2,3,4] is the Hadamard combination",
      fwht([1, 2, 3, 4]) == [10, -2, -4, 0])

# --- round-trip identity ----------------------------------------------------
rng = LCG(2026)
rt_ok = zt_ok = st_ok = True
for _ in range(200):
    k = rng.randint(0, 4)             # length 1..16
    n = 1 << k
    arr = [rng.randint(-9, 9) for _ in range(n)]
    if fwht(fwht(arr), invert=True) != arr:
        rt_ok = False
        break
    if mobius_transform(zeta_transform(arr)) != arr:
        zt_ok = False
        break
    if superset_mobius(superset_zeta(arr)) != arr:
        st_ok = False
        break
check("FWHT inverse undoes FWHT (200 random arrays)", rt_ok)
check("Mobius undoes zeta / subset-sum (200 random arrays)", zt_ok)
check("superset Mobius undoes superset-sum (200 random arrays)", st_ok)

# --- exhaustive validation vs brute for all three convolutions -------------
rng = LCG(777)
xor_ok = or_ok = and_ok = True
for _ in range(300):
    k = rng.randint(1, 4)
    n = 1 << k
    a = [rng.randint(-6, 6) for _ in range(n)]
    b = [rng.randint(-6, 6) for _ in range(n)]
    if xor_convolution(a, b) != brute_convolution(a, b, XOR):
        xor_ok = False
        break
    if or_convolution(a, b) != brute_convolution(a, b, OR):
        or_ok = False
        break
    if and_convolution(a, b) != brute_convolution(a, b, AND):
        and_ok = False
        break
check("XOR convolution matches brute force (300 random pairs)", xor_ok)
check("OR convolution matches brute force (300 random pairs)", or_ok)
check("AND convolution matches brute force (300 random pairs)", and_ok)

# --- linearity of the transform --------------------------------------------
rng = LCG(999)
lin_ok = True
for _ in range(100):
    n = 1 << rng.randint(1, 4)
    a = [rng.randint(-5, 5) for _ in range(n)]
    b = [rng.randint(-5, 5) for _ in range(n)]
    alpha = rng.randint(-3, 3)
    lhs = fwht([alpha * a[i] + b[i] for i in range(n)])
    fa, fb = fwht(a), fwht(b)
    rhs = [alpha * fa[i] + fb[i] for i in range(n)]
    if lhs != rhs:
        lin_ok = False
        break
check("FWHT is linear: T(alpha*a + b) == alpha*T(a) + T(b)", lin_ok)

# --- convolution is commutative --------------------------------------------
rng = LCG(555)
comm_ok = True
for _ in range(100):
    n = 1 << rng.randint(1, 4)
    a = [rng.randint(0, 9) for _ in range(n)]
    b = [rng.randint(0, 9) for _ in range(n)]
    if xor_convolution(a, b) != xor_convolution(b, a):
        comm_ok = False
        break
check("XOR convolution is commutative", comm_ok)

# --- XOR-convolving with a delta at 0 is the identity ----------------------
rng = LCG(4321)
delta_ok = True
for _ in range(50):
    n = 1 << rng.randint(1, 4)
    a = [rng.randint(-9, 9) for _ in range(n)]
    delta = [0] * n
    delta[0] = 1
    if xor_convolution(a, delta) != a:
        delta_ok = False
        break
check("XOR convolution with delta_0 is the identity", delta_ok)

# --- probability interpretation: two random bitmasks -----------------------
# a and b are distributions over {0,1,2,3}; their XOR distribution sums to 1
a = [0.1, 0.2, 0.3, 0.4]
b = [0.25, 0.25, 0.25, 0.25]
conv = xor_convolution(a, b)
check("XOR convolution of two distributions still sums to ~1",
      abs(sum(conv) - 1.0) < 1e-9)

# --- non-power-of-two length is rejected -----------------------------------
try:
    fwht([1, 2, 3])
    raised = False
except ValueError:
    raised = True
check("non-power-of-two length raises ValueError", raised)

# --- larger transform is fast and exact ------------------------------------
n = 1 << 16
a = [(i * 2654435761) & 0xFF for i in range(n)]
b = [(i * 40503) & 0xFF for i in range(n)]
c = xor_convolution(a, b)
# total mass identity: sum(c) == sum(a) * sum(b) for any bit-convolution
check("2^16 XOR convolution: total mass equals product of input sums",
      sum(c) == sum(a) * sum(b))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all walsh_hadamard tests passed")
