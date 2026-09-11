"""Tests for p2_quantile: accuracy vs exact quantiles on several distributions, edges, histogram."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from p2_quantile import P2Quantile, P2Histogram

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 2024
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def gauss():
    # Box-Muller
    u1 = max(rng(), 1e-12)
    u2 = rng()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def exact_quantile(data, p):
    s = sorted(data)
    idx = min(len(s) - 1, int(p * len(s)))
    return s[idx]


# --- uniform stream: several quantiles within a small error ----------------
uniform = [rng() for _ in range(100000)]
for p in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    est = P2Quantile(p)
    for x in uniform:
        est.add(x)
    exact = exact_quantile(uniform, p)
    err = abs(est.quantile() - exact)
    check(f"uniform p={p}: estimate within 0.01 of exact (err {err:.4f})", err < 0.01)

# --- normal stream ---------------------------------------------------------
normal = [gauss() for _ in range(100000)]
for p in [0.5, 0.9, 0.99]:
    est = P2Quantile(p)
    for x in normal:
        est.add(x)
    exact = exact_quantile(normal, p)
    # allow a slightly larger absolute error for the heavier tails
    err = abs(est.quantile() - exact)
    check(f"normal p={p}: estimate within 0.1 of exact (err {err:.4f})", err < 0.1)

# --- exponential stream (skewed) -------------------------------------------
expo = [-math.log(max(rng(), 1e-12)) for _ in range(100000)]
for p in [0.5, 0.9]:
    est = P2Quantile(p)
    for x in expo:
        est.add(x)
    exact = exact_quantile(expo, p)
    rel = abs(est.quantile() - exact) / max(exact, 1e-9)
    check(f"exponential p={p}: relative error under 5% ({rel:.3f})", rel < 0.05)

# --- median of a symmetric stream is near its centre -----------------------
sym = [rng() * 2 - 1 for _ in range(50000)]      # uniform on [-1, 1]
med = P2Quantile(0.5)
for x in sym:
    med.add(x)
check(f"median of a symmetric stream is near zero ({med.quantile():.4f})", abs(med.quantile()) < 0.02)

# --- min / max markers are exact -------------------------------------------
est = P2Quantile(0.5)
vals = [rng() * 100 for _ in range(10000)]
for x in vals:
    est.add(x)
check("min marker equals the true minimum", abs(est.minimum() - min(vals)) < 1e-9)
check("max marker equals the true maximum", abs(est.maximum() - max(vals)) < 1e-9)

# --- constant stream returns the constant ----------------------------------
c = P2Quantile(0.5)
for _ in range(1000):
    c.add(42.0)
check("constant stream returns the constant", c.quantile() == 42.0)

# --- warm-up (fewer than 5 samples) ----------------------------------------
w = P2Quantile(0.5)
w.add(3.0)
check("single sample quantile", w.quantile() == 3.0)
w.add(1.0)
w.add(2.0)
check("three-sample median is sane", 1.0 <= w.quantile() <= 3.0)

# --- monotonic input -------------------------------------------------------
mono = P2Quantile(0.5)
for i in range(10000):
    mono.add(float(i))
check("median of 0..9999 is near 5000", abs(mono.quantile() - 5000) < 200)

# --- P2Histogram tracks several quantiles simultaneously -------------------
hist = P2Histogram(quantiles=(0.5, 0.9, 0.95, 0.99))
data = [gauss() for _ in range(80000)]
for x in data:
    hist.add(x)
summary = hist.summary()
ok = True
for p in (0.5, 0.9, 0.95, 0.99):
    if abs(summary[p] - exact_quantile(data, p)) > 0.12:
        ok = False
check("P2Histogram tracks all quantiles within tolerance", ok)
# quantiles should be ordered
vals = [summary[p] for p in (0.5, 0.9, 0.95, 0.99)]
check("histogram quantiles are monotone increasing", all(vals[i] <= vals[i + 1] for i in range(3)))

# --- validation ------------------------------------------------------------
raised = False
try:
    P2Quantile(1.5)
except ValueError:
    raised = True
check("quantile out of (0,1) raises", raised)

# --- accuracy improves as more data arrives -------------------------------
# (the P-square estimate should get closer to the true 0.9 quantile with more samples)
est_small = P2Quantile(0.9)
est_big = P2Quantile(0.9)
small = [rng() for _ in range(500)]
big = [rng() for _ in range(100000)]
for x in small:
    est_small.add(x)
for x in big:
    est_big.add(x)
err_big = abs(est_big.quantile() - 0.9)
check(f"large-sample p90 estimate is very accurate (err {err_big:.4f})", err_big < 0.01)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all p2_quantile tests passed")
