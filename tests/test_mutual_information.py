"""Tests for mutual_information: independence, copy, identities, KL divergence, normalization."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mutual_information as mi

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- entropy against hand values -------------------------------------------
check("entropy of a fair coin is 1 bit", approx(mi.entropy([0.5, 0.5]), 1.0, 1e-12))
check("entropy of 4 equal is 2 bits", approx(mi.entropy([0.25] * 4), 2.0, 1e-12))
check("entropy of a certain event is 0", approx(mi.entropy([1.0]), 0.0, 1e-12))
check("entropy from counts", approx(mi.entropy_counts([1, 1, 1, 1]), 2.0, 1e-12))
check("entropy of samples", approx(mi.entropy_samples([0, 0, 1, 1]), 1.0, 1e-12))

# --- LCG samples -----------------------------------------------------------
state = 1


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- independent variables: mutual information ~ 0 -------------------------
xi = [int(rng() * 2) for _ in range(5000)]
yi = [int(rng() * 2) for _ in range(5000)]
check("independent variables have ~zero MI", mi.mutual_information(xi, yi) < 0.01)

# --- a deterministic copy: I(X;X) = H(X), maximal --------------------------
xc = [int(rng() * 4) for _ in range(5000)]
yc = list(xc)
hx = mi.entropy_samples(xc)
check("copy has MI equal to H(X)", approx(mi.mutual_information(xc, yc), hx, 1e-9))
check("copy MI is the maximum (= min of the two entropies)",
      approx(mi.mutual_information(xc, yc), min(hx, mi.entropy_samples(yc)), 1e-9))

# --- MI identities ---------------------------------------------------------
hy = mi.entropy_samples(yc)
hxy = mi.joint_entropy(xc, yc)
info = mi.mutual_information(xc, yc)
check("I = H(X) + H(Y) - H(X,Y)", approx(info, hx + hy - hxy, 1e-9))
check("I = H(X) - H(X|Y)", approx(info, hx - mi.conditional_entropy(xc, yc), 1e-9))
check("joint entropy >= each marginal", hxy >= hx - 1e-9 and hxy >= hy - 1e-9)
check("conditional entropy >= 0", mi.conditional_entropy(xc, yc) >= -1e-9)
check("MI is symmetric", approx(mi.mutual_information(xc, yc), mi.mutual_information(yc, xc), 1e-12))

# --- a noisy channel: partial information ----------------------------------
# Y = X flipped 20% of the time -> 0 < I < H(X)
xn = [int(rng() * 2) for _ in range(6000)]
yn = [(x if rng() > 0.2 else 1 - x) for x in xn]
info_n = mi.mutual_information(xn, yn)
check("noisy copy has partial MI", 0.1 < info_n < mi.entropy_samples(xn))

# --- mutual information from an explicit joint distribution ----------------
# perfectly correlated 2x2 -> 1 bit
check("correlated joint has MI 1 bit",
      approx(mi.mutual_information_from_joint({(0, 0): 0.5, (1, 1): 0.5}), 1.0, 1e-12))
# independent joint (product) -> 0
indep = {(x, y): 0.25 for x in (0, 1) for y in (0, 1)}
check("independent joint has MI 0", approx(mi.mutual_information_from_joint(indep), 0.0, 1e-12))

# --- KL divergence ---------------------------------------------------------
check("KL of equal distributions is 0", approx(mi.kl_divergence([0.5, 0.5], [0.5, 0.5]), 0.0, 1e-12))
check("KL is nonnegative", mi.kl_divergence([0.9, 0.1], [0.5, 0.5]) > 0)
check("KL is asymmetric",
      not approx(mi.kl_divergence([0.9, 0.1], [0.5, 0.5]),
                 mi.kl_divergence([0.5, 0.5], [0.9, 0.1]), 1e-6))
check("KL infinite when q lacks support", mi.kl_divergence([0.5, 0.5], [1.0, 0.0]) == float("inf"))
# hand value: D([1,0]||[0.5,0.5]) = 1*log2(1/0.5) = 1 bit
check("KL hand value", approx(mi.kl_divergence([1.0, 0.0], [0.5, 0.5]), 1.0, 1e-12))

# --- normalized mutual information in [0,1] --------------------------------
check("NMI of a copy is 1", approx(mi.normalized_mutual_information(xc, yc), 1.0, 1e-9))
check("NMI of independents is ~0", mi.normalized_mutual_information(xi, yi) < 0.02)
for m in ("sqrt", "min", "sum"):
    v = mi.normalized_mutual_information(xn, yn, method=m)
    check(f"NMI ({m}) in [0,1]", 0.0 <= v <= 1.0)

# --- information gain equals mutual information ----------------------------
labels = [0, 0, 1, 1, 0, 1, 1, 0]
feature = [0, 0, 1, 1, 0, 1, 1, 0]     # perfectly predictive
check("information gain = MI(feature; label)",
      approx(mi.information_gain(feature, labels), mi.mutual_information(feature, labels), 1e-12))
check("perfectly predictive feature has full gain",
      approx(mi.information_gain(feature, labels), mi.entropy_samples(labels), 1e-9))
useless = [0, 1, 0, 1, 0, 1, 0, 1]     # uncorrelated with label
check("useless feature has ~zero gain", mi.information_gain(useless, labels) < 0.3)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all mutual_information tests passed")
