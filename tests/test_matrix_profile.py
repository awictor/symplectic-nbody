"""Tests for the matrix profile: MASS vs brute force, motif/discord recovery, z-norm invariance."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matrix_profile as MP  # noqa: E402


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


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def main():
    rnd = _lcg(7)
    t = [rnd() * 2 - 1 for _ in range(90)]

    # ---- 1. MASS distance profile matches direct z-normalized Euclidean -----------------
    q = t[10:18]
    dm = MP.mass(t, q)
    db = MP.brute_distance_profile(t, q)
    check("MASS == brute distance profile", max(abs(dm[i] - db[i]) for i in range(len(dm))) < 1e-9,
          f"{max(abs(dm[i] - db[i]) for i in range(len(dm)))}")

    # ---- 2. distance from a window to itself is zero ------------------------------------
    dp = MP.mass(t, t[20:28])
    # MASS computes the distance via an FFT sliding dot product, so the self-match is ~0
    # up to floating-point noise rather than exactly 0.
    check("self-distance is ~zero", abs(dp[20]) < 1e-6, f"{dp[20]}")

    # ---- 3. full matrix profile: MASS == brute-force ------------------------------------
    m = 8
    pf_mass, idx_mass = MP.matrix_profile(t, m, use_mass=True)
    pf_brute, idx_brute = MP.matrix_profile(t, m, use_mass=False)
    check("matrix profile MASS == brute",
          max(abs(pf_mass[i] - pf_brute[i]) for i in range(len(pf_mass))) < 1e-9)

    # ---- 4. planted motif is recovered --------------------------------------------------
    pat = [math.sin(2 * math.pi * k / 8) for k in range(8)]
    series = [rnd() * 0.3 for _ in range(60)]
    for k in range(8):
        series[10 + k] = pat[k]
        series[40 + k] = pat[k]
    i, j, d = MP.top_motif(series, 8)
    check("motif recovers the planted repeated pattern", sorted((i, j)) == [10, 40],
          f"got {sorted((i, j))}, dist {d}")
    check("motif distance is ~0 (identical copies)", d < 1e-6, f"{d}")

    # ---- 5. planted discord (anomaly) is recovered --------------------------------------
    sig = [math.sin(2 * math.pi * k / 12) for k in range(72)]
    for k in range(12):
        sig[30 + k] = 2.0                    # flat anomaly, length = window
    di, dd = MP.top_discord(sig, 12)
    check("discord locates the planted anomaly", di == 30, f"got {di}")
    check("discord distance is large", dd > 3.0, f"{dd}")

    # ---- 6. z-normalization: profile invariant to adding a constant ---------------------
    base = [math.sin(2 * math.pi * k / 10) + 0.1 * (rnd() - 0.5) for k in range(50)]
    pf0, _ = MP.matrix_profile(base, 10)
    shifted = [x + 5.0 for x in base]
    pf1, _ = MP.matrix_profile(shifted, 10)
    check("profile invariant to a constant offset",
          max(abs(pf0[i] - pf1[i]) for i in range(len(pf0))) < 1e-7)

    # ---- 7. z-normalization: profile invariant to positive scaling ----------------------
    scaled = [x * 3.0 for x in base]
    pf2, _ = MP.matrix_profile(scaled, 10)
    check("profile invariant to positive scaling",
          max(abs(pf0[i] - pf2[i]) for i in range(len(pf0))) < 1e-7)

    # ---- 8. profile index points at a genuinely close window ----------------------------
    # for the motif series, window 10's nearest neighbour should be window 40
    pf, idx = MP.matrix_profile(series, 8)
    check("profile index points at the matching copy", idx[10] == 40 and idx[40] == 10,
          f"idx[10]={idx[10]} idx[40]={idx[40]}")

    # ---- 9. exclusion zone prevents trivial self-matches --------------------------------
    # every profile-index neighbour must be outside the exclusion zone
    excl = max(1, 8 // 2)
    ok = all(idx[i] == -1 or abs(i - idx[i]) >= excl for i in range(len(idx)))
    check("neighbours respect the exclusion zone", ok)

    # ---- 10. constant series: all windows identical, profile ~0 -------------------------
    flat = [3.0] * 40
    pf_flat, _ = MP.matrix_profile(flat, 8)
    check("constant series has ~zero profile", all(abs(v) < 1e-9 for v in pf_flat if v != math.inf))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
