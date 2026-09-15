"""Validate Hopfield: energy descent, stored patterns as fixed points, recall, capacity, spurious states."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import hopfield as hf


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Hopfield network tests")

    # --- weight matrix symmetric with zero diagonal ---
    pats = hf.random_patterns(3, 20, seed=1)
    w = hf.hebbian_weights(pats)
    n = len(w)
    sym = all(abs(w[i][j] - w[j][i]) < 1e-12 for i in range(n) for j in range(n))
    diag = all(abs(w[i][i]) < 1e-12 for i in range(n))
    check("weights symmetric", sym)
    check("zero diagonal", diag)

    # --- stored patterns are fixed points (few patterns, well below capacity) ---
    all_fixed = all(hf.is_fixed_point(w, p) for p in pats)
    check("stored patterns are fixed points", all_fixed)

    # --- and so are their negations (sign symmetry) ---
    negs_fixed = all(hf.is_fixed_point(w, [-x for x in p]) for p in pats)
    check("negated patterns are fixed points", negs_fixed)

    # --- stored patterns have lower energy than random states ---
    rng = hf._Rng(99)
    stored_e = hf.energy(w, pats[0])
    rand_states = [[rng.bit() for _ in range(n)] for _ in range(20)]
    rand_e = sum(hf.energy(w, s) for s in rand_states) / len(rand_states)
    check(f"stored energy {stored_e:.3f} < mean random {rand_e:.3f}", stored_e < rand_e)

    # --- energy never increases under an asynchronous flip ---
    # start from a random state, apply single flips toward the field, track energy
    s = [rng.bit() for _ in range(n)]
    e_prev = hf.energy(w, s)
    ok = True
    for _ in range(200):
        i = rng.randint(n)
        h = hf.local_field(w, s, i)
        new = hf._sign(h)
        if new != 0 and new != s[i]:
            s[i] = new
            e_now = hf.energy(w, s)
            if e_now > e_prev + 1e-9:
                ok = False
                break
            e_prev = e_now
    check("energy non-increasing under async flips", ok)

    # --- convergence: async update reaches a fixed point ---
    rng2 = hf._Rng(7)
    start = [rng2.bit() for _ in range(n)]
    final, converged, sweeps = hf.update_async(w, start, rng2, sweeps=50)
    check(f"async update converges ({sweeps} sweeps)", converged and hf.is_fixed_point(w, final))

    # --- recall from a corrupted cue: flip a few bits, recover the pattern exactly ---
    # use a well-under-capacity net (N=100, 3 patterns -> p/N=0.03) so the basin is deep
    rpats = hf.random_patterns(3, 100, seed=4)
    rw = hf.hebbian_weights(rpats)
    target = rpats[0]
    cue = hf.flip_bits(target, 12, seed=5)  # 12 of 100 bits wrong
    rng3 = hf._Rng(123)
    out = hf.recall(rw, cue, rng3, sweeps=30)
    m = hf.overlap(out, target)
    check(f"recall from 12-bit-corrupted cue (overlap {m:.3f})", abs(m) > 0.99)

    # --- overlap sanity: identical=1, negation=-1 ---
    check("overlap(x,x) == 1", abs(hf.overlap(target, target) - 1.0) < 1e-12)
    check("overlap(x,-x) == -1", abs(hf.overlap(target, [-t for t in target]) + 1.0) < 1e-12)

    # --- capacity: recall accuracy high below ~0.138N, degrades above ---
    N = 80
    low_load = max(1, int(0.05 * N))   # 4 patterns
    high_load = int(0.40 * N)          # 32 patterns, well over capacity
    acc_low = hf.recall_accuracy(low_load, N, flips=8, seed=2)
    acc_high = hf.recall_accuracy(high_load, N, flips=8, seed=2)
    check(f"recall accurate below capacity (acc {acc_low:.3f} at p/N=0.05)", acc_low > 0.95)
    check(f"recall degrades above capacity (acc {acc_high:.3f} at p/N=0.40 < {acc_low:.3f})",
          acc_high < acc_low - 0.05)

    # --- capacity ratio constant exposed and sane ---
    check("capacity ratio ~0.138", abs(hf.CAPACITY_RATIO - 0.138) < 1e-9)

    # --- spurious mixture state is (typically) stable below capacity ---
    # store 3 patterns in a larger net, build sign(xi1+xi2+xi3), check it is a fixed point
    p3 = hf.random_patterns(3, 60, seed=11)
    w3 = hf.hebbian_weights(p3)
    mix = hf.spurious_mixture(p3, [1, 1, 1])
    # the mixture is not one of the stored patterns
    is_stored = any(mix == p for p in p3)
    check("mixture is a novel state (not stored)", not is_stored)
    check("odd 3-mixture is a stable spurious attractor", hf.is_fixed_point(w3, mix))

    # --- determinism: same seed -> same recall ---
    r_a = hf.recall(w, cue, hf._Rng(42), sweeps=30)
    r_b = hf.recall(w, cue, hf._Rng(42), sweeps=30)
    check("recall deterministic per seed", r_a == r_b)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
