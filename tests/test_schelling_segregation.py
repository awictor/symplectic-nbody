"""Validate Schelling: segregation emerges above tau, happiness rises, higher tau more segregation, conservation."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import schelling_segregation as ss


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Schelling segregation tests")

    # --- segregation emerges: final similarity far above the tolerance threshold ---
    tau = 1.0 / 3
    res = ss.simulate(30, 30, empty_frac=0.1, tau=tau, seed=1)
    check(f"final similarity >> tau ({res['mean_similarity']:.3f} vs tau={tau:.2f})",
          res["mean_similarity"] > tau + 0.25)
    check(f"segregation increased ({res['initial_similarity']:.3f} -> {res['mean_similarity']:.3f})",
          res["mean_similarity"] > res["initial_similarity"] + 0.1)

    # --- happy fraction rises toward 1 ---
    res = ss.simulate(30, 30, empty_frac=0.1, tau=tau, seed=1, track=True)
    happies = [h[1] for h in res["history"]]
    check(f"happy fraction rises ({happies[0]:.3f} -> {happies[-1]:.3f})", happies[-1] > happies[0])
    check(f"ends nearly all happy ({res['happy_fraction']:.3f})", res["happy_fraction"] > 0.9)

    # --- higher tolerance threshold -> more segregation ---
    s_low = ss.simulate(30, 30, 0.1, tau=0.3, seed=2)["mean_similarity"]
    s_high = ss.simulate(30, 30, 0.1, tau=0.6, seed=2)["mean_similarity"]
    check(f"higher tau -> more segregation ({s_low:.3f} < {s_high:.3f})", s_high > s_low)

    # --- moves conserve empty count and each type ---
    grid, rng = ss.make_grid(20, 20, 0.15, seed=3)
    before = ss.type_counts(grid)
    for _ in range(5):
        ss.step(grid, 0.4, rng)
    after = ss.type_counts(grid)
    check("moves conserve empty/type counts", before == after)

    # --- tau=0: everyone always happy, essentially no segregation change ---
    res0 = ss.simulate(30, 30, 0.1, tau=0.0, seed=1, track=True)
    check("tau=0: no moves (all happy)", res0["rounds"] == 1)  # one round finds zero unhappy -> stops
    check("tau=0: similarity ~ random start", abs(res0["mean_similarity"] - res0["initial_similarity"]) < 1e-9)

    # --- similarity is a fraction in [0,1] ---
    check("mean similarity in [0,1]", 0 <= res["mean_similarity"] <= 1)

    # --- an isolated agent (no occupied neighbours) counts as similarity 1 (happy) ---
    lone = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    check("isolated agent similarity 1", ss.similarity(lone, 0, 0) == 1.0)

    # --- similarity computes correctly on a hand case ---
    # center type-1 with neighbours: 3 type-1, 2 type-2, rest empty
    g = [[1, 1, 2],
         [1, 1, 2],
         [0, 0, 0]]
    # center (1,1) neighbours: (0,0)=1,(0,1)=1,(0,2)=2,(1,0)=1,(1,2)=2,(2,0)=0,(2,1)=0,(2,2)=0
    # occupied: 1,1,2,1,2 -> 3 of 5 same -> 0.6
    check("hand similarity 3/5", abs(ss.similarity(g, 1, 1) - 0.6) < 1e-12)

    # --- happy fraction 1.0 on a fully segregated grid ---
    seg = [[1, 1, 0, 2, 2],
           [1, 1, 0, 2, 2],
           [1, 1, 0, 2, 2]]
    check("segregated grid all happy", ss.happy_fraction(seg, 0.5) == 1.0)

    # --- deterministic ---
    a = ss.simulate(15, 15, 0.1, 0.4, seed=42)
    b = ss.simulate(15, 15, 0.1, 0.4, seed=42)
    check("deterministic", a["grid"] == b["grid"] and a["rounds"] == b["rounds"])

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
