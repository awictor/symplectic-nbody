"""Tests for cellular_automaton: elementary Wolfram rules."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cellular_automaton as ca

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Rule table: rule 90 = 01011010 -> bits at k=1,3,4,6.
t90 = ca.rule_table(90)
check("rule 90 table", t90 == [0, 1, 0, 1, 1, 0, 1, 0])
# Rule 0 all zero, rule 255 all one.
check("rule 0 all zero", ca.rule_table(0) == [0] * 8)
check("rule 255 all one", ca.rule_table(255) == [1] * 8)

# Output for a neighbourhood.
# Rule 90 is XOR of the two neighbours: out = left XOR right.
for l in (0, 1):
    for c in (0, 1):
        for r in (0, 1):
            check(f"rule 90 = left XOR right ({l}{c}{r})",
                  ca.output_for_neighbourhood(90, l, c, r) == (l ^ r))

# Rule 110 output table = 01101110.
check("rule 110 table", ca.rule_table(110) == [0, 1, 1, 1, 0, 1, 1, 0])

# apply_rule length preserved.
row = ca.single_seed_row(11)
check("single seed has one live cell", ca.population(row) == 1)
nxt = ca.apply_rule(row, 90)
check("apply_rule preserves width", len(nxt) == 11)

# Rule 90 from a single seed: after one step, the two neighbours light up (XOR structure).
check("rule 90 seed -> two cells", ca.population(ca.apply_rule(ca.single_seed_row(21), 90)) == 2)

# Rule 0 kills everything in one step.
check("rule 0 dies immediately", ca.is_all_zero(ca.apply_rule([1, 0, 1, 1, 0], 0)))
# Rule 255 fills everything.
check("rule 255 fills", ca.apply_rule([0, 0, 0], 255) == [1, 1, 1])

# evolve returns generations+1 rows.
rows = ca.evolve(ca.single_seed_row(31), 90, 15)
check("evolve length gens+1", len(rows) == 16)

# Rule 90 draws the Sierpinski triangle: the pattern spreads outward from the seed (peak
# population climbs even though it oscillates step to step -- 1,2,2,4,2,4,4,8,...).
pops = [ca.population(r) for r in rows]
check("rule 90 pattern spreads (peak grows)", max(pops[8:12]) > max(pops[:3]))
# Rule 90 population is a power of 2 (Sierpinski / XOR structure) at each generation.
check("rule 90 population is a power of two",
      all((p & (p - 1)) == 0 for p in pops[:12] if p > 0))

# Rule 30 is chaotic: from a single seed it does NOT die and grows asymmetrically.
rows30 = ca.evolve(ca.single_seed_row(61), 30, 30)
check("rule 30 does not die out", not ca.is_all_zero(rows30[-1]))
check("rule 30 grows from seed", ca.population(rows30[-1]) > 1)
# Rule 30 centre column looks aperiodic: not constant over the run.
centre = [rows30[g][30] for g in range(len(rows30))]
check("rule 30 centre column varies", 0 in centre and 1 in centre)

# Rule 110 sustains structure from a random-ish start (does not die, does not fill).
start = [1 if i % 3 == 0 else 0 for i in range(50)]
rows110 = ca.evolve(start, 110, 40)
check("rule 110 sustains activity", not ca.is_all_zero(rows110[-1]))
check("rule 110 not saturated", ca.population(rows110[-1]) < 50)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all cellular_automaton tests passed")
