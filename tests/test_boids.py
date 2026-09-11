"""Tests for boids: emergent flocking from three local rules."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import boids

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Random flock: right size, positions in box, speed ~ given.
flock = boids.random_flock(30, size=100.0, speed=2.0, seed=1)
check("flock has n boids", len(flock) == 30)
check("positions in box", all(0 <= b[0] <= 100 and 0 <= b[1] <= 100 for b in flock))
check("initial speed ~2", all(abs(math.hypot(b[2], b[3]) - 2.0) < 1e-6 for b in flock))

# Polarization of a random flock is low; of an aligned flock is 1.
check("random flock low polarization", boids.polarization(flock) < 0.6)
aligned = [(i * 5.0, 0.0, 1.0, 0.0) for i in range(10)]   # all heading +x
check("aligned flock polarization = 1", abs(boids.polarization(aligned) - 1.0) < 1e-9)
anti = [(0.0, 0.0, 1.0, 0.0), (5.0, 0.0, -1.0, 0.0)]      # opposite headings
check("antiparallel pair polarization = 0", abs(boids.polarization(anti)) < 1e-9)

# Separation: two boids on top of each other steer apart.
close = [(50.0, 50.0, 0.0, 0.0), (52.0, 50.0, 0.0, 0.0)]
ax, ay = boids.steering(close, 0)
check("separation pushes boid 0 away (-x)", ax < 0.0)

# Alignment: a boid among neighbours all heading +x (spaced beyond the separation radius, so
# only alignment/cohesion act) is steered toward +x.
group = [(50.0, 50.0, 0.0, 1.0)] + [(50.0 + 10 * i, 50.0, 2.0, 0.0) for i in range(1, 2)]
axg, ayg = boids.steering(group, 0, radius=30.0, sep_radius=5.0)
check("alignment steers toward neighbour heading (+x)", axg > 0.0)

# step keeps flock size and bounds speed.
f2 = boids.step(flock, max_speed=3.0, size=100.0)
check("step preserves flock size", len(f2) == 30)
check("speed capped at max", all(math.hypot(b[2], b[3]) <= 3.0 + 1e-9 for b in f2))
check("positions stay in box after step", all(0 <= b[0] < 100 and 0 <= b[1] < 100 for b in f2))

# Flocking emerges: polarization rises from a random start after many steps.
p0 = boids.polarization(flock)
evolved = boids.evolve(flock, 120, max_speed=3.0, size=100.0)
p1 = boids.polarization(evolved)
check("polarization increases (flock aligns)", p1 > p0)
check("evolved flock fairly aligned", p1 > 0.5)

# Separation keeps boids apart: mean nearest distance stays positive (not all collapsed).
mnd = boids.mean_nearest_distance(evolved)
check("boids keep some spacing", mnd > 0.5)

# mean_nearest_distance sensible.
check("nearest distance of a pair", abs(boids.mean_nearest_distance([(0,0,0,0),(3,4,0,0)]) - 5.0) < 1e-9)
check("single boid nearest distance 0", boids.mean_nearest_distance([(0,0,0,0)]) == 0.0)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all boids tests passed")
