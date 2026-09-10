"""Tests for metropolis: MCMC sampling of the 2D Ising model."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import metropolis as mp

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Acceptance rule: always accept downhill, Boltzmann uphill.
check("downhill move always accepted", mp.accept_probability(-1.0, 2.0) == 1.0)
check("zero-energy move accepted", mp.accept_probability(0.0, 2.0) == 1.0)
check("uphill accepted with exp(-dE/T)", abs(mp.accept_probability(2.0, 1.0) - math.exp(-2.0)) < 1e-9)
check("colder temperature suppresses uphill more",
      mp.accept_probability(2.0, 0.5) < mp.accept_probability(2.0, 2.0))
check("acceptance in (0,1]", 0 < mp.accept_probability(3.0, 1.5) <= 1.0)

# Ising energy: fully aligned grid is the minimum; each of 2 N^2 bonds contributes -J.
n = 8
aligned = [[1] * n for _ in range(n)]
check("aligned energy = -2 J N^2", abs(mp.ising_energy(aligned) - (-2.0 * n * n)) < 1e-9)
# A single flipped spin raises the energy.
one_flip = [row[:] for row in aligned]
one_flip[0][0] = -1
check("flipping a spin raises energy", mp.ising_energy(one_flip) > mp.ising_energy(aligned))

# Magnetization: aligned = 1, checkerboard = 0.
check("aligned magnetization = 1", abs(mp.magnetization(aligned) - 1.0) < 1e-9)
checker = [[1 if (r + c) % 2 == 0 else -1 for c in range(n)] for r in range(n)]
check("checkerboard magnetization = 0", abs(mp.magnetization(checker)) < 1e-9)

# Onsager T_c ~ 2.269 (in J/k_B).
check("Onsager T_c ~ 2.269", abs(mp.ONSAGER_TC - 2.269) < 0.001)

# Full run: ordered (m ~ 1) well below T_c, disordered (m ~ 0) well above.
m_cold = mp.run(16, 1.0, sweeps=120, seed=1)
m_hot = mp.run(16, 4.0, sweeps=120, seed=1)
check("ordered below T_c (m high)", m_cold > 0.85)
check("disordered above T_c (m low)", m_hot < 0.3)
check("magnetization drops across T_c", m_cold > m_hot)

# Monotone-ish: colder is more ordered than near-critical.
m_mid = mp.run(16, 2.3, sweeps=120, seed=1)
check("near-T_c between ordered and disordered", m_hot < m_mid < m_cold + 1e-9)

# Energy after a hot run is well above the ground state (disorder costs energy).
rng = mp._Rng(5)
g = [[1] * 12 for _ in range(12)]
for _ in range(50):
    mp.sweep(g, 5.0, rng)
check("hot configuration well above ground energy",
      mp.ising_energy(g) > mp.ising_energy([[1] * 12 for _ in range(12)]) + 100)

# A cold run keeps the grid essentially aligned (energy near the minimum).
rng2 = mp._Rng(5)
gc = [[1] * 12 for _ in range(12)]
for _ in range(80):
    mp.sweep(gc, 0.5, rng2)
check("cold run stays near ground state", mp.ising_energy(gc) < -2.0 * 12 * 12 * 0.85)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all metropolis tests passed")
