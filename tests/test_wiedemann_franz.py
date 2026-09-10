"""Tests for wiedemann_franz: the thermal/electrical conductivity ratio."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import wiedemann_franz as wf

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Lorenz number is 2.44e-8 W ohm / K^2.
check("Lorenz number ~2.44e-8", abs(wf.lorenz_number() - 2.44e-8) < 0.02e-8)

# Copper: sigma ~ 5.96e7 S/m at 300 K -> kappa ~ 436 W/(m K) (measured ~400, WF a bit high).
kappa_cu = wf.thermal_conductivity(5.96e7, 300.0)
check("copper thermal conductivity ~400-440", 390.0 < kappa_cu < 445.0)

# kappa = L sigma T: scales with sigma and T.
check("kappa linear in sigma",
      abs(wf.thermal_conductivity(2 * 5.96e7, 300.0) - 2.0 * kappa_cu) < 1e-6)
check("kappa linear in temperature",
      abs(wf.thermal_conductivity(5.96e7, 600.0) - 2.0 * kappa_cu) < 1e-6)

# electrical_conductivity inverts thermal_conductivity.
sigma_back = wf.electrical_conductivity(kappa_cu, 300.0)
check("electrical_conductivity inverts", abs(sigma_back - 5.96e7) / 5.96e7 < 1e-9)

# Effective Lorenz from consistent kappa, sigma equals the Sommerfeld value.
check("effective Lorenz recovers L",
      abs(wf.effective_lorenz(kappa_cu, 5.96e7, 300.0) - wf.lorenz_number()) < 1e-12)

# Copper obeys the law (using its actual measured kappa ~ 401).
check("copper obeys Wiedemann-Franz", wf.obeys_law(401.0, 5.96e7, 300.0))

# Silver: sigma ~ 6.3e7, measured kappa ~ 429 -> obeys.
check("silver obeys the law", wf.obeys_law(429.0, 6.3e7, 300.0))

# A material with heat carried very differently (tiny kappa for its sigma) violates the law.
check("suppressed kappa violates the law", not wf.obeys_law(50.0, 5.96e7, 300.0))
# Effective Lorenz far below Sommerfeld in that case.
check("effective Lorenz below L when suppressed",
      wf.effective_lorenz(50.0, 5.96e7, 300.0) < wf.lorenz_number())

# An insulator's phonon heat conduction (high kappa, tiny sigma) gives a huge effective L,
# i.e. the electronic law does not apply.
check("phonon insulator: effective L >> Sommerfeld",
      wf.effective_lorenz(30.0, 1e-6, 300.0) > 100.0 * wf.lorenz_number())
check("insulator does not obey electronic WF law",
      not wf.obeys_law(30.0, 1e-6, 300.0))

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all wiedemann_franz tests passed")
