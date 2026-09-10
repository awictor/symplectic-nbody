"""Tests for stark: electric-field splitting of spectral lines."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import stark as st

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Linear Stark shift proportional to field, n, and k.
E = 1e7   # 10 MV/m
check("linear shift proportional to field",
      abs(st.linear_stark_shift(2, 1, 2 * E) - 2 * st.linear_stark_shift(2, 1, E)) < 1e-40)
check("linear shift proportional to k",
      abs(st.linear_stark_shift(2, 2, E) - 2 * st.linear_stark_shift(2, 1, E)) < 1e-40)
check("k=0 gives no shift", st.linear_stark_shift(3, 0, E) == 0.0)
check("shift flips sign with k",
      abs(st.linear_stark_shift(2, -1, E) + st.linear_stark_shift(2, 1, E)) < 1e-40)

# Linear Stark pattern: n=2 -> 3 components (k = -1, 0, 1), symmetric about zero.
p2 = st.linear_stark_pattern(2, E)
check("n=2 gives 3 components", len(p2) == 3)
check("pattern symmetric about zero", abs(p2[0] + p2[-1]) < 1e-40)
check("middle component unshifted", abs(p2[1]) < 1e-40)
# n=3 -> 5 components.
check("n=3 gives 5 components", len(st.linear_stark_pattern(3, E)) == 5)
# Components equally spaced.
p3 = st.linear_stark_pattern(3, E)
gaps = [p3[i + 1] - p3[i] for i in range(len(p3) - 1)]
check("components equally spaced", max(gaps) - min(gaps) < 1e-40)

# Quadratic Stark shift: negative, proportional to E^2 and polarizability.
alpha = 1e-40   # SI polarizability
q = st.quadratic_stark_shift(alpha, E)
check("quadratic shift negative", q < 0.0)
check("quadratic shift proportional to E^2",
      abs(st.quadratic_stark_shift(alpha, 2 * E) - 4 * q) < 1e-50)
check("quadratic shift proportional to polarizability",
      abs(st.quadratic_stark_shift(2 * alpha, E) - 2 * q) < 1e-50)

# Induced dipole p = alpha E.
check("induced dipole = alpha E", abs(st.induced_dipole(alpha, E) - alpha * E) < 1e-50)
check("stronger field, bigger dipole", st.induced_dipole(alpha, 2 * E) > st.induced_dipole(alpha, E))

# Field ionization: higher binding -> higher threshold field.
bind_ground = st.hydrogen_binding_energy(1)
bind_rydberg = st.hydrogen_binding_energy(30)
check("Rydberg binding much smaller than ground", bind_rydberg < bind_ground / 100)
check("Rydberg ionizes at lower field",
      st.field_ionization_threshold(bind_rydberg) < st.field_ionization_threshold(bind_ground))
check("threshold scales as binding^2",
      abs(st.field_ionization_threshold(2 * bind_ground)
          - 4 * st.field_ionization_threshold(bind_ground)) < 1e-3 * st.field_ionization_threshold(4 * bind_ground) + 1e6)

# Hydrogen ground binding ~13.6 eV = 2.18e-18 J.
check("hydrogen ground binding ~2.18e-18 J", abs(bind_ground - 2.18e-18) < 0.02e-18)
# n scaling: binding ~ 1/n^2.
check("binding ~ 1/n^2",
      abs(st.hydrogen_binding_energy(2) - bind_ground / 4) < 1e-25)

# A Rydberg atom (n=30) ionizes at a modest lab field (< 1e6 V/m); ground H needs ~1e11.
check("Rydberg n=30 ionizes below 1e7 V/m",
      st.field_ionization_threshold(bind_rydberg) < 1e7)
check("ground-state H needs a huge field (>1e10 V/m)",
      st.field_ionization_threshold(bind_ground) > 1e10)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all stark tests passed")
