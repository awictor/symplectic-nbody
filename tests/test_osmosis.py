"""Tests for osmosis: van't Hoff osmotic pressure, checked against seawater and blood."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import osmosis as osm

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Seawater: ~1.1 mol/kg of dissolved ions at 288 K -> ~27 atm (classic desalination figure).
# Model as ~0.6 mol/L NaCl with i=2 -> 1.2 osmol/L = 1200 osmol/m^3.
pi_sea = osm.osmotic_pressure(600.0, 288.0, vant_hoff_i=2.0)
check("seawater ~27 atm", 24.0 < pi_sea / osm.ATM < 30.0)

# Blood plasma ~0.30 osmol/L at 310 K -> ~7.6 atm.
pi_blood = osm.osmotic_pressure(300.0, 310.0, vant_hoff_i=1.0)   # 300 mol/m^3 = 0.30 osmol/L
check("blood plasma ~7.6 atm", 7.0 < pi_blood / osm.ATM < 8.2)

# van't Hoff Pi = i c R T exactly.
check("Pi = i c R T", abs(osm.osmotic_pressure(100.0, 300.0, 2.0)
                          - 2.0 * 100.0 * osm.R_GAS * 300.0) < 1e-6)

# NaCl (i=2) gives twice the pressure of glucose (i=1) at equal molarity.
check("NaCl double glucose pressure",
      abs(osm.osmotic_pressure(50.0, 298.0, 2.0)
          - 2.0 * osm.osmotic_pressure(50.0, 298.0, 1.0)) < 1e-6)

# concentration_from_pressure inverts osmotic_pressure.
pi = osm.osmotic_pressure(42.0, 298.0, 1.0)
check("concentration_from_pressure inverts",
      abs(osm.concentration_from_pressure(pi, 298.0, 1.0) - 42.0) < 1e-9)

# Osmometry: 10 kg/m^3 (10 g/L) of a protein giving a small pressure -> large molar mass.
# 10 g/L of 60 kg/mol protein at 298 K: c = 10/60000 mol/L = 1.667e-4 mol/L = 0.1667 mol/m^3.
c = 10.0 / 60.0                      # mol/m^3
pi_prot = osm.osmotic_pressure(c, 298.0, 1.0)
M = osm.molar_mass_from_pressure(10.0, pi_prot, 298.0, 1.0)
check("osmometry recovers 60 kg/mol", abs(M - 60.0) < 1e-6)

# Higher T raises pressure at fixed concentration.
check("pressure rises with temperature",
      osm.osmotic_pressure(100.0, 350.0) > osm.osmotic_pressure(100.0, 250.0))

# Osmolarity counts dissociation.
check("osmolarity = i*c", abs(osm.osmolarity(100.0, 3.0) - 300.0) < 1e-9)

# Tonicity: 0.9% saline (~0.15 mol/L NaCl, i=2 -> 0.30 osmol/L) is isotonic with blood.
check("normal saline isotonic with blood",
      osm.tonicity(150.0, 2.0, 300.0, 1.0) == "isotonic")
check("pure water is hypotonic",
      osm.tonicity(0.0, 1.0, 300.0, 1.0) == "hypotonic")
check("concentrated brine is hypertonic",
      osm.tonicity(1000.0, 2.0, 300.0, 1.0) == "hypertonic")

# Reverse osmosis must exceed the feed osmotic pressure.
check("RO pressure = feed osmotic pressure",
      abs(osm.reverse_osmosis_pressure(600.0, 288.0, 2.0) - pi_sea) < 1e-6)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all osmosis tests passed")
