"""Tests for kolmogorov: the turbulent energy cascade."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import kolmogorov as ko

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Dissipation rate epsilon = u^3 / L.
check("epsilon = u^3 / L", abs(ko.dissipation_rate(1.0, 0.1) - 1.0 / 0.1) < 1e-9)
check("epsilon scales as u^3",
      abs(ko.dissipation_rate(2.0, 0.1) - 8.0 * ko.dissipation_rate(1.0, 0.1)) < 1e-9)

# Kolmogorov microscales for a lab flow: nu=1e-6 (water), epsilon=1e-2 W/kg.
nu, eps = 1e-6, 1e-2
eta = ko.kolmogorov_length(nu, eps)
# eta = (1e-18/1e-2)^0.25 = (1e-16)^0.25 = 1e-4 m = 0.1 mm.
check("Kolmogorov length ~0.1 mm", abs(eta - 1e-4) < 1e-6)

tau = ko.kolmogorov_time(nu, eps)
# tau = (1e-6/1e-2)^0.5 = (1e-4)^0.5 = 1e-2 s.
check("Kolmogorov time ~0.01 s", abs(tau - 1e-2) < 1e-4)

u_eta = ko.kolmogorov_velocity(nu, eps)
# u = (1e-6*1e-2)^0.25 = (1e-8)^0.25 = 1e-2 m/s.
check("Kolmogorov velocity ~0.01 m/s", abs(u_eta - 1e-2) < 1e-4)

# The Kolmogorov-scale Reynolds number is exactly 1.
check("Kolmogorov Re = 1", abs(u_eta * eta / nu - 1.0) < 1e-9)
# Consistency: eta = u_eta * tau.
check("eta = u_eta * tau", abs(eta - u_eta * tau) < 1e-12)

# Energy spectrum: E(k) = C eps^(2/3) k^(-5/3).
E1 = ko.energy_spectrum(1.0, eps)
E2 = ko.energy_spectrum(2.0, eps)
check("spectrum -5/3 slope: E(2k)/E(k) = 2^(-5/3)", abs(E2 / E1 - 2.0 ** (-5.0 / 3.0)) < 1e-9)
check("more dissipation, more energy at a scale", ko.energy_spectrum(1.0, 2 * eps) > E1)
# Spectrum falls with wavenumber (small eddies hold less energy).
check("spectrum falls with k", ko.energy_spectrum(10.0, eps) < ko.energy_spectrum(1.0, eps))

# Eddy turnover: smaller eddies turn over faster.
check("small eddies turn over faster",
      ko.eddy_turnover_time(0.01, eps) < ko.eddy_turnover_time(1.0, eps))
# tau(l) scales as l^(2/3).
check("turnover time scales as l^(2/3)",
      abs(ko.eddy_turnover_time(8.0, eps) / ko.eddy_turnover_time(1.0, eps) - 8.0 ** (2.0 / 3.0)) < 1e-6)

# Scale separation L/eta ~ Re^(3/4).
check("L/eta = Re^(3/4)", abs(ko.scale_separation(1e4) - 1e4 ** 0.75) < 1e-3)
check("higher Re, wider inertial range", ko.scale_separation(1e6) > ko.scale_separation(1e4))
# Re = 1e6 -> L/eta ~ 31623 (a huge span of scales).
check("Re=1e6 spans ~3e4 scales", abs(ko.scale_separation(1e6) - 31623.0) < 5.0)

# Physical sanity: dissipation-scale eta is far smaller than the stirring scale for Re>>1.
L = 1.0
u = 1.0
eps_flow = ko.dissipation_rate(u, L)
Re = u * L / nu
eta_flow = ko.kolmogorov_length(nu, eps_flow)
check("eta << L for high Re", eta_flow < L / 1000.0)
check("L/eta matches Re^3/4 within factor 2",
      0.5 < (L / eta_flow) / ko.scale_separation(Re) < 2.0)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all kolmogorov tests passed")
