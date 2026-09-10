"""Fick's laws: how a random walk spreads a concentration out.

A drop of dye in still water, heat through a bar, dopant into silicon -- all spread the same
way, because each molecule executes a random walk and the net flux runs downhill in
concentration. Fick's first law makes that precise:

    J = -D dC/dx,                      (flux proportional to the gradient)

and combining it with conservation of particles gives Fick's second law, the diffusion
equation:

    dC/dt = D d^2C/dx^2.

Two exact solutions carry most of the intuition. A point release of N particles spreads into
a Gaussian whose width grows as sqrt(t):

    C(x, t) = N / sqrt(4 pi D t) * exp(-x^2 / (4 D t)),

so the rms spread is sigma = sqrt(2 D t) in 1-D -- the signature diffusive sqrt(t), never
the ballistic t of a directed motion. A step interface (one region full, one empty) relaxes
through an error-function profile:

    C(x, t) = (C0/2) erfc(x / sqrt(4 D t)),

which is how a doped junction or a quenched temperature front smooths out. The characteristic
distance reached in time t is the diffusion length L = sqrt(D t), and the time to cross a
length L scales as L^2/D -- the reason diffusion rules the microscopic and small but loses
to flow and convection on human scales.

This module gives the flux, the Gaussian and erfc profiles, the rms spread, the diffusion
length and time, and the Stokes-Einstein diffusion coefficient of a sphere, and reproduces
the sqrt(t) spreading and the conservation of the released amount. SI units. Pure stdlib
(erf/erfc from math); the transport companion to the random-walk, osmosis and heat notes.
"""

from __future__ import annotations

import math

K_B = 1.380649e-23             # Boltzmann constant (J/K)


def flux(diffusivity: float, dC: float, dx: float) -> float:
    """Fick's first law J = -D dC/dx: the diffusive flux (mol/(m^2 s)) down a concentration
    gradient dC over distance dx. Negative because flux runs from high to low C."""
    return -diffusivity * dC / dx


def gaussian_profile(x: float, t: float, diffusivity: float, amount: float = 1.0) -> float:
    """Concentration at position x, time t from a point release of `amount` at the origin
    (1-D): C = amount / sqrt(4 pi D t) exp(-x^2/(4 D t)). Units: amount per length."""
    if t <= 0.0:
        return 0.0
    return amount / math.sqrt(4.0 * math.pi * diffusivity * t) * \
        math.exp(-x * x / (4.0 * diffusivity * t))


def rms_spread(t: float, diffusivity: float) -> float:
    """Root-mean-square spread of a 1-D diffusing cloud, sigma = sqrt(2 D t) (m). The
    hallmark sqrt(t) growth of diffusion."""
    return math.sqrt(2.0 * diffusivity * t)


def step_profile(x: float, t: float, diffusivity: float, c0: float = 1.0) -> float:
    """Concentration from an initial step (C = c0 for x < 0, 0 for x > 0) relaxing by
    diffusion: C = (c0/2) erfc(x / sqrt(4 D t)). At x=0 it holds c0/2 for all t."""
    if t <= 0.0:
        return c0 if x < 0.0 else 0.0
    return 0.5 * c0 * math.erfc(x / math.sqrt(4.0 * diffusivity * t))


def diffusion_length(t: float, diffusivity: float) -> float:
    """Characteristic distance reached by diffusion in time t: L = sqrt(D t) (m)."""
    return math.sqrt(diffusivity * t)


def diffusion_time(length: float, diffusivity: float) -> float:
    """Time to diffuse across a length L: t = L^2 / D (s). The quadratic scaling that makes
    diffusion fast in a cell and hopeless across a room."""
    return length * length / diffusivity


def stokes_einstein(temperature: float, viscosity: float, radius: float) -> float:
    """Stokes-Einstein diffusion coefficient of a sphere: D = k_B T / (6 pi eta r) (m^2/s),
    for radius r in a fluid of viscosity eta. Gives ~2e-9 m^2/s for a small molecule in
    water and links Brownian motion to temperature and drag."""
    return K_B * temperature / (6.0 * math.pi * viscosity * radius)
