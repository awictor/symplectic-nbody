# symplectic-nbody

A dependency-free (pure-Python stdlib) N-body gravitational simulator built to
make one deep fact of numerical physics *observable and testable*:

> **Symplectic integrators conserve energy over exponentially long times.
> Higher local accuracy (RK4) does not save you — its energy drifts secularly.**

This is why real celestial-mechanics codes integrate the solar system for
billions of years with a humble 2nd-order leapfrog instead of a fancy
adaptive Runge-Kutta.

## Quick look

```
$ python examples/energy_drift_demo.py

Eccentric two-body (e=0.7), dt=0.01, 40k steps, G=1

method        max |dE/E|      net dE/E        drift shape
------------------------------------------------------------------------
verlet        4.916e-03       -1.112e-07      # @ # # * * + = = - : : . . .
forest_ruth   1.080e-05       1.114e-11       # @ # # * * + + = - - : : : . . .
rk4           1.379e-04       1.379e-04              ...:::---===++++****####@@@
```

Look at the `net dE/E` column and the drift shape. The symplectic methods'
energy **oscillates around zero and returns**. RK4's energy **walks off in one
direction** — a monotone ramp. That ramp is the secular drift that eventually
ruins a long non-symplectic integration.

## What's inside

| File | Contents |
|------|----------|
| `src/integrators.py` | `velocity_verlet` (2nd-order symplectic), `forest_ruth` (4th-order symplectic), `rk4` (non-symplectic foil) |
| `src/nbody.py` | Newtonian forces, kinetic/potential energy, linear & angular momentum, softening |
| `src/barnes_hut.py` | O(N log N) octree force solver with opening-angle theta criterion |
| `src/adaptive.py` | Dormand-Prince RK45 with PI error-controlled adaptive step size |
| `src/kepler.py` | Exact analytic two-body orbit (Kepler-equation solver) -- the ground truth |
| `src/cr3bp.py` | Circular restricted 3-body problem: Lagrange points, Jacobi constant |
| `src/solar_system.py` | The real 8-planet solar system from published orbital elements |
| `src/relativity.py` | First post-Newtonian gravity & Mercury's perihelion precession |
| `src/lyapunov.py` | Largest Lyapunov exponent (Benettin shadow-trajectory method) |
| `src/systems.py` | Test systems: two-body, figure-eight, pythagorean 3-body, **Plummer-sphere star cluster** (any N) |
| `tests/test_conservation.py` | Automated checks of every conservation claim |
| `tests/test_barnes_hut.py` | Tree force validated against exact O(N^2) summation |
| `src/render_svg.py` | Zero-dependency SVG trajectory renderer (orbit paths -> standalone .svg) |
| `examples/energy_drift_demo.py` | The ASCII energy-drift demo above |
| `examples/scaling_benchmark.py` | Direct vs Barnes-Hut timing & empirical scaling exponent |
| `examples/plot_orbits.py` | Render figure-eight / eccentric / pythagorean orbits to SVG |
| `examples/adaptive_demo.py` | Adaptive DP45 vs fixed RK4: step adaptation & force-eval savings |
| `examples/convergence_demo.py` | Measured convergence order of each method vs the exact orbit |
| `examples/lagrange_demo.py` | Lagrange points + zero-velocity curves rendered to SVG |
| `examples/solar_system_demo.py` | Integrate the real solar system, recover Kepler's third law |
| `examples/precession_demo.py` | Mercury's 43"/century precession + a relativistic rosette SVG |
| `examples/chaos_demo.py` | Lyapunov exponent + two trajectories diverging 6 orders of magnitude |
| `examples/build_dashboard.py` | Assemble all demos into one self-contained `index.html` |

## Barnes-Hut: scaling to many bodies

Direct summation is O(N^2). For a star cluster of thousands of bodies that is
hopeless. `barnes_hut.py` builds an octree, collapses distant groups of bodies
to their centre of mass, and uses them wholesale when the opening angle
`theta = cell_width / distance` is small enough. Same acceleration interface, so
it drops straight into the same symplectic integrators.

```
$ python examples/scaling_benchmark.py

     N    direct (ms)     bh (ms)   speedup
---------------------------------------------
   200           9.15       12.19      0.8x
  3200        2680.78     1126.60      2.4x

empirical scaling exponent  direct ~ N^2.04   barnes-hut ~ N^1.64
```

`theta=0` reproduces direct summation to machine precision; `theta=0.5` is the
classic accuracy/speed sweet spot (matches exact forces to a few percent).
Tightening `theta` provably reduces the error — all checked in the tests. The
`plummer_sphere(n=...)` generator builds an equilibrium cluster (positions from
the Plummer inverse-CDF, velocities by rejection sampling the exact distribution
function) using a tiny built-in LCG, so it's deterministic and dependency-free.

## Chaos: why the three-body problem is unpredictable

The deepest fact in dynamics: exact equations can still defy long-term
prediction. `lyapunov.py` estimates the largest Lyapunov exponent by the
Benettin shadow-trajectory method -- evolve a twin orbit an infinitesimal
distance away, measure the growth, renormalize, repeat.

```
$ python examples/chaos_demo.py examples/output

  pythagorean 3-body : lambda = 0.523   Lyapunov time ~ 1.9 time units
  regular two-body   : lambda = 0.035   (decays toward 0 with T)

separation of two trajectories started 1e-9 apart (log scale):
  ..::----------===========++++++++********************#############@
  grew from 1e-9 to ~1.3e-03 -- 6 orders of magnitude.
```

A positive Lyapunov exponent *is* the definition of chaos: a 1e-9 uncertainty
amplifies to order unity in a few Lyapunov times. The tests confirm the chaotic
system's exponent dwarfs a regular one's, and that the regular estimate decays
toward zero as `1/T` (linear, non-exponential separation) while the chaotic one
stays large. The demo also renders two nearly-identical runs peeling apart.

## Mercury's perihelion: the first triumph of general relativity

Newtonian two-body orbits are closed ellipses -- they never precess. The 43
arcsec/century advance of Mercury's perihelion was the anomaly that general
relativity explained. `relativity.py` adds the first post-Newtonian correction
to the acceleration and both derives and integrates the result:

```
$ python examples/precession_demo.py examples/output

  analytic advance at the real speed of light: 42.98 arcsec/century
  observed / GR-predicted value:               ~43 arcsec/century

numeric integration reproduces 6*pi*GM/(c^2 a(1-e^2)):
    c factor   numeric/orbit  analytic/orbit    ratio
  c/300         4.514734e-02    4.517161e-02   0.9995
  c/600         1.798247e-01    1.806865e-01   0.9952
```

The famous number comes straight out of the closed form. To *see* the effect,
the demo amplifies GR (shrinks c) so the ellipse visibly rotates into a rosette
and renders it to SVG. The tests confirm the 43"/century value, that numeric
integration matches the analytic advance, that Newtonian orbits don't precess,
and that the precession scales as 1/c^2.

## The real solar system, and Kepler's third law for free

`solar_system.py` builds all eight planets from published orbital elements in
AU / years / solar masses (so `G = 4*pi^2`). Integrate, measure each period, and
the third law appears on its own:

```
$ python examples/solar_system_demo.py examples/output

planet      a [AU]  T measured   T Kepler    T real   T^2/a^3
-------------------------------------------------------------
Mercury      0.387      0.2410     0.2408    0.2408    1.0010
Earth        1.000      1.0005     1.0000    1.0000    1.0010
Jupiter      5.204     11.9080    11.8724   11.8620    1.0060
Neptune     30.070    164.9740   164.8916  164.7900    1.0010
```

`T^2/a^3` is constant across two orders of magnitude in orbital radius -- that
constant *is* Kepler's third law, and it drops out of Newtonian gravity plus a
symplectic step with no fitting. Measured periods match the real sidereal
periods to better than half a percent. The tests verify the T-vs-a log-log slope
is exactly 3/2 and that the full eight-body system conserves energy.

## One-page dashboard

```
python examples/build_dashboard.py examples/output
```

Runs every demo, captures its output, inlines all the SVGs, and writes a single
self-contained `examples/output/index.html` -- no external assets, no JavaScript,
no build step. Open it in a browser or point GitHub Pages at `examples/output/`.
It's the whole library on one page: energy conservation, convergence order,
adaptive stepping, Barnes-Hut scaling, the orbit gallery, and the Lagrange points.

## Lagrange points: where spacecraft park

Move to the frame co-rotating with two orbiting primaries and five equilibrium
points appear -- the Lagrange points. JWST sits at Sun-Earth L2; Trojan asteroids
cluster at Sun-Jupiter L4/L5. `cr3bp.py` builds the circular restricted 3-body
problem, locates all five points, and exposes the conserved Jacobi constant.

```
$ python examples/lagrange_demo.py examples/output

Earth-Moon CR3BP (mu = 0.01215)
point            x           y      Jacobi C
--------------------------------------------
L1        0.836918    0.000000      3.188336
L2        1.155680    0.000000      3.172156
L3       -1.005062    0.000000      3.012147
L4        0.487850    0.866025      2.987998
L5        0.487850   -0.866025      2.987998
```

The collinear points L1/L2/L3 are found by 1-D root-finding on the effective
potential; L4/L5 are the exact equilateral-triangle points. The tests verify
every point is a true equilibrium (|grad Omega| < 1e-9), the Jacobi constant is
conserved along trajectories (~1e-11), and -- the elegant part -- **L4 stability
flips at the Routh mass ratio** mu ~ 0.0385: below it a nudged particle librates
in a bounded loop (like the Trojans), above it the particle escapes. The demo
also renders the zero-velocity (Hill) curves to SVG via marching squares.

## Ground truth: convergence against the exact Kepler orbit

The two-body problem has a closed-form solution (solve Kepler's equation
`M = E - e sin E` for the eccentric anomaly). `kepler.py` gives the *exact*
position at any time, so we can measure an integrator's true error -- and read
its convergence order straight off the data by halving the step:

```
$ python examples/convergence_demo.py

verlet:                      forest_ruth:                 rk4:
   steps     error  order       steps     error  order      steps     error  order
    1000  1.13e-04   2.00        1000  5.07e-08   4.00       1000  5.80e-09   4.10
    2000  2.81e-05   2.00        2000  3.17e-09   4.00       2000  3.50e-10   4.05
    4000  7.03e-06   2.00        4000  1.98e-10   4.00       4000  2.15e-11   4.03
```

Halving the step cuts verlet's error 4x (order 2) and forest_ruth/rk4's error
16x (order 4) -- exactly as theory predicts, confirmed empirically. forest_ruth
and rk4 share an order, but only forest_ruth is symplectic, so only it also keeps
energy bounded forever. Order buys short-term accuracy; symplecticity buys
long-term stability. This repo measures both.

## Adaptive stepping: same accuracy, far less work

Symplectic methods win the *long-term energy* game. But when you just need a
high-accuracy trajectory over a bounded time, an error-controlled adaptive step
wins the *efficiency* game -- it spends tiny steps at pericenter (where the orbit
moves fast) and long steps at apocenter (where nothing happens).

```
$ python examples/adaptive_demo.py

adaptive step size over the orbit (small=pericenter, large=apocenter):
  .=+**###@##*+==---::..::::.......  ..........::::::--===++*#######*-
  steps accepted=109 rejected=13  h_min=1.00e-03 h_max=1.17e-01 ratio=117x

method                     force evals     end error
----------------------------------------------------
adaptive DP45                      854      3.95e-08
fixed RK4 (2000 steps)            8000      8.02e-09

adaptive reaches the same accuracy with 9.4x fewer force evaluations.
```

`adaptive.DormandPrince` is the Dormand-Prince 5(4) embedded pair (the method
behind MATLAB's `ode45` / SciPy's `RK45`): two solutions of different order share
the same stages, their difference estimates the local error, and the step grows
or shrinks to hold that error near tolerance. It is **not** symplectic, so it's
the right tool for bounded high-accuracy runs, not billion-year integrations --
the complement to the leapfrog family, and the contrast makes the tradeoff clear.

## Plotting orbits (no dependencies)

```
python examples/plot_orbits.py examples/output
```

Writes standalone SVGs you can open in any browser -- see `examples/output/`.
`render_svg.py` projects the 3D trajectory onto a chosen plane, draws each body's
path as a polyline (hollow marker = start, filled = end), and stamps the net
energy drift. The figure-eight closes on itself to 1 part in 1e13; the softened
pythagorean 3-body stays energy-stable (drift ~1e-9) through its close encounters.

It also writes an `*_animated.svg` for each system: the bodies actually **orbit**
along their computed paths using SMIL `<animateMotion>` -- pure declarative SVG
animation, no JavaScript, no dependencies. Open `figure_eight_animated.svg` in a
browser and watch three masses chase each other around the shared figure-eight.

## The claims, checked automatically

Run:

```
python tests/test_conservation.py
```

1. **Verlet keeps energy bounded** — relative drift stays below 1e-3 over 20k steps.
2. **Forest-Ruth beats Verlet** — 4th order has smaller max energy error than 2nd.
3. **RK4 drifts secularly** — on an eccentric orbit, the least-squares *slope* of
   RK4's energy error exceeds Verlet's by >3x (Verlet only oscillates; RK4 ramps).
4. **Momentum conserved** — linear & angular momentum held to ~1e-9.

## The math, briefly

A symplectic integrator exactly conserves a *shadow Hamiltonian* H̃ = H + O(dt^p)
that stays close to the true H. Because H̃ is conserved exactly, the true energy H
can only oscillate within O(dt^p) of its start — it can never drift away. RK4
conserves no nearby Hamiltonian, so nothing pins its energy, and truncation error
accumulates into a one-way walk.

`velocity_verlet` uses the kick-drift-kick (leapfrog) form; `forest_ruth` uses the
Forest & Ruth (1990) triple-jump coefficients w1 = 1/(2 - 2^(1/3)).

## Design notes

- **Zero dependencies.** Pure stdlib `math`. Runs on any Python 3.
- **G = 1 units**, the convention for celestial-mechanics test problems.
- **Softening** parameter tames the r -> 0 singularity for chaotic close encounters.
- O(N^2) direct summation — clear over clever. A Barnes-Hut tree is the natural
  next step for large N.
