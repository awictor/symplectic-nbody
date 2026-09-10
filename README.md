# symplectic-nbody

A dependency-free (pure-Python stdlib) **computational-gravity library**. It
began as an N-body integrator built to make one deep fact observable —

> **Symplectic integrators conserve energy over exponentially long times.
> Higher local accuracy (RK4) does not save you — its energy drifts secularly.**

— and grew into a tour of gravity across every regime, from the leapfrog that
keeps the solar system stable for billions of years to black-hole geodesics,
gravitational waves, dark matter, and the expansion of the universe. Every claim
is produced by code here and checked against an analytic result or a famous
measured value (1.75″ light bending, Mercury's 43″/century, the 1.44 M☉ white-
dwarf limit, ξ₁ = π, a 13.5 Gyr universe).

**No dependencies** — pure Python stdlib, no numpy, no matplotlib. All figures
are hand-built SVG; the whole thing runs anywhere Python does.

### Live dashboard

Every result on one self-contained page (no JavaScript, all SVG):
**https://awictor.github.io/symplectic-nbody/** — or `python examples/build_dashboard.py`.

### What's in it

| Domain | Modules |
|--------|---------|
| **Numerics** | 5 integrators (Verlet, Forest-Ruth, RK4, adaptive Dormand-Prince, Hermite), Barnes-Hut O(N log N) tree, exact-Kepler convergence tests |
| **Celestial mechanics** | real solar system + Kepler's 3rd law, Lagrange points, coorbital tadpole/horseshoe orbits, mean-motion resonance, Kozai-Lidov cycles, Tisserand & gravity assists |
| **Chaos** | Lyapunov exponents, Poincaré sections, three-body stability maps, the Sitnikov route to chaos |
| **General relativity** | Mercury perihelion precession, Schwarzschild orbits (ISCO, photon sphere), gravitational lensing |
| **Gravitational waves** | inspiral chirp (Peters energy loss), eccentric-binary circularization |
| **Galaxies & cosmology** | galaxy-collision tidal tails, rotation curves & dark matter, Friedmann expansion, cosmic distances & acceleration |
| **Stellar physics** | Jeans collapse, Lane-Emden structure, Chandrasekhar mass, TOV neutron stars, Sedov-Taylor blast waves, the virial theorem |
| **Visualization** | dependency-free SVG renderer (static + SMIL-animated), one-page HTML dashboard |

The rest of this README walks through each result. This is also the reason real
celestial-mechanics codes integrate the solar system for billions of years with
a humble 2nd-order leapfrog instead of a fancy adaptive Runge-Kutta.

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
| `src/hermite.py` | 4th-order Hermite predictor-corrector with analytic jerk (1 force call/step) |
| `src/kepler.py` | Exact analytic two-body orbit (Kepler-equation solver) -- the ground truth |
| `src/cr3bp.py` | Circular restricted 3-body problem: Lagrange points, Jacobi constant |
| `src/solar_system.py` | The real 8-planet solar system from published orbital elements |
| `src/relativity.py` | First post-Newtonian gravity & Mercury's perihelion precession |
| `src/lyapunov.py` | Largest Lyapunov exponent (Benettin shadow-trajectory method) |
| `src/stability_map.py` | Three-body escape-time scan over a grid of initial conditions |
| `src/virial.py` | Virial theorem & violent relaxation of a self-gravitating cluster |
| `src/cluster.py` | Galaxy-cluster virial temperature, M-T relation & X-ray scaling |
| `src/roche.py` | Roche limit & tidal disruption of a rubble-pile satellite |
| `src/tidal_heating.py` | Tidal heating: Io's volcanic power from orbital flexing |
| `src/roche_lobe.py` | Roche lobes & binary mass-transfer stability (Eggleton) |
| `src/kozai.py` | Kozai-Lidov secular cycles: eccentricity <-> inclination in a triple |
| `src/resonance.py` | Mean-motion resonance: period locks & librating resonant arguments |
| `src/coorbital.py` | Tadpole & horseshoe coorbital orbits in the CR3BP |
| `src/tisserand.py` | Tisserand parameter: the near-invariant of a gravity assist |
| `src/lensing.py` | Gravitational lensing: deflection, Einstein ring, microlensing |
| `src/rotation_curve.py` | Galaxy rotation curves: Keplerian disk vs flat dark-halo curve |
| `src/mond.py` | MOND: modified gravity, flat curves & baryonic Tully-Fisher |
| `src/schwarzschild.py` | Black-hole orbits: effective potential, ISCO, photon sphere, plunge |
| `src/kerr.py` | Rotating black holes: horizons, ergosphere, spin-dependent ISCO |
| `src/penrose.py` | Penrose process: irreducible mass & extractable spin energy |
| `src/hawking.py` | Black-hole thermodynamics: Hawking temperature, entropy, evaporation |
| `src/eddington.py` | Eddington luminosity, accretion rate & Salpeter black-hole growth |
| `src/hohmann.py` | Hohmann transfer: mission delta-v budgets & launch windows |
| `src/oberth.py` | Oberth effect: why rockets burn deep in a gravity well |
| `src/gr_time.py` | Gravitational redshift, GPS clock correction, Shapiro delay |
| `src/lense_thirring.py` | Frame-dragging & geodetic precession (Gravity Probe B) |
| `src/pulsar.py` | Hulse-Taylor binary-pulsar orbital decay (first GW evidence) |
| `src/friedmann.py` | Friedmann cosmology: scale factor a(t), expansion eras, age of the universe |
| `src/saha.py` | Saha equation & cosmic recombination (the CMB release) |
| `src/cmb.py` | CMB acoustic scale: sound horizon & the l~220 first peak |
| `src/bbn.py` | Big Bang nucleosynthesis: n/p freeze-out & primordial helium |
| `src/lane_emden.py` | Lane-Emden stellar structure: polytrope profiles & surface radii |
| `src/distances.py` | Cosmological distances: luminosity/angular-diameter, cosmic acceleration |
| `src/chandrasekhar.py` | White-dwarf structure & the Chandrasekhar mass (~1.44 M_sun) |
| `src/tov.py` | Neutron-star structure via the TOV equation & the GR maximum mass |
| `src/jeans.py` | Jeans instability: the gravitational-collapse / star-formation threshold |
| `src/sedov.py` | Sedov-Taylor blast wave: supernova remnants and the Trinity yield |
| `src/poincare.py` | Poincare surface-of-section for the CR3BP (tori vs chaos) |
| `src/sitnikov.py` | The Sitnikov problem: on-axis test particle, integrable-to-chaotic |
| `src/galaxy.py` | Disk-galaxy generator and two-galaxy tidal encounters |
| `src/dynamical_friction.py` | Chandrasekhar friction: satellites sinking into galaxies |
| `src/gravwave.py` | 2.5PN radiation reaction: gravitational-wave inspiral & chirp |
| `src/gw_strain.py` | GW strain amplitude, chirp mass & LIGO arm-length change |
| `src/systems.py` | Test systems: two-body, figure-eight, pythagorean 3-body, **Plummer-sphere star cluster** (any N) |
| `tests/test_conservation.py` | Automated checks of every conservation claim |
| `tests/test_barnes_hut.py` | Tree force validated against exact O(N^2) summation |
| `src/render_svg.py` | Zero-dependency SVG trajectory renderer (orbit paths -> standalone .svg) |
| `examples/energy_drift_demo.py` | The ASCII energy-drift demo above |
| `examples/scaling_benchmark.py` | Direct vs Barnes-Hut timing & empirical scaling exponent |
| `examples/plot_orbits.py` | Render figure-eight / eccentric / pythagorean orbits to SVG |
| `examples/adaptive_demo.py` | Adaptive DP45 vs fixed RK4: step adaptation & force-eval savings |
| `examples/hermite_demo.py` | Hermite vs RK4/Forest-Ruth accuracy at a fixed force budget |
| `examples/convergence_demo.py` | Measured convergence order of each method vs the exact orbit |
| `examples/lagrange_demo.py` | Lagrange points + zero-velocity curves rendered to SVG |
| `examples/solar_system_demo.py` | Integrate the real solar system, recover Kepler's third law |
| `examples/precession_demo.py` | Mercury's 43"/century precession + a relativistic rosette SVG |
| `examples/chaos_demo.py` | Lyapunov exponent + two trajectories diverging 6 orders of magnitude |
| `examples/stability_map_demo.py` | Parallel escape-time heatmap revealing the fractal chaos boundary |
| `examples/virial_demo.py` | Equilibrium vs cold cluster: running 2T/U converging on -1 |
| `examples/cluster_demo.py` | Cluster M-T table + the kT ~ M^2/3 relation curve |
| `examples/roche_demo.py` | Survival curve across the Roche limit + a tidal-stream SVG |
| `examples/tidal_heating_demo.py` | Galilean-moon heating table + heating-vs-eccentricity curve |
| `examples/roche_lobe_demo.py` | Lobe radius & transfer stability vs mass ratio |
| `examples/kozai_demo.py` | e/i oscillations vs analytic e_max, out-of-phase time series SVG |
| `examples/resonance_demo.py` | 2:1 resonant argument: libration (locked) vs circulation (free) |
| `examples/coorbital_demo.py` | Tadpole & horseshoe paths in the rotating frame (SVG) |
| `examples/tisserand_demo.py` | a & e jump across a flyby while Tisserand stays flat |
| `examples/lensing_demo.py` | Microlensing light curve + Einstein-ring image diagram (SVG) |
| `examples/rotation_curve_demo.py` | Visible (declining) vs disk+halo (flat) rotation curves |
| `examples/mond_demo.py` | MOND (flat) vs Newton-on-baryons (declining) + Tully-Fisher |
| `examples/schwarzschild_demo.py` | Precessing & plunging black-hole orbits with ISCO/photon sphere |
| `examples/kerr_demo.py` | ISCO-vs-spin curves + horizon/ergosphere diagram |
| `examples/penrose_demo.py` | Extractable-energy fraction vs spin + the area theorem |
| `examples/hawking_demo.py` | Temperature & evaporation time across black-hole masses |
| `examples/eddington_demo.py` | L_Edd across masses + Eddington-limited growth to a quasar |
| `examples/hohmann_demo.py` | LEO->GEO & Earth->Mars delta-v budgets + transfer diagram |
| `examples/oberth_demo.py` | Periapsis-vs-apoapsis burn: escape speed vs burn radius |
| `examples/gr_time_demo.py` | Pound-Rebka, GPS gain, Sun redshift + Shapiro-delay curve |
| `examples/lense_thirring_demo.py` | GP-B geodetic & frame-drag rates vs orbit radius |
| `examples/pulsar_demo.py` | Hulse-Taylor dP/dt vs measured + the periastron-shift parabola |
| `examples/friedmann_demo.py` | Scale-factor curves for radiation/matter/dark-energy/LCDM |
| `examples/saha_demo.py` | Ionization fraction plunging to zero at recombination |
| `examples/cmb_demo.py` | Sound horizon, acoustic angle & the l~220 peak comb |
| `examples/bbn_demo.py` | n/p freeze-out chain and the Y_p ~ 0.25 helium fraction |
| `examples/lane_emden_demo.py` | Polytrope density profiles + surface-radius / mass table |
| `examples/distances_demo.py` | Hubble diagram (LCDM vs decelerating) + D_A turnover |
| `examples/chandrasekhar_demo.py` | White-dwarf mass-radius curve approaching 1.44 M_sun |
| `examples/tov_demo.py` | Neutron-star mass-radius curve with a maximum mass; GR vs Newton |
| `examples/jeans_demo.py` | Dispersion relation: sound waves vs collapse across the Jeans length |
| `examples/sedov_demo.py` | SNR radius/shock-speed history + the Trinity yield estimate |
| `examples/poincare_demo.py` | Overlaid surface-of-section: KAM tori amid the chaotic sea |
| `examples/sitnikov_demo.py` | Stroboscopic maps: circular binary (tori) vs eccentric (chaos) |
| `examples/galaxy_collision_demo.py` | Two disk galaxies collide, grow tidal tails (Barnes-Hut) |
| `examples/dynamical_friction_demo.py` | Sinking times by mass + drag-vs-speed curve |
| `examples/gravwave_demo.py` | Inspiral chirp, energy loss validated against Peters (1964) |
| `examples/gw_strain_demo.py` | GW150914 strain & arm change + strain-vs-distance curve |
| `examples/circularization_demo.py` | Peters (a, e) tracks: all binaries circularize before merger |
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

## Hulse-Taylor pulsar: gravitational waves before LIGO

Two decades before LIGO, the binary pulsar PSR B1913+16 proved gravitational
waves exist -- its orbit shrinks as it radiates them. `pulsar.py` computes the
decay from the Peters formula:

```
$ python examples/pulsar_demo.py examples/output

  dP/dt (GR predicted): -2.4031e-12 s/s
  dP/dt (measured)    : -2.423e-12 s/s
  agreement           : 99.2% of measured
  cumulative shift over 30 yr : -38.6 s
```

The orbital period drops ~76 microseconds a year, and the cumulative shift in
periastron time traces a parabola whose data points fall on the general-
relativity curve -- the plot that won the 1993 Nobel Prize. The tests confirm the
dP/dt match to under 1%, the `t^2` cumulative shift, the eccentricity and
period dependence, and the ~370 Myr decay timescale.

## GW strain: the number LIGO measures

The wave that reaches Earth stretches space by a fractional strain `h`.
`gw_strain.py` computes it from the chirp mass:

```
$ python examples/gw_strain_demo.py examples/output

  GW150914 (36 + 29 M_sun, 410 Mpc, f_gw~150 Hz):
    chirp mass    = 28.1 M_sun
    strain h      = 2.13e-21
    LIGO arm move = 8.50e-18 m (1.1% of a proton width)
```

`h ~ (G M_c/c^2)^{5/3}(pi f/c)^{2/3}/d`, falling as `1/d` and rising as `f^{2/3}`.
For GW150914 it is ~1e-21, which moves LIGO's 4 km arms by ~1e-18 m -- a
thousandth of a proton's width, and why LIGO is among the most sensitive
instruments ever built. The tests reproduce the ~28 M_sun chirp mass, the ~1e-21
strain, the sub-proton arm change, and the distance/frequency scalings.

## Gravitational waves: the LIGO chirp from first principles

`gravwave.py` adds the 2.5PN radiation-reaction force to a binary. Orbital energy
bleeds into gravitational waves, the orbit shrinks, and the frequency sweeps up
-- the chirp LIGO heard from GW150914.

```
$ python examples/gravwave_demo.py examples/output

  energy-loss rate dE/dt: measured ...  Peters ...  ratio ~1.0
  separation: 1.00 -> 0.38  (orbit shrinks as it radiates)
  orbital frequency chirps up 4.2x:
    ...............:::::::---==+*@
```

The energy-loss rate matches Peters' (1964) circular formula
`dE/dt = -(32/5) G^4 mu^2 M^3 / (c^5 a^5)` to a few percent, and the tests verify
the orbit only ever shrinks, the frequency chirps upward, the loss scales as
`1/c^5`, and switching radiation off leaves a closed orbit. As with the
perihelion demo, `c` is shrunk to bring the effect into view.

## Gravitational waves circularize binaries (Peters 1964)

The companion to the chirp: `gravwave.py` also carries the orbit-averaged Peters
equations for the coupled decay of semi-major axis and eccentricity. Both shrink,
but `e` falls faster and faster near merger, so binaries are driven toward
circular orbits:

```
$ python examples/circularization_demo.py examples/output

    e0     a_final     e_final   e reduction
  0.20      0.0098      0.0001       1406.2x
  0.60      0.0098      0.0009        688.1x
  0.90      0.0099      0.0099         90.9x
```

The `(a, e)` tracks all bend toward `e = 0`. The rates stiffen dramatically as
`a -> 0` and `e -> 1` (the `(1-e^2)^{-7/2}` factor), so `peters_evolve` uses an
adaptive step scaled to the local `a/|da/dt|` timescale. The tests confirm both
rates are negative, a circular orbit stays circular, the eccentricity decreases
monotonically, and the orbit-averaged `de/da` agrees with a full 2.5PN
integration. This is why LIGO's merger templates can assume circular orbits.

## Dynamical friction: satellites spiralling in

A massive body moving through a sea of stars focuses them into a trailing wake
whose pull drags it backward. `dynamical_friction.py` gives Chandrasekhar's drag
and the resulting sinking time:

```
$ python examples/dynamical_friction_demo.py examples/output

  satellite mass       sinking time
  globular cluster (1e8)     292.55 Gyr
  LMC-scale (1e10)             2.93 Gyr
  massive dwarf (1e11)         0.29 Gyr
```

The drag goes as `M^2 rho / v^2`, so the sinking time scales as `1/M` -- a heavy
satellite merges in a few Gyr while a light globular cluster survives a Hubble
time. The velocity dependence is non-monotonic (zero at rest, peaking near the
dispersion, falling as `1/v^2` when fast). This is what drags massive black holes
to galactic centres. The tests verify the `1/M` sinking time, the linear
mass/density dependence, and the velocity profile.

## Galaxy collisions: tidal tails from gravity alone

`galaxy.py` builds a disk galaxy -- a heavy central mass wrapped in a cold disk
of light tracers on circular orbits -- and sends two of them past each other.
Differential tidal force stretches the disks into the bridges and tails seen in
real interacting galaxies, exactly the Toomre & Toomre (1972) restricted N-body
picture. Forces use the Barnes-Hut tree, so ~1000 bodies run quickly.

```
python examples/galaxy_collision_demo.py examples/output
# Two-galaxy encounter: 1002 bodies, Barnes-Hut forces
# 487 tracer particles pulled into tidal bridges/tails.
```

The demo writes a time sequence of SVG snapshots (approach -> close passage ->
tails). The tests confirm an isolated disk is stable (cold circular orbits don't
fly apart), the tracers are massless with all mass in the two cores, and a close
passage strips a substantial fraction of the disk into tails.

## The Sitnikov problem: a dial from order to chaos

The cleanest chaos in celestial mechanics. Two equal masses orbit on a Kepler
ellipse; a massless body sits on the axis through their barycentre, and its whole
dynamics is one equation, `z'' = -z / (z^2 + r(t)^2)^{3/2}`. The binary's
eccentricity `e` is the only knob:

```
python examples/sitnikov_demo.py examples/output
# e=0.0: 10 orbits, mean z-spread   1.83   (integrable -- nested tori)
# e=0.3: 10 orbits, mean z-spread 198.81   (chaotic -- orbits diffuse outward)
```

At `e = 0` the forcing is constant, energy is conserved, and the stroboscopic
`(z, vz)` map is a set of smooth nested curves. Turn `e` up and the periodic
forcing tears the inner curves into a chaotic layer -- this is the system Moser
used to prove that chaotic (symbolic-dynamics) orbits exist. The tests verify
energy conservation at `e=0`, that `z=0` is an equilibrium, the force is odd in
`z`, and that sensitivity to initial conditions explodes 100x+ once `e > 0`.

## Poincare sections: order and chaos at the same energy

The tool Poincare invented for the three-body problem. A CR3BP trajectory lives
on a 3-D energy surface in 4-D phase space; slice it with the plane `y = 0` and
record `(x, vx)` at each upward crossing. The 4-D flow collapses to a 2-D map
whose structure is unmistakable:

```
python examples/poincare_demo.py examples/output
# integrated 19 orbits; 2 trace tight closed curves (KAM tori),
# the rest fill chaotic regions.
```

A quasi-periodic orbit pierces the plane on a smooth closed loop -- an invariant
KAM torus. A chaotic orbit at the *same Jacobi energy* sprinkles the plane with
diffuse dust. `poincare.py` reconstructs `vy` from the energy so each section
point is a full initial condition, and the tests confirm the Jacobi constant is
conserved along each orbit (faithful section), regular orbits stay on tight
curves, and chaotic ones scatter more than 3x wider.

## Sedov-Taylor: supernova remnants and the Trinity bomb

A sudden energy release `E` in a medium of density `rho` drives a self-similar
shock -- the only length you can build from `E`, `rho`, `t` is
`(E t^2/rho)^{1/5}`. `sedov.py`:

```
$ python examples/sedov_demo.py examples/output

  Trinity: fireball R=130 m at 25 ms in air -> yield ~ 9 kilotons (device ~21 kt)
    age (yr)   radius (pc)  shock (km/s)      T (K)
         100          2.05          8018    8.8e+08
       10000         12.93           506    3.5e+06
```

The shock radius grows as `t^{2/5}` and decelerates as `t^{-3/5}`, dating
supernova remnants (parsec-scale, thousands of km/s, X-ray-hot). Run the law
backwards and an observed radius-and-time gives the explosion energy -- exactly
how G. I. Taylor recovered the classified Trinity yield from a photograph. The
tests verify the scalings, the exact energy inversion (independent of which time
you sample), the SNR scale, and the Trinity order of magnitude.

## Jeans instability: when a cloud becomes a star

Star formation begins when a gas cloud's self-gravity overwhelms its pressure.
`jeans.py` gives the linear dispersion relation for a self-gravitating gas:

```
$ python examples/jeans_demo.py examples/output

  Jeans wavenumber k_J : 3.5449   (c_s = rho0 = G = 1)
    k/k_J     omega^2       behaviour
     0.50      -9.425        collapse
     1.00      -0.000        marginal
     2.00      37.699      sound wave
```

`omega^2 = c_s^2 k^2 - 4 pi G rho0`: short-wavelength modes have `omega^2 > 0`
and just oscillate as sound waves, but long-wavelength modes (`k < k_J`) have
`omega^2 < 0` and grow exponentially -- the cloud collapses. The crossover is the
Jeans length, and a cloud above the corresponding Jeans mass forms stars. Denser
or colder gas has a smaller Jeans length, so it fragments more easily. The tests
verify the two branches, the marginal mode at `k_J`, the growth rate approaching
`sqrt(4 pi G rho)`, and the density/temperature scalings.

## Neutron stars: the TOV equation and the mass that makes black holes

In a neutron star, gravity is so strong that Newtonian hydrostatics is wrong --
you need the general-relativistic Tolman-Oppenheimer-Volkoff equation.
`tov.py` integrates it for a polytropic equation of state:

```
$ python examples/tov_demo.py examples/output

       rho_c    R (km)    M_TOV    M_Newton
     1.0e-03     10.37    0.946        1.70
     5.2e-03      7.09    1.342        8.91   <- near the peak
     7.6e-01      4.94    0.965     1287.71
  TOV maximum mass       : 1.351 M_sun (the sequence turns over)
  Newtonian, densest star: 1698 M_sun (no limit -- grows forever)
```

The relativistic corrections make gravity effectively stronger, so the
mass-radius curve **turns over**: there is a maximum neutron-star mass, above
which no static star exists and collapse to a black hole is inevitable. The
Newtonian version has no such limit -- its mass grows without bound. The tests
check the neutron-star scale (R ~ 10 km, M ~ 1 M_sun), the TOV turnover, the
absence of a Newtonian maximum, and that GR caps the mass below Newton.

## The Chandrasekhar mass: the limit of a white dwarf

Electron degeneracy pressure holds up a white dwarf -- but only up to a point.
`chandrasekhar.py` computes the limit from fundamental constants (via the n=3
Lane-Emden mass factor) and integrates the full relativistic degenerate
equation of state to trace the mass-radius relation:

```
$ python examples/chandrasekhar_demo.py examples/output

  Chandrasekhar mass (mu_e=2): 1.435 M_sun (the famous 1.44)
    rho_c (kg/m^3)   radius (km)  mass (M_sun)
           1.0e+09         10780         0.395
           1.0e+11          4140         1.157
           1.0e+13          1080         1.357
```

As the central density rises the electrons turn relativistic, the equation of
state softens to `P ~ rho^{4/3}` (a polytrope of index 3), the star shrinks, and
its mass climbs toward `~1.44 M_sun` but never past it. Above the limit no stable
white dwarf exists -- it collapses, the trigger for type-Ia supernovae. The tests
verify the 1.44 value, the `1/mu_e^2` scaling, and that a denser dwarf is smaller
and more massive, approaching the limit from below.

## Lane-Emden: the structure of a star

A star in hydrostatic equilibrium with a polytropic equation of state
`P = K rho^{1+1/n}` has a density profile set by the Lane-Emden equation.
`lane_emden.py` integrates it for any index n:

```
$ python examples/lane_emden_demo.py examples/output

     n    surface xi_1    mass -xi1^2 theta'    meaning
   0.0           2.449                 4.899    uniform-density sphere
   1.0           3.142 (pi)            3.142    analytic sin(xi)/xi
   3.0           6.897                 2.018    Eddington standard model
   5.0             inf                   n/a    infinite radius
```

Three indices have closed forms the integrator reproduces to ~1e-9: `n=0`
(`theta = 1 - xi^2/6`, surface `sqrt(6)`), `n=1` (`sin(xi)/xi`, surface `pi`),
and `n=5` (`1/sqrt(1 + xi^2/3)`, which never reaches zero -- finite mass, infinite
radius). `n=3` is the Eddington standard model with the tabulated `xi_1 = 6.897`
and mass factor `2.018`. The tests check all four against their known values.

## Cosmic distances: how dark energy was found

Every cosmological distance is one integral of `1/E(z)` over redshift.
`distances.py` builds the comoving, luminosity, and angular-diameter distances
and the distance modulus:

```
$ python examples/distances_demo.py examples/output

      z     mu LCDM      mu EdS    Delta mu (fainter)
    0.5      42.261      41.862                +0.399
    1.0      44.100      43.502                +0.598
  angular-diameter distance peaks at z = 1.61
```

A dark-energy universe (LCDM) puts a given redshift at a larger distance than a
decelerating Einstein-de Sitter universe, so type-Ia supernovae look ~0.4 mag
**fainter** at `z~0.5` -- exactly the excess faintness Riess and Perlmutter found
in 1998 (2011 Nobel Prize). The angular-diameter distance is non-monotonic,
peaking near `z~1.6`, which is why the CMB's acoustic spots subtend about a
degree. The tests check the low-z Hubble law, the acceleration signal, the
turnover, and the Etherington duality `D_L = (1+z)^2 D_A`.

## Big Bang nucleosynthesis: the primordial helium

In the first few minutes the universe forged the light elements. `bbn.py` gets
the headline number -- the ~25% helium -- from a short chain:

```
$ python examples/bbn_demo.py examples/output

  stage                              n/p
  equilibrium at 10 MeV (t~0.01 s)  0.879
  freeze-out at 0.8 MeV (t~1 s)     0.199
  after neutron decay (t~200 s)     0.152
  primordial helium mass fraction Y_p = 0.264
```

Neutrons and protons start nearly equal (`n/p = exp(-Delta m/kT)`), the ratio
freezes at ~1/6 when the weak interaction shuts off, decays toward ~1/7 as free
neutrons beta-decay, and then almost every surviving neutron is locked into
helium-4, giving `Y_p = 2(n/p)/(1+n/p) ~ 0.25`. That quarter-helium abundance,
observed everywhere in the universe, is one of the strongest confirmations of the
hot Big Bang. The tests verify the equilibrium limits, the freeze-out ratio, the
decay, and the ~0.25 helium fraction.

## The CMB acoustic scale: the 1-degree spots

The sound horizon at recombination -- the farthest a pressure wave travels in the
photon-baryon plasma before the CMB is released -- is a standard ruler. Seen
across the distance to last scattering it subtends a fixed angle, the first
acoustic peak. `cmb.py`:

```
$ python examples/cmb_demo.py examples/output

  sound horizon r_s          : 192 Mpc
  distance to last scattering: 13734 Mpc
  acoustic angle theta       : 0.80 deg
  first acoustic peak        : l ~ 225   (WMAP/Planck: 220)
```

Reusing the recombination redshift (`saha`) and the comoving distance
(`friedmann`/`distances`), the first peak comes out at `l ~ 220` -- features about
a degree across, exactly what WMAP and Planck measured. Because that angle
depends on the geometry the light traveled through, its position is what pins the
universe to be spatially flat. The tests verify the peak multipole, the ~14000
Mpc distance to last scattering, and the sound-speed limits.

## Cosmic recombination: the birth of the CMB

The universe became neutral and transparent -- releasing the cosmic microwave
background -- when electrons and protons combined into hydrogen. `saha.py` finds
when, via the Saha equation:

```
$ python examples/saha_demo.py examples/output

  naive guess (kT = 13.6 eV)     : 157821 K
  actual recombination (x = 0.5) : z = 1379, T = 3760 K
    redshift    temp (K)   ionized x
        1600        4363      0.9925
        1200        3273      0.0339
```

Recombination happens at ~3700 K, ~40x cooler than the naive `kT = 13.6 eV`
estimate, because there are ~1.6 billion photons per baryon and the hot tail of
that bath keeps hydrogen ionized far below its binding energy. The ionization
fraction plunges from 1 to 0 across `z ~ 1400`; below it, photons free-stream to
us as the CMB. The tests verify the recombination redshift and temperature, that
it is far below the naive value, and the monotonic ionization curve.

## Friedmann cosmology: the expanding universe

The same gravity that binds orbits governs the expansion of the whole universe.
`friedmann.py` integrates the scale factor `a(t)` under the Friedmann equation
`(a_dot/a)^2 = H0^2 (Om_r/a^4 + Om_m/a^3 + Om_k/a^2 + Om_L)`:

```
$ python examples/friedmann_demo.py examples/output

  age of a flat LCDM universe : 0.964/H0   (~13.5 Gyr for H0=70)
  radiation                         n ~ 0.50   (a ~ t^1/2)
  matter                            n ~ 0.67   (a ~ t^2/3)
  dark energy     exponential (accelerating)
  flat LCDM                         n ~ 1.14
```

Each component dilutes differently as the universe grows -- radiation as
`a^-4`, matter as `a^-3`, dark energy not at all -- so the expansion passes
through radiation, matter, and dark-energy eras with distinct power laws. The
age comes out as a look-back integral to `~0.96/H0`, the measured ~13.8 Gyr. The
tests verify each era's exponent, the exponential dark-energy growth, and the
LCDM age.

## The Oberth effect: burn low and fast

A burn's energy gain is `dE = v dv + dv^2/2`, so the same `dv` buys more energy
where the ship already moves fast -- deep in the gravity well. `oberth.py`:

```
$ python examples/oberth_demo.py examples/output

  same 1500 m/s burn on a 300 km x 35786 km orbit:
    at periapsis (v=10.2 km/s): v_inf = 4.05 km/s (escapes)
    at apoapsis  (v=1.6 km/s):  v_inf = 0.00 km/s (still bound)
    energy-gain advantage of the periapsis burn: 4.6x
```

The identical burn escapes from periapsis but leaves the ship bound at apoapsis.
This is why interplanetary probes dive toward a planet before their escape burn,
and why a powered gravity assist (an Oberth maneuver at closest approach) far
outperforms the same burn in deep space. The tests verify the `v dv` energy law,
the periapsis advantage, and the escape-only-from-periapsis result.

## Hohmann transfer: the delta-v to get there

The cheapest two-burn maneuver between circular orbits sets every mission's fuel
budget. `hohmann.py` computes it:

```
$ python examples/hohmann_demo.py examples/output

  LEO (200 km) -> GEO: total = 3932 m/s, 5.26 hours, m0/mf = 2.44 (LH2/LOX)
  Earth -> Mars: total = 5.60 km/s, 259 days, launch phase angle = 44.4 deg
```

Both burns and the half-ellipse transfer time come straight from the vis-viva
equation, and the launch phase angle -- the lead the target must have when you
depart -- is why Mars windows open only every ~26 months. Feed the delta-v to
Tsiolkovsky's rocket equation and you get the propellant mass. The tests
reproduce the standard LEO->GEO (~3.9 km/s), Earth->Mars (~5.6 km/s, 259 day,
44 deg) figures used in real mission design.

## The Eddington luminosity: the brightness limit of accretion

Radiation carries momentum, so an accreting object cannot outshine the point
where radiation pressure balances gravity. `eddington.py` gives that limit:

```
$ python examples/eddington_demo.py examples/output

  Salpeter e-folding time: 45.0 Myr
  object          M (Msun)   L_Edd (L_sun)  Mdot (Msun/yr)
  Sun                    1        3.28e+04        2.22e-08
  quasar               1e9        3.28e+13        2.22e+01
```

`L_Edd = 4 pi G M m_p c / sigma_T` is linear in mass and independent of radius.
It caps the steady accretion rate and hence the growth rate: an Eddington-limited
black hole e-folds its mass every ~45 Myr (the Salpeter time), so growing a
10-solar-mass seed into a billion-solar-mass quasar takes ~0.8 Gyr -- just barely
possible in the early universe, which is why the first quasars are a puzzle. The
tests verify the solar value, the mass linearity, the 45 Myr Salpeter time, and
the quasar growth time.

## Hawking radiation: black holes are not black

Quantum effects at the horizon give a black hole a temperature and an entropy.
`hawking.py` computes both from fundamental constants:

```
$ python examples/hawking_demo.py examples/output

  primordial mass evaporating in a Hubble time: 1.73e11 kg
  object                       T (K)   t_evap (yr)       S/k_B
  1 solar mass              6.17e-08      2.10e+67    1.05e+77
  M87* (6.5e9 Msun)         9.49e-18      5.76e+96    4.43e+96
```

`T_H ~ 1/M` (big holes are colder), the Bekenstein-Hawking entropy is one
quarter of the horizon area in Planck units, and the evaporation time scales as
`M^3`. A solar-mass hole is ~60 nanokelvin and lives ~10^67 years -- effectively
eternal -- while a ~1.7e11 kg primordial hole is ending its life in a burst right
now. The tests verify the `1/M` temperature, the `M^3` lifetime, the primordial
mass, and the quarter-area entropy law.

## The Penrose process: mining a black hole's spin

Inside a Kerr ergosphere an object can have negative energy as seen from
infinity, so splitting one there lets a fragment escape with more energy than
went in -- extracted from the hole's rotation. `penrose.py` does the bookkeeping
with the irreducible mass:

```
$ python examples/penrose_demo.py examples/output

     a/M   M_irr/M  E_rot fraction  horizon area
    0.00    1.0000            0.0%         50.27
    0.90    0.8473           15.3%         36.09
    1.00    0.7071           29.3%         25.13
```

`M_irr = sqrt((M + sqrt(M^2 - a^2))/2)` is the Schwarzschild-equivalent mass you
are left with; the rest, up to **29% of Mc^2** for an extremal hole, is
extractable rotational energy. Crucially, extracting spin only ever GROWS the
irreducible mass and the horizon area -- Hawking's area theorem, the second law
of black-hole mechanics. The tests verify the extremal `1/sqrt(2)` mass, the
29.3% ceiling, the monotonic spin dependence, and the area theorem.

## Kerr black holes: spin drags spacetime

A rotating black hole is richer than a static one. `kerr.py` gives the exact
horizon, ergosphere, and Bardeen-Press-Teukolsky ISCO as functions of spin `a`:

```
$ python examples/kerr_demo.py examples/output

     a/M   horizon   ISCO pro  ISCO retro   Omega_H
    0.00     2.000      6.000       6.000     0.000
    0.90     1.436      2.321       8.717     0.313
    1.00     1.000      1.000       9.000     0.500
```

Spinning the hole up shrinks the horizon (2M → M), opens an ergosphere where
frame-dragging forbids standing still (the region behind the Penrose process and
Blandford-Znajek jets), and splits the ISCO: prograde orbits reach down toward
`1M` while retrograde ones recede to `9M`. Because the ISCO sets the inner edge
of the accretion disk, measuring it is how astronomers weigh black-hole spin.
The tests check the Schwarzschild limit, the extremal `1M`/`9M` values, the
monotonic spin dependence, and cosmic censorship (`a > M` is rejected).

## Frame-dragging: Gravity Probe B

A gyroscope in orbit precesses two ways in general relativity, both measured by
Gravity Probe B. `lense_thirring.py`:

```
$ python examples/lense_thirring_demo.py examples/output

  effect                     predicted    measured
  geodetic (de Sitter)      6638 mas/yr        6602
  frame-dragging (LT)       41.1 mas/yr        37.2
```

Geodetic precession comes from the curvature of space the gyro is carried
through (`~ r^{-5/2}`); frame-dragging comes from Earth's rotation twisting
spacetime around it (`~ r^{-3}`). The frame-dragging term is ~180x smaller, which
is why measuring it needed near-perfect gyroscopes in a dedicated satellite. The
tests reproduce the ~6600 and ~40 mas/yr values and both radius scalings.

## Gravitational time: redshift, GPS, and the Shapiro delay

Gravity slows clocks and delays light. `gr_time.py` reproduces three classic
tests:

```
$ python examples/gr_time_demo.py examples/output

  Pound-Rebka (22.5 m tower)  : z = 2.45e-15   (measured 2.5e-15)
  GPS clock gain              : +38.5 us/day
  Sun surface redshift        : z = 2.12e-06
  Shapiro delay (past the Sun): 281 us      (Cassini ~240-280 us)
```

A photon climbing out of a well is redshifted by `Delta Phi / c^2`; GPS
satellites' clocks gain ~38 microseconds a day (uncorrected, positions drift
kilometers daily); and radar grazing the Sun is delayed a few hundred
microseconds -- the Shapiro effect, the tightest Solar-System test of GR. The
tests check the Pound-Rebka value, the GPS gain, the solar redshift, the Shapiro
scale and its mass-linearity, and that the exact Schwarzschild redshift reduces
to `g h / c^2`.

## Schwarzschild orbits: strong-field general relativity

Outside a non-rotating black hole, a particle's radial motion is governed by the
effective potential `V(r) = (1 - 2M/r)(1 + L^2/r^2)` (units of M). `schwarzschild.py`
locates the landmark radii and integrates the exact orbit shape:

```
$ python examples/schwarzschild_demo.py examples/output

  event horizon  : r = 2 M
  photon sphere  : r = 3 M
  ISCO           : r = 6 M
  bound orbit (r: 10-20 M): perihelion advance 126.2 deg/orbit
  plunging orbit: falls from r=12 M through the horizon to r=0.01 M
```

The **ISCO at 6M** is the inner edge of accretion disks; the **photon sphere at
3M** is the light ring seen in the M87*/Sgr A* images. Bound orbits precess by
tens of degrees *per orbit* (versus Mercury's 43 arcsec per century), and orbits
that pass inside the potential barrier plunge to `r -> 0`. The demo renders a
precessing rosette and a plunging geodesic with the horizon, photon sphere, and
ISCO marked. The tests verify the ISCO/photon-sphere radii, the circular-orbit
angular momentum `sqrt(12) M` at the ISCO, the plunge, and that the weak-field
precession recovers the classic `6*pi*M/(a(1-e^2))`.

## MOND: flat curves without dark matter

The rival explanation to the dark halo. Rather than add unseen mass, MOND
(Milgrom 1983) modifies gravity below `a0 ~ 1.2e-10 m/s^2`. `mond.py`:

```
$ python examples/mond_demo.py examples/output

  asymptotic flat speed v = (G M a0)^1/4 = 175.8 km/s   (baryons = 6e10 M_sun)
   r (kpc)   Newton (km/s)   MOND (km/s)
        10           160.7         215.4
        80            56.8         180.5
```

Solving `g mu(g/a0) = g_N` makes a bare baryonic galaxy's rotation curve flatten
on its own -- no dark matter -- and yields the baryonic Tully-Fisher relation
`v_flat^4 = G M a0`, which ties a galaxy's flat speed to its visible mass with
remarkably little scatter. This is the same flat-curve observation as the dark-
matter section, explained the opposite way; the repo lets you compare them side
by side. The tests check the interpolating-function limits, the deep-MOND
`sqrt(g_N a0)` law, the flattening, and the `v ~ M^{1/4}` Tully-Fisher slope.

## Galaxy rotation curves: the case for dark matter

A star orbits on the mass enclosed within its radius, so `v_c(r) = sqrt(G
M(<r)/r)`. `rotation_curve.py` builds the curve from an exponential disk and an
optional NFW dark halo:

```
$ python examples/rotation_curve_demo.py examples/output

  visible disk only : outer slope -0.490  (Keplerian decline = -0.50)
  disk + dark halo  : outer slope -0.000  (flat = 0.00)
  visible-only v(r): .::-------------------::::::::::::::::::::::::......
  disk+halo   v(r): :-=+++*****#############################@##########
```

Past the visible edge the disk's enclosed mass saturates, so the disk-only curve
falls off Keplerian (slope -1/2). An NFW halo's enclosed mass keeps growing as
`~r`, which holds the total curve flat -- exactly what observations show. The gap
between the two curves is the dark matter. The tests verify the Keplerian
decline, the halo-flattened curve, and that disk mass saturates while halo mass
keeps climbing.

## Gravitational lensing: bending light with gravity

Mass deflects light by `alpha = 4GM/(c^2 b)` -- exactly twice the Newtonian
value, the prediction Eddington confirmed at the 1919 eclipse. `lensing.py`
reproduces it and solves the point-mass lens equation:

```
$ python examples/lensing_demo.py examples/output

  light deflection at the Sun's limb : 1.751 arcsec  (measured ~1.75)
  microlensing light curve (u0=0.2):
  ....::::----====++***####@@####***++====----::::....
  peak magnification A_max = 5.07 at closest approach
```

A point-mass lens splits a source into two images; at perfect alignment they
merge into an Einstein ring at `theta_E`, and as the source drifts past, the
total magnification `A(u) = (u^2+2)/(u sqrt(u^2+4))` traces the symmetric
microlensing curve used to find exoplanets and dark compact objects. The tests
verify the 1.75-arcsec deflection, that both images solve the lens equation, the
Einstein ring at `beta=0`, and that the summed image magnification equals the
closed-form total.

## Tisserand parameter: the invariant of a gravity assist

A planetary flyby can drastically reshape a small body's orbit, yet the
combination `T = a_p/a + 2 sqrt(a/a_p (1-e^2)) cos i` stays almost fixed -- it is
the CR3BP Jacobi constant written in heliocentric elements. `tisserand.py`
integrates a flyby and measures it:

```
$ python examples/tisserand_demo.py examples/output

              before     after    change
a (AU)         1.281     0.890    -0.391
e              0.330     0.265    -0.065
T_planet      2.9404    2.9432  +0.00277
```

The semi-major axis shifts ~30% and the eccentricity moves too, but the
Tisserand parameter changes by ~0.1%. This is how Tisserand recognized returning
comets whose orbits Jupiter had scrambled, how small-body populations are
classified (`T_J > 3` asteroids, `2 < T_J < 3` Jupiter-family comets), and what
bounds a single gravity assist. The tests verify `T = 3` for a planet-matching
orbit, the near-invariance across a real flyby, and the classification ordering.

## Coorbital orbits: tadpoles and horseshoes

A body sharing a planet's orbit does not sit still in the rotating frame -- it
slowly librates. `coorbital.py` (built on the CR3BP) produces both families:

```
$ python examples/coorbital_demo.py examples/output

  tadpole   (near L4): angular range    78 deg -> tadpole
  horseshoe (near L3): angular range   315 deg -> horseshoe
```

A **tadpole** loops a single triangular Lagrange point (Jupiter's Trojan
asteroids); a **horseshoe** sweeps around L3, enclosing both L4 and L5 and
turning back before it reaches the planet (Saturn's coorbital moons Janus and
Epimetheus, Earth's companion 3753 Cruithne). The classifier keys off the
angular range about the primary-secondary line -- under 180 deg for a tadpole,
over 180 for a horseshoe. The demo renders both paths in the rotating frame.

## Mean-motion resonance: periods that lock

When two planets' orbital periods approach an integer ratio `p:q`, they can lock
into resonance. `resonance.py` integrates a Sun + two-planet system and tracks
the resonant argument `phi = p*theta_out - q*theta_in - (p-q)*varpi_in`:

```
$ python examples/resonance_demo.py examples/output

  resonant (2:1 spacing) : phi range 2.10 rad -> LIBRATES (locked)
  off-resonance          : phi range 6.27 rad -> circulates (2pi=6.28)
  resonant phi  : ====++=====-----:-----======+====---::-----===
  off-res  phi  : -.#+-.*=: *=:#+-.*=: *=:#+-.*+: *=: +-.#=: *=:#
```

At the `2:1` spacing `a_out = a_in * 2^{2/3}` the measured period ratio comes out
to ~2 and `phi` librates in a bounded band -- the definition of a resonance lock.
Move the outer planet off that spacing and `phi` circulates through the full
`2*pi`. This is the mechanism behind the Kirkwood gaps and the Laplace 4:2:1
resonance of Io-Europa-Ganymede. The tests verify the Kepler spacing, the period
ratio, and the libration-vs-circulation distinction.

## Kozai-Lidov cycles: trading eccentricity for inclination

In a hierarchical triple -- a tight inner binary orbited by a distant third body
-- the inner orbit's eccentricity and inclination undergo large coupled
oscillations. `kozai.py` integrates the secular (doubly-averaged) quadrupole
Hamiltonian flow:

```
$ python examples/kozai_demo.py examples/output

  critical inclination      : 39.23 deg
  e_max measured / analytic : 0.975 / 0.975   (start e=0.01, i=80 deg)
  inclination swings        : 39.3 - 80.0 deg
  eccentricity :  # #  :+*=.  =: ...
  inclination  : @ #=####*#######- ...   (out of phase with e)
```

The conserved quantity `Theta = sqrt(1-e^2) cos i` (the inner orbit's z-angular
momentum) is held to machine precision, so the orbit trades eccentricity for
inclination and back. Above the critical inclination `arccos(sqrt(3/5)) ~ 39.2
deg` the eccentricity is driven to `e_max = sqrt(1 - (5/3) cos^2 i0)`; below it,
nothing happens. This mechanism drives hot-Jupiter migration and merges compact
binaries. The tests verify the conserved quantity, the critical angle, the
analytic `e_max`, and the e-i anticorrelation.

## Roche lobes: how close binaries feed each other

Each star in a binary owns a Roche lobe; the lobes meet at the inner Lagrange
point L1. `roche_lobe.py` gives the Eggleton lobe radius and the mass-transfer
stability:

```
$ python examples/roche_lobe_demo.py examples/output

   q = M_donor/M_acc    R_L/a   d ln a/d ln M    transfer
                 0.5    0.321            -1.0      stable
                 1.0    0.379             0.0    unstable
                 2.0    0.440             2.0    unstable
```

When a star fills its lobe (by swelling or orbital shrinkage), gas spills through
L1 onto its companion. From a lighter donor conservative transfer widens the
orbit and is stable -- the steady accretion of cataclysmic variables and X-ray
binaries; from a heavier donor the orbit shrinks and runs away, a path to mergers
and type-Ia supernovae. The tests verify the ~0.38 a equal-mass lobe, the L1
midpoint, and the stability flip at `q = 1`.

## Tidal heating: why Io has volcanoes

A moon on an eccentric orbit is flexed by the changing tide, and an imperfectly
elastic body dissipates that flexing as heat. `tidal_heating.py`:

```
$ python examples/tidal_heating_demo.py examples/output

  moon         power (W)  flux (W/m^2)
  Io            9.33e+13         2.238
  Europa        6.37e+12         0.208
  Ganymede      5.48e+10         0.001
```

`dE/dt = (21/2)(k2/Q) G M_p^2 R^5 n e^2 / a^6` -- it scales as `e^2`, `R^5`, and
`a^{-15/2}`. For Io it comes to ~1e14 W, about 40x Earth's internal heat flux,
which is why Io is the most volcanically active body in the Solar System; Europa's
lower but real heating keeps a subsurface ocean liquid. The eccentricity that
powers it is maintained by the Laplace 4:2:1 resonance (see the resonance
section), so orbital resonance and volcanism are the same story. The tests verify
the ~1e14 W scale, the surface flux, and all three power-law scalings.

## The Roche limit: tearing a moon into a ring

`roche.py` models a satellite as a rubble pile -- a cloud of particles bound only
by mutual gravity -- orbiting a heavy primary. Self-gravity holds it together
until the tidal field wins, inside the Roche limit
`d = 2.44 R (rho_primary / rho_satellite)^{1/3}`:

```
$ python examples/roche_demo.py examples/output

 d / d_Roche  surviving bound fraction
        0.40  ------------------------------ 0.00
        0.80  ##########################---- 0.88
        1.50  #############################- 0.98
        3.00  ############################## 1.00
```

Holding mass, size, and integration time fixed so the comparison is purely
tidal, the surviving bound fraction drops sharply as the orbit crosses inside the
Roche limit. The demo renders a satellite stretching into a tidal stream -- the
process behind planetary rings and the fragment chain of comet Shoemaker-Levy 9.
The tests check the Roche formula's scaling and the disruption gradient.

## Galaxy clusters: virial temperature and X-rays

The virial theorem applied to the largest bound objects. `cluster.py` turns a
cluster's mass into the temperature of its gas:

```
$ python examples/cluster_demo.py examples/output

     mass (M_sun)  radius (Mpc)  kT (keV)       T (K)
           1e+14          0.93      1.45     1.7e+07
           1e+15          2.00      6.74     7.8e+07
```

Gas falling into a `10^15`-solar-mass well virializes at `kT ~ G M mu m_p / (2R)`
-- a few keV, `~10^8` K -- hot enough to emit thermal-bremsstrahlung X-rays, which
is how clusters are found. Because `kT ~ M^{2/3}` at fixed overdensity, an X-ray
temperature measures the cluster's total (mostly dark) mass. The tests reproduce
Coma's ~8 keV, the dispersion estimate, the `M^{2/3}` relation, and the
mass inversion.

## The virial theorem and violent relaxation

A bound gravitational system in equilibrium obeys `2<T> + <U> = 0`, i.e.
`2T/U = -1`. `virial.py` measures the running virial ratio along an N-body
integration:

```
$ python examples/virial_demo.py examples/output

  equilibrium Plummer : running <2T/U> -> -0.992  (target -1)
  cold cluster        : start -0.083 -> running <2T/U> -0.952
```

An equilibrium Plummer sphere sits right at `-1`. A **cold** cluster (velocities
scaled down, far sub-virial) collapses, overshoots, and relaxes toward `-1` --
"violent relaxation" -- forgetting its initial state. The tests confirm the
equilibrium value, the cold-start relaxation, that scaling velocities scales `T`
as `v^2`, and that total energy is conserved throughout the collapse.

## Stability maps: chaos drawn in initial-condition space

`stability_map.py` drops a third body at rest at every point of a grid between
two primaries and integrates each one, recording how long the system stays bound
before a body escapes. Colouring the grid by escape time draws the boundary
between order and chaos directly:

```
python examples/stability_map_demo.py examples/output 64   # 64x64 = 4096 integrations
```

Rows run in parallel across CPU cores. The output SVG shows a mirror-symmetric
pattern (the primaries sit on the x-axis, so `y -> -y` is an exact symmetry --
verified in the tests) with a fractal-edged escape boundary. Bright regions are
long-lived, near-periodic configurations; dark regions ionize almost at once.
The intricate filigree at the edge is the fingerprint of chaos: neighbouring
starting points can have wildly different fates.

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

A full build re-runs every simulation (~4 min, several are O(N^2)). Each demo's
text output is cached to `examples/output/_<name>.txt`, so a rebuild that only
touched the page layout can reuse them:

```
python examples/build_dashboard.py examples/output --fast   # ~0.1s, uses caches
```

`--fast` reuses the cached text and the already-rendered SVGs and just
re-assembles the HTML. Delete a `_<name>.txt` to force that one demo to re-run.

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

## Hermite: fourth order for one force call

RK4 and Forest-Ruth reach 4th order but pay 4 and 3 force evaluations per step.
The Hermite predictor-corrector (`hermite.py`) reaches 4th order with a *single*
force+jerk evaluation, by computing the analytic jerk `da/dt` and Hermite-
interpolating acceleration and jerk. It's the integrator of real collisional
star-cluster codes.

```
$ python examples/hermite_demo.py

method           steps   f-evals     end error   order
------------------------------------------------------
rk4               6000     24000     4.230e-12    4.10
forest_ruth       8000     24000     1.242e-11    4.00
hermite          24000     24000     1.964e-13    4.01
```

For the same force-evaluation budget, Hermite takes 4x as many steps as RK4 and
lands ~20x more accurate. The tests confirm the analytic jerk matches a finite-
difference of the acceleration, the scheme is 4th order against the exact Kepler
orbit, and energy stays well controlled.

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
