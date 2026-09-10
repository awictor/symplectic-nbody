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
| `src/exoplanet.py` | Exoplanet detection: transit depth & radial-velocity wobble |
| `src/habitable_zone.py` | Habitable zone: equilibrium temperature & liquid-water bounds |
| `src/focusing.py` | Gravitational focusing: enhanced collision cross-section & runaway growth |
| `src/relativity.py` | First post-Newtonian gravity & Mercury's perihelion precession |
| `src/lyapunov.py` | Largest Lyapunov exponent (Benettin shadow-trajectory method) |
| `src/stability_map.py` | Three-body escape-time scan over a grid of initial conditions |
| `src/virial.py` | Virial theorem & violent relaxation of a self-gravitating cluster |
| `src/cluster.py` | Galaxy-cluster virial temperature, M-T relation & X-ray scaling |
| `src/sz.py` | Sunyaev-Zeldovich effect: Compton-y CMB distortion by cluster gas |
| `src/bremsstrahlung.py` | Free-free X-ray emissivity & cluster-gas cooling time |
| `src/pair_production.py` | Photon-photon pair production & the gamma-ray horizon |
| `src/axial_precession.py` | Precession of the equinoxes: the 26,000-year luni-solar wobble |
| `src/alfven.py` | Alfven waves, plasma beta & the Alfven surface of the solar wind |
| `src/parker_spiral.py` | The Parker spiral: the Sun's field wound up by its rotation |
| `src/magnetic_braking.py` | Magnetic braking & gyrochronology: a star's age from its spin |
| `src/tidal_locking.py` | Tidal locking timescale: why the Moon shows one face |
| `src/jeans_escape.py` | Jeans escape: which gases a world keeps, which leak to space |
| `src/snow_line.py` | The snow line: disk temperature & the rocky/icy divide at ~3 AU |
| `src/poynting_robertson.py` | Poynting-Robertson drag: dust spiralling into the Sun |
| `src/toomre.py` | Toomre Q: when a rotating disk fragments into clumps and arms |
| `src/accretion_disk.py` | Shakura-Sunyaev disk: the X-ray/UV glow of accreting black holes |
| `src/fermi_acceleration.py` | Diffusive shock acceleration & the universal E^(-2) cosmic-ray spectrum |
| `src/opacity.py` | Stellar opacity: electron scattering, Kramers law & the photon mean free path |
| `src/brunt_vaisala.py` | Brunt-Vaisala buoyancy frequency & the Schwarzschild convection criterion |
| `src/ram_pressure.py` | Ram-pressure stripping: how clusters strip spirals of their gas |
| `src/free_fall.py` | Free-fall & dynamical time: the 1/sqrt(G rho) clock of gravity |
| `src/shock_jump.py` | Sound speed & the Rankine-Hugoniot shock jumps |
| `src/stromgren.py` | The Stromgren sphere: the ionized HII bubble around a hot star |
| `src/relaxation_time.py` | Two-body relaxation & evaporation: collisional clusters vs collisionless galaxies |
| `src/parker_wind.py` | The Parker transonic solar wind through the sonic critical point |
| `src/greenhouse.py` | The greenhouse effect: surface warming from infrared optical depth |
| `src/rossby.py` | Rossby number & geostrophic balance: why weather spins |
| `src/rayleigh_benard.py` | Rayleigh-Benard convection: the Ra_c ~ 1708 onset & Nusselt transport |
| `src/terminal_velocity.py` | Terminal velocity & drag: Stokes vs quadratic regimes |
| `src/snr_phases.py` | Supernova-remnant evolution: free expansion, Sedov, snowplow, merge |
| `src/magnetic_mirror.py` | The magnetic mirror & loss cone: trapping charged particles |
| `src/debye.py` | Debye shielding & the plasma frequency: what makes a plasma |
| `src/line_broadening.py` | Spectral line broadening: Doppler, natural & pressure widths |
| `src/curve_of_growth.py` | Curve of growth: equivalent width vs column density in three regimes |
| `src/sackur_tetrode.py` | Sackur-Tetrode equation: the absolute entropy of an ideal gas |
| `src/maxwell_boltzmann.py` | Maxwell-Boltzmann speed distribution & the three characteristic speeds |
| `src/gamow.py` | The Gamow peak: the narrow energy window where stars fuse |
| `src/parallax.py` | Parallax, proper motion & space velocity: the geometry of stellar distance |
| `src/standard_candle.py` | Standard candles: distance modulus, Cepheids & the distance ladder |
| `src/tully_fisher.py` | Tully-Fisher: a spiral galaxy's luminosity from its rotation speed |
| `src/tolman.py` | Tolman surface-brightness dimming: the (1+z)^4 test of expansion |
| `src/olbers.py` | Olbers' paradox: why the dark night sky reveals a finite-age universe |
| `src/bi_elliptic.py` | Bi-elliptic transfer: when three burns beat the Hohmann two |
| `src/gravity_assist.py` | Gravity assist: the slingshot boost from a planetary flyby |
| `src/synodic.py` | Synodic periods: how often planets line up |
| `src/black_hole_shadow.py` | The black-hole shadow: the dark disk the EHT imaged |
| `src/hill_sphere.py` | The Hill sphere: how far a planet's gravity keeps its moons |
| `src/j2_precession.py` | J2 orbital precession: nodal regression, apsidal drift & sun-synchronous orbits |
| `src/solar_sail.py` | Solar sails & radiation pressure: the lightness number beta |
| `src/beaming.py` | Relativistic beaming: Doppler boosting & one-sided jets |
| `src/relativistic_rocket.py` | The relativistic rocket: interstellar travel at constant 1 g |
| `src/relativistic_doppler.py` | Relativistic Doppler: longitudinal, transverse & redshift-velocity |
| `src/de_broglie.py` | The de Broglie wavelength: matter as waves |
| `src/bohr.py` | The Bohr model: the hydrogen spectrum from a quantized orbit |
| `src/roche.py` | Roche limit & tidal disruption of a rubble-pile satellite |
| `src/tidal_heating.py` | Tidal heating: Io's volcanic power from orbital flexing |
| `src/roche_lobe.py` | Roche lobes & binary mass-transfer stability (Eggleton) |
| `src/kozai.py` | Kozai-Lidov secular cycles: eccentricity <-> inclination in a triple |
| `src/resonance.py` | Mean-motion resonance: period locks & librating resonant arguments |
| `src/coorbital.py` | Tadpole & horseshoe coorbital orbits in the CR3BP |
| `src/tisserand.py` | Tisserand parameter: the near-invariant of a gravity assist |
| `src/lensing.py` | Gravitational lensing: deflection, Einstein ring, microlensing |
| `src/rotation_curve.py` | Galaxy rotation curves: Keplerian disk vs flat dark-halo curve |
| `src/faber_jackson.py` | Faber-Jackson L ~ sigma^4 relation for elliptical galaxies |
| `src/mond.py` | MOND: modified gravity, flat curves & baryonic Tully-Fisher |
| `src/schwarzschild.py` | Black-hole orbits: effective potential, ISCO, photon sphere, plunge |
| `src/kerr.py` | Rotating black holes: horizons, ergosphere, spin-dependent ISCO |
| `src/penrose.py` | Penrose process: irreducible mass & extractable spin energy |
| `src/hawking.py` | Black-hole thermodynamics: Hawking temperature, entropy, evaporation |
| `src/eddington.py` | Eddington luminosity, accretion rate & Salpeter black-hole growth |
| `src/bondi.py` | Bondi accretion: spherical feeding rate onto a compact object |
| `src/hohmann.py` | Hohmann transfer: mission delta-v budgets & launch windows |
| `src/oberth.py` | Oberth effect: why rockets burn deep in a gravity well |
| `src/cosmic_velocities.py` | Orbital/escape/Solar-System speeds & the Schwarzschild link |
| `src/atmosphere.py` | Jeans atmospheric escape: which worlds keep which gases |
| `src/gr_time.py` | Gravitational redshift, GPS clock correction, Shapiro delay |
| `src/lense_thirring.py` | Frame-dragging & geodetic precession (Gravity Probe B) |
| `src/pulsar.py` | Hulse-Taylor binary-pulsar orbital decay (first GW evidence) |
| `src/friedmann.py` | Friedmann cosmology: scale factor a(t), expansion eras, age of the universe |
| `src/saha.py` | Saha equation & cosmic recombination (the CMB release) |
| `src/cmb.py` | CMB acoustic scale: sound horizon & the l~220 first peak |
| `src/blackbody.py` | Blackbody radiation: Planck law, Wien peak, Stefan-Boltzmann |
| `src/compton.py` | Compton & inverse-Compton scattering (photon-electron energy exchange) |
| `src/larmor.py` | Larmor formula: power radiated by an accelerating charge |
| `src/synchrotron.py` | Synchrotron radiation: critical frequency, power, spectral index |
| `src/optical_depth.py` | Optical depth & radiative transfer: the tau~2/3 photosphere |
| `src/bbn.py` | Big Bang nucleosynthesis: n/p freeze-out & primordial helium |
| `src/lane_emden.py` | Lane-Emden stellar structure: polytrope profiles & surface radii |
| `src/main_sequence.py` | Main sequence: mass-luminosity relation, lifetimes, HR diagram |
| `src/kelvin_helmholtz.py` | Kelvin-Helmholtz thermal timescale (gravity vs fusion) |
| `src/distances.py` | Cosmological distances: luminosity/angular-diameter, cosmic acceleration |
| `src/degeneracy.py` | Fermi degeneracy pressure: the quantum support of dead stars |
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
| `examples/exoplanet_demo.py` | Transit depths & RV wobbles + a transit light-curve dip |
| `examples/habitable_zone_demo.py` | HZ bounds by stellar type + the zone-vs-luminosity band |
| `examples/focusing_demo.py` | Cross-section enhancement vs encounter speed (runaway growth) |
| `examples/precession_demo.py` | Mercury's 43"/century precession + a relativistic rosette SVG |
| `examples/chaos_demo.py` | Lyapunov exponent + two trajectories diverging 6 orders of magnitude |
| `examples/stability_map_demo.py` | Parallel escape-time heatmap revealing the fractal chaos boundary |
| `examples/virial_demo.py` | Equilibrium vs cold cluster: running 2T/U converging on -1 |
| `examples/cluster_demo.py` | Cluster M-T table + the kT ~ M^2/3 relation curve |
| `examples/sz_demo.py` | Compton y & CMB decrement across cluster masses |
| `examples/bremsstrahlung_demo.py` | Emissivity & cooling time vs density (cooling flows) |
| `examples/pair_production_demo.py` | Threshold gamma energy vs background photon energy |
| `examples/axial_precession_demo.py` | Sun/Moon precession rates + the wandering-pole circle |
| `examples/alfven_demo.py` | v_A & beta across environments + the Alfven-surface crossing |
| `examples/parker_spiral_demo.py` | Garden-hose angle Sun->Saturn + spiral field lines |
| `examples/magnetic_braking_demo.py` | Gyro ages for clusters + the Skumanich age-period curve |
| `examples/tidal_locking_demo.py` | Locking times across the solar system + the a^6 curve |
| `examples/jeans_escape_demo.py` | Gas-retention table + the escape-vs-thermal-speed shoreline |
| `examples/snow_line_demo.py` | Disk T at each planet + the frost-line temperature profile |
| `examples/poynting_robertson_demo.py` | Inspiral time vs grain size with blow-out & solar age |
| `examples/toomre_demo.py` | Q across the galactic disk with the unstable band shaded |
| `examples/accretion_disk_demo.py` | T(r) for stellar-mass vs supermassive disks + wavebands |
| `examples/fermi_acceleration_demo.py` | Spectral index vs Mach + power-law spectra toward p=2 |
| `examples/opacity_demo.py` | Opacity by region + the Kramers/electron-floor T profile |
| `examples/brunt_vaisala_demo.py` | N & buoyancy period by layer + the N^2-vs-lapse-rate curve |
| `examples/ram_pressure_demo.py` | Surviving gas radius by environment + the R_strip(v) curves |
| `examples/free_fall_demo.py` | Free-fall time from clouds to neutron stars + the rho^(-1/2) line |
| `examples/shock_jump_demo.py` | Jump ratios vs Mach + the density-4 ceiling and M^2 divergence |
| `examples/stromgren_demo.py` | Radius & ionized mass by star/density + the R ~ n^(-2/3) curves |
| `examples/relaxation_time_demo.py` | Crossing/relax/evap times by system + t_relax(N) vs Hubble time |
| `examples/parker_wind_demo.py` | Sound speed/critical radius/1 AU speed + transonic profiles |
| `examples/greenhouse_demo.py` | T_eq/T_surf/warming for Venus-Earth-Mars + the warming curve |
| `examples/rossby_demo.py` | Ro & regime for tornado-to-gyre flows + the Ro(L) curve |
| `examples/rayleigh_benard_demo.py` | Ra & state from lab cell to Sun + the Nu(Ra) onset curve |
| `examples/terminal_velocity_demo.py` | Speeds fog-to-skydiver + the v(r) Stokes/quadratic bend |
| `examples/snr_phases_demo.py` | R/v/phase from centuries to Myr + the radius-vs-age track |
| `examples/magnetic_mirror_demo.py` | Loss-cone angle & trapping by mirror ratio + the alpha(R_m) curve |
| `examples/debye_demo.py` | lambda_D/N_D/f_p by environment + the f_p(n) radio-cutoff curve |
| `examples/line_broadening_demo.py` | Doppler/pressure widths by environment + Gaussian vs Lorentzian profiles |
| `examples/curve_of_growth_demo.py` | W & regime vs optical depth + the three-segment curve |
| `examples/sackur_tetrode_demo.py` | Predicted vs measured noble-gas entropy + S(T) curves |
| `examples/maxwell_boltzmann_demo.py` | Characteristic speeds by gas + the f(v) distributions |
| `examples/gamow_demo.py` | Peak energy by reaction + the tail x tunnelling = peak curves |
| `examples/parallax_demo.py` | Distances & space velocities of nearby stars + the parallax geometry |
| `examples/standard_candle_demo.py` | Moduli of landmark objects + the ladder-rung modulus curve |
| `examples/tully_fisher_demo.py` | L/M_abs/M_baryon by rotation speed + the slope-4 log-log line |
| `examples/tolman_demo.py` | Dimming vs z (expanding vs tired-light) + the magnitude curves |
| `examples/olbers_demo.py` | Mean free path/horizon/covered fraction + the covering-vs-distance curve |
| `examples/bi_elliptic_demo.py` | Hohmann vs bi-elliptic delta-v by ratio + the crossover curves |
| `examples/gravity_assist_demo.py` | Turn angle & boost by flyby depth/speed + the boost(r_p) curves |
| `examples/synodic_demo.py` | Synodic period & conjunction cadence per planet + the S(P) curve |
| `examples/black_hole_shadow_demo.py` | Shadow size for M87*/Sgr A* + the nested-radii diagram |
| `examples/hill_sphere_demo.py` | Hill radius & stable-moon limit per planet + the r_H(a) plot |
| `examples/j2_precession_demo.py` | Nodal/apsidal rates per orbit + the rate-vs-inclination curves |
| `examples/solar_sail_demo.py` | Pressure/accel/beta per sail + the beta-vs-area/mass curve |
| `examples/beaming_demo.py` | Doppler/boost/jet-ratio per gamma & angle + the D(theta) curves |
| `examples/relativistic_rocket_demo.py` | Ship/Earth time & v to each destination + the divergence plot |
| `examples/relativistic_doppler_demo.py` | Receding/approaching/transverse z per speed + the z(beta) curves |
| `examples/de_broglie_demo.py` | Matter wavelengths electron-to-baseball + the lambda(E) curves |
| `examples/bohr_demo.py` | Energy levels & series wavelengths + the level diagram |
| `examples/roche_demo.py` | Survival curve across the Roche limit + a tidal-stream SVG |
| `examples/tidal_heating_demo.py` | Galilean-moon heating table + heating-vs-eccentricity curve |
| `examples/roche_lobe_demo.py` | Lobe radius & transfer stability vs mass ratio |
| `examples/kozai_demo.py` | e/i oscillations vs analytic e_max, out-of-phase time series SVG |
| `examples/resonance_demo.py` | 2:1 resonant argument: libration (locked) vs circulation (free) |
| `examples/coorbital_demo.py` | Tadpole & horseshoe paths in the rotating frame (SVG) |
| `examples/tisserand_demo.py` | a & e jump across a flyby while Tisserand stays flat |
| `examples/lensing_demo.py` | Microlensing light curve + Einstein-ring image diagram (SVG) |
| `examples/rotation_curve_demo.py` | Visible (declining) vs disk+halo (flat) rotation curves |
| `examples/faber_jackson_demo.py` | L ~ sigma^4 from dwarf to giant ellipticals |
| `examples/mond_demo.py` | MOND (flat) vs Newton-on-baryons (declining) + Tully-Fisher |
| `examples/schwarzschild_demo.py` | Precessing & plunging black-hole orbits with ISCO/photon sphere |
| `examples/kerr_demo.py` | ISCO-vs-spin curves + horizon/ergosphere diagram |
| `examples/penrose_demo.py` | Extractable-energy fraction vs spin + the area theorem |
| `examples/hawking_demo.py` | Temperature & evaporation time across black-hole masses |
| `examples/eddington_demo.py` | L_Edd across masses + Eddington-limited growth to a quasar |
| `examples/bondi_demo.py` | Accretion rate vs gas temperature and mass |
| `examples/hohmann_demo.py` | LEO->GEO & Earth->Mars delta-v budgets + transfer diagram |
| `examples/oberth_demo.py` | Periapsis-vs-apoapsis burn: escape speed vs burn radius |
| `examples/cosmic_velocities_demo.py` | Orbital/escape speeds from the Moon to a white dwarf |
| `examples/atmosphere_demo.py` | Gas-retention grid across bodies + escape-vs-thermal plot |
| `examples/gr_time_demo.py` | Pound-Rebka, GPS gain, Sun redshift + Shapiro-delay curve |
| `examples/lense_thirring_demo.py` | GP-B geodetic & frame-drag rates vs orbit radius |
| `examples/pulsar_demo.py` | Hulse-Taylor dP/dt vs measured + the periastron-shift parabola |
| `examples/friedmann_demo.py` | Scale-factor curves for radiation/matter/dark-energy/LCDM |
| `examples/saha_demo.py` | Ionization fraction plunging to zero at recombination |
| `examples/cmb_demo.py` | Sound horizon, acoustic angle & the l~220 peak comb |
| `examples/blackbody_demo.py` | Peak wavelengths (CMB->B-star) + Planck spectra |
| `examples/compton_demo.py` | Compton shift/energy vs angle + inverse-Compton boost |
| `examples/larmor_demo.py` | Radiated power vs gamma (gamma^4 circular, gamma^6 linear) |
| `examples/synchrotron_demo.py` | Critical frequency/power/cooling vs energy + spectral index |
| `examples/optical_depth_demo.py` | Transmission vs tau + the photosphere at tau=2/3 |
| `examples/bbn_demo.py` | n/p freeze-out chain and the Y_p ~ 0.25 helium fraction |
| `examples/lane_emden_demo.py` | Polytrope density profiles + surface-radius / mass table |
| `examples/main_sequence_demo.py` | Mass-L-lifetime table + the main sequence on an HR diagram |
| `examples/kelvin_helmholtz_demo.py` | Kelvin-Helmholtz vs nuclear timescale by mass |
| `examples/distances_demo.py` | Hubble diagram (LCDM vs decelerating) + D_A turnover |
| `examples/degeneracy_demo.py` | Degeneracy-pressure laws vs density with the relativistic transition |
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

## Fermi degeneracy pressure: the quantum floor

The Pauli exclusion principle makes a cold, dense electron gas resist
compression even at zero temperature -- the pressure that holds up white dwarfs.
`degeneracy.py`:

```
$ python examples/degeneracy_demo.py examples/output

  relativistic transition density: 5.87e35 /m^3
      n (/m^3)          regime        P (Pa)
         1e+35         non-rel      5.03e+20
         1e+36    relativistic      2.45e+22
```

Filling momentum states to the Fermi momentum gives `P ~ n^{5/3}` while
electrons are non-relativistic, softening to `P ~ n^{4/3}` once they turn
relativistic (above `n ~ 6e35 /m^3`, white-dwarf densities). That softer exponent
is precisely why gravity eventually overwhelms a massive white dwarf -- the seed
of the Chandrasekhar mass in the next section. The tests verify both scalings,
the relativistic transition, and that metals are non-relativistic while white
dwarfs are.

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

## Kelvin-Helmholtz time: why the Sun can't run on gravity

Before fusion was known, Kelvin and Helmholtz argued the Sun shines by
contracting. `kelvin_helmholtz.py` shows why that fails:

```
$ python examples/kelvin_helmholtz_demo.py examples/output

  Sun's KH time: 31 Myr -- vs Earth's 4500 Myr age.
  mass (M_sun)    t_KH (Myr)   t_nuclear (Myr)
           1.0         31.42           10000.0
          10.0          0.16              31.6
```

The thermal time `t_KH = G M^2 / (R L)` -- how long a star could shine by
radiating its gravitational binding energy -- is only ~30 Myr for the Sun,
hundreds of times shorter than the geological age of the Earth. That mismatch was
the historic proof that stars must be powered by nuclear fusion (whose
main-sequence lifetime is ~10 Gyr). `t_KH` survives as the timescale on which a
protostar contracts before ignition. The tests reproduce the 30 Myr solar value,
the mismatch with Earth's age, the `M^2/(RL)` scaling, and the binding energy.

## The main sequence and the HR diagram

Mass sets a star's whole life. `main_sequence.py` encodes the empirical
relations:

```
$ python examples/main_sequence_demo.py examples/output

  mass (M_sun)   L (L_sun)  T_eff (K)      lifetime
           0.3         0.0       3258     202.9 Gyr
           1.0         1.0       5772      10.0 Gyr
          10.0      3162.3      17232        32 Myr
          30.0    147885.1      29037         2 Myr
```

`L ~ M^{3.5}` makes massive stars blindingly bright, so they exhaust their fuel
in a few Myr (`t ~ M/L ~ M^{-2.5}`), while a red dwarf sips hydrogen for hundreds
of Gyr -- far longer than the current age of the universe. Coupling
`L = 4 pi R^2 sigma T^4` places each star on the Hertzsprung-Russell diagram; the
plot of luminosity vs temperature is the main sequence. The tests verify the
mass-luminosity slope, the 10 Gyr solar lifetime, the `M^{-2.5}` lifetime
scaling, and the temperature ordering.

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

## The Larmor formula: radiation from acceleration

The root of every classical radiation process. `larmor.py` gives the power an
accelerating charge emits:

```
$ python examples/larmor_demo.py examples/output

     gamma    perp (gamma^4)  parallel (gamma^6)
        10          5.71e-10            5.71e-08
      1000          5.71e-02            5.71e+04
  classical hydrogen atom collapse time: 1.55e-11 s
```

`P = q^2 a^2 / (6 pi eps0 c^3)` is quadratic in acceleration; relativistically a
circular accelerator boosts it by `gamma^4` and a linear one by `gamma^6`. The
same formula predicts a classical hydrogen atom collapses in `~1.6e-11 s` -- the
catastrophe quantum mechanics had to resolve -- and, boosted by `gamma^4`, is the
engine of synchrotron radiation in the next section. The tests verify the `a^2`
law, the `gamma^4`/`gamma^6` boosts, and the atom-collapse time.

## Optical depth: where a star's surface is

Light crossing matter is attenuated as `exp(-tau)`. `optical_depth.py`:

```
$ python examples/optical_depth_demo.py examples/output

     tau   transmitted          regime
    0.67         0.513   thin (photosphere)
    1.00         0.368            thin
    3.00         0.050           thick
```

The optical depth `tau = n sigma L` counts mean free paths; `tau << 1` is
transparent, `tau >> 1` opaque. A star has no solid surface -- its photosphere is
simply the layer where the inward optical depth reaches `tau ~ 2/3`
(Eddington-Barbier), the depth photons escape from and that fixes the effective
temperature. The tests verify the `exp(-tau)` law, the linearity in
density/path, the mean free path, and the `tau = 2/3` photosphere.

## Synchrotron radiation: the cosmic radio glow

Relativistic electrons spiralling in magnetic fields power most cosmic radio
emission. `synchrotron.py`:

```
$ python examples/synchrotron_demo.py examples/output   (B = 1 nT)

     gamma     nu_c (Hz)       cooling
     1e+04      4.20e+09      2.5e+07 yr
     1e+06      4.20e+13      2.5e+05 yr
  p=2.5: alpha=0.75  (S(nu) ~ nu^-0.75)
```

The critical frequency `nu_c ~ gamma^2 B` puts `gamma ~ 1e4` electrons in
microgauss fields at GHz radio; the single-electron power goes as `gamma^2 B^2`,
so high-energy electrons cool fastest. A power-law electron population
`N(E) ~ E^{-p}` radiates a power-law spectrum of index `(p-1)/2` -- the observed
radio slope (~0.75) that reveals the electron distribution in jets, radio
galaxies, and supernova remnants. The tests verify the radio frequency, the
`gamma^2 B` and `gamma^2 B^2` scalings, faster cooling at higher energy, and the
spectral index.

## Compton and inverse-Compton scattering

Photons exchange energy with electrons. `compton.py`:

```
$ python examples/compton_demo.py examples/output

   angle (deg)  shift (pm)   E scattered (keV)   (500 keV photon)
             0       0.000               500.0
            90       2.426               252.7
           180       4.853               169.1
  inverse Compton gamma=1000: boost x1.3e6  (1 meV CMB photon -> 1333 eV)
```

A photon off a stationary electron lengthens by `lambda_C(1 - cos theta)` with
`lambda_C = h/m_e c = 2.426 pm`, losing the most energy at back-scattering.
Inverse Compton runs it the other way: a relativistic electron kicks a photon up
in energy by `~gamma^2`, turning CMB and starlight into X-rays and gamma-rays --
the engine of the Sunyaev-Zeldovich effect and high-energy astrophysics. The
tests verify the 2.426 pm wavelength, the 511 keV electron rest energy, the shift
at each angle, and the `gamma^2` boost.

## Blackbody radiation: Planck, Wien, Stefan-Boltzmann

The universal thermal spectrum. `blackbody.py` gives the Planck law, Wien's peak,
and the Stefan-Boltzmann flux:

```
$ python examples/blackbody_demo.py examples/output

  object              T (K)     peak    flux (W/m^2)
  CMB                     3   1.06 mm       3.13e-06
  Sun (G)              5772    502 nm       6.29e+07
  hot star (B)        20000    145 nm       9.07e+09
```

`lambda_max T = 2.9 mm K` (Wien) makes hot bodies blue and cool ones red, and
`j = sigma T^4` (Stefan-Boltzmann) makes a 20000 K star outshine the Sun 144x per
unit area. The Sun peaks in the visible at ~500 nm; the 2.725 K CMB peaks in the
microwave. The tests reproduce the Sun/CMB/body peaks, the numeric-vs-Wien
agreement, the `T^4` flux, and the solar luminosity.

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

## Atmospheric escape: which worlds keep air

Whether a planet holds a gas is a race between gravity and heat. `atmosphere.py`
uses the Jeans escape parameter `lambda = v_esc^2 / v_th^2`:

```
$ python examples/atmosphere_demo.py examples/output

  body            H2      He     H2O      N2     CO2
  Moon          lose    lose    lose    lose    keep
  Earth         lose    lose    keep    keep    keep
  Jupiter       keep    keep    keep    keep    keep
```

A species is retained over the age of the Solar System when the escape speed
exceeds ~6x the molecules' most-probable thermal speed (`lambda >~ 36`). That is
why Earth keeps `N2`/`O2` but slowly loses `H2` and `He`, the Moon and Mars lose
the light gases, and Jupiter keeps even hydrogen -- exactly the atmospheres we
observe. The tests reproduce those cases and the mass/temperature scalings of the
escape parameter.

## Escape and cosmic velocities

The speed thresholds of spaceflight, from one formula. `cosmic_velocities.py`:

```
$ python examples/cosmic_velocities_demo.py examples/output

  body          v_orbit (km/s)  v_escape (km/s)
  Earth                   7.91            11.19
  Sun                   436.82           617.75
  white dwarf          5000.33          7071.53
  leaving the Solar System from Earth's orbit: 42.1 km/s
  set v_escape = c -> Schwarzschild radius: 2954 m for the Sun.
```

The circular-orbit speed is `v1 = sqrt(GM/r)`, escape is `v2 = sqrt(2) v1`
(11.2 km/s from Earth), and ~42 km/s leaves the Solar System from Earth's orbit.
Pushing the escape speed to `c` in `sqrt(2GM/r) = c` recovers the Schwarzschild
radius `2GM/c^2` -- the point where not even light escapes. The tests reproduce
the Earth values, the universal `sqrt(2)` ratio, the Solar-System speed, and the
Schwarzschild radii of the Sun (~3 km) and Earth (~9 mm).

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

## Bondi accretion: feeding on ambient gas

The companion to the Eddington limit -- the supply side. `bondi.py` gives the
spherical accretion rate onto a body at rest in gas:

```
$ python examples/bondi_demo.py examples/output

   gas T (K)  c_s (km/s)  r_B (AU)  Mdot (Msun/yr)
       1e+02         1.2    6450.0        9.09e-11
       1e+04        11.7      64.5        9.09e-14
       1e+07       370.9       0.1        2.87e-18
```

`Mdot = 4 pi lambda (G M)^2 rho / c_s^3` runs away as `M^2` (bigger holes eat
faster), rises with density, and falls steeply with temperature (`c_s^{-3} ~
T^{-3/2}`) -- a black hole in a 100 K molecular cloud accretes millions of times
faster than one in 10^7 K coronal gas. Compared with the Eddington rate it tells
you whether growth is supply-limited or radiation-limited. The tests verify all
three scalings and the ~tens-of-AU Bondi radius.

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

## Faber-Jackson: ellipticals from their motion

The elliptical-galaxy analog of Tully-Fisher. `faber_jackson.py`:

```
$ python examples/faber_jackson_demo.py examples/output

  sigma (km/s)     L (L_sun)  virial M (M_sun)
            50      7.81e+07          2.91e+09
           200      2.00e+10          4.65e+10
           400      3.20e+11          1.86e+11
```

Luminosity climbs as the fourth power of the stellar velocity dispersion,
`L ~ sigma^4` -- a consequence of the virial theorem with a roughly constant
mass-to-light ratio and surface brightness. Because the dependence is so steep, a
spectral line width fixes a galaxy's luminosity and, against its apparent
brightness, its distance. The tests verify the `sigma^4` slope, the `L*` and
giant-elliptical luminosities, the virial-mass scale, and the inversion.

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

## Bremsstrahlung: the X-rays of cluster gas

Free electrons braking in ion Coulomb fields radiate -- the free-free X-rays that
make galaxy clusters glow. `bremsstrahlung.py`:

```
$ python examples/bremsstrahlung_demo.py examples/output   (T = 5e7 K)

    n_e (/m^3)  emissivity (W/m^3)  t_cool (Gyr)   flow?
         1e+03            9.9e-31          66.3      no
         1e+04            9.9e-29           6.6      yes
```

Emissivity goes as `n^2 sqrt(T)`, so dense cores glow brightest, and the cooling
time `~ sqrt(T)/n` drops below a Hubble time only in those cores -- the cooling
flows -- while the tenuous outskirts effectively never cool. This is the X-ray
emission whose CMB imprint is the SZ effect, and it's rooted in the same Larmor
radiation as synchrotron. The tests verify the `n^2` and `sqrt(T)` scalings, the
cooling-time scalings, and the cluster cooling-flow regime.

## Pair production: why the gamma-ray sky has a horizon

Turn the emission around: sufficiently energetic photons don't just scatter, they
collide and *become matter*. Two photons make an electron-positron pair once
`E1 E2 (1 - cos theta) >= 2 (m_e c^2)^2`. `pair_production.py`:

```
$ python examples/pair_production_demo.py examples/output

     background photon    E (eV)   threshold gamma
  ------------------------------------------------
                   CMB   6.0e-04           435 TeV
        infrared (EBL)   1.0e-01             3 TeV
         optical (EBL)   2.0e+00           131 GeV
                 X-ray   1.0e+03           261 MeV
```

Head-on, two 511 keV gammas just reach threshold (`E >= m_e c^2` each). A lone
high-energy gamma pair-produces off a soft background photon above
`(m_e c^2)^2 / E_bg`, so a TeV photon from a distant blazar is annihilated by the
optical/infrared extragalactic background light and a PeV photon by the meV CMB.
The universe is opaque to gamma rays beyond a horizon that shrinks as their energy
rises -- the same `m_e c^2` scale that sets Compton scattering, run in reverse.
The tests verify the 511 keV threshold, the `(m_e c^2)^2 / E_bg` partner energy,
the angle dependence, and the TeV-EBL / PeV-CMB absorption cases.

## Precession of the equinoxes: the 26,000-year wobble

Earth is an oblate spheroid whose equatorial bulge is tilted 23.4 degrees to the
ecliptic, so the Sun and Moon pull harder on the near side than the far side. That
torque makes the spin axis sweep out a cone -- a leaning gyroscope, not a toppling
one. `axial_precession.py`:

```
$ python examples/axial_precession_demo.py examples/output

      source     arcsec/yr     period (yr)
  ----------------------------------------
         Sun         15.95           81269
        Moon         34.70           37352
    Sun+Moon         50.64           25591
```

Torque scales as `M / r^3`, so the nearby Moon out-torques the vastly heavier Sun by
`(M_moon/M_sun)(AU/r_moon)^3 ~ 2.2`. Their sum, ~50.6 arcsec/yr (measured 50.29),
carries the celestial pole around a 47-degree circle once every ~25,600 years
(measured ~25,772) -- the "Great Year." That is why Polaris is only a temporary North
Star (Vega had the job ~12,000 BC and gets it back ~14,000 AD), why the tropical year
is ~20 minutes shorter than the sidereal year, and why the equinox has slipped a whole
zodiac sign since the constellations were named. The tests verify the arcsec rate, the
period, the Moon-beats-Sun ratio, the `M/r^3` scaling, and the `cos(obliquity)`
dependence.

## Alfven waves: the magnetized plasma's plucked string

A magnetic field threading a conducting plasma acts like tensioned strings -- displace
the field lines and magnetic tension springs them back, with the frozen-in plasma
supplying the inertia. The transverse wave runs along `B` at the Alfven speed
`v_A = B / sqrt(mu0 rho)`, the magnetic analogue of the sound speed. `alfven.py`:

```
$ python examples/alfven_demo.py examples/output

             environment     B (T)    n (/m^3)  v_A (km/s)      beta
  ------------------------------------------------------------------
           active corona   1.0e-02     1.0e+15      6897.6    0.0007
            quiet corona   1.0e-03     1.0e+14      2181.2    0.0035
         solar wind 1 AU   5.0e-09     5.0e+06        48.8    0.6940
                warm ISM   5.0e-10     1.0e+06        10.9    1.1104
```

The plasma beta `= p_gas/p_mag` says who is in charge: `beta << 1` in the corona means
the field channels the plasma and stores the energy that heats it and drives flares;
`beta > 1` in dense interiors means gas pressure drags the field around. The solar wind
starts sub-Alfvenic -- so the Sun's field co-rotates the plasma and magnetically brakes
the spin -- then crosses the Alfven surface near ~15 R_sun (where Parker Solar Probe
found it) and coasts out super-Alfvenic, decoupled from the Sun's rotation. The tests
verify the coronal speed, the field- vs gas-dominated regimes, the super-Alfvenic wind,
and the `v_A ~ B`, `v_A ~ 1/sqrt(rho)`, `beta ~ 1/B^2` scalings.

## The Parker spiral: the Sun's field wound up by its rotation

The solar wind drags the Sun's magnetic field radially outward, but the field's feet
stay rooted in a Sun that turns once every ~25 days -- a rotating sprinkler. Each plasma
parcel flies straight out, yet the field line joining them is an Archimedean spiral.
`parker_spiral.py`:

```
$ python examples/parker_spiral_demo.py examples/output

        location   r (AU)  angle (deg)   |B| (nT)
  -----------------------------------------------
         Mercury     0.39         22.7      35.63
           Venus     0.72         37.7      12.18
           Earth     1.00         47.0       7.33
            Mars     1.52         58.5       4.14
         Jupiter     5.20         79.8       1.05
          Saturn     9.58         84.4       0.56
```

The garden-hose angle `tan(psi) = Omega r sin(theta) / u` grows with distance: nearly
radial near the Sun, ~45 degrees at Earth (measured value), nearly azimuthal past
Jupiter. Flux conservation splits the field into a radial part falling as `1/r^2` and an
azimuthal part falling only as `1/r`, so the distant heliospheric field is mostly the
wound-up azimuthal component -- and Earth's ~7 nT total field matches. This geometry is
why solar energetic particles from a western-limb flare reach Earth best (they are
magnetically connected along the spiral). The tests verify the ~45-degree angle at 1 AU,
the radial/azimuthal regimes near the Sun and at Jupiter, the `1/r^2` and `1/r` scalings,
and that a faster wind winds the spiral less tightly.

## Magnetic braking & gyrochronology: a star's age from its spin

That magnetized wind is also a superb brake. Plasma leaving the star stays locked to
the field out to the Alfven radius (~15 R_sun), so it is forced to corotate on a long
lever arm and carries off angular momentum -- a feeble ~1e-14 Msun/yr mass loss bleeds
a large spin. Fast rotators brake hardest, so a broad spread of young spins converges
onto one age-period sequence. `magnetic_braking.py`:

```
$ python examples/magnetic_braking_demo.py examples/output

                 rotator  P (days)  gyro age (Gyr)
  ------------------------------------------------
         Pleiades member       3.0            0.06
        young field star       6.0            0.25
           Hyades member       8.5            0.51
                 the Sun      25.4            4.57
     old thick-disk star      35.0            8.67
```

Skumanich's empirical law has the surface rotation decay as `Omega ~ t^(-1/2)`, so the
period grows as `P ~ t^(1/2)` and inverts to an age: `t = t_sun (P/P_sun)^2`. That is
gyrochronology -- the Sun's 25-day spin reads 4.6 Gyr, a 3-day Pleiad reads ~60 Myr, and
a single rotation measurement dates a field star or open cluster. Underneath sits the
Weber-Davis wind torque `dJ/dt = (2/3) Mdot Omega r_A^2` with the Alfven radius as the
lever arm. The tests verify the `t^(1/2)` period law, the solar recovery, the young
fast-rotator, the many-radii Alfven lever arm, and the torque's `Omega` and `r_A^2`
scalings.

## Tidal locking: why the Moon shows one face

Internal friction drags a body's tidal bulge slightly out of line with its primary, and
that misaligned bulge feels a torque that despins the body toward synchronous rotation.
`tidal_locking.py`:

```
$ python examples/tidal_locking_demo.py examples/output

           body -> primary  t_lock (Gyr)   locked?
  ------------------------------------------------
             Moon -> Earth       0.00933       yes
            Phobos -> Mars      7.87e-11       yes
             Io -> Jupiter      1.01e-07       yes
             Earth -> Moon          17.5        no
            Mercury -> Sun         0.349       yes
```

The locking time `t ~ a^6 I Q / (G M_p^2 k2 R^5)` is dominated by the brutal `a^6`
distance factor, so close-in moons lock in a geological blink while distant bodies never
do. The Moon locked to Earth long ago (~9 Myr from a fast primordial spin), but the Earth,
braking only on the Moon's far weaker tide, needs ~17 Gyr -- longer than the universe is
old -- which is why our days are still lengthening (~1.8 ms/century) rather than frozen.
Mercury dodged full locking into a 3:2 spin-orbit resonance, and hot Jupiters at a few
stellar radii are all assumed synchronous. The tests verify the Moon-locked / Earth-free
contrast, the `a^6` and `1/M_p^2` scalings, and the locking-zone boundary.

## Jeans escape: which gases a world keeps

At the top of an atmosphere (the exobase) molecules follow a Maxwell-Boltzmann speed
distribution; any moving upward faster than escape speed leave for good. Light, hot gases
have a fatter high-speed tail, so the escape parameter `lambda = v_esc^2/v_th^2 =
G M m/(R k_B T)` decides who keeps an air. `jeans_escape.py`:

```
$ python examples/jeans_escape_demo.py examples/output

      body    v_esc     H2     He    H2O     N2     O2    CO2   (Y=kept, .=lost)
  -------------------------------------------------------------------
     Earth    11.2k      .      .      Y      Y      Y      Y
      Moon     2.4k      .      .      .      .      .      Y
      Mars     5.0k      .      .      Y      Y      Y      Y
     Titan     2.6k      .      .      Y      Y      Y      Y
   Jupiter    60.2k      Y      Y      Y      Y      Y      Y
```

A rough rule keeps a gas over geologic time when `v_esc >~ 6 v_th` (lambda >~ 36), and
the Jeans flux carries a steep `exp(-lambda)` Boltzmann factor. So Earth keeps its heavy
N2/O2/CO2 and water but lost its primordial H2 and He, the hot low-gravity Moon holds
almost nothing, cold Titan clings even to nitrogen, and giant Jupiter retains hydrogen
itself. Plotting escape speed against thermal speed draws the "cosmic shoreline" that
separates worlds with atmospheres from airless ones. The tests verify Earth's keep/lose
split, the airless Moon, the `lambda` scalings with mass/gravity/temperature, and the
exponential suppression of heavy-gas escape.

## The snow line: why the inner solar system is rocky

A protoplanetary disk is heated by its star, so for a grain in radiative equilibrium the
temperature falls as `T(r) = (L / 16 pi sigma r^2)^(1/4) ~ r^(-1/2)`. The snow line is
where the disk cools past water ice's ~160 K condensation point. `snow_line.py`:

```
$ python examples/snow_line_demo.py examples/output

      planet   r (AU)    T (K)     state
  --------------------------------------
     Mercury     0.39    445.7      rock
       Venus     0.72    328.0      rock
       Earth     1.00    278.3      rock
        Mars     1.52    225.8      rock
     Jupiter     5.20    122.1  rock+ice
      Saturn     9.58     89.9  rock+ice
      Uranus    19.20     63.5  rock+ice
     Neptune    30.10     50.7  rock+ice
```

The water snow line lands at ~3 AU, right between Mars and Jupiter. Inside it water is
vapour, so only rock and metal condense and the terrestrial planets stayed small and dry;
outside it ice roughly triples the solid surface density, letting Jupiter's core grow fast
enough to seize nebular gas before the disk dissipated. Each ice has its own frost line --
CO2 near ~16 AU, CO out past ~190 AU -- sorting the disk by composition, and a brighter
star pushes the whole pattern outward as `sqrt(L)`. The tests verify the ~280 K
temperature at 1 AU, the `r^(-1/2)` profile, the ~3 AU water line, the frost-line ordering,
and the `sqrt(L)` scaling.

## Poynting-Robertson drag: dust spiralling into the Sun

A dust grain absorbs radially-incoming sunlight and re-emits it isotropically in its own
frame, but aberration makes those photons carry off a little forward momentum in the Sun's
frame -- a headwind of the grain's own thermal radiation that drains orbital angular
momentum. `poynting_robertson.py`:

```
$ python examples/poynting_robertson_demo.py examples/output

    grain size     beta     fate / t_PR from 1 AU
  ------------------------------------------------
        0.1 um    1.914       blown out (unbound)
        0.3 um    0.638       blown out (unbound)
        0.5 um    0.383                  1,046 yr
          1 um    0.191                  2,092 yr
         10 um    0.019                 20,923 yr
        100 um    0.002                209,231 yr
          1 mm    0.000              2,092,311 yr
```

The size-dependent `beta = 3 L / (16 pi G M c rho s)` is the ratio of radiation pressure to
gravity; grains with `beta > 1/2` (below ~0.4 micron here) are unbound the moment they are
released and blown out as "beta meteoroids." Bound grains spiral in on `t_PR ~ r^2 s`, so
micron dust at 1 AU is gone in a few thousand years -- thousands of times less than the age
of the solar system. The zodiacal cloud therefore cannot be primordial; it must be
continuously resupplied by comet trails and asteroid collisions. The tests verify the
`beta ~ 1/s` law, the ~0.1-0.5 micron blow-out size, the `r^2` and `s` inspiral scalings,
and the fast micron-grain infall.

## Toomre Q: when a rotating disk fragments

A thin, rotating disk is caught between self-gravity (which collapses overdense patches),
pressure or velocity dispersion (which resists collapse on small scales) and rotation via
the epicyclic frequency kappa (which resists it on large scales). Toomre's `Q` sets the
balance in one number. `toomre.py`:

```
$ python examples/toomre_demo.py examples/output

    R (kpc)  kappa (/Gyr)    gas Q       state
  --------------------------------------------
          2         159.1     2.61      stable
          4          79.6     1.82      stable
          6          53.0     1.69      stable
          8          39.8     1.77      stable
         10          31.8     1.98      stable
         14          22.7     2.75      stable
         18          17.7     4.17      stable
```

With `Q = c_s kappa / (pi G Sigma)` for gas (and `sigma_R kappa / 3.36 G Sigma` for stars),
`Q > 1` is stable at every wavelength while `Q < 1` opens an unstable band that fragments
into clumps and spiral arms; the most-unstable Toomre wavelength `4 pi^2 G Sigma / kappa^2`
sets their ~kpc size. The Milky Way hovers at `Q ~ 1.5-2` -- marginally stable, and not by
chance: a disk cooling toward `Q < 1` forms stars that heat it back up, so disks
self-regulate to the stability line. The tests verify the marginally-stable solar
neighbourhood, the stability verdict, the `c_s` and `Sigma` scalings, the critical
dispersion at `Q = 1`, and the kpc-scale Toomre wavelength.

## Accretion disks: why black holes glow in X-rays

Gas carrying angular momentum cannot fall straight in -- it forms a disk and spirals inward
only as viscosity ferries momentum outward, dissipating gravitational energy as heat that
each ring radiates as a blackbody. `accretion_disk.py`:

```
$ python examples/accretion_disk_demo.py examples/output

              object    r_in (km)   peak T (K)     peak kT   L_Edd (Lsun)
  -----------------------------------------------------------------------
     stellar-mass BH         88.6     8.64e+06    0.74 keV       3.28e+05
     supermassive BH  886237966.5     1.54e+05       13 eV       3.28e+12
```

The Shakura-Sunyaev profile `T ~ r^(-3/4)` makes the inner edge (the ISCO at 3 Schwarzschild
radii) hottest, and because the characteristic scale goes as `M^(-1/4)` a heavier hole runs
a cooler disk: a 10-solar-mass black hole peaks around a keV in soft X-rays (how X-ray
binaries are found) while a billion-solar-mass one peaks in the ultraviolet -- the quasar
"big blue bump." Both cap out at the Eddington luminosity `L_Edd = 4 pi G M m_p c / sigma_T`
and convert `eta ~ 6%` of infalling rest mass to light via `L = eta Mdot c^2`, about eight
times more efficient than hydrogen fusion, which is why accreting black holes are the
brightest steady engines in the universe. The tests verify the ISCO radius, the `r^(-3/4)`
profile vanishing at the inner edge, the X-ray-hot stellar disk and UV quasar disk, the
Eddington scaling, and the efficiency advantage over fusion.

## Fermi acceleration: the universal cosmic-ray spectrum

Fermi's answer to how cosmic rays reach vast energies was repeated small kicks: a charged
particle crossing a shock front gains energy each round trip and has a fixed escape chance,
and "multiply the energy by a constant per cycle, lose a constant fraction per cycle"
produces a scale-free power law `N(E) ~ E^(-p)`. `fermi_acceleration.py`:

```
$ python examples/fermi_acceleration_demo.py examples/output

     Mach   compression r   index p
  --------------------------------
        2          2.286     3.333
        3          3.000     2.500
        5          3.571     2.167
       10          3.883     2.040
       50          3.995     2.002
     1000          4.000     2.000
```

The beauty of *first-order* (diffusive shock) acceleration is that the index depends only
on the shock compression ratio, `p = (r+2)/(r-1)`, not on the messy microphysics. Every
strong shock hits the Rankine-Hugoniot limit `r -> 4` (gamma=5/3), so `p -> 2` -- the
near-universal `E^(-2)` spectrum injected by supernova remnants across the Galaxy. And
because the per-cycle gain is `~(4/3) beta` at a shock versus `~(4/3) beta^2` for Fermi's
original random magnetic clouds, shocks win by a factor `1/beta`, which is why they, not
clouds, are the cosmic-ray engines. The tests verify the `r -> 4` strong-shock limit, the
`p = 2` universal index, steeper spectra from weaker shocks, and the first- over
second-order advantage.

## Stellar opacity: how slowly light escapes

Opacity `kappa` (m^2/kg) sets the photon mean free path `1/(kappa rho)` and so how slowly a
star leaks its luminosity. `opacity.py`:

```
$ python examples/opacity_demo.py examples/output

            region       rho     T (K)  kappa_es   Kramers    total
  -----------------------------------------------------------------
      solar centre   1.5e+05   1.5e+07     0.034     0.074    0.108
    radiative zone   2.0e+04   5.0e+06     0.034     0.461    0.494
      near surface   1.0e-03   1.0e+05     0.034     0.020    0.054
       photosphere   1.0e-04   6.0e+03     0.034    38.468   38.502
```

Electron (Thomson) scattering is a flat floor, `~0.034 m^2/kg`, independent of density and
temperature -- the opacity the Eddington luminosity rests on. Kramers bound-free and
free-free absorption follow `kappa ~ rho T^(-7/2)`, so the cool outer layers are hundreds of
times more opaque than the blazing core; that steep fall-off is what flips a stellar
envelope from radiative diffusion to convection. At the solar centre the total is ~0.1
m^2/kg and the photon mean free path is only ~60 microns, so light random-walks out over
~100,000 years. The tests verify the electron-scattering value and constancy, the `T^(-7/2)`
and density scalings, the solar-centre magnitude, the high-T approach to the floor, and the
sub-millimetre mean free path.

## Brunt-Vaisala frequency: buoyancy waves and convection

Displace a fluid parcel upward in a stratified medium: if it ends up denser than its new
surroundings gravity pulls it back and it overshoots, oscillating at the Brunt-Vaisala
frequency N; if it ends up lighter, buoyancy runs away and the layer convects.
`brunt_vaisala.py`:

```
$ python examples/brunt_vaisala_demo.py examples/output

                   layer  dT/dz (K/km)   N (1/s)      period        state
  -----------------------------------------------------------------------
        strong inversion          10.0    0.0266     3.9 min       stable
            stratosphere           2.0    0.0229     4.6 min       stable
              isothermal           0.0    0.0187     5.6 min       stable
     typical troposphere          -6.5    0.0108     9.7 min       stable
           dry adiabatic          -9.8    0.0000         --    CONVECTIVE
          superadiabatic         -15.0    0.0000         --    CONVECTIVE
```

For an ideal gas `N^2 = (g/T)(dT/dz + g/c_p)`, comparing the environmental temperature
gradient to the dry adiabatic lapse rate `g/c_p ~ 9.8 K/km`. When `N^2 > 0` the layer is
stably stratified and rings with internal gravity waves (buoyancy period `2 pi / N`, ~5-10
min in the troposphere and threading the Sun's radiative core as g-modes); when `N^2 < 0`
it convects. That sign change is precisely the Schwarzschild convection criterion that
decides how stars and atmospheres carry heat. The tests verify the lapse rate, the stable
subadiabatic and isothermal layers, the superadiabatic convective instability, the absence
of a real oscillation frequency when unstable, and the `2 pi / N` period.

## Ram-pressure stripping: how clusters strip spirals of their gas

A galaxy plunging through the hot intracluster medium feels a wind of ram pressure
`P = rho_icm v^2`. The Gunn-Gott criterion strips its interstellar gas wherever that wind
beats the disk's gravitational hold `2 pi G Sigma_star Sigma_gas`. `ram_pressure.py`:

```
$ python examples/ram_pressure_demo.py examples/output

             environment   n (/cc)  v (km/s)  R_strip (kpc)
  ---------------------------------------------------------
        field / isolated     1e-05       300          16.62
              poor group     1e-04       500          11.63
       cluster outskirts     5e-04      1000           7.14
            rich cluster     1e-03      1500           4.88
        dense core, fast     3e-03      2000           2.37
```

A galaxy keeps only the gas inside the stripping radius, where its self-gravity still wins;
for exponential disks that radius is `R_strip = (h/2) ln(2 pi G Sigma0_s Sigma0_g / rho v^2)`.
Because the ram pressure scales as `rho v^2` and clusters are both dense and dynamically hot
(v ~ 1000-2000 km/s), an infalling Milky-Way-like spiral is stripped down to a few kpc on a
single pass -- its star formation quenches from the outside in, helping turn field spirals
into the gas-poor S0 and elliptical galaxies that crowd cluster cores. The tests verify the
`rho v^2` scaling, a finite partial-strip radius in a cluster, inner gas surviving while the
outskirts go, deeper stripping in harsher environments, and full stripping (R=0) in extreme
conditions.

## Free-fall: the universal clock of gravity

Take away a body's pressure support and it collapses in the free-fall time
`t_ff = sqrt(3 pi / 32 G rho)`, which depends only on mean density -- not size or mass.
`free_fall.py`:

```
$ python examples/free_fall_demo.py examples/output

                  system   rho (kg/m^3)              t_ff
  ---------------------------------------------------------
   giant molecular cloud       3.85e-19          3.39 Myr
        dense cloud core       3.85e-17         339.4 kyr
       protostellar core       3.85e-13           3.4 kyr
          the Sun (mean)       1.41e+03          29.5 min
               the Earth       5.51e+03          14.9 min
             white dwarf       1.00e+09             2.1 s
            neutron star       5.00e+17           0.09 ms
```

Because size and mass drop out, a galaxy and a raindrop of equal density collapse in the
same time, and the dynamical time and surface-orbit period share the identical
`1/sqrt(G rho)` scaling -- which is why every low Earth orbit is ~90 minutes regardless of
altitude, a cloud core forms stars in a few hundred kyr, and a neutron star's dynamical time
is under a millisecond. The Sun would free-fall in ~30 minutes if fusion switched off; that
it shines for billions of years instead is the whole point of the Kelvin-Helmholtz and
main-sequence modules. The tests verify the ~30-minute solar free-fall, the `rho^(-1/2)`
scaling, size-independence, the cloud-core collapse time, and the ~84-minute surface orbit.

## Shock jumps: the Rankine-Hugoniot conditions

Move faster than the sound speed `c_s = sqrt(gamma P / rho)` and the gas cannot get out of
the way -- a shock forms, a razor-thin front across which everything jumps. `shock_jump.py`:

```
$ python examples/shock_jump_demo.py examples/output

     Mach   rho2/rho1       P2/P1       T2/T1   M2 (down)
  -------------------------------------------------------
        1       1.000         1.0         1.0       1.000
      1.5       1.714         2.6         1.5       0.716
        2       2.286         4.8         2.1       0.607
        3       3.000        11.0         3.7       0.522
        5       3.571        31.0         8.7       0.475
       10       3.883       124.8        32.1       0.454
       30       3.987      1124.8       282.1       0.448
      100       3.999     12499.8      3125.9       0.447
```

Mass, momentum and energy conservation fix every downstream quantity from the upstream Mach
number. Density compression saturates at `(gamma+1)/(gamma-1) = 4` for `gamma=5/3`, but the
pressure and temperature jumps climb as `M^2` without limit -- which is why a strong shock
heats gas to millions of kelvin (supernova remnants, atmospheric re-entry plasma) while
packing it only fourfold, and the downstream flow is always subsonic (`M2 < 1`), a one-way
valve. This is the same compression ratio the Fermi-acceleration module feeds on and the
jump that drives the Sedov blast wave. The tests verify the ~340 m/s air sound speed, the
Mach-1 identity, the compression ceiling of 4, the unbounded `M^2` pressure growth, and the
subsonic post-shock flow.

## The Stromgren sphere: the ionized bubble around a hot star

A hot, massive star floods its surroundings with photons above 13.6 eV, carving out a sphere
of ionized hydrogen (an HII region). Its size is set by balance: every ionizing photon the
star emits replaces one recombination inside the sphere. `stromgren.py`:

```
$ python examples/stromgren_demo.py examples/output

        star (Q, /s)   n (/cc)  R_s (pc)  M_ion (Msun)
  ----------------------------------------------------
           O5 (5e49)        10     25.00         16172
           O5 (5e49)       100      5.39          1617
           O5 (5e49)      1000      1.16           162
           O9 (5e48)       100      2.50           162
           B0 (1e48)       100      1.46            32
```

Setting the star's output `Q` equal to the enclosed recombination rate
`(4/3) pi R^3 n^2 alpha_B` gives `R_s = (3Q / 4 pi n^2 alpha_B)^(1/3)`. Because `R ~ Q^(1/3)`
and `R ~ n^(-2/3)`, an O star lights up a ~25 pc bubble in diffuse gas but only a fraction of
a parsec in a dense clump (a compact HII region). These are the pink emission-line nebulae --
Orion, the Rosette, the Eagle -- that trace where massive stars formed in the last few
million years, and the same physics sizes the ionized bubbles of cosmic reionization. The
tests verify the ~pc-scale O-star sphere, the smaller B-star sphere, the `Q^(1/3)` and
`n^(-2/3)` scalings, the recombination-output balance, and the ~1000-year recombination time.

## Two-body relaxation: collisional clusters vs collisionless galaxies

Every stellar flyby deflects a star a little; the accumulated random kicks change its
velocity by order itself in the relaxation time. `relaxation_time.py`:

```
$ python examples/relaxation_time_demo.py examples/output

                system         N    t_cross     t_relax           state
  ---------------------------------------------------------------------
          open cluster     1e+03   1956 kyr      35 Myr     collisional
      globular cluster     1e+05    978 kyr    1062 Myr     collisional
  nuclear star cluster     1e+07     49 kyr       4 Gyr     collisional
          dwarf galaxy     1e+08     33 Myr   22117 Gyr   collisionless
             Milky Way     1e+11     73 Myr   3e+06 t_H   collisionless
```

With `t_relax ~ (N / 8 ln N) t_cross`, the Coulomb logarithm counting the many weak distant
encounters, relaxation grows almost linearly with `N`. So a globular cluster relaxes in ~1
Gyr, mass-segregates, and slowly evaporates (a roughly constant ~1% of stars escape per
relaxation time, giving a dissolution time ~100 t_relax), while the Milky Way's relaxation
time is millions of Hubble times -- it is effectively collisionless, which is why galaxies
keep their spiral arms, tidal streams and cold disks intact for a Hubble time. The tests
verify the ~1 Gyr globular relaxation, the collisionless galaxy, the near-linear `N`
dependence, the large evaporation multiple, and faster relaxation for shorter crossing times.

## The Parker wind: why the corona cannot stay still

Parker showed a hot corona cannot sit in hydrostatic equilibrium -- an isothermal atmosphere
keeps a finite pressure at infinity, far above interstellar space, so it must expand.
`parker_wind.py`:

```
$ python examples/parker_wind_demo.py examples/output

    T (MK)  c_s (km/s)  r_c (R_sun)   v(1 AU) km/s  Mach(1 AU)
  ------------------------------------------------------------
       1.0       117.3         6.94            430        3.67
       1.5       143.7         4.62            559        3.89
       2.0       165.9         3.47            671        4.05
       3.0       203.2         2.31            864        4.25
```

The physically-correct solution passes smoothly through Mach 1 at the sonic critical radius
`r_c = GM/2c_s^2` (a few solar radii), staying subsonic inside and supersonic outside, and is
solved here from the Parker integral by bisection on the right branch. A hotter corona has a
larger sound speed but a smaller critical radius (`r_c ~ 1/c_s^2`), so its wind goes supersonic
sooner and reaches a higher terminal speed -- a few hundred km/s by 1 AU, the solar wind that
Mariner 2 confirmed over the static-corona camp. The tests verify the ~140 km/s coronal sound
speed, the few-solar-radii critical radius, exact Mach 1 at `r_c`, the subsonic/supersonic
branches, monotonic outward acceleration, the 1 AU speed, and the hotter-faster trend.

## The greenhouse effect: why planets beat their sunlight

An atmosphere transparent to sunlight but opaque in the infrared lets light in and traps the
outgoing heat, so the surface runs hotter than the equilibrium temperature. `greenhouse.py`:

```
$ python examples/greenhouse_demo.py examples/output

    planet  T_eq (K)  T_surf (K)   warming  tau needed
  ----------------------------------------------------
     Venus     226.8       737.0     510.2       147.3
     Earth     254.7       288.0      33.3         0.8
      Mars     209.9       210.0       0.1         0.0
     Titan      84.6        94.0       9.4         0.7
```

For a grey atmosphere of infrared optical depth tau the surface warms to
`T_surf = T_eq (1 + 3 tau/4)^(1/4)`. Earth's modest tau ~ 0.8 lifts its 255 K skin temperature
to a life-friendly 288 K -- a 33 K blanket that keeps the oceans liquid. Venus, wrapped in a
dense CO2 atmosphere of tau ~ 150, runs away from a 227 K equilibrium to a lead-melting 737 K,
while nearly airless Mars sits at its equilibrium temperature. The tests verify Earth's ~255 K
equilibrium and ~33 K greenhouse, the tau=0 airless limit, Venus's runaway optical depth, the
monotonic rise with tau, and the optical-depth inversion.

## Rossby number: why weather spins

On a rotating planet the Coriolis force deflects any moving parcel at rate
`f = 2 Omega sin(lat)`, and the Rossby number `Ro = U/(fL)` decides whether that matters.
`rossby.py`:

```
$ python examples/rossby_demo.py examples/output

                  flow  U (m/s)           L          Ro          regime
  ---------------------------------------------------------------------
         bathtub drain      0.2       0.1 m    1.94e+04    ageostrophic
               tornado    100.0     100.0 m     9.7e+03    ageostrophic
            sea breeze      5.0       20 km        2.42    ageostrophic
             hurricane     50.0      500 km        0.97    ageostrophic
    cyclone (synoptic)     10.0     1000 km       0.097     geostrophic
            ocean gyre      0.1     2000 km    0.000485     geostrophic
```

When `Ro << 1` the Coriolis and pressure-gradient forces balance (geostrophic), so the wind
blows along the isobars rather than across them -- which is why cyclones and ocean gyres are
persistent rotating vortices instead of simple radial flows, and why a 1 mb / 100 km gradient
drives a ~8 m/s geostrophic wind. When `Ro >> 1` rotation is negligible: tornadoes and bathtub
drains (Ro ~ 10^4) spin from local vorticity, not the Coriolis force -- the draining-sink story
is a myth. The tests verify the equator/pole Coriolis parameter, the ~1e-4 mid-latitude value,
the geostrophic cyclone vs ageostrophic tornado, the geostrophic-wind scale, the hemisphere sign
flip, and the ~17-hour inertial period.

## Rayleigh-Benard convection: when a heated layer churns

Heat a fluid from below and buoyancy fights viscosity and thermal diffusion. The Rayleigh
number measures the contest. `rayleigh_benard.py`:

```
$ python examples/rayleigh_benard_demo.py examples/output

                  system          Ra           state      Nu
  ----------------------------------------------------------
   lab cell (near onset)     3.6e+01      conducting         1
           mug of coffee     4.7e+07      convecting        30
          pot on a stove     3.6e+08      convecting        60
          Earth's mantle     2.2e+08      convecting        50
   solar convection zone     7.8e+30      convecting     2e+09
```

With `Ra = g alpha dT d^3 / (nu kappa)`, convection switches on abruptly above the critical
value `Ra_c ~ 1708` for rigid plates (`(27/4) pi^4 ~ 657.5` for stress-free ones) -- one of
the cleanest predictions in fluid dynamics, confirmed to under a percent. Below onset the
layer only conducts (`Nu = 1`); above it the heat enhancement climbs as
`Nu ~ (Ra/Ra_c)^(1/3)`. Astrophysical layers run at `Ra` of `10^20` and up, so they are
violently turbulent -- the granulation on the Sun and the mantle convection that drifts
continents both follow. The tests verify the free-free critical value, the supercritical
water pot, the critical temperature difference that gives `Ra = Ra_c`, the conductive
sub-onset limit, the 1/3 Nusselt law, and that deeper layers convect at a smaller `dT`.

## Terminal velocity: how fast things fall

A falling body speeds up until drag balances gravity, then coasts at terminal velocity. Which
drag law applies is set by the Reynolds number. `terminal_velocity.py`:

```
$ python examples/terminal_velocity_demo.py examples/output

              object        v_term          Re      regime
  --------------------------------------------------------
    fog droplet 10um    12.03 mm/s       0.016      Stokes
       drizzle 100um       2.1 m/s          29   quadratic
        raindrop 2mm       9.5 m/s     2.6e+03   quadratic
       hailstone 1cm      20.2 m/s     2.7e+04   quadratic
      steel ball 1cm      59.5 m/s     8.1e+04   quadratic
```

At low Reynolds number viscous Stokes drag `F = 6 pi mu r v` gives `v ~ r^2`, so a fog
droplet 200x smaller than a raindrop falls ~40000x slower and effectively floats. At high
Reynolds number quadratic drag `F = (1/2) C_d rho A v^2` gives `v ~ sqrt(r)`, so a raindrop
settles at ~9 m/s and a belly-down skydiver tops out near 50 m/s. The regime boundary is
`Re ~ 1`. The tests verify the skydiver speed, the raindrop speed and its quadratic regime,
the Stokes fog droplet, the `r^2` and `sqrt(r)` scalings, and that drag balances weight at
terminal velocity.

## Supernova-remnant phases: the four ages of a blast wave

A supernova dumps ~10^51 erg into the ISM, and the shell evolves through four distinct
phases as it sweeps up mass and radiates. `snr_phases.py`:

```
$ python examples/snr_phases_demo.py examples/output

      age (yr)    R (pc)    v (km/s)           phase
  --------------------------------------------------
           100       1.0       10000  free expansion
           300       3.1       10000  free expansion
          1000       5.0        1947    Sedov-Taylor
          5000       9.5         741    Sedov-Taylor
         20000      16.5         323    Sedov-Taylor
         50000      23.8         186        snowplow
        200000      41.4          81        snowplow
       1000000      78.9          31          merged
```

The blast coasts ballistically (`R ~ t`) until it sweeps up its own ejecta mass at a few
parsecs, then enters the long adiabatic Sedov-Taylor phase (`R ~ t^(2/5)`, covered in detail
by the `sedov` module). Once the shell cools it radiates efficiently and coasts on momentum
(the snowplow, `R ~ t^(2/7)`), finally merging into the ISM near 100 pc when the shock slows
to the ~10 km/s turbulent velocity after ~10^6 yr -- seeding the galaxy with the elements it
forged. The tests verify the few-pc sweep-up radius, the early end of free expansion, the
`t^(2/5)` Sedov scaling and deceleration, the four-phase ordering, and the ~100 pc merge.

## The magnetic mirror: trapping charged particles

A charged particle spiraling along a field line conserves its magnetic moment
`mu = m v_perp^2 / 2B`; drifting into stronger field forces `v_perp` up and `v_parallel`
down until it reflects. `magnetic_mirror.py`:

```
$ python examples/magnetic_mirror_demo.py examples/output

    mirror ratio R_m   loss cone (deg)   20 deg   60 deg
  ------------------------------------------------------
                   2              45.0     lost     trap
                   4              30.0     lost     trap
                  10              18.4     trap     trap
                  50               8.1     trap     trap
                1000               1.8     trap     trap
```

Trapping depends only on the equatorial pitch angle: a particle mirrors if
`sin^2(alpha) > B_min/B_max = 1/R_m`, and otherwise falls into the loss cone and escapes
through the throat. A larger mirror ratio gives a narrower loss cone and holds more
particles. This is exactly how Earth's dipole traps the Van Allen belts -- particles bounce
pole to pole, reflected where the field tightens -- and how mirror-machine fusion devices
try to confine a plasma; particles scattered into the loss cone rain into the atmosphere and
light the aurora. The tests verify moment conservation (`v_perp ~ sqrt(B)`), the 30-degree
loss cone at `R_m=4`, its shrinking with ratio, the trapped/escaping pitch angles, and the
`B_min/sin^2(alpha)` mirror point.

## Debye shielding: what makes a plasma a plasma

Drop a test charge into ionized gas and the electrons rearrange to screen it, cutting the
potential off beyond the Debye length. `debye.py`:

```
$ python examples/debye_demo.py examples/output

         environment    n (/m^3)    T (K)    lambda_D       N_D        f_p
  -------------------------------------------------------------------------
    solar wind (1 AU)      5e+06    1e+05       9.8 m   1.9e+10     20 kHz
           ionosphere      1e+12    1e+03     2.18 mm   4.4e+04    9.0 MHz
         solar corona      1e+15    2e+06     3.09 mm   1.2e+08  283.9 MHz
   lab (tokamak edge)      1e+18    1e+05     0.02 mm   4.4e+04    9.0 GHz
          fusion core      1e+20    1e+08     0.07 mm   1.4e+08   89.8 GHz
```

The screening length is `lambda_D = sqrt(eps0 kT / n e^2)`, and the gas behaves as a
collective plasma only when the system is larger than `lambda_D` and many particles sit
inside a Debye sphere (`N_D >> 1`). Disturb the electrons and they ring at the plasma
frequency `omega_p = sqrt(n e^2 / eps0 m_e)`; EM waves below it cannot propagate and are
reflected. That is why the ionosphere's ~9 MHz cutoff bounces AM radio (~1 MHz) around the
curve of the Earth while FM (~100 MHz) and TV pass straight through to space. The tests
verify the ~9 MHz ionospheric plasma frequency, the millimetre Debye length, the well-
populated Debye sphere, the `lambda_D ~ sqrt(T/n)` and `omega_p ~ sqrt(n)` scalings, and the
critical-density inversion.

## Spectral line broadening: why lines have width

An atomic line is never infinitely sharp; three mechanisms give it width, and their sizes
read out the gas. `line_broadening.py`:

```
$ python examples/line_broadening_demo.py examples/output

              environment    T (K)     Doppler     pressure    dominant
  -------------------------------------------------------------------
              HII region    10000     19.57G      0.000G     Doppler
       solar photosphere     6000     15.16G      0.159G     Doppler
               red giant     4000     12.38G      0.002G     Doppler
       white-dwarf atmos    10000     19.57G   1591.549G    pressure
          cool ISM cloud      100      1.96G      0.000G     Doppler
```

Thermal Doppler motion gives a Gaussian of width `(nu0/c) sqrt(2kT/m)` -- widening with
temperature and favouring light atoms, the standard plasma thermometer. The finite
excited-state lifetime gives an irreducible natural (Lorentzian) width `A/4pi`, and
collisions add a pressure (Lorentzian) width that grows with density -- so a dense
white-dwarf photosphere shows hugely broadened, Lorentzian-winged lines while a thin HII
region is purely Doppler, letting line shape diagnose surface gravity. The observed profile
is the Voigt convolution of the Gaussian core and Lorentzian wings. The tests verify the
~20 pm H-alpha Doppler width, the `sqrt(T)` and `1/sqrt(m)` scalings, the density-driven
switch from Doppler to pressure dominance, and the linear addition of Lorentzian widths.

## The curve of growth: reading abundances from line strength

How an absorption line's equivalent width `W` grows with the column density `N` of
absorbing atoms follows three distinct regimes. `curve_of_growth.py`:

```
$ python examples/curve_of_growth_demo.py examples/output

            tau0     W / dnu_D        regime
  ----------------------------------------
            0.01          0.01        linear
             0.1           0.1        linear
               1             1     saturated
              10         3.097     saturated
            1000         5.257     saturated
          100000         17.72        damped
           1e+09          1772        damped
```

Weak lines are optically thin and grow linearly (`W ~ N`). Once the core saturates the line
is already black, so `W` creeps up only as `sqrt(ln N)` -- the flat plateau where abundances
are hardest to pin down. At enormous columns the Lorentzian damping wings go optically thick
and growth revives as `W ~ sqrt(N)`. Matching a measured equivalent width to the appropriate
segment of this curve is exactly how stellar and interstellar abundances are read from
absorption spectra. The tests verify the linear `W ~ N` rise, the flat saturated plateau, the
`sqrt(N)` damping tail (100x column -> 10x W), the monotonic overall growth, the regime
ordering, and the linear-regime column inversion.

## Sackur-Tetrode: the absolute entropy of a gas

Quantum state-counting fixes the entropy constant that classical thermodynamics leaves free.
`sackur_tetrode.py`:

```
$ python examples/sackur_tetrode_demo.py examples/output

     gas  mass (amu)   S predicted   S measured    error
  ------------------------------------------------------
      He       4.003         126.0        126.2   -0.12%
      Ne      20.180         146.2        146.3   -0.06%
      Ar      39.948         154.7        154.8   -0.04%
      Kr      83.798         164.0        164.1   -0.08%
      Xe     131.290         169.6        169.7   -0.07%
```

Counting the microstates of `N` indistinguishable atoms with phase-space cells of size `h^3`
gives `S = N k_B [ln((V/N)(4 pi m U / 3 N h^2)^(3/2)) + 5/2]`. Planck's constant appears
explicitly -- without it the log's argument would be dimensional -- and the `1/N!` for
indistinguishable atoms makes entropy extensive and resolves the Gibbs paradox. Predicted
from nothing but atomic mass, temperature and pressure, it matches the calorimetric standard
molar entropies of the noble gases to under a fifth of a percent, a direct confirmation that
entropy is the logarithm of countable microstates. The tests verify argon's and helium's
measured molar entropies, the heavier-is-higher trend, the picometre thermal wavelength, the
classical (non-degenerate) STP gas, and the rise with temperature and volume.

## Maxwell-Boltzmann: how fast gas molecules move

Molecular speeds in a gas follow `f(v) = 4 pi (m/2 pi kT)^(3/2) v^2 exp(-mv^2/2kT)`.
`maxwell_boltzmann.py`:

```
$ python examples/maxwell_boltzmann_demo.py examples/output

       gas  mass (amu)     v_p     <v>    v_rms    >3 v_p
  ---------------------------------------------------------
        H2           2     1579    1782     1934   4.4e-04
        He           4     1117    1260     1368   4.4e-04
        N2          28      422     476      517   4.4e-04
        O2          32      395     446      484   4.4e-04
       CO2          44      337     380      412   4.4e-04
```

Three characteristic speeds fall out and always keep the same ratio,
`v_p : <v> : v_rms = 1 : 1.128 : 1.225`, independent of gas or temperature. Because they all
scale as `1/sqrt(m)`, hydrogen zips along about four times faster than nitrogen at the same
temperature -- which is why light gases escape atmospheres and why sound (set by `~v_rms`)
travels faster in helium. The mean translational kinetic energy is `(3/2) k_B T` for any gas,
and the high-speed `exp(-v^2)` tail leaves only ~0.04% of molecules above `3 v_p` -- yet it is
exactly that thin tail that governs Jeans escape and lets nuclei overcome the Coulomb barrier
to fuse. The tests verify the universal speed ratios, the ~500 m/s nitrogen `v_rms`, the
`1/sqrt(m)` and `sqrt(T)` scalings, the unit normalization, and the smallness of the tail.

## The Gamow peak: where stars fuse

Nuclei must beat an ~MeV Coulomb barrier to fuse, yet the Sun's core is only ~1.3 keV.
`gamow.py`:

```
$ python examples/gamow_demo.py examples/output

      reaction  Z1 Z2     T (K)   E_G (keV)    peak E0
  ----------------------------------------------------
         p + p      1   1.5e+07         493       5.9k
       p + N14      7   1.5e+07       45105      26.6k
       He + He      4   1.0e+08       31560      83.7k
         C + C     36   5.0e+08     7669118    1526.8k
```

Two factors save fusion, pulling opposite ways in energy: the Maxwell-Boltzmann tail
`exp(-E/kT)` supplies fewer particles as `E` rises, while quantum tunnelling
`exp(-sqrt(E_G/E))` becomes far more likely. Their product peaks sharply at the Gamow energy
`E0 = (E_G (kT)^2/4)^(1/3)`, far out on the thermal tail but well below the barrier -- for
solar p-p fusion, ~6 keV, several times the mean 1.3 keV. Because the Gamow energy scales as
`(Z1 Z2)^2`, higher-charge reactions need dramatically hotter cores (carbon burning at
~5x10^8 K versus hydrogen at 1.5x10^7 K), which is the thermostat that orders the stages of
stellar nucleosynthesis. The tests verify the ~6 keV solar p-p peak, that it sits well above
`kT`, that the reaction integrand is maximal at `E0`, the `T^(2/3)` and `(Z1 Z2)^2` scalings,
and that heavier nuclei need higher temperatures.

## Parallax & proper motion: the geometry of stellar distance

As Earth orbits the Sun a nearby star shifts against the background by the parallax angle p,
and the parsec is defined so the trigonometry is trivial. `parallax.py`:

```
$ python examples/parallax_demo.py examples/output

            star     p (")   d (pc)   d (ly)   mu ("/yr)   v_space
  ----------------------------------------------------------------
       Proxima Cen   0.7687     1.30      4.2       3.85     32.6k
    Barnard's Star   0.5469     1.83      6.0      10.36    142.0k
            Sirius   0.3792     2.64      8.6       1.34     17.6k
              Vega   0.1305     7.66     25.0       0.35     18.8k
        Betelgeuse   0.0055   181.82    593.0       0.03     33.9k
```

Distance follows directly, `d (pc) = 1/p (arcsec)` -- the first rung of the cosmic distance
ladder, measured by Hipparcos and Gaia for over a billion stars. A star's proper motion mu
(the angular drift per year) converts to a tangential velocity `v_t = 4.74 mu d`, which
combines in quadrature with the Doppler radial velocity into the total space velocity.
Barnard's Star, with the largest known proper motion (10.4"/yr at 1.83 pc), races across the
sky at ~90 km/s tangential and 142 km/s through space. The tests verify the 1 pc = 1"
definition, Proxima's 1.30 pc / 4.24 ly, the inverse parallax-distance relation, Barnard's
~90 km/s tangential velocity, and the quadrature space velocity.

## Standard candles: the cosmic distance ladder

Know an object's true luminosity and its apparent brightness gives its distance -- it is a
standard candle. `standard_candle.py`:

```
$ python examples/standard_candle_demo.py examples/output

                object      distance   modulus
  ----------------------------------------------
           10 pc (M = m)         10 pc      0.00
          Hyades cluster         47 pc      3.36
         Galactic centre       8200 pc     14.57
                     LMC        50 kpc     18.49
         Andromeda (M31)       778 kpc     24.45
           Virgo cluster      16.5 Mpc     31.09
           SN Ia horizon    1000.0 Mpc     40.00
```

The distance modulus `m - M = 5 log10(d/10 pc)` inverts to distance, and five magnitudes is
exactly a factor of 100 in flux. The trick is knowing `M`: Cepheid variables supply it
through Leavitt's period-luminosity law (`M_V ~ -2.81 log10 P - 1.43`, longer period =
brighter), and Type Ia supernovae (`M ~ -19.3`) extend the same logic to hundreds of Mpc and
revealed cosmic acceleration. Chaining parallax to Cepheids to supernovae is the distance
ladder. The tests verify the 10 pc zero point, the LMC's ~18.5 modulus, the modulus/distance
inversion, the 100-per-5-magnitudes rule, the Cepheid period-luminosity trend, and a Cepheid
distance round-trip.

## Tully-Fisher: a spiral's brightness from its spin

Spiral galaxies obey a tight scaling between luminosity and flat rotation speed.
`tully_fisher.py`:

```
$ python examples/tully_fisher_demo.py examples/output

       galaxy type   v_flat    L (Lsun)    M_abs    M_baryon
  ------------------------------------------------------------
        dwarf spiral       80    5.12e+08   -14.38    2.05e+09
        small spiral      120    2.59e+09   -16.05    1.04e+10
      Milky Way-like      220    2.93e+10   -18.55    1.17e+11
      massive spiral      300    1.01e+11   -19.83    4.05e+11
        giant spiral      400    3.20e+11   -21.02    1.28e+12
```

The relation is `L ~ v_flat^4`: because `v^2 = GM/R` and spirals hold roughly constant surface
brightness and mass-to-light ratio, mass, spin and light all rise together, so a doubling of
rotation speed brightens a spiral 16-fold. The baryonic version `M_baryon ~ v^4` is tighter
still and probes dark matter. Since the rotation width is easy to measure from the 21-cm line,
Tully-Fisher is a redshift-independent distance indicator that reaches far beyond resolvable
Cepheids -- the spiral-galaxy cousin of the Faber-Jackson relation for ellipticals. The tests
verify the `v^4` luminosity and baryonic-mass scalings, the Milky-Way luminosity and mass, the
faster-is-brighter trend, and the luminosity/rotation-speed inversion.

## The Tolman test: surface-brightness dimming

Surface brightness -- flux per unit solid angle -- is distance-independent in a static
Euclidean universe: flux and angular area fall together. `tolman.py`:

```
$ python examples/tolman_demo.py examples/output

       z     expanding     mag    tired-light   ratio E/T
  -----------------------------------------------------
     0.5           1/5    1.76         1/1.5      0.296
     1.0          1/16    3.01         1/2.0      0.125
     2.0          1/81    4.77         1/3.0      0.037
     3.0         1/256    6.02         1/4.0      0.016
     5.0        1/1296    7.78         1/6.0      0.005
```

Expansion breaks the static invariance with four factors of `(1+z)` -- photon redshift, time
dilation, and the `D_A/D_L` geometry -- so `SB ~ (1+z)^-4`: a z=1 galaxy is dimmed 16x per
square arcsecond, a z=3 galaxy 256x. A static "tired-light" universe would dim only as
`(1+z)^-1`, so the expanding prediction is `(1+z)^-3` fainter. Observations confirm the
`(1+z)^4` exponent, one of the most direct pieces of evidence that the cosmological redshift
is genuine expansion rather than photons losing energy en route. The tests verify the 1/16
dimming at z=1, the 1/256 at z=3, the ~3 mag magnitude form, the single tired-light factor,
the expanding-vs-tired ratio, and the exponent recovery from an observed ratio.

## Olbers' paradox: why the night sky is dark

In an infinite, eternal, static universe uniformly filled with stars, every line of sight
would end on a stellar surface and the whole sky would blaze as bright as the Sun.
`olbers.py`:

```
$ python examples/olbers_demo.py examples/output

          distance     d / mfp     sky covered
  --------------------------------------------
    cosmic horizon    6.76e-07       6.757e-07
      100x horizon    6.76e-05       6.757e-05
           0.1 mfp    1.00e-01         0.09516
             1 mfp    1.00e+00          0.6321
             5 mfp    5.00e+00          0.9933
            20 mfp    2.00e+01               1
```

Summing shells of stars, each contributing r-independent brightness, diverges; the finite
scale is the mean free path to a star `lambda = 1/(n sigma)`, and the sky-covering fraction
out to distance d is `1 - exp(-d/lambda)`. For realistic star densities `lambda ~ 10^16`
light-years, but the cosmic horizon `c x age ~ 1.4x10^10` ly is a million times closer, so
only ~`10^-6` of the sky is covered and night is dark. The paradox is real -- every sight
line *would* eventually hit a star -- but "eventually" lies far beyond the horizon: the
finite age of the universe, not infinite space, is the resolution. The tests verify the
astronomically large mean free path, the `1 - 1/e` covering at one mfp and its saturation,
the horizon scale, and the tiny sky fraction covered within the observable universe.

## Bi-elliptic transfer: when three burns beat two

The Hohmann two-burn transfer is cheapest for modest orbit changes, but for large radius
ratios a three-burn bi-elliptic transfer wins. `bi_elliptic.py`:

```
$ python examples/bi_elliptic_demo.py examples/output

     R = r2/r1   Hohmann dv   bi-elliptic       winner
  ---------------------------------------------------
          5.00        3622m        4482m      Hohmann
         10.00        3998m        4120m      Hohmann
         11.94        4030m        4050m      Hohmann
         13.00        4039m        4020m  bi-elliptic
         16.00        4046m        3953m  bi-elliptic
         30.00        3980m        3810m  bi-elliptic
         60.00        3836m        3735m  bi-elliptic
```

Flinging the craft far beyond the target and dropping back costs less total delta-v because
the mid-course burn happens where orbital speeds -- and so the cost of changing them -- are
tiny. Below the exact crossover `R = 11.94` the Hohmann transfer always wins; above
`R = 15.58` bi-elliptic always does; between them it depends on the detour radius. The
saving is paid for with a far longer, sometimes years-long, transfer, so bi-elliptic is
reserved for the most extreme orbit raises. The tests verify Hohmann winning below the
crossover, bi-elliptic winning above it, the intermediate band needing a large detour, the
11.94 crossover value, and agreement with the vis-viva Hohmann result.

## Gravity assist: the slingshot

A spacecraft flying past a planet follows a hyperbola in the planet's frame: it leaves at the
same speed but bent by the turn angle. `gravity_assist.py`:

```
$ python examples/gravity_assist_demo.py examples/output

    v_inf (km/s)   periapsis   turn (deg)   boost (km/s)
  ---------------------------------------------------
              5     2 R_jup       153.1          9.73
              5     5 R_jup       138.2          9.34
             10     2 R_jup       127.9         17.97
             10     5 R_jup       102.5         15.60
             15     2 R_jup       105.8         23.92
             15     5 R_jup        75.4         18.35
```

With `sin(delta/2) = 1/e` and `e = 1 + r_p v_inf^2 / mu`, a slower or deeper pass bends the
trajectory more. Energy is conserved in the planet's frame, but in the Sun's frame the planet
is moving, so rotating the excess-velocity vector adds up to `2 v_inf` of free heliocentric
speed -- a slow deep pass steals the most of the planet's orbital motion. Voyager 2 chained
Jupiter, Saturn, Uranus and Neptune this way to reach Solar-System-escape speed on a fraction
of the fuel a direct burn would need, while the planets lost a laughably tiny bit of orbital
energy. The tests verify that deeper and slower passes bend more, the Voyager-scale Jupiter
boost, that the gain never exceeds `2 v_inf`, the near-180-degree limit, and that a trailing
pass adds speed.

## Synodic periods: how often planets line up

The geometry we see -- oppositions, launch windows, new Moons -- repeats not on a planet's
sidereal period but on its synodic period, the beat between two orbital rates. `synodic.py`:

```
$ python examples/synodic_demo.py examples/output

      planet  sidereal (d)   synodic (d)   per year
  ------------------------------------------------
     Mercury          88.0         115.9     3.152
       Venus         224.7         583.9     0.626
        Mars         687.0         779.9     0.468
     Jupiter        4332.6         398.9     0.916
      Saturn       10759.2         378.1     0.966
     Neptune       60190.0         367.5     0.994
```

From `1/S = |1/P_planet - 1/P_earth|`, Mars returns to opposition every ~780 days -- exactly
the ~26-month cadence of Mars launch windows -- and the synodic month is 29.5 days, longer
than the 27.3-day sidereal month because the Earth-Moon system also circles the Sun. Fast
inner planets lap Earth often; distant planets barely move, so their synodic period settles
toward one Earth year (Earth does the lapping). Right at Earth's own orbit the synodic period
diverges: two bodies at the same distance never change their alignment. The tests verify the
Mars/Venus/Mercury synodic periods, the 29.5-day synodic month, the divergence at equal
periods, the one-year limit for distant planets, and the conjunction cadence.

## The black-hole shadow: what the EHT imaged

A black hole casts a dark disk larger than its horizon: light inside the critical impact
parameter is captured, and lensing magnifies the boundary. `black_hole_shadow.py`:

```
$ python examples/black_hole_shadow_demo.py examples/output

      object    mass (Msun)    distance   shadow (uas)
  --------------------------------------------------
        M87*      6.50e+09    16.8 Mpc          39.7
      Sgr A*      4.15e+06    8.15 kpc          52.2
  stellar BH      1.00e+01       3 kpc       3.4e-04
```

The unstable photon sphere sits at `1.5 r_s` and the shadow's apparent radius is
`b_crit = 3 sqrt(3) GM/c^2`, so the dark disk is `3 sqrt(3) ~ 5.196 r_s` across -- larger than
the `2 r_s` horizon because gravity bends the light around it. Plugging in M87* and Sgr A*
gives ~40 and ~52 microarcseconds, the sizes the Event Horizon Telescope measured by linking
radio dishes across the whole Earth (the shadow is the angular size of an orange on the Moon).
The tests verify the 1.5 r_s photon sphere, the 5.196 r_s shadow, the M87* and Sgr A* angular
sizes, the `b_crit` formula, and the M and 1/D scalings.

## The Hill sphere: how far a planet holds its moons

A moon is bound to its planet only inside the Hill sphere, where the planet's pull beats the
star's tide. `hill_sphere.py`:

```
$ python examples/hill_sphere_demo.py examples/output

      planet   a (AU)  mass (Me)   r_H (Mkm)  stable limit
  --------------------------------------------------------
     Mercury    0.387      0.055        0.22        0.11M
       Venus    0.723      0.815        1.01        0.51M
       Earth    1.000      1.000        1.50        0.75M
        Mars    1.524      0.107        1.08        0.54M
     Jupiter    5.203    317.800       53.13       26.57M
      Saturn    9.537     95.200       65.16       32.58M
     Neptune   30.070     17.100      115.93       57.96M
```

With `r_H = a (m/3M)^(1/3)`, a bigger orbit or heavier planet widens the domain, so Jupiter
commands ~53 million km while Mercury holds barely 0.22. Real moons survive out to only
~1/2 r_H prograde -- the Moon at 0.384 Mkm sits well inside Earth's 0.75 Mkm limit, and the
Moon's own ~60,000 km Hill sphere is why it has no sub-moons. The same balance sets the
feeding zone of a forming planet and the mutual Hill spacing (systems need roughly >10) that
keeps planetary orbits stable. The tests verify Earth's ~1.5 Mkm Hill radius, the bound Moon,
the `a` and `m^(1/3)` scalings, the Moon's own Hill sphere, and the Earth-Venus mutual spacing.

## J2 orbital precession: reading the equatorial bulge

A planet's oblateness (coefficient J2) makes satellite orbits precess in two ways.
`j2_precession.py`:

```
$ python examples/j2_precession_demo.py examples/output

                 orbit  incl (deg)  nodal (deg/d)  apsidal (deg/d)
  ------------------------------------------------------------------
            equatorial        0.00         -6.921           13.842
              ISS-like       51.60         -4.299            3.215
    critical (Molniya)       63.43         -3.095            0.000
       sun-synchronous       98.19          0.986           -3.110
                 polar       90.00         -0.000           -3.461
```

The line of nodes regresses at `-(3/2) J2 (R/p)^2 n cos i` and the apsides advance at
`(3/4) J2 (R/p)^2 n (5 cos^2 i - 1)`. Tuning the inclination so the nodal drift equals the
Sun's `0.9856 deg/day` gives a sun-synchronous orbit (~98 deg, retrograde) that crosses the
equator at the same local time each pass -- the workhorse of imaging and weather satellites.
Setting `5 cos^2 i - 1 = 0` (i = 63.4 deg) freezes the apsides, the Molniya orbit that parks
apogee over high latitudes for long dwell. The tests verify the ISS nodal drift, the ~98-deg
retrograde sun-synchronous inclination and its Sun-matching drift, the 63.43-deg critical
inclination with zero apsidal rate, and the vanishing nodal drift of a polar orbit.

## Solar sails: pushing spacecraft with sunlight

Light carries momentum: a mirror sail feels a pressure `2F/c`, ~9 uPa at 1 AU.
`solar_sail.py`:

```
$ python examples/solar_sail_demo.py examples/output

                sail   area (m^2)   mass (kg)   accel (mm/s^2)     beta
  -----------------------------------------------------------------
              IKAROS        196        315         0.00565 0.000953
         LightSail 2         32          5         0.05812   0.0098
           NEA Scout         86         14         0.05578   0.0094
       Starshot chip         16      0.001           145.3     24.5
       beta = 1 sail        653          1            5.93        1
```

Because sunlight and gravity both fall as `1/r^2`, the lightness number
`beta = radiation force / solar gravity` is a distance-independent property of the sail,
`beta = (1+Q) L_sun (A/m) / (4 pi c G M_sun)`. `beta = 1` -- an area-to-mass ratio of ~653
m^2/kg, a ~1.5 g/m^2 mirror -- exactly cancels the Sun's pull, and `beta > 1` escapes the
Solar System on sunlight alone (the Starshot lightsail chip reaches beta ~ 25). Real sails
like IKAROS and LightSail 2 sit at beta ~ 0.01: a gentle push, but propellant-free and
endless. The tests verify the ~1361 W/m^2 flux and ~9 uPa mirror pressure, the mirror-is-twice-black
force, the `1/r^2` falloff, the ~1.5 g/m^2 critical loading giving beta = 1, and the
area-to-mass scaling.

## Relativistic beaming: why one jet is bright

Radiation from a source moving near light speed is swept forward and Doppler-boosted.
`beaming.py`:

```
$ python examples/beaming_demo.py examples/output

    gamma   theta   Doppler D   flux boost      jet/cj   v_app/c
  ------------------------------------------------------------
        2      5d       3.64         119     1.6e+04       0.5
        5      5d       8.36    2.58e+03     1.2e+07       3.6
       10      5d      11.37    8.05e+03     5.2e+08       9.9
       10     20d       1.54        4.92     2.8e+05       5.2
```

Aberration sweeps the emission into a cone of half-angle `~1/gamma`, and the Doppler factor
`D = 1/(gamma(1 - beta cos theta))` boosts the observed flux by `D^(3+alpha)` (a discrete
blob) or `D^(2+alpha)` (a steady jet). An approaching jet is brightened hundreds of times
while its receding twin is dimmed by the same powers -- which is why M87's jet looks
one-sided, its counter-jet beamed away. The same geometry gives apparent superluminal motion
(`v_app` peaks near `gamma*beta`), an illusion of light-travel time, not a real
faster-than-light speed. The tests verify the `1/gamma` cone, the approaching boost and
receding dimming, the enormous jet/counter-jet ratio, superluminal motion, and the
`gamma*beta` peak.

## The relativistic rocket: the Galaxy in a lifetime

A ship at constant proper acceleration follows a hyperbolic worldline. `relativistic_rocket.py`:

```
$ python examples/relativistic_rocket_demo.py examples/output

           destination     distance   ship (yr)   Earth (yr)   v_peak/c
  ------------------------------------------------------------------
      Proxima Centauri      4.37 ly        2.3         5.25    0.9834
                  Vega     25.00 ly        3.9           26    0.9993
       Galactic centre       27 kly       10.6      2.7e+04    1.0000
       Andromeda (M31)      2.5 Mly       15.0      2.5e+06    1.0000
    edge of observable   4.6e+04 Mly       24.5      4.6e+10    1.0000
```

Velocity `c tanh(a tau/c)` saturates just short of `c`, but distance and Earth time use
`cosh`/`sinh`, so proper time grows only logarithmically with distance -- a 1-g ship reaches
the galactic centre in ~10 crew-years (27,000 pass on Earth), Andromeda in ~15, and the edge
of the observable universe in ~25. Rapidity `phi = a tau/c` is the additive velocity whose
`tanh` is `v/c`. The impossible part is fuel: an ideal photon drive needs `exp(2 phi)` times
the payload mass to accelerate and stop. The tests verify the ~0.77c after one year at 1 g,
that velocity never reaches c, the distance/proper-time inversion, the ~10-crew-year galactic
centre with ~27,000 Earth-years, and the exponential mass ratio.

## Relativistic Doppler: colour shifts of fast sources

A moving light source shifts in frequency by the classical Doppler effect times time
dilation. `relativistic_doppler.py`:

```
$ python examples/relativistic_doppler_demo.py examples/output

     beta   receding z   approaching z   transverse z
  --------------------------------------------------
     0.10       0.1055        -0.0955         0.0050
     0.50       0.7321        -0.4226         0.1547
     0.90       3.3589        -0.7706         1.2942
     0.99      13.1067        -0.9291         6.0888
```

Radial motion gives `f_obs/f_src = sqrt((1-beta)/(1+beta))` receding (redshift) and its
reciprocal approaching (blueshift). The purely relativistic surprise is the transverse shift:
a source moving exactly across the line of sight has no classical Doppler component, yet its
light still reddens by `1/gamma` because its clock runs slow -- the effect Ives and Stilwell
measured in 1938, direct proof of time dilation. The general angle gives the same Doppler
factor `1/(gamma(1 - beta cos theta))` that beams jets, and a measured redshift inverts to a
speed, `beta = ((1+z)^2-1)/((1+z)^2+1)`. The tests verify the receding redshift and
approaching blueshift, the frequency/wavelength reciprocity, the `1/gamma` transverse
time-dilation shift, the reduction of the general angle to the radial and transverse cases,
and the redshift-velocity round-trip.

## de Broglie: matter as waves

Every particle has a wavelength inversely proportional to its momentum. `de_broglie.py`:

```
$ python examples/de_broglie_demo.py examples/output

                    object          lambda
  --------------------------------------------
     100 keV microscope e-         3.88 pm
             1 eV electron         1.23 nm
   thermal neutron (300 K)          100 pm
   thermal He atom (300 K)         50.2 pm
       100 m/s N2 molecule          141 pm
         baseball (40 m/s)      1.14e-34 m
```

`lambda = h/p`, so heavy or fast objects have vanishingly short waves -- a baseball's
`10^-34 m` is undetectable. But a 100 keV electron's ~4 pm is thousands of times finer than
visible light, which is why electron microscopes resolve atoms, and a thermal neutron's
~0.1 nm matches crystal spacing, making neutron diffraction a structural probe. Non-relativistically
`lambda = h/sqrt(2 m E)`, and the thermal de Broglie wavelength `h/sqrt(2 pi m k_B T)` marks
where quantum statistics take over -- the same condition behind degeneracy pressure and
Bose-Einstein condensation. The tests verify the 100 keV electron and thermal-neutron
wavelengths, the negligible baseball, the `1/p` and `1/sqrt(E)` scalings, the energy/momentum
form agreement, and the momentum-wavelength inversion.

## The Bohr model: the hydrogen spectrum

Quantizing angular momentum (`L = n hbar`) forces the electron onto discrete orbits.
`bohr.py`:

```
$ python examples/bohr_demo.py examples/output

  spectral series (to lower level n1):
          Lyman (UV) to n=1:  121.5, 102.5, 97.2 nm ...
    Balmer (visible) to n=2:  656.1, 486.0, 433.9 nm ...
        Paschen (IR) to n=3:  1874.6, 1281.5, 1093.5 nm ...
```

Balancing Coulomb attraction against the quantized orbit fixes `E_n = -13.6/n^2 eV`,
`r_n = n^2 a_0` (`a_0 ~ 52.9 pm`), and the Rydberg formula `1/lambda = R_H(1/n1^2 - 1/n2^2)`.
The ground state at -13.6 eV is the ionization energy; transitions emit fixed-energy photons,
the sharp hydrogen lines -- Lyman in the UV, Balmer in the visible (H-alpha at 656 nm, the red
of nebulae), Paschen in the infrared. The n=1 orbital speed is `alpha*c`, giving the
fine-structure constant `v/c ~ 1/137`. Though superseded by full quantum mechanics, the model
gets the hydrogen energies exactly right. The tests verify the -13.6 eV ground state, the
52.9 pm Bohr radius, the 656.3 nm H-alpha and 121.6 nm Lyman-alpha lines, the `-1/n^2` and
`n^2` scalings, and the 1/137 fine-structure constant.

## The Sunyaev-Zeldovich effect: clusters shadowing the CMB

The same hot cluster gas that glows in X-rays also inverse-Compton scatters
passing CMB photons, imprinting a tiny spectral distortion. `sz.py`:

```
$ python examples/sz_demo.py examples/output

  cluster        n_e (/m^3)  kT (keV)         y     dT (uK)
  Coma-like           1e+03       8.0  3.21e-05      -175.1
  massive             3e+03      12.0  2.89e-04     -1576.3
```

The Compton `y = integral (k_B T_e/m_e c^2) sigma_T n_e dl` is the line-of-sight
electron pressure; in the Rayleigh-Jeans band the cluster is a cold spot,
`dT/T = -2y`, a few hundred microkelvin. Crucially the SZ signal is a fractional
distortion of the CMB, so it is **redshift-independent** -- it does not dim with
distance, which is why SZ surveys find clusters clear across the universe. The
tests verify the y-parameter scale, its linearity in density/temperature/path,
the RJ decrement, and the microkelvin signal.

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

## Gravitational focusing: how planets grow fast

Colliding bodies don't need a bullseye -- gravity curves distant trajectories
into a hit. `focusing.py`:

```
$ python examples/focusing_demo.py examples/output   (100 km planetesimal, v_esc=130 m/s)

   v_inf (m/s)    focusing    Safronov      regime
             1     16775.3     8387.17     runaway
            50         7.7        3.35     runaway
          1000         1.0        0.01   geometric
```

The cross-section is `pi R^2 (1 + v_esc^2/v_inf^2)`. In a dynamically cold swarm
(`v_inf << v_esc`) the enhancement is enormous, so the biggest bodies sweep up
mass fastest -- runaway growth that builds planetary embryos. Stir the swarm up
and only direct hits count, reverting to the geometric `pi R^2`. The boundary is
the Safronov number `Theta = v_esc^2/(2 v_inf^2) = 1`. The tests verify the
geometric limit, the runaway regime, the `Theta = 1` boundary, and the
`1/v_inf^2` scaling.

## The habitable zone: where water can be liquid

`habitable_zone.py` turns a star's luminosity into the band of orbits where a
planet's equilibrium temperature allows surface water:

```
$ python examples/habitable_zone_demo.py examples/output

  Earth's equilibrium temperature: 255 K   (greenhouse -> 288 K surface)
  star              L (L_sun)   HZ inner   HZ outer
  Sun (1.0)              1.00     0.47 AU    0.87 AU
  A star (2.0)         11.31     1.57 AU    2.93 AU
```

The equilibrium temperature `T_eq ~ L^{1/4} / sqrt(d)` gives Earth 255 K (the
greenhouse effect warms the surface the rest of the way to 288 K). Setting `T_eq`
to the liquid-water bounds gives the habitable zone, whose distance scales as
`sqrt(L_star)` -- tucked in close for a dim red dwarf, far out for a luminous
star. The tests verify Earth's 255 K, the `L^{1/4}/sqrt(d)` scalings, and the
`sqrt(L)` march of the zone.

## Exoplanet detection: transits and radial velocity

The two workhorse methods, both simple geometry plus Kepler. `exoplanet.py`:

```
$ python examples/exoplanet_demo.py examples/output

  planet          a (AU)    depth   RV K (m/s)   period
  hot Jupiter      0.050  1.01e-02      127.07    4.1 d
  Jupiter          5.204  1.01e-02       12.46   11.9 yr
  Earth            1.000  8.39e-05        0.09  365.2 d
```

A transit dims the star by `(R_p/R_star)^2` -- ~1% for Jupiter, 0.008% for Earth
-- and the star wobbles at a radial-velocity semi-amplitude `K` (12 m/s for
Jupiter, 9 cm/s for Earth). Hot Jupiters, being big, close, and fast, give the
strongest signals in both channels, which is why they were the first exoplanets
found. The tests reproduce the Jupiter/Earth transit depths, the RV amplitudes,
and Jupiter's 11.9-year period.

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
