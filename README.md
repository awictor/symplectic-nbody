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
| `src/photoelectric.py` | The photoelectric effect: light quantized into photons |
| `src/uncertainty.py` | The Heisenberg uncertainty principle & zero-point energy |
| `src/tunneling.py` | Quantum tunneling: barrier transmission, WKB & the STM |
| `src/particle_box.py` | The particle in a box: quantized levels & quantum-dot colour |
| `src/harmonic_oscillator.py` | The quantum harmonic oscillator: evenly-spaced levels & zero-point energy |
| `src/rutherford.py` | Rutherford scattering: the Coulomb cross section that found the nucleus |
| `src/radioactive_decay.py` | Radioactive decay: half-lives, dating & Bateman decay chains |
| `src/mass_formula.py` | Semi-empirical mass formula: nuclear binding & the iron peak |
| `src/q_value.py` | Nuclear Q-value: the energy released when nuclei rearrange |
| `src/quantum_stats.py` | Quantum statistics: Fermi-Dirac, Bose-Einstein & the classical limit |
| `src/debye_heat.py` | Debye specific heat: the T^3 law & the Dulong-Petit plateau |
| `src/carnot.py` | The Carnot cycle: the efficiency limit & heat-pump COP |
| `src/adiabatic.py` | Adiabatic processes: PV^gamma, compression heating & the speed of sound |
| `src/van_der_waals.py` | The van der Waals gas: real-gas EOS, critical point & corresponding states |
| `src/joule_thomson.py` | The Joule-Thomson effect: throttling cooling & gas liquefaction |
| `src/clausius_clapeyron.py` | Clausius-Clapeyron: vapor pressure, boiling point & latent heat |
| `src/reynolds.py` | The Reynolds number: laminar vs turbulent flow & Hagen-Poiseuille |
| `src/bernoulli.py` | Bernoulli's principle: the Venturi effect, Pitot airspeed & Torricelli |
| `src/surface_tension.py` | Surface tension: capillary rise (Jurin), Young-Laplace droplet/bubble pressure |
| `src/ekman.py` | The Ekman spiral: wind-driven rotating boundary layer, transport & depth |
| `src/milankovitch.py` | Milankovitch cycles: daily insolation, obliquity/eccentricity/precession forcing |
| `src/equipartition.py` | Equipartition: (1/2)kT per DOF, gas C_V/C_P/gamma, the H2 heat-capacity staircase |
| `src/osmosis.py` | Osmotic pressure: van't Hoff Pi=icRT, tonicity, osmometry, reverse osmosis |
| `src/diffusion.py` | Fick's laws: Gaussian/erfc profiles, sqrt(t) spread, diffusion length, Stokes-Einstein |
| `src/peclet.py` | Peclet number Pe=UL/D + Prandtl/Schmidt/Lewis: advection vs diffusion |
| `src/convection.py` | Convective heat transfer: Newton cooling, Nusselt correlations, Biot, lumped cooling |
| `src/stefan.py` | The Stefan problem: melting/freezing front X=2 lambda sqrt(alpha t), latent heat |
| `src/capillary.py` | Capillary length, Bond & Weber numbers: surface tension vs gravity vs inertia |
| `src/froude.py` | Froude number: flow regime, hull speed, hydraulic jump, Kelvin wake |
| `src/mach_cone.py` | Mach cone: cone angle, sonic-boom timing, Prandtl-Glauert & Prandtl-Meyer |
| `src/nozzle.py` | de Laval nozzle: isentropic ratios, area-Mach, choked flow, exhaust velocity |
| `src/blasius.py` | Blasius boundary layer: delta~sqrt(x), skin friction, plate drag, transition |
| `src/strouhal.py` | Strouhal number: vortex-shedding frequency, Roshko fit, aeolian tone, lock-in |
| `src/cluster_mass.py` | Virial cluster mass from velocity dispersion, M/L ratio, dark-matter fraction |
| `src/sersic.py` | Sersic surface-brightness profile: b_n, total luminosity, enclosed light |
| `src/grashof.py` | Grashof number: natural convection, Rayleigh Nu correlations, buoyancy vs forced |
| `src/womersley.py` | Womersley number: pulsatile flow, penetration depth, phase lag, pulse-wave speed |
| `src/marangoni.py` | Marangoni effect: surface-tension-gradient flow, onset, dynamic Bond number |
| `src/kutta_joukowski.py` | Kutta-Joukowski lift: circulation, 2 pi lift-slope, Magnus force, induced drag |
| `src/knudsen.py` | Knudsen number: mean free path, flow regimes, continuum-to-free-molecular |
| `src/richardson.py` | Richardson number: stratified-shear stability, Kelvin-Helmholtz onset |
| `src/kolmogorov.py` | Kolmogorov cascade: -5/3 spectrum, dissipation microscales, Re^(3/4) range |
| `src/casimir.py` | Casimir effect: vacuum pressure/force/energy between plates, d^-4 law |
| `src/hall_effect.py` | Hall effect: Hall voltage, coefficient, carrier density/sign/mobility |
| `src/wiedemann_franz.py` | Wiedemann-Franz law: Lorenz number, thermal-from-electrical conductivity |
| `src/bragg.py` | Bragg diffraction: n lambda = 2 d sin(theta), Miller spacings, max order |
| `src/diffraction_limit.py` | Diffraction limit: Rayleigh resolution, Abbe limit, grating resolving power |
| `src/snell.py` | Snell's law: refraction, critical angle/TIR, Brewster, fibre numerical aperture |
| `src/thin_film.py` | Thin-film interference: bubble colours, AR coatings, Newton's rings |
| `src/malus.py` | Malus's law: polarizer transmission, three-polarizer trick, wave plates |
| `src/cherenkov.py` | Cherenkov radiation: threshold, cone angle, velocity from the ring |
| `src/zeeman.py` | Zeeman effect: normal/anomalous line splitting, Lande g-factor, field readout |
| `src/rabi.py` | Rabi oscillations: two-level flopping, generalized Rabi, pi/pi-2 pulses |
| `src/franck_hertz.py` | Franck-Hertz: quantized excitation dips, emission wavelength |
| `src/moseley.py` | Moseley's law: K-alpha X-ray energy vs Z, elemental identification |
| `src/stark.py` | Stark effect: linear/quadratic line shifts, field ionization of Rydberg atoms |
| `src/aharonov_bohm.py` | Aharonov-Bohm phase, flux quantum, SQUID field sensitivity |
| `src/josephson.py` | Josephson junction: DC/AC supercurrent, Shapiro steps, volt standard |
| `src/quantum_hall.py` | Quantum Hall effect: von Klitzing constant, plateaus, Landau levels |
| `src/bcs.py` | BCS superconductivity: gap-to-Tc ratio, gap(T), Tc from coupling, isotope effect |
| `src/london.py` | London/Meissner: penetration depth, field expulsion, type I/II classification |
| `src/ising_mft.py` | Mean-field Ising: Curie temperature, spontaneous magnetization, Curie-Weiss |
| `src/percolation.py` | Site percolation: union-find clusters, spanning test, threshold sweep |
| `src/polya.py` | Polya random walk: return probability by dimension, recurrence, simulation |
| `src/langevin_para.py` | Langevin paramagnetism: L(x), Curie-law susceptibility, saturation |
| `src/buffon.py` | Buffon's needle: crossing probability, Monte Carlo pi, 1/sqrt(N) convergence |
| `src/metropolis.py` | Metropolis MCMC on the 2D Ising model: acceptance rule, Onsager T_c |
| `src/logistic_map.py` | Logistic map: period doubling, attractor, Lyapunov exponent, Feigenbaum |
| `src/henon.py` | Henon map: strange attractor, area contraction, fixed points, Lyapunov |
| `src/lorenz.py` | Lorenz attractor: RK4 flow, volume contraction, fixed points, Lyapunov |
| `src/double_pendulum.py` | Double pendulum: RK4 equations of motion, energy, chaotic divergence |
| `src/mandelbrot.py` | Mandelbrot set: escape time, membership, cardioid/bulb tests |
| `src/van_der_pol.py` | Van der Pol oscillator: limit cycle, amplitude, relaxation period |
| `src/duffing.py` | Duffing oscillator: double-well potential, regimes, resonance backbone |
| `src/kuramoto.py` | Kuramoto model: order parameter, critical coupling, sync transition |
| `src/sandpile.py` | Abelian sandpile: toppling dynamics, avalanches, self-organized criticality |
| `src/cellular_automaton.py` | Elementary CA: rule table, evolution, Wolfram rules 30/90/110 |
| `src/game_of_life.py` | Conway's Game of Life: B3/S23 update, still lifes, blinker, glider |
| `src/reaction_diffusion.py` | Gray-Scott reaction-diffusion: Turing spots/stripes from a seed |
| `src/boids.py` | Boids flocking: separation/alignment/cohesion, polarization order parameter |
| `src/dla.py` | Diffusion-limited aggregation: fractal growth, mass-radius dimension D~1.71 |
| `src/benford.py` | Benford's law: log10(1+1/d) leading digits, chi-square goodness-of-fit |
| `src/coupon_collector.py` | Coupon collector: E[T]=n H_n, variance, completion CDF, Monte-Carlo |
| `src/secretary.py` | Secretary problem: 1/e optimal-stopping rule, win probability, Monte-Carlo |
| `src/birthday.py` | Birthday problem: collision probability, sqrt(d) law, birthday-attack cost |
| `src/gamblers_ruin.py` | Gambler's ruin: ruin probability & duration, fair/biased, infinite house |
| `src/parrondo.py` | Parrondo's paradox: two losing games win when mixed, Markov-chain drift |
| `src/galton.py` | Galton board: binomial slot law, CLT Gaussian limit, 1/sqrt(n) convergence |
| `src/monty_hall.py` | Monty Hall: stay 1/N vs switch (N-1)/N, informed-vs-random host, Monte-Carlo |
| `src/bayes_test.py` | Bayes & base-rate fallacy: PPV/NPV, likelihood ratios, retest odds, Monte-Carlo |
| `src/shannon.py` | Shannon entropy & Huffman coding: H = -sum p log p, optimal prefix code, H<=L<H+1 |
| `src/kelly.py` | Kelly criterion: optimal bet fraction f*=p-q/b, log-growth rate, fractional Kelly |
| `src/hamming.py` | Hamming codes: (7,4) SEC + SECDED, syndrome decoding, exhaustively verified |
| `src/rsa.py` | RSA: Miller-Rabin, extended Euclid, keygen, encrypt/decrypt/sign/verify |
| `src/diffie_hellman.py` | Diffie-Hellman: safe primes, generators, key exchange, BSGS discrete log |
| `src/crc.py` | CRC-8/16/32: GF(2) polynomial division, frame check, matches zlib.crc32 |
| `src/lz77.py` | LZ77: sliding-window dictionary coding, lossless round-trip, compression ratio |
| `src/bloom.py` | Bloom filter: probabilistic membership, no false negatives, optimal m/k |
| `src/hyperloglog.py` | HyperLogLog: distinct-count in fixed memory, error ~1.04/sqrt(m), mergeable |
| `src/fenwick.py` | Fenwick tree: O(log n) prefix sums & point updates, cumulative select |
| `src/union_find.py` | Union-Find: path compression + union by rank, components, Kruskal MST |
| `src/dijkstra.py` | Dijkstra shortest paths: from-scratch min-heap, path reconstruction, Bellman-Ford check |
| `src/kdtree.py` | k-d tree: nearest / k-nearest / radius search, brute-force verified in 2D & 3D |
| `src/boyer_moore.py` | Boyer-Moore string search: bad-character + good-suffix skips, sublinear |
| `src/astar.py` | A* pathfinding: f=g+h heuristic search, grid heuristics, Dijkstra-verified optimal |
| `src/toposort.py` | Topological sort: Kahn + DFS, cycle detection, critical-path scheduling |
| `src/levenshtein.py` | Edit distance: DP table, alignment backtrace, similarity, Damerau variant |
| `src/knapsack.py` | 0/1 knapsack DP: optimal value + item reconstruction, subset-sum, unbounded |
| `src/lcs.py` | Longest common subsequence: DP + backtrace, diff edit-script, indel distance |
| `src/quickselect.py` | Quickselect + median-of-medians: O(n) k-th smallest, median, percentile |
| `src/aho_corasick.py` | Aho-Corasick: trie + failure links, all patterns in one pass |
| `src/floyd_warshall.py` | Floyd-Warshall all-pairs shortest paths: negative edges, cycle detection, closure |
| `src/misra_gries.py` | Misra-Gries frequent items: heavy hitters over n/k, majority vote, k-1 counters |
| `src/reservoir.py` | Reservoir sampling: uniform k-sample in one pass, weighted variant, streaming |
| `src/count_min.py` | Count-Min sketch: frequency estimates in sublinear memory, never underestimates |
| `src/alias_method.py` | Alias method: O(1) weighted sampling after O(n) setup, Vose construction |
| `src/fisher_yates.py` | Fisher-Yates shuffle: unbiased permutation, k-sample, Sattolo cyclic variant |
| `src/box_muller.py` | Box-Muller: uniform->Gaussian transform, Marsaglia polar, moment-verified |
| `src/rejection_sampling.py` | Rejection sampling: sample any evaluable density, box + general proposal |
| `src/welford.py` | Welford online mean/variance: one stable pass, higher moments, mergeable |
| `src/kahan.py` | Kahan/Neumaier compensated summation: bounded error, pairwise sum, dot product |
| `src/horner.py` | Horner's method: O(n) polynomial eval, synthetic division, Newton roots |
| `src/rootfind.py` | Bracketing root-finders: bisection, secant, false position, Brent |
| `src/quadrature.py` | Numerical integration: trapezoid, Simpson, Romberg, adaptive, Gauss-Legendre |
| `src/spline.py` | Cubic spline interpolation: natural/clamped C^2, Thomas solve, vs Lagrange |
| `src/fft.py` | Fast Fourier Transform: radix-2 Cooley-Tukey, inverse, FFT convolution |
| `src/linsolve.py` | Gaussian elimination / LU: partial pivoting, solve, determinant, inverse |
| `src/qr.py` | QR decomposition: modified Gram-Schmidt, least squares, orthonormal Q |
| `src/eigen.py` | Power iteration eigenvalues: Rayleigh quotient, inverse iteration, deflation |
| `src/conjugate_gradient.py` | Conjugate gradient: iterative SPD solver, Jacobi preconditioning |
| `src/svd.py` | SVD & PCA: A=USV^T via eigen(A^TA), low-rank approx, principal components |
| `src/kmeans.py` | k-means clustering: Lloyd's algorithm, k-means++ init, silhouette |
| `src/regression.py` | Linear & logistic regression: QR + gradient descent, R^2, accuracy, L2 |
| `src/decision_tree.py` | CART decision-tree classifier: Gini/entropy splits, rules, feature importance |
| `src/random_forest.py` | Random forest: bagged CART trees, feature subsampling, out-of-bag score |
| `src/gmm.py` | Gaussian mixture by EM: soft clustering, log-sum-exp, AIC/BIC model selection |
| `src/hmm.py` | Hidden Markov model: forward, Viterbi decode, forward-backward, Baum-Welch EM |
| `src/kalman.py` | Kalman filter + RTS smoother: predict/update, Kalman gain, self-contained matrix ops |
| `src/pagerank.py` | PageRank: sparse power iteration, damping, dangling nodes, personalized teleport |
| `src/lu.py` | LU (partial pivot) & Cholesky: solve, determinant, inverse, positive-definite test |
| `src/gaussian_process.py` | Gaussian process regression: RBF kernel, posterior mean/variance, marginal likelihood |
| `src/bayes_opt.py` | Bayesian optimization: GP surrogate, Expected Improvement, beats random search |
| `src/dbscan.py` | DBSCAN density clustering: core/border/noise, arbitrary shapes, auto k, k-distance |
| `src/hierarchical.py` | Agglomerative clustering: single/complete/average/Ward linkage, dendrogram, tree cut |
| `src/naive_bayes.py` | Naive Bayes: Gaussian & multinomial, log-space, Laplace smoothing, class posteriors |
| `src/knn.py` | k-nearest-neighbours: classify & regress, uniform/distance weights, standardize, LOO CV |
| `src/gradient_boosting.py` | Gradient boosting: sequential regression trees, squared-error & log-loss, shrinkage |
| `src/spectral_clustering.py` | Spectral clustering: affinity graph, Laplacian eigenvectors, non-convex shapes |
| `src/particle_filter.py` | Particle filter: bootstrap SIR, systematic resampling, adaptive ESS, nonlinear tracking |
| `src/simulated_annealing.py` | Simulated annealing: Metropolis criterion, cooling schedules, TSP 2-opt solver |
| `src/genetic_algorithm.py` | Genetic algorithm: tournament selection, crossover, mutation, elitism, knapsack |
| `src/particle_swarm.py` | Particle swarm optimization: inertia/cognitive/social velocity, benchmark functions |
| `src/reed_solomon.py` | Reed-Solomon codes: GF(256), encode, syndrome/Berlekamp-Massey/Chien/Forney decode |
| `src/mutual_information.py` | Mutual information: joint/conditional entropy, KL divergence, normalized MI, info gain |
| `src/lru_cache.py` | LRU & LFU caches: O(1) get/put via hash map + linked list / frequency buckets |
| `src/trie.py` | Trie: O(len) insert/search/prefix, autocomplete, delete-with-pruning, suffix index |
| `src/sorting.py` | Comparison sorts: insertion/merge/quick/heap + binary heap, stability, comparison counts |
| `src/newton_nd.py` | Newton's method in n-D: Jacobian (analytic/finite-diff), damping, Broyden, LU step |
| `src/differential_evolution.py` | Differential evolution DE/rand/1/bin: difference-vector mutation, bound reflection |
| `src/nelder_mead.py` | Nelder-Mead simplex: derivative-free reflect/expand/contract/shrink, restarts |
| `src/hits.py` | HITS hubs & authorities: mutual-reinforcement power iteration, eigenvector of A'A / AA' |
| `src/mds.py` | Classical MDS: double-centering, eigen-embedding from distances, Procrustes align |
| `src/skiplist.py` | Skip list: probabilistic O(log n) ordered map, express lanes, range queries |
| `src/ant_colony.py` | Ant colony optimization: pheromone-trail TSP solver, evaporation, elitist deposit |
| `src/avl_tree.py` | AVL self-balancing BST: rotations, O(log n) ordered map, range queries |
| `src/segment_tree.py` | Segment tree + lazy propagation: O(log n) range sum/min/max query and range-add |
| `src/convex_hull.py` | Convex hull (Andrew's monotone chain): area, perimeter, point-in-hull, diameter |
| `src/closest_pair.py` | Closest pair of points: O(n log n) divide-and-conquer with the strip merge |
| `src/segment_intersection.py` | Segment intersection: orientation predicate, crossing point, simple-polygon test |
| `src/point_in_polygon.py` | Point-in-polygon: ray casting + winding number, signed area, centroid, boundary |
| `src/polygon_clip.py` | Sutherland-Hodgman polygon clipping against a convex window, area |
| `src/marching_squares.py` | Marching squares: iso-contour extraction from a scalar grid, 16-case + interpolation |
| `src/bresenham.py` | Bresenham rasterization: integer-only line, midpoint circle, filled disk |
| `src/flood_fill.py` | Flood fill: queue/stack/scanline, 4/8-connectivity, connected-component labeling |
| `src/bezier.py` | Bezier curves: de Casteljau, derivative, subdivision, degree elevation, arc length |
| `src/bwt.py` | Burrows-Wheeler transform + inverse, move-to-front, RLE, the bzip2-style pipeline |
| `src/arithmetic_coding.py` | Arithmetic coding: integer range coder, renormalization, beats Huffman on skew |
| `src/sequence_alignment.py` | Needleman-Wunsch & Smith-Waterman: global/local DP, traceback, scoring |
| `src/string_matching.py` | KMP prefix function, Z-algorithm, Manacher longest palindrome, linear-time |
| `src/suffix_array.py` | Suffix array (prefix doubling) + LCP (Kasai): search, longest repeated/common substring |
| `src/quadtree.py` | Point-region quadtree: rectangle/circle range queries, nearest neighbour |
| `src/max_flow.py` | Maximum flow (Edmonds-Karp), min-cut theorem, bipartite matching by reduction |
| `src/de_bruijn.py` | De Bruijn sequences B(k,n) via Eulerian circuits (Hierholzer); general Eulerian path finder |
| `src/rotating_calipers.py` | Rotating calipers: diameter, width, minimum-area bounding rectangle from the hull |
| `src/ternary_search_tree.py` | Ternary search tree: autocomplete, longest-prefix, and '.'-wildcard string search |
| `src/delaunay.py` | Delaunay triangulation (Bowyer-Watson, exact in-circle) and the dual Voronoi diagram |
| `src/simplex.py` | Two-phase simplex method for linear programs (Bland's rule, mixed constraints, duality) |
| `src/minhash.py` | MinHash Jaccard estimation + banded LSH near-duplicate detection (universal hashing) |
| `src/cma_es.py` | CMA-ES derivative-free optimizer with full covariance adaptation (Jacobi eigensolver) |
| `src/lbfgs.py` | L-BFGS limited-memory quasi-Newton optimizer (two-loop recursion, Wolfe line search) |
| `src/tsne.py` | t-SNE nonlinear dimensionality reduction (perplexity calibration, KL-gradient descent) |
| `src/wavelet_tree.py` | Wavelet tree: rank/select/quantile/range-count over a sequence in O(log sigma) |
| `src/fibonacci_heap.py` | Fibonacci heap (O(1) amortized decrease-key) + Dijkstra built on it |
| `src/treap.py` | Treap: randomized balanced BST with split/merge and order statistics (select/rank) |
| `src/splay_tree.py` | Splay tree: self-adjusting BST with the working-set property (hot keys near root) |
| `src/van_emde_boas.py` | Van Emde Boas tree: integer set with O(log log u) successor/predecessor |
| `src/sparse_table.py` | Sparse table (O(1) range min/max/gcd) + binary-lifting LCA with tree distance |
| `src/pollard_rho.py` | Pollard's rho / p-1 factorization + Miller-Rabin, totient, and divisor count |
| `src/perlin.py` | Perlin gradient noise (1-D/2-D) + fractal Brownian motion for procedural fields |
| `src/wave_function_collapse.py` | Tiled WFC: constraint-propagation procedural generation with contradiction restart |
| `src/hungarian.py` | Hungarian algorithm: optimal O(n^3) assignment (Kuhn-Munkres), min or max |
| `src/dtw.py` | Dynamic time warping: distance + warping path, Sakoe-Chiba band, multi-dimensional |
| `src/p2_quantile.py` | P-square streaming quantile estimation (p50/p95/p99) in O(1) memory |
| `src/tarjan_scc.py` | Tarjan's strongly connected components, condensation DAG, topological sort |
| `src/two_sat.py` | 2-SAT solver via the implication graph + SCCs (linear time, with assignment) |
| `src/rk45.py` | Dormand-Prince RK45 adaptive-step ODE solver (embedded error control, FSAL) |
| `src/cordic.py` | CORDIC: cos/sin/atan2/hypot/exp/ln/sqrt with only shifts and additions |
| `src/savitzky_golay.py` | Savitzky-Golay filter: peak-preserving smoothing and noisy-data differentiation |
| `src/lzw.py` | LZW adaptive dictionary compression/decompression (GIF-style, capped code width) |
| `src/convolutional_code.py` | Convolutional encoder + Viterbi maximum-likelihood decoder for noisy channels |
| `src/ear_clipping.py` | Ear-clipping polygon triangulation (concave polygons, n-2 triangles) |
| `src/mlp.py` | Multi-layer perceptron + backpropagation (sigmoid/tanh/ReLU, momentum SGD) |
| `src/mdp.py` | Markov decision process: value iteration, policy iteration, gridworld builder |
| `src/bandit.py` | Multi-armed bandit: epsilon-greedy, UCB1, Thompson sampling with regret tracking |
| `src/q_learning.py` | Model-free RL: tabular Q-learning and SARSA over a gridworld environment |
| `src/autodiff.py` | Reverse-mode automatic differentiation (a Value graph with backward, like autograd) |
| `src/shamir.py` | Shamir's (k,n) secret sharing over a prime field (split + Lagrange reconstruct) |
| `src/merkle.py` | Merkle hash tree with O(log n) inclusion proofs (SHA-256, domain-separated) |
| `src/tsp.py` | Traveling salesman: exact Held-Karp DP + nearest-neighbour and 2-opt heuristics |
| `src/poisson.py` | 2-D Poisson/Laplace by relaxation (Jacobi, Gauss-Seidel, SOR) with Dirichlet BCs |
| `src/manacher.py` | Manacher's O(n) longest palindromic substring + palindrome counting |
| `src/stoer_wagner.py` | Stoer-Wagner global minimum cut of a weighted undirected graph (O(V^3)) |
| `src/chebyshev.py` | Chebyshev polynomial approximation (Clenshaw eval, cures the Runge phenomenon) |
| `src/gibbs.py` | Gibbs sampling for multivariate Gaussians and generic conditionals |
| `src/louvain.py` | Louvain community detection by modularity optimization (weighted graphs) |
| `src/matrix_chain.py` | Matrix-chain optimal parenthesization by interval DP (O(n^3)) |
| `src/lis.py` | Longest increasing subsequence via patience sorting (O(n log n), with witness) |
| `src/continued_fraction.py` | Continued-fraction expansion, convergents, and best rational approximation |
| `src/crt.py` | Chinese Remainder Theorem (coprime + general) with extended Euclid and mod inverse |
| `src/tonelli_shanks.py` | Modular square root (Tonelli-Shanks) + Legendre symbol / residue test |
| `src/discrete_log.py` | Baby-step giant-step discrete logarithm (O(sqrt n)) + multiplicative order |
| `src/welzl.py` | Welzl's smallest enclosing circle (expected O(n), iterative move-to-front) |
| `src/cuckoo_filter.py` | Cuckoo filter: approximate membership with deletion (cuckoo hashing) |
| `src/dsu_rollback.py` | Rollback disjoint-set union (snapshot/undo) for offline dynamic connectivity |
| `src/interval_scheduling.py` | Weighted interval scheduling DP + greedy activity selection |
| `src/coin_change.py` | Coin change: minimum coins + ways to make change (combinations/sequences) |
| `src/combinatorial_rank.py` | Combinatorial ranking: permutation/combination rank-unrank + Gray code |
| `src/bridges.py` | Bridges & articulation points (Tarjan) + 2-edge-connected components |
| `src/bipartite_matching.py` | Maximum bipartite matching (Hopcroft-Karp) + Konig cover + Hall test |
| `src/min_cost_flow.py` | Minimum-cost maximum flow (SPFA successive shortest paths) + assignment |
| `src/sprague_grundy.py` | Sprague-Grundy nimbers (mex + XOR) for Nim, subtraction games, Kayles |
| `src/walsh_hadamard.py` | Fast Walsh-Hadamard transform + XOR/OR/AND convolutions (integer-exact) |
| `src/dpll.py` | DPLL SAT solver: unit propagation + pure-literal elimination + backtracking |
| `src/berlekamp_massey.py` | Shortest linear recurrence of a sequence over Q and GF(2) (LFSR) |
| `src/suffix_automaton.py` | Suffix automaton: distinct substrings, occurrences, LRS, LCS in O(n) |
| `src/lyndon.py` | Lyndon words: Duval factorisation, least rotation, FKM generation |
| `src/eertree.py` | Eertree (palindromic tree): all distinct palindromic substrings in O(n) |
| `src/li_chao.py` | Li Chao tree: lower/upper envelope of lines, convex-hull-trick DP |
| `src/mo_algorithm.py` | Mo's algorithm: offline range distinct-count & power-sum in O((n+q)vn) |
| `src/arborescence.py` | Chu-Liu/Edmonds minimum spanning arborescence (directed MST) |
| `src/steiner_tree.py` | Steiner tree (Dreyfus-Wagner): cheapest tree connecting terminals |
| `src/tree_isomorphism.py` | AHU tree isomorphism: linear-time canonical form + center rooting |
| `src/eulerian.py` | Eulerian paths & circuits (Hierholzer), undirected + directed, existence tests |
| `src/yen_ksp.py` | Yen's K shortest loopless paths in a directed weighted graph |
| `src/karger.py` | Karger & Karger-Stein randomized global minimum cut |
| `src/bron_kerbosch.py` | Bron-Kerbosch maximal-clique enumeration (pivoting + degeneracy) |
| `src/dinic.py` | Dinic's max-flow (blocking flows on the level graph) + min cut + matching |
| `src/graph_coloring.py` | Graph coloring: greedy, DSATUR, exact chromatic number |
| `src/k_core.py` | k-core decomposition: coreness, degeneracy, k-shells |
| `src/prufer.py` | Prufer sequences: labeled-tree <-> string bijection + Cayley's formula |
| `src/gale_shapley.py` | Gale-Shapley stable matching (deferred acceptance) + stability check |
| `src/degree_sequence.py` | Graphic degree sequences: Havel-Hakimi + Erdos-Gallai + realize |
| `src/matrix_tree.py` | Matrix-Tree theorem: count spanning trees via the Laplacian cofactor |
| `src/hirschberg.py` | Hirschberg linear-space optimal alignment + LCS |
| `src/centroid_decomposition.py` | Centroid tree + distance-pair counting on a tree |
| `src/meet_in_middle.py` | Meet-in-the-middle subset sum / closest sum / count for huge values |
| `src/fenwick_2d.py` | 2D Fenwick tree: point update + rectangle sum in O(log R log C) |
| `src/linear_sieve.py` | Linear sieve: primes + SPF + Euler totient + Mobius in O(N) |
| `src/weighted_dsu.py` | Weighted union-find: difference constraints + parity/bipartite |
| `src/durand_kerner.py` | Durand-Kerner: all complex roots of a polynomial simultaneously |
| `src/tanh_sinh.py` | Tanh-sinh (double-exponential) quadrature for endpoint singularities |
| `src/ntt.py` | Number-theoretic transform: exact integer convolution + big-int multiply |
| `src/karatsuba.py` | Karatsuba & Toom-3 fast multiplication + Karatsuba polynomial multiply |
| `src/strassen.py` | Strassen sub-cubic matrix multiplication (7 block products, padded) |
| `src/dancing_links.py` | Dancing Links (DLX): Algorithm X exact cover + N-queens |
| `src/walksat.py` | WalkSAT: randomized local-search SAT solver (incomplete) |
| `src/householder_qr.py` | Householder QR by reflections + least squares + solve |
| `src/jacobi_eigen.py` | Jacobi symmetric eigendecomposition by Givens rotations |
| `src/kitamasa.py` | Kitamasa: N-th linear-recurrence term in O(k^2 log n) |
| `src/rational_rref.py` | Exact rational RREF: rank, null space, exact linear solve |
| `src/rabin_karp.py` | Rabin-Karp rolling-hash search + multi-pattern + longest common substring |
| `src/half_plane_intersection.py` | Half-plane intersection: feasible convex region of linear constraints |
| `src/bentley_ottmann.py` | Sweep-line segment intersection: all crossings via x-ordered sweep |
| `src/gjk.py` | GJK convex collision detection + Minkowski difference |
| `src/ks_test.py` | Kolmogorov-Smirnov one- and two-sample tests + reference CDFs |
| `src/bootstrap.py` | Bootstrap CIs (percentile + BCa) + jackknife + standard error |
| `src/permutation_test.py` | Permutation tests: exact enumeration + Monte-Carlo + paired sign-flip |
| `src/lll.py` | LLL lattice reduction (exact rational Gram-Schmidt) + integer relations |
| `src/levinson_durbin.py` | O(n^2) Toeplitz solver + autoregressive (Yule-Walker) fit + predictor |
| `src/poisson_disk.py` | Bridson blue-noise Poisson-disk sampling (2D + n-D) |
| `src/low_discrepancy.py` | Van der Corput / Halton / Hammersley + quasi-Monte-Carlo integration |
| `src/thompson_nfa.py` | Linear-time regex engine (Thompson NFA, no catastrophic backtracking) |
| `src/bluestein.py` | Bluestein chirp-z DFT: O(n log n) transform at any length, even primes |
| `src/vp_tree.py` | Vantage-point tree: metric-space nearest neighbours (points, strings, vectors) |
| `src/tdigest.py` | t-digest: streaming quantiles over the whole distribution, sharp tails, mergeable |
| `src/myers_diff.py` | Myers O(ND) diff: shortest edit script + LCS + unified diff (the git algorithm) |
| `src/push_relabel.py` | Push-relabel (Goldberg-Tarjan) max flow + min cut + bipartite matching |
| `src/butterworth.py` | Butterworth IIR filter design (low/high-pass) + filtfilt + frequency response |
| `src/unscented_kalman.py` | Unscented Kalman filter: nonlinear state estimation via sigma points |
| `src/dct.py` | Discrete cosine transform (DCT-II/III, 1D + 2D) with energy compaction |
| `src/quaternion.py` | Quaternion rotation: Hamilton product, slerp, axis-angle/matrix/Euler conversions |
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
| `examples/photoelectric_demo.py` | Threshold & stopping voltage per metal + the V_stop(f) lines |
| `examples/uncertainty_demo.py` | Confinement energy by box size + the E(dx) electron/nucleon curves |
| `examples/tunneling_demo.py` | Transmission by width/height + STM gap sensitivity + the T(L) curves |
| `examples/particle_box_demo.py` | Levels & quantum-dot colours + the level/wavefunction diagram |
| `examples/harmonic_oscillator_demo.py` | Vibrational quanta per molecule + the parabolic-well level diagram |
| `examples/rutherford_demo.py` | Cross section & impact parameter by angle + the 1/sin^4 curve |
| `examples/radioactive_decay_demo.py` | Dating ages + a parent/daughter chain and Bateman curve |
| `examples/mass_formula_demo.py` | B/A for landmark nuclei + the binding-energy curve peaking at iron |
| `examples/q_value_demo.py` | Reaction Q-values + the chemical-to-annihilation energy-density chart |
| `examples/quantum_stats_demo.py` | Occupation vs (E-mu)/kT + the FD/BE/MB distribution curves |
| `examples/debye_heat_demo.py` | C_V per material + the universal C_V/3R vs T/Theta_D curve |
| `examples/carnot_demo.py` | Engine efficiencies & COP + the efficiency-vs-temperature-ratio curve |
| `examples/adiabatic_demo.py` | Compression temperatures + the adiabat-vs-isotherm P-V diagram |
| `examples/van_der_waals_demo.py` | Critical constants per gas + the reduced isotherms with the loop |
| `examples/joule_thomson_demo.py` | Inversion temperatures per gas + the mu_JT(T) crossings |
| `examples/clausius_clapeyron_demo.py` | Boiling point vs altitude + the vapor-pressure curve |
| `examples/reynolds_demo.py` | Re & regime from bacterium to whale + the log-Re transition chart |
| `examples/bernoulli_demo.py` | Pitot/Torricelli speeds + the Venturi velocity/pressure diagram |
| `examples/surface_tension_demo.py` | Capillary rise table + rise-vs-radius log-log plot |
| `examples/ekman_demo.py` | Current-vs-depth table + the Ekman spiral hodograph |
| `examples/milankovitch_demo.py` | 65N-summer sensitivity table + seasonal insolation map |
| `examples/equipartition_demo.py` | Gas heat-capacity table + the H2 C_V staircase plot |
| `examples/osmosis_demo.py` | Everyday-solution pressure table + Pi-vs-concentration plot |
| `examples/diffusion_demo.py` | Diffusion length/time table + the spreading-Gaussian fan |
| `examples/peclet_demo.py` | Per-system Peclet table + the advection-diffusion regime map |
| `examples/convection_demo.py` | Cooling-regime table + the Newtonian cooling curves |
| `examples/stefan_demo.py` | Ice-growth table by frost severity + the sqrt(t) front curves |
| `examples/capillary_demo.py` | Per-liquid capillary length + the drop-shape crossover figure |
| `examples/froude_demo.py` | Flow-regime & hull-speed tables + the hydraulic-jump profile |
| `examples/mach_cone_demo.py` | Cone-angle & boom-timing tables + the Mach-cone figure |
| `examples/nozzle_demo.py` | Area-ratio/exit-Mach table + the converging-diverging bell figure |
| `examples/blasius_demo.py` | Thickness/skin-friction table + the growing-boundary-layer figure |
| `examples/strouhal_demo.py` | Shedding-frequency table + the von Karman vortex-street figure |
| `examples/cluster_mass_demo.py` | Per-cluster virial-mass table + the mass-to-light ladder |
| `examples/sersic_demo.py` | Per-index profile table + the surface-brightness curves |
| `examples/grashof_demo.py` | Natural-convection table + the h-vs-height / transition figure |
| `examples/womersley_demo.py` | Vascular-tree alpha table + the parabola-to-plug profile figure |
| `examples/marangoni_demo.py` | Onset/regime table + the Marangoni-vs-buoyancy regime map |
| `examples/kutta_joukowski_demo.py` | Lift/Magnus tables + the lift-slope & induced-drag figure |
| `examples/knudsen_demo.py` | Per-system regime table + the size-pressure regime map |
| `examples/richardson_demo.py` | Per-layer stability table + the Ri map & KH-billow sketch |
| `examples/kolmogorov_demo.py` | Per-flow microscale table + the -5/3 energy-spectrum figure |
| `examples/casimir_demo.py` | Pressure/force-vs-gap table + the d^-4 pressure figure |
| `examples/hall_effect_demo.py` | Per-material Hall table + the Hall-bar schematic |
| `examples/wiedemann_franz_demo.py` | Predicted-vs-measured kappa table + the Lorenz-line figure |
| `examples/bragg_demo.py` | Per-plane Bragg-angle table + the reflection-geometry figure |
| `examples/diffraction_limit_demo.py` | Per-instrument resolution table + the Rayleigh-limit figure |
| `examples/snell_demo.py` | Per-medium critical/Brewster table + the refraction/TIR ray figure |
| `examples/thin_film_demo.py` | Soap-colour/AR-coating table + colour-vs-thickness & Newton's-rings figure |
| `examples/malus_demo.py` | Transmission/rescue/stack table + cos^2 & three-polarizer figure |
| `examples/cherenkov_demo.py` | Per-radiator threshold table + cone-angle & cone-geometry figure |
| `examples/zeeman_demo.py` | Splitting/g-factor table + the triplet-fan & sublevel-ladder figure |
| `examples/rabi_demo.py` | Pulse/detuning table + the flopping & Lorentzian-resonance figure |
| `examples/franck_hertz_demo.py` | Dip/excitation table + the current-vs-voltage sawtooth figure |
| `examples/moseley_demo.py` | Per-element K-alpha table + the Moseley sqrt(f)-vs-Z line |
| `examples/stark_demo.py` | Splitting/ionization table + the Stark fan & ionization-field figure |
| `examples/aharonov_bohm_demo.py` | Flux/phase table + the fringe-shift & phase-winding figure |
| `examples/josephson_demo.py` | V-f/Shapiro table + the I-phi sine & Shapiro-staircase figure |
| `examples/quantum_hall_demo.py` | Plateau/Landau table + the R_xy staircase figure |
| `examples/bcs_demo.py` | Gap/isotope/Tc table + the gap(T) & Tc-vs-coupling figure |
| `examples/london_demo.py` | Penetration/type table + the Meissner-decay & type-boundary figure |
| `examples/ising_mft_demo.py` | Magnetization/susceptibility table + the m(T) & chi figure |
| `examples/percolation_demo.py` | Spanning/cluster table + the threshold curve & lattice snapshots |
| `examples/polya_demo.py` | Return/escape/visits table + the return-probability-vs-dimension figure |
| `examples/langevin_para_demo.py` | Alignment/Curie table + the L(x) & 1/T susceptibility figure |
| `examples/buffon_demo.py` | Convergence table + the needle-scatter & pi-estimate figure |
| `examples/metropolis_demo.py` | Simulated m(T) table + the transition curve & spin-snapshot figure |
| `examples/logistic_map_demo.py` | Period/Lyapunov table + the bifurcation diagram & Lyapunov figure |
| `examples/henon_demo.py` | Fixed-point/Lyapunov table + the attractor & fractal-zoom figure |
| `examples/lorenz_demo.py` | Fixed-point/Lyapunov table + the butterfly & trajectory-divergence figure |
| `examples/double_pendulum_demo.py` | Energy/divergence table + the bob-trace & two-pendulum figure |
| `examples/mandelbrot_demo.py` | Escape-time table + ASCII view & the escape-time-coloured set |
| `examples/van_der_pol_demo.py` | Amplitude/period table + the phase-portrait & waveform figure |
| `examples/duffing_demo.py` | Regime/backbone table + the double-well & resonance-backbone figure |
| `examples/kuramoto_demo.py` | Synchrony-vs-coupling table + the r(K) transition & phase-circle figure |
| `examples/sandpile_demo.py` | Topple/avalanche table + the relaxed-pattern & size-distribution figure |
| `examples/cellular_automaton_demo.py` | Rule-table/class table + the rule-90/30/110 space-time figure |
| `examples/game_of_life_demo.py` | Pattern table + glider ASCII animation & the phases/board figure |
| `examples/reaction_diffusion_demo.py` | Growth table + ASCII field & the pattern-over-time figure |
| `examples/boids_demo.py` | Polarization table + the scatter-to-flock & polarization-over-time figure |
| `examples/dla_demo.py` | Size/dimension table + the cluster & log-log mass-radius scaling figure |
| `examples/benford_demo.py` | Digit-frequency table + the Benford curve with Fibonacci vs uniform bars |
| `examples/coupon_collector_demo.py` | E[T] vs simulation table + the progress curve & completion CDF |
| `examples/secretary_demo.py` | Optimal cutoff vs simulation table + the P(win) curve & 1/e convergence |
| `examples/birthday_demo.py` | Collision-probability table + the P vs k curve & sqrt(days) crossover |
| `examples/gamblers_ruin_demo.py` | Ruin vs simulation table + the ruin curves & sample walk paths |
| `examples/parrondo_demo.py` | Per-game drift vs simulation + capital trajectories & drift-vs-mix curve |
| `examples/galton_demo.py` | Slot histogram vs binomial + the Gaussian overlay & 1/sqrt(n) convergence |
| `examples/monty_hall_demo.py` | Stay/switch win rates vs simulation + the bars & (N-1)/N scaling curve |
| `examples/bayes_test_demo.py` | Rare-disease posterior vs simulation + the PPV-vs-prevalence curve & cohort |
| `examples/shannon_demo.py` | Huffman code table + the binary-entropy curve & codeword-length figure |
| `examples/kelly_demo.py` | Growth-rate table vs simulation + the g(f) curve & bankroll trajectories |
| `examples/hamming_demo.py` | Syndrome-locates-error table + the parity-coverage grid & code-rate curve |
| `examples/rsa_demo.py` | Keygen + encrypt/decrypt/sign walkthrough + the key-flow & modexp-cost figure |
| `examples/diffie_hellman_demo.py` | Exchange walkthrough + the flow diagram & attacker-vs-honest cost figure |
| `examples/crc_demo.py` | Check values + frame verify/corrupt + the frame layout & miss-probability figure |
| `examples/lz77_demo.py` | Token stream + ratio-vs-repetition table + the stream & compression-ratio figure |
| `examples/bloom_demo.py` | No-false-negative check + FP-rate table + the fill & optimal-k figure |
| `examples/hyperloglog_demo.py` | Estimate-vs-true table + merge demo + the accuracy & error-band figure |
| `examples/fenwick_demo.py` | Prefix/range/select walkthrough + the coverage-range & cost figure |
| `examples/union_find_demo.py` | Component-merge trace + Kruskal MST + the MST-edge & merge figure |
| `examples/dijkstra_demo.py` | Distances vs Bellman-Ford + a grid maze + the distance-flood & route figure |
| `examples/kdtree_demo.py` | Nearest/k-NN/radius vs brute force + the point-cloud query figure |
| `examples/boyer_moore_demo.py` | Match + bad-char table + the comparison-count vs pattern-length figure |
| `examples/astar_demo.py` | A* vs Dijkstra cost/expansions + the side-by-side explored-cells figure |
| `examples/toposort_demo.py` | Kahn/DFS order + critical path + the layered DAG figure |
| `examples/levenshtein_demo.py` | Alignment + spell-check ranking + the DP-table heatmap figure |
| `examples/knapsack_demo.py` | Chosen items vs brute force + the DP-table heatmap & value-vs-capacity figure |
| `examples/lcs_demo.py` | LCS + a real line-diff + the DP-table heatmap with the match diagonal |
| `examples/quickselect_demo.py` | Order statistics + the comparison-count vs sort figure |
| `examples/aho_corasick_demo.py` | Overlapping matches + log scan + the trie-with-failure-links figure |
| `examples/floyd_warshall_demo.py` | Distance matrix vs Dijkstra + path/cycle/closure + the matrix heatmap |
| `examples/misra_gries_demo.py` | Heavy hitters vs exact + majority + the approx-count & memory figure |
| `examples/reservoir_demo.py` | Uniformity chi-square + weighted proportions + the frequency & weight figure |
| `examples/count_min_demo.py` | Estimate vs true + error-vs-width + the scatter & error-decay figure |
| `examples/alias_method_demo.py` | Target vs sampled + the alias table + the frequency & column figure |
| `examples/fisher_yates_demo.py` | Uniform-vs-biased permutation counts + the frequency-histogram figure |
| `examples/box_muller_demo.py` | Moments + 68-95-99.7 + the histogram-vs-Gaussian-density figure |
| `examples/rejection_sampling_demo.py` | Acceptance rate + the accepted/rejected darts & histogram figure |
| `examples/welford_demo.py` | Running stats + naive-vs-Welford offset table + the convergence figure |
| `examples/kahan_demo.py` | Error-vs-n table + cancellation case + the error-growth figure |
| `examples/horner_demo.py` | Eval + synthetic division + roots + the op-count & Newton-convergence figure |
| `examples/rootfind_demo.py` | Method comparison + transcendental roots + the convergence-rate figure |
| `examples/quadrature_demo.py` | Method accuracy + convergence-order table + the error-vs-samples figure |
| `examples/spline_demo.py` | Spline vs polynomial on Runge + the through-the-knots curve figure |
| `examples/fft_demo.py` | Two-tone spectrum + convolution + the signal & spectrum & cost figure |
| `examples/linsolve_demo.py` | Solve + LU factors + reuse + det/inverse + the L/U heatmap figure |
| `examples/qr_demo.py` | QR + line/quadratic least-squares fits + the best-fit-line & residual figure |
| `examples/eigen_demo.py` | Dominant + inverse + full spectrum + the convergence & spectrum figure |
| `examples/conjugate_gradient_demo.py` | CG vs steepest descent + the residual-decay figure |
| `examples/svd_demo.py` | SVD + low-rank + PCA + the data-cloud axes & singular-value figure |
| `examples/kmeans_demo.py` | Cluster recovery + elbow/silhouette + the coloured-clusters & elbow figure |
| `examples/regression_demo.py` | Linear + logistic fits + log-loss decay + the line & sigmoid figure |
| `examples/decision_tree_demo.py` | CART on 3 blobs: learned rules, importances, depth-cap sweep + region figure |
| `examples/random_forest_demo.py` | Forest vs overfit tree on noisy data: OOB score, importances + boundary figure |
| `examples/gmm_demo.py` | EM on 3 unequal-spread clusters: recovered params, BIC curve + soft-responsibility figure |
| `examples/hmm_demo.py` | Dishonest casino: Viterbi decode, posterior P(loaded) ribbon, Baum-Welch relearn + figure |
| `examples/kalman_demo.py` | Noisy tracking: filter/smoother beat raw measurements, variance-collapse + track figure |
| `examples/pagerank_demo.py` | Small web graph: ranks, personalization, geometric convergence + node-sized graph figure |
| `examples/lu_demo.py` | P A = L U and A = L L' factorizations, multi-RHS solves, SPD test + shaded factor grids |
| `examples/gaussian_process_demo.py` | GP fit to sparse noisy data: length-scale tuning, 2-sigma confidence band figure |
| `examples/bayes_opt_demo.py` | BO of a multimodal function: surrogate + EI figure, convergence vs random search |
| `examples/dbscan_demo.py` | Two moons + outliers: cluster recovery, noise flagging, k-distance elbow figure |
| `examples/hierarchical_demo.py` | Blobs + dendrogram figure, gap-based k selection, single-vs-complete chaining contrast |
| `examples/naive_bayes_demo.py` | Gaussian decision regions + multinomial spam filter with per-word log-odds figure |
| `examples/knn_demo.py` | Decision boundary jagged at k=1 vs smooth LOO-selected k, regression on a noisy sine |
| `examples/gradient_boosting_demo.py` | Fit sharpening from 1 to 120 trees + loss curve, learning-rate/n-trees trade |
| `examples/spectral_clustering_demo.py` | Concentric rings split correctly + the eigenvector embedding that untangles them |
| `examples/particle_filter_demo.py` | Nonlinear tracking beats raw sensor + ESS collapse-vs-healthy resampling figure |
| `examples/simulated_annealing_demo.py` | TSP greedy-vs-annealed tour + cost-cooling curve, multimodal global min |
| `examples/genetic_algorithm_demo.py` | OneMax + real optimization fitness curves, 0/1 knapsack solve |
| `examples/particle_swarm_demo.py` | Sphere/Rastrigin/Rosenbrock solves, swarm scatter + inertia-decay convergence |
| `examples/reed_solomon_demo.py` | Corrupt bytes in a message and recover it; byte-grid error/repair figure |
| `examples/mutual_information_demo.py` | MI vs channel noise (matches 1-H(f)), nonlinear catch, feature ranking, KL |
| `examples/lru_cache_demo.py` | LRU eviction trace + LRU-vs-LFU hit rates across uniform/skewed/looping workloads |
| `examples/trie_demo.py` | Autocomplete (alpha + frequency-ranked), longest-prefix, substring index + tree figure |
| `examples/sorting_demo.py` | Comparison-count scaling (log-log), stability contrast, heap priority queue |
| `examples/newton_nd_demo.py` | Quadratic convergence curve, damping rescue, Broyden, 3-variable system |
| `examples/differential_evolution_demo.py` | Benchmark convergence curves, vs random, F sweep, scaling to 20-D |
| `examples/nelder_mead_demo.py` | Rosenbrock simplex crawl + convergence curve, restart refinement, eval scaling |
| `examples/hits_demo.py` | Hub vs authority rankings on a small web + PageRank comparison, dual graph figure |
| `examples/mds_demo.py` | Rebuild a city map from a distance table, Procrustes-aligned, + eigenvalue scree |
| `examples/skiplist_demo.py` | Express-lane tower figure, search-path trace, geometric level histogram |
| `examples/ant_colony_demo.py` | TSP tour over a pheromone field, convergence curve, alpha/beta balance |
| `examples/avl_tree_demo.py` | AVL vs naive-BST height on sorted input, four rotation cases, tree figure |
| `examples/segment_tree_demo.py` | Range sum/min/max + lazy range-add, O(log n) full-array update, tree figure |
| `examples/convex_hull_demo.py` | Hull of a scatter with area/perimeter/diameter + outlined-boundary figure |
| `examples/closest_pair_demo.py` | Closest pair highlighted, D&C-vs-brute scaling, agreement across sizes |
| `examples/segment_intersection_demo.py` | Crossing/touch/parallel classification + simple-vs-self-crossing polygons |
| `examples/point_in_polygon_demo.py` | Grid membership on a concave arrow + pentagram ray-vs-winding divergence |
| `examples/polygon_clip_demo.py` | Concave polygon clipped to rectangle/triangle/diamond windows + overlay figure |
| `examples/marching_squares_demo.py` | Circular contours of a radial field + Gaussian-terrain iso-lines figure |
| `examples/bresenham_demo.py` | ASCII line-fan and circle rasterization, sub-pixel accuracy, circumference scaling |
| `examples/flood_fill_demo.py` | Paint-bucket inside a wall, three strategies agree, connected components + connectivity |
| `examples/bezier_demo.py` | Quadratic/cubic curves with control polygons, subdivision, elevation, arc length |
| `examples/bwt_demo.py` | BWT runniness gain, full pipeline compression, worked banana example |
| `examples/arithmetic_coding_demo.py` | Bits/symbol vs Huffman vs entropy across distributions, ~49% saving on skew |
| `examples/sequence_alignment_demo.py` | Global vs local alignments with match rulers + DP-matrix traceback figure |
| `examples/string_matching_demo.py` | Prefix function, KMP/Z/brute agreement, overlapping matches, Manacher palindromes |
| `examples/suffix_array_demo.py` | Sorted-suffix + LCP table, binary-search patterns, longest repeated/common substring |
| `examples/quadtree_demo.py` | Point cloud with adaptive cell boundaries + rectangle/circle/nearest queries |
| `examples/max_flow_demo.py` | Six-node flow network with capacities, min-cut edges highlighted, bipartite matching |
| `examples/de_bruijn_demo.py` | B(2,3)/B(2,4)/PIN-pad B(10,4) sequences + the B(2,3) De Bruijn graph with Eulerian circuit |
| `examples/rotating_calipers_demo.py` | Point cloud with hull, diameter, and minimum-area rectangle (rotated box beats the AABB) |
| `examples/ternary_search_tree_demo.py` | Autocomplete, longest-prefix, and wildcard search on a dictionary + the TST structure |
| `examples/delaunay_demo.py` | Delaunay triangulation overlaid with its dual Voronoi diagram, empty-circumcircle checked live |
| `examples/simplex_demo.py` | Production LP with the feasible polytope, objective gradient, and optimal vertex drawn |
| `examples/minhash_demo.py` | Document near-duplicate detection + the estimate error tracking the 1/sqrt(k) curve |
| `examples/cma_es_demo.py` | CMA-ES convergence on sphere/Rosenbrock/Rastrigin/ellipsoid vs random search |
| `examples/lbfgs_demo.py` | L-BFGS vs gradient descent on an ill-conditioned quadratic + a logistic-regression fit |
| `examples/tsne_demo.py` | 8-D clusters embedded into a clear 2-D map, with the KL divergence falling over training |
| `examples/wavelet_tree_demo.py` | Rank/select/quantile/range-count queries + the recursive alphabet-partition tree |
| `examples/fibonacci_heap_demo.py` | Decrease-key/merge/Dijkstra + max root degree staying within the log_phi(n) bound |
| `examples/treap_demo.py` | Order statistics + split/merge + height near 2log2(n) even under sorted insertion |
| `examples/splay_tree_demo.py` | Working-set property: average access depth sinking below log2(n) as access skews |
| `examples/van_emde_boas_demo.py` | O(log log u) recursion depth staying flat as the universe explodes vs a BST's log u |
| `examples/sparse_table_demo.py` | Range-minimum query on an array + lowest-common-ancestor on a tree |
| `examples/pollard_rho_demo.py` | Factoring RSA-style semiprimes + iterations tracking the sqrt(p) trend |
| `examples/perlin_demo.py` | 2-D fBm terrain heightfield + a 1-D fBm cross-section |
| `examples/wave_function_collapse_demo.py` | A coastline map where land never touches sea + a forced checkerboard |
| `examples/hungarian_demo.py` | Worker-job assignment beating the greedy heuristic, with the cost matrix drawn |
| `examples/dtw_demo.py` | Two speed-varying signals aligned, DTW 6.6x smaller than Euclidean, warp path drawn |
| `examples/p2_quantile_demo.py` | Latency p50/p90/p95/p99 from a 200k stream in 20 floats, estimate converging |
| `examples/tarjan_scc_demo.py` | A directed graph's SCCs colored + the condensation DAG beside it |
| `examples/two_sat_demo.py` | A satisfiable feature-constraint instance + the canonical unsatisfiable formula |
| `examples/rk45_demo.py` | Van der Pol oscillator with step ticks clustering at the sharp transitions |
| `examples/cordic_demo.py` | The rotation spiralling to a target angle; cos/sin/exp/ln/sqrt vs the math library |
| `examples/savitzky_golay_demo.py` | Noisy two-peak signal: SG keeps the peaks where a moving average flattens them |
| `examples/lzw_demo.py` | Compression ratio improving with repetition + repetitive vs random comparison |
| `examples/convolutional_code_demo.py` | Error correction over a noisy channel + coding-gain curve vs uncoded |
| `examples/ear_clipping_demo.py` | A star, L-shape, and arrow triangulated with exact area conservation |
| `examples/mlp_demo.py` | XOR solved + a learned circular decision boundary a linear model can't draw |
| `examples/mdp_demo.py` | A gridworld solved to optimality: value heatmap + optimal-action arrows |
| `examples/bandit_demo.py` | Regret curves for epsilon-greedy/UCB1/Thompson vs random selection |
| `examples/q_learning_demo.py` | Learning curve + a learned gridworld policy matching value iteration |
| `examples/autodiff_demo.py` | Exact gradients vs finite diff + a model trained with no hand-derived gradients |
| `examples/shamir_demo.py` | Splitting a secret + the polynomial geometry with the secret at f(0) |
| `examples/merkle_demo.py` | An inclusion proof + tamper detection + the tree with its authentication path |
| `examples/tsp_demo.py` | Held-Karp exact solve + 2-opt untangling a 60-city nearest-neighbour tour |
| `examples/poisson_demo.py` | Steady-state heat on a plate + Jacobi/Gauss-Seidel/SOR convergence comparison |
| `examples/manacher_demo.py` | Longest palindrome + the self-similar radius profile of abacabadabacaba |
| `examples/stoer_wagner_demo.py` | Global min cut of a graph + two clusters severed at their weak links |
| `examples/chebyshev_demo.py` | Geometric convergence + Chebyshev taming Runge's function where equispaced blows up |
| `examples/gibbs_demo.py` | A correlated Gaussian sampled coordinate-wise, cloud filling the covariance ellipse |
| `examples/louvain_demo.py` | Four planted communities recovered, coloured, with the bridging edges highlighted |
| `examples/matrix_chain_demo.py` | Optimal vs left-to-right cost + the DP cost table as a heatmap |
| `examples/lis_demo.py` | The LIS highlighted on a bar chart + the patience-sorting piles |
| `examples/continued_fraction_demo.py` | pi/e/phi/sqrt2 expansions + convergent error vs denominator |
| `examples/crt_demo.py` | Sunzi's puzzle + reconstructing a secret from residues + non-coprime handling |
| `examples/tonelli_shanks_demo.py` | Modular square roots + the residue split + EC point decompression |
| `examples/discrete_log_demo.py` | Breaking a toy Diffie-Hellman + BSGS vs brute-force work curve |
| `examples/welzl_demo.py` | A point cloud's smallest enclosing circle with its support points |
| `examples/cuckoo_filter_demo.py` | Deletion + false-positive rate shrinking with fingerprint bits |
| `examples/dsu_rollback_demo.py` | Component count over edge additions and rollbacks on a timeline |
| `examples/interval_scheduling_demo.py` | A Gantt chart with the optimal-value jobs vs greedy heuristics |
| `examples/coin_change_demo.py` | The greedy trap + min-coins-per-amount curve for three coin systems |
| `examples/combinatorial_rank_demo.py` | Permutation/combination ranking tables + a 5-bit Gray-code bit-flip SVG |
| `examples/bridges_demo.py` | A 3-cluster network with its failure edges/nodes highlighted in red |
| `examples/bipartite_matching_demo.py` | Staffing 5 workers onto 5 jobs with the matched edges in green |
| `examples/min_cost_flow_demo.py` | A factory-to-store shipping network with per-pipe flow labels |
| `examples/sprague_grundy_demo.py` | Grundy-number colour strips revealing subtraction/Kayles periodicity |
| `examples/walsh_hadamard_demo.py` | Two 3-bit distributions combined by XOR/OR/AND as bar panels |
| `examples/dpll_demo.py` | A SAT model plus the pigeonhole principle proven UNSAT as bars |
| `examples/berlekamp_massey_demo.py` | Famous sequence recurrences + a length-5 LFSR keystream cracked |
| `examples/suffix_automaton_demo.py` | The 'abracadabra' automaton drawn as states by substring length |
| `examples/lyndon_demo.py` | Duval factorisations as coloured Lyndon-word segments |
| `examples/eertree_demo.py` | Palindromes of a rich word tiled by length, brightness by frequency |
| `examples/li_chao_demo.py` | A bundle of lines with their lower envelope highlighted |
| `examples/mo_algorithm_demo.py` | Query windows drawn in Mo's block-snake processing order |
| `examples/arborescence_demo.py` | A directed broadcast tree with the chosen min-cost edges in green |
| `examples/steiner_tree_demo.py` | Four terminals linked through a cheap hub, saving over the perimeter |
| `examples/tree_isomorphism_demo.py` | Two relabelled trees vs a different-shape one, centers marked |
| `examples/eulerian_demo.py` | Konigsberg and friends classified; a bowtie circuit numbered in walk order |
| `examples/yen_ksp_demo.py` | The four cheapest A-to-F routes, each road coloured by its best route |
| `examples/karger_demo.py` | Random contraction converging to a two-cluster graph's min cut |
| `examples/bron_kerbosch_demo.py` | A friendship network's maximal cliques, the largest highlighted |
| `examples/dinic_demo.py` | A pipe network at max flow with the min-cut pipes highlighted |
| `examples/graph_coloring_demo.py` | An exam-conflict graph coloured into the minimum time slots |
| `examples/k_core_demo.py` | A network peeled into onion rings sized by coreness |
| `examples/prufer_demo.py` | A tree beside its Prufer code + Cayley's count enumerated |
| `examples/gale_shapley_demo.py` | Applicants stably matched to schools with each side's rank shown |
| `examples/degree_sequence_demo.py` | Realizability tests + a Havel-Hakimi witness graph |
| `examples/matrix_tree_demo.py` | A graph beside its Laplacian and its spanning-tree count |
| `examples/hirschberg_demo.py` | An optimal alignment as a match/gap track + the memory saving |
| `examples/centroid_decomposition_demo.py` | A tree tinted by centroid-decomposition level + pair counts |
| `examples/meet_in_middle_demo.py` | Two half-sum lists combining to hit a target on a number line |
| `examples/fenwick_2d_demo.py` | A grid heatmap with a query rectangle and its dynamic sum |
| `examples/linear_sieve_demo.py` | The totient curve and Mobius bars from one linear-sieve pass |
| `examples/weighted_dsu_demo.py` | Difference constraints accepted/rejected + an odd-cycle contradiction |
| `examples/durand_kerner_demo.py` | Polynomial roots found at once + the 8th roots of unity plotted |
| `examples/tanh_sinh_demo.py` | Singular integrals nailed vs Simpson + the clustering abscissae |
| `examples/ntt_demo.py` | Exact polynomial product + big-integer multiply by digit convolution |
| `examples/karatsuba_demo.py` | Big-int products + the complexity-exponent curves of each method |
| `examples/strassen_demo.py` | The seven block products + n^3 vs n^2.807 cost curves |
| `examples/dancing_links_demo.py` | Knuth's exact-cover example + a 6-queens board on a chessboard |
| `examples/walksat_demo.py` | A 3-SAT solve with the falling conflict-count trajectory |
| `examples/householder_qr_demo.py` | A least-squares line fit with residual stubs |
| `examples/jacobi_eigen_demo.py` | The off-diagonal norm plunging to zero over rotations |
| `examples/kitamasa_demo.py` | Huge recurrence terms + the O(n) vs O(log n) cost gap |
| `examples/rational_rref_demo.py` | An exact RREF grid with pivot/free columns + the three system kinds |
| `examples/rabin_karp_demo.py` | Rolling-hash matches across a text + multi-pattern + sentence LCS |
| `examples/half_plane_intersection_demo.py` | A 5-constraint feasible region with its LP optimum vertex |
| `examples/bentley_ottmann_demo.py` | Six segments with all their crossings marked |
| `examples/gjk_demo.py` | Two shapes + their Minkowski difference with the origin inside |
| `examples/ks_test_demo.py` | Two empirical CDFs with the maximal-gap KS statistic marked |
| `examples/bootstrap_demo.py` | The bootstrap distribution of the mean with its 95% CI band |
| `examples/permutation_test_demo.py` | The permutation null distribution with the rejection region shaded |
| `examples/lll_demo.py` | A skewed lattice basis vs its short, near-orthogonal LLL reduction |
| `examples/levinson_durbin_demo.py` | An AR(2) one-step-ahead prediction tracking a synthesised signal |
| `examples/poisson_disk_demo.py` | Blue-noise Poisson-disk points vs clumpy uniform random, side by side |
| `examples/low_discrepancy_demo.py` | Halton vs pseudo-random points and QMC-vs-MC pi convergence |
| `examples/thompson_nfa_demo.py` | Linear match-time curve of the catastrophic-backtracking pattern |
| `examples/bluestein_demo.py` | Bluestein vs direct DFT timing at prime lengths (6x-36x speedup) |
| `examples/vp_tree_demo.py` | A k-NN query touching under 10% of points, plus edit-distance lookup |
| `examples/tdigest_demo.py` | Streaming p50-p9999 from 500k samples in 64 centroids, plus a merge |
| `examples/myers_diff_demo.py` | A git-style unified diff and the edit graph with its shortest path |
| `examples/push_relabel_demo.py` | A max-flow network with per-edge utilisation and the min cut drawn |
| `examples/butterworth_demo.py` | Filter magnitude responses and a 7x denoising of a buried sine |
| `examples/unscented_kalman_demo.py` | A projectile tracked 4x better than raw range/bearing readings |
| `examples/dct_demo.py` | Energy compaction and lossy reconstruction of a signal from few coefficients |
| `examples/quaternion_demo.py` | SLERP vs component-LERP: even arc spacing vs bunching |
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

## The photoelectric effect: light in photon lumps

Light ejects electrons from a metal only above a threshold frequency, no matter how bright a
redder beam is -- Einstein's proof that light arrives in photons of energy hf.
`photoelectric.py`:

```
$ python examples/photoelectric_demo.py examples/output

       metal  phi (eV)   threshold  V_stop @254nm  V_stop @400nm
  --------------------------------------------------------------
      cesium      2.14       579nm         2.74 V         0.96 V
      sodium      2.28       544nm         2.60 V         0.82 V
        zinc      4.31       288nm         0.57 V           none
    platinum      6.35       195nm           none           none
```

One photon gives all its energy to one electron, `K_max = hf - phi`, so below the threshold
`f0 = phi/h` nothing is emitted and above it the stopping voltage `V_stop = K_max/e` climbs
linearly with frequency at the universal slope `h/e` -- only the intercept (the work function)
differs between metals. Brighter light ejects more electrons, not faster ones. Millikan's
measurement of that line pinned down Planck's constant. The tests verify the sodium threshold
wavelength, the no-emission-below-threshold rule, the linear stopping voltage, the slope being
Planck's constant independent of metal, and the work-function ordering.

## The uncertainty principle: why quantum things can't sit still

Position and momentum cannot both be sharp: `dx dp >= hbar/2`. `uncertainty.py`:

```
$ python examples/uncertainty_demo.py examples/output

           confinement      size    electron E     nucleon E
  ----------------------------------------------------------
              molecule      1 nm     0.0095 eV    5.2e-06 eV
                  atom    0.1 nm       0.95 eV    0.00052 eV
        atomic nucleus      5 fm   3.8e+02 MeV      0.21 MeV
          nucleon core      1 fm   9.5e+03 MeV       5.2 MeV
```

Confining a particle to a box forces a momentum spread `dp >= hbar/(2 dx)` and thus an
irreducible zero-point kinetic energy `E ~ hbar^2/(m dx^2)`. Squeeze an electron to atomic
size (~0.1 nm) and it carries ~1 eV -- which is why it does not spiral into the nucleus -- and
minimizing that confinement energy against the Coulomb pull reproduces hydrogen's 13.6 eV
binding from the uncertainty principle alone. A nucleon confined to a femtometre nucleus
carries MeV, the nuclear energy scale. The same `dE dt >= hbar/2` gives a short-lived state a
broad natural line width. The tests verify the `dx dp = hbar/2` bound, the eV atomic and MeV
nuclear confinement energies, the 13.6 eV hydrogen estimate, and the lifetime-linewidth
relation.

## Quantum tunneling: through a barrier you cannot climb

A particle with `E < V` can still leak through a barrier -- its wavefunction decays as
`exp(-kappa x)` inside. `tunneling.py`:

```
$ python examples/tunneling_demo.py examples/output

    V - E (eV)   width (nm)     transmission
  ----------------------------------------
             4          0.2        1.66e-02
             4          0.5        3.55e-05
             4          1.0        1.26e-09

  STM tip-surface gap sensitivity (4 eV work function):
    +0.1 nm gap  ->  current x 1.29e-01  (1/8)
    +0.2 nm gap  ->  current x 1.66e-02  (1/60)
```

Transmission `T ~ exp(-2 kappa L)` with `kappa = sqrt(2m(V-E))/hbar` plunges exponentially
with width and with `sqrt(V-E)`, so a nanometre barrier is essentially opaque -- yet shaving
an Angstrom raises the tunneling current ~8x. That razor sensitivity is how a scanning
tunneling microscope feels individual atoms, and the same barrier penetration drives alpha
decay (tunneling out of the nuclear Coulomb barrier) and stellar fusion (tunneling in, the
Gamow module). The WKB integral extends it to a general barrier shape. The tests verify the
exponential width dependence, the exact-to-thick-limit reduction, the WKB match on a
rectangular barrier, and the ~order-of-magnitude-per-Angstrom STM sensitivity.

## The particle in a box: the simplest quantized system

Trap a particle in an infinite square well and only standing waves with a whole number of
half-wavelengths fit, quantizing the energy. `particle_box.py`:

```
$ python examples/particle_box_demo.py examples/output

  quantum-dot n=1->2 emission (smaller box = bluer):
    5 nm dot  ->  27477 nm
    3 nm dot  ->  9892 nm
    2 nm dot  ->  4396 nm
    1 nm dot  ->  1099 nm
```

`E_n = n^2 h^2 / (8 m L^2)`: levels rise as `n^2`, the ground state is nonzero (the
confinement zero-point energy of the uncertainty principle), and every level scales as
`1/L^2`. That last dependence is why quantum dots -- nanoscale electron boxes -- have
size-tunable colour: shrink the dot and the level gaps widen, shifting emission toward the
blue, exploited in displays and biological markers. The eigenfunctions
`psi_n = sqrt(2/L) sin(n pi x/L)` are orthonormal with `n-1` nodes. The tests verify the
`n^2` levels, the nonzero ground state, the `1/L^2` scaling, the smaller-dot-bluer trend, the
wavefunction normalization, and the box-width-for-gap inversion.

## The quantum harmonic oscillator: evenly-spaced rungs

Near any potential minimum a system behaves as a spring, so the oscillator is the workhorse
of quantum mechanics. `harmonic_oscillator.py`:

```
$ python examples/harmonic_oscillator_demo.py examples/output

    molecule   k (N/m)   hbar omega (eV)   IR wavelength
  ------------------------------------------------------
          H2       570            0.5454         2.27 um
          CO      1902            0.2690         4.61 um
          N2      2294            0.2924         4.24 um
         HCl       516            0.3721         3.33 um
```

`E_n = (n + 1/2) hbar omega` with `omega = sqrt(k/m)`: the levels are EVENLY spaced by
`hbar omega` (unlike the box's `n^2` or the atom's `-1/n^2`), so a molecule absorbs one sharp
infrared line per vibrational quantum -- CO at 4.6 microns, the workhorse of IR spectroscopy.
The ground state is nonzero: the zero-point energy `(1/2) hbar omega` is forced by the
uncertainty principle and is physically real -- it keeps helium liquid at absolute zero, shifts
chemical bond energies, and sets each field mode's vacuum energy. The tests verify the equal
level spacing, the nonzero zero-point energy, the CO vibrational quantum and 4.6-micron line,
the `sqrt(k/m)` frequency, and the spring-constant round-trip.

## Rutherford scattering: finding the nucleus

Firing alpha particles at gold foil, a few bounced almost straight back -- impossible off
diffuse charge. `rutherford.py`:

```
$ python examples/rutherford_demo.py examples/output

     angle   dsigma/dOmega (rel)    impact b (fm)
  ----------------------------------------------
       10d              4.33e+03           260.1
       30d                  55.7            84.9
       90d                     1            22.8
      150d                 0.287             6.1
      179d                  0.25             0.2
```

The Coulomb cross section `dsigma/dOmega = (Z1 Z2 e^2 / 16 pi eps0 E)^2 / sin^4(theta/2)`
soars at small angles (grazing passes) but stays nonzero at 180 degrees -- exactly the rare
hard bounces observed, which are impossible off spread-out charge and revealed a tiny dense
nucleus. The impact parameter maps to angle via `b = (k/2E) cot(theta/2)`, and the head-on
closest approach `r_min = Z1 Z2 e^2 / (4 pi eps0 E)` is ~45 fm for 5 MeV alphas on gold --
Rutherford's upper bound on the nuclear size. The tests verify the `1/sin^4` dependence, the
nonzero back-scatter, the small-angle divergence, the ~45 fm closest approach, the
impact-angle inversion, and the `1/E^2` cross-section scaling.

## Radioactive decay: half-lives, dating, and chains

Unstable nuclei decay at a constant per-nucleus rate, so a population falls exponentially.
`radioactive_decay.py`:

```
$ python examples/radioactive_decay_demo.py examples/output

  carbon-14 dating (t_half = 5730 yr):
     % remaining   half-lives    age (yr)
  ----------------------------------------
             90%         0.15         871
             50%         1.00        5730
             25%         2.00       11460
             10%         3.32       19035
              1%         6.64       38069
```

`N = N0 2^(-t/t_half)` with `lambda = ln2/t_half`, so the surviving fraction dates a sample:
`t = t_half log2(N0/N)`. In a parent-daughter chain the daughter follows the Bateman equation,
`N_D = N0 lambda_P/(lambda_D-lambda_P)(e^{-lambda_P t} - e^{-lambda_D t})`, starting at zero,
rising to a peak, then tracking the parent -- secular equilibrium, where daughter activity
equals parent activity independent of the daughter's half-life. That balance runs medical
radioisotope generators (Mo-99/Tc-99m) and the radon that seeps from long-lived uranium. The
tests verify the one-half-life halving, the C-14 dating age, the number of half-lives, the
Bateman rise-and-fall from zero, and secular equilibrium.

## Nuclear binding: the semi-empirical mass formula

Weizsacker's liquid-drop model sums competing terms into the nuclear binding energy.
`mass_formula.py`:

```
$ python examples/mass_formula_demo.py examples/output

       nucleus    Z    A   B/A (MeV)
  --------------------------------
          He-4    2    4       5.710
         Fe-56   26   56       8.846
         Ni-62   28   62       8.863
        Pb-208   82  208       7.857
         U-238   92  238       7.625
```

`B = a_V A - a_S A^(2/3) - a_C Z(Z-1)/A^(1/3) - a_A (A-2Z)^2/A + delta`: the volume term
wants a big nucleus, surface and Coulomb penalize it, asymmetry wants N=Z, and pairing
favours even-even nuclei. Their balance gives the binding-energy-per-nucleon curve, peaking
near iron/nickel at ~8.8 MeV/nucleon -- which is exactly why fusion releases energy up to
iron and fission releases it beyond, and why a massive star's iron core cannot burn and
collapses into a supernova. Minimizing over Z traces the valley of stability, drifting to
neutron excess in heavy nuclei (U-238 at Z=92, Pb-208 at Z=82). The tests verify the ~8.8
MeV iron binding, the iron-group peak, light-rises/heavy-falls, the valley of stability, the
growing neutron excess, and the pairing signs.

## Nuclear Q-value: the energy in mass

A nuclear reaction releases energy equal to its mass defect times c^2. `q_value.py`:

```
$ python examples/q_value_demo.py examples/output

              reaction    Q (MeV)   mass converted
  --------------------------------------------------
      D + T -> He4 + n      17.59          0.375 %
       p-p chain (net)      26.70          0.711 %
         U-235 fission     197.01          0.090 %

  energy density of fuels (joules per kilogram):
          TNT (chemical): 4.60e+06 J/kg
           U-235 fission: 8.05e+13 J/kg
              D-T fusion: 3.38e+14 J/kg
       matter-antimatter: 8.99e+16 J/kg
```

With `Q = (m_reactants - m_products) c^2` and the shortcut `1 amu = 931.494 MeV`, D-T fusion
yields 17.6 MeV and U-235 fission ~200 MeV -- only a fraction of a percent of the mass, yet
`c^2` makes it millions of times a chemical bond: fission is ~2 million times TNT, fusion ~4x
fission, and total matter-antimatter annihilation converts 100% of the mass at the ultimate
`c^2 ~ 9x10^16 J/kg`. The tests verify the 931.494 MeV/amu equivalence, the D-T and U-235
Q-values, the exothermic sign, the mass-energy consistency, the binding-energy route, and the
~10^14 J/kg fusion energy density.

## Quantum statistics: how identical particles share states

Identical quantum particles come in two kinds, and the tiny +1/-1 in their occupation number
decides everything. `quantum_stats.py`:

```
$ python examples/quantum_stats_demo.py examples/output

     (E-mu)/kT     Fermi-Dirac   Bose-Einstein   Maxwell-Boltz
  --------------------------------------------------------
          -4.0        0.9820             inf         54.5982
           0.5        0.3775           1.541          0.6065
           1.0        0.2689           0.582          0.3679
           4.0        0.0180           0.019          0.0183
           8.0        0.0003           0.000          0.0003
```

Fermions obey `<n> = 1/(exp((E-mu)/kT)+1)`, never exceeding one per state (Pauli), so at T=0
they fill a sharp step up to the Fermi level -- electron degeneracy and white-dwarf pressure.
Bosons obey `<n> = 1/(exp((E-mu)/kT)-1)`, which diverges as `E -> mu`, so below a critical
temperature a macroscopic fraction condenses into the ground state (BEC, superfluid helium).
Far above the chemical potential (`E - mu >> kT`) both fade into the classical Maxwell-Boltzmann
exponential, and a photon mode (`mu=0`) at `E=kT` has the Planck occupation `1/(e-1) ~ 0.58`.
The tests verify the 0.5-at-mu Fermi value, the T=0 step, the never-above-one bound, the bosonic
divergence, the classical high-energy limit, and the photon Planck factor.

## Debye specific heat: why solids go cold

Classically a solid stores 3R of heat per mole (Dulong-Petit), but measured heat capacities
plunge toward zero in the cold. `debye_heat.py`:

```
$ python examples/debye_heat_demo.py examples/output

      material  Theta_D (K)   C_V @300K   % of 3R
  ------------------------------------------------
          lead          105       24.79     99.4%
        copper          343       23.39     93.8%
     aluminium          428       22.58     90.5%
       diamond         2230        4.13     16.6%
```

Debye treated the lattice vibrations as a gas of quantized phonons with a maximum frequency
set by the interatomic spacing (the Debye temperature Theta_D), giving
`C_V = 9R (T/Theta_D)^3 integral x^4 e^x/(e^x-1)^2 dx`. It has two clean limits: the classical
`3R` for `T >> Theta_D`, and the Debye `T^3` law `(12 pi^4/5) R (T/Theta_D)^3` for
`T << Theta_D`, the fingerprint of freezing out high-frequency phonon modes. Every solid
follows one universal curve in `T/Theta_D`, so a stiff light lattice like diamond
(Theta_D ~ 2230 K) stays "cold" at room temperature -- only ~1/6 of 3R -- while soft heavy
lead has long reached the plateau. The tests verify the high-T Dulong-Petit limit, the low-T
`T^3` law and its scaling, copper's ~94% at room temperature, diamond staying below 3R, and
the monotonic rise.

## The Carnot cycle: the limit on heat engines

No heat engine between reservoirs at T_h and T_c can beat `eta = 1 - T_c/T_h`. `carnot.py`:

```
$ python examples/carnot_demo.py examples/output

                engine   T_hot (K)   T_cold (K)   max eta
  -------------------------------------------------------
            car engine        2000          300     0.850
     steam power plant         800          300     0.625
            geothermal         450          300     0.333
  ocean thermal (OTEC)         298          278     0.067
```

The second law forces some heat to be dumped to the cold reservoir, so no engine reaches
100% -- a steam plant is capped at 62% (real losses cut it to ~40%), and an ocean-thermal
gradient of only 20 K yields 7%. Work plus rejected heat equals the input, and over a full
reversible cycle the total entropy change is zero. Reversed, the cycle is a fridge or heat
pump with `COP_fridge = T_c/(T_h-T_c)` and `COP_heatpump = COP_fridge + 1`, both large for a
small gap -- a heat pump delivers several times the heat of the work it draws, which is why
it beats resistive heating. The tests verify the 62.5% plant efficiency, efficiency below
one, energy conservation, always-rejected heat, the heat-pump-is-fridge-plus-one relation,
and the zero-entropy reversible cycle.

## Adiabatic processes: heat-free compression and expansion

Compress or expand a gas with no time to shed heat and it obeys `P V^gamma = const` and
`T V^(gamma-1) = const`. `adiabatic.py`:

```
$ python examples/adiabatic_demo.py examples/output

     ratio V1/V2    T2 (K)     P2/P1
  ----------------------------------
               2       396       2.6
               5       571       9.5
              10       754      25.1
              22      1033      75.8
              50      1435     239.1
```

Because no heat escapes, the temperature moves with the volume: a diesel engine's 22:1
squeeze reaches ~1000 K and ignites fuel without a spark, while expansion cools the gas
(rising air, released spray). The adiabat is steeper than an isotherm through the same point
precisely because T also changes. Sound waves compress air adiabatically, so the speed of
sound is `c = sqrt(gamma R T / M)` -- Laplace's `sqrt(gamma)` factor that fixed Newton's ~18%
error (343 vs 290 m/s). The tests verify the ~343 m/s air sound speed, the Laplace-over-Newton
ratio, diesel-compression ignition, the `PV^gamma` and `TV^(gamma-1)` invariants, the P/V
temperature consistency, and the positive expansion work.

## The van der Waals gas: a real gas that condenses

Give the ideal gas molecules a finite size and mutual attraction and it can condense.
`van_der_waals.py`:

```
$ python examples/van_der_waals_demo.py examples/output

         gas   T_c (K)   P_c (MPa)   Pc Vc / R Tc
  ------------------------------------------------
      helium       5.2        0.23         0.3750
         CO2     304.0        7.40         0.3750
       water     647.0       22.06         0.3750
```

The correction `(P + a n^2/V^2)(V - nb) = nRT` -- attraction `a` lowering the pressure, size
`b` shrinking the volume -- gives the gas a liquid-vapour transition the ideal law can never
have. Below the critical temperature the isotherm develops an unstable loop where pressure
would rise with volume, and the gas condenses across it. The critical constants follow from
`a` and `b` alone (`T_c = 8a/27Rb`, `P_c = a/27b^2`, `V_c = 3nb`), reproducing CO2's 304 K and
water's 647 K, and the compressibility `Pc Vc / R Tc = 3/8` is universal -- the law of
corresponding states, under which every gas collapses onto one reduced curve. The tests verify
CO2's critical temperature and pressure, the universal 3/8 compressibility, the ideal-gas
large-volume limit, attraction lowering Z, and the reduced critical point.

## The Joule-Thomson effect: cooling by throttling

Push a real gas through a valve at constant enthalpy and its temperature changes.
`joule_thomson.py`:

```
$ python examples/joule_thomson_demo.py examples/output

       gas   T_inv (K)      at 300 K   mu (K/MPa) @300K
  ----------------------------------------------------
       CO2        2052         cools             6.717
        N2         852         cools             2.445
        H2         224         warms            -0.235
        He          35         warms            -1.011
```

The JT coefficient `mu = (dT/dP)_H ~ (1/C_p)(2a/RT - b)` pits attraction (which cools an
expanding gas) against finite molecular size (which warms it). Below the inversion
temperature `T_inv = 2a/Rb = (27/4) T_c` attraction wins, so nitrogen and CO2 cool when
throttled and liquefy by repeated expansion at room temperature -- but hydrogen and helium
have low inversion temperatures and *warm* on throttling, so they must be pre-cooled first
(a real early-liquefaction hazard). An ideal gas has `mu = 0` exactly. The tests verify
nitrogen cooling and hydrogen/helium warming at room temperature, the `(27/4) T_c` inversion
ratio, the sign flip at the inversion temperature, the vanishing ideal-gas coefficient, and
the inversion-temperature ordering.

## Clausius-Clapeyron: vapor pressure and boiling

Along a liquid-vapour coexistence line pressure and temperature are locked together.
`clausius_clapeyron.py`:

```
$ python examples/clausius_clapeyron_demo.py examples/output

          location   altitude (m)   pressure (kPa)   boils at (C)
  --------------------------------------------------------------
         sea level             0           101.3          100.0
            Denver          1609            83.7           94.6
            La Paz          3640            65.7           88.1
    Everest summit          8848            35.3           72.2
```

Integrating `dP/dT = L/(T dV)` with an ideal vapour gives `P(T) = P0 exp(-(L/R)(1/T - 1/T0))`,
a steep exponential -- vapor pressure roughly doubles every ~15 K. Boiling happens where the
vapor pressure equals the ambient pressure, so the thinner air atop Everest lets water boil
at 72 C (too cold to cook an egg) while a pressure cooker raises it to ~121 C. Two measured
`(P, T)` points recover the latent heat, and the same saturation curve governs atmospheric
humidity and cloud formation. The tests verify the 100 C sea-level boiling, the Everest and
pressure-cooker points, the steep exponential rise, the latent-heat recovery, and the
boiling/vapor-pressure inversion.

## The Reynolds number: laminar vs turbulent

One dimensionless ratio decides whether a flow is smooth or chaotic. `reynolds.py`:

```
$ python examples/reynolds_demo.py examples/output

            system   v (m/s)     L (m)          Re        regime
  ----------------------------------------------------------------
         bacterium     3e-05     1e-06     3.0e-05       laminar
  water tap (pipe)         1      0.02     2.0e+04     turbulent
    swimming human       1.5       1.8     2.7e+06     turbulent
        blue whale        10        25     2.5e+08     turbulent
```

`Re = rho v L / mu` weighs inertia against viscosity: below the critical value viscosity
damps disturbances and the flow stays laminar, above it inertia tears it into turbulent
eddies, with pipe flow transitioning near `Re ~ 2300`. It spans thirteen orders of magnitude
-- a bacterium at `Re ~ 1e-5` swims in what feels like honey and cannot coast, while a whale
at `Re ~ 1e8` glides on inertia. Laminar pipe flow obeys Hagen-Poiseuille's `Q ~ r^4 dP` law,
so halving a pipe's radius cuts the flow sixteen-fold. The tests verify the laminar/turbulent
transition, the bacterium and whale Reynolds numbers, the `r^4` Poiseuille scaling, the
kinematic/dynamic form agreement, and the critical-velocity inversion.

## Bernoulli's principle: fast flow, low pressure

Along a streamline `P + 1/2 rho v^2 + rho g h` is constant. `bernoulli.py`:

```
$ python examples/bernoulli_demo.py examples/output

  Pitot airspeed (air):        Torricelli efflux (water):
     50 m/s  ->  1.53 kPa        depth  1 m  ->   4.4 m/s
    250 m/s  -> 38.28 kPa        depth 20 m  ->  19.8 m/s
```

Speeding up a flow drops its static pressure. A narrowing Venturi pipe is fastest and
lowest-pressure at the throat (continuity `A1 v1 = A2 v2` plus Bernoulli), which draws fuel
into a carburettor, reads flow in a meter, and -- with circulation -- helps lift a wing. A
Pitot tube runs it backwards, stopping the flow to convert dynamic pressure into a
measurable stagnation pressure, giving `v = sqrt(2(P_stag - P_static)/rho)`; Torricelli's
`sqrt(2gh)` jet is the same equation with the pressure terms cancelled, equal to free-fall
from that depth. The tests verify faster-flow-lower-pressure, the Pitot round-trip, the
Torricelli/free-fall identity, the Venturi throat velocity, and conservation of the Bernoulli
constant along a streamline.

## Surface tension: why water climbs and beads

A liquid surface costs energy per unit area (`gamma`), so it acts like a stretched skin.
`surface_tension.py`:

```
$ python examples/surface_tension_demo.py examples/output

  Capillary rise (Jurin's law):        Young-Laplace overpressure:
    0.500 mm tube  ->  2.98 cm           1.0 mm droplet  ->  145.6 Pa
    0.001 mm tube  ->  14.9 m            1.0 mm bubble   ->  291.2 Pa
```

In a thin tube surface tension lifts water against gravity by Jurin's law
`h = 2 gamma cos(theta) / (rho g r)` -- a 1 mm bore climbs ~1.5 cm, a 1 micron root pore
tens of metres, which is how sap and groundwater wick upward. A curved surface also holds a
pressure jump `2 gamma/r` across a droplet, or `4 gamma/r` across a soap bubble's two films,
so smaller drops sit at higher pressure and empty into larger ones when connected. Mercury,
whose contact angle exceeds 90 deg, is pushed *down* instead. The tests reproduce water's
~1.5 cm rise in a 1 mm tube, the `1/r` scaling, mercury depression, the bubble-is-twice-a-
droplet relation, and a water strider's weight held on the surface.

## The Ekman spiral: wind, rotation, and the ocean

Steady wind over a rotating ocean drives a current that Coriolis deflects. `ekman.py`:

```
$ python examples/ekman_demo.py examples/output

  depth      speed        direction (from wind)
   -0.0 m   4.30 cm/s      -45 deg   (right of wind, surface)
  -24.5 m   1.96 cm/s      -90 deg
  -97.8 m   0.19 cm/s     +135 deg   (~one Ekman depth, e^-pi weaker)
```

Balancing friction against the Coriolis force, the surface current turns exactly 45 degrees
to the right of the wind (northern hemisphere), and with depth it spirals clockwise while
decaying over the Ekman depth `D = pi sqrt(2 A_z/|f|)`. The classic hodograph -- the tip of
the velocity vector tracing a shrinking spiral -- is the signature of the layer. Summed over
depth, the net *Ekman transport* points 90 degrees to the right of the wind with magnitude
`tau/(rho |f|)`, independent of the eddy viscosity; this cross-wind pumping is what drives
coastal upwelling and helps spin the ocean gyres. The tests verify the 45-degree surface
angle (and its hemisphere flip), the `e^-pi` decay at one Ekman depth, the clockwise
rotation, and the viscosity-independent transport.

## Milankovitch cycles: orbits and the ice ages

Earth's orbit slowly reshapes, redistributing sunlight and pacing the glacial cycles.
`milankovitch.py`:

```
$ python examples/milankovitch_demo.py examples/output

  orbital state                            65N June (W/m^2)
  present day                                    477.8
  low obliquity 22.1 deg (cool summers)          456.5   (-21.4)
  high ecc 0.05, summer at aphelion              446.6   (-31.2)
  high ecc 0.05, summer at perihelion            545.6   (+67.8)
```

From the standard astronomical insolation formula -- declination
`delta = arcsin(sin eps sin lambda)`, distance factor `(1 + e cos nu)^2/(1 - e^2)`, and the
sunrise hour angle `H0 = arccos(-tan phi tan delta)` -- the module computes daily top-of-
atmosphere sunlight anywhere, giving the polar midnight-sun and polar-night limits for free.
The Milankovitch control knob is Northern-Hemisphere high-latitude *summer* insolation (the
famous 65N June value, ~478 W/m^2 today): weak summers let winter snow survive and ice
sheets grow. Obliquity (~41 kyr) sets season strength, and climatic precession `e sin(omega)`
(~23 kyr) sets which season falls at perihelion, all modulated by the ~100 kyr eccentricity
envelope. The tests check the 65N peak, the solstice/equinox declinations, the polar
day/night limits, and the obliquity and precession sensitivities.

## Equipartition: half a kT per degree of freedom

Classical statistical mechanics puts `(1/2) k_B T` into every quadratic degree of freedom.
`equipartition.py`:

```
$ python examples/equipartition_demo.py examples/output

  gas / solid              f   C_V/R   C_P/R   gamma
  monatomic (He, Ar)       3    1.50    2.50   1.667
  diatomic, room T (N2)    5    2.50    3.50   1.400
  solid (Dulong-Petit)     6    3.00    4.00   1.333

  H2 staircase:  20 K -> 1.77 R   300 K -> 2.49 R   10000 K -> 3.47 R
```

Summing the modes gives `C_V = (f/2)R`, `C_P = C_V + R`, and `gamma = (f+2)/f` -- exactly the
5/3 of a monatomic gas, the 7/5 of a diatomic, and the `3R` Dulong-Petit value of a solid.
But equipartition is the *classical, high-temperature* limit: a mode only contributes once
`k_B T` exceeds its energy quantum, so a real diatomic gas climbs a heat-capacity staircase
as it warms -- translation always on, rotation thawing near its `theta_rot ~ 85 K`, and
vibration only near `theta_vib ~ 6000 K`. The module uses the Einstein two-state activation
factor for the freeze-out, and the tests verify the monatomic/diatomic capacities, Mayer's
relation, Dulong-Petit, the N2 rms speed, and the monotonic H2 staircase from 3R/2 through
5R/2 toward 7R/2.

## Osmotic pressure: van't Hoff across a membrane

Dissolved particles push solvent across a semipermeable membrane. `osmosis.py`:

```
$ python examples/osmosis_demo.py examples/output

  solution                          conc   i   Pi (atm)
  blood plasma (~0.30 osmol/L)     0.30 M   1     7.63
  normal saline 0.9%               0.15 M   2     7.63   (isotonic)
  seawater (~0.6 M NaCl-equiv)     0.60 M   2    28.36
```

Van't Hoff's law `Pi = i c R T` is the ideal-gas law with solute particles playing the gas,
so a salt that dissociates into `i` ions pushes `i` times as hard as an equal molarity of
sugar. It reproduces seawater's ~27 atm -- the pressure reverse-osmosis desalination must
overcome to force water back out -- and blood plasma's ~7.6 atm, which is why IV fluids and
0.9% saline are formulated isotonic (get it wrong and red cells swell and burst or shrivel).
Because the pressure is colligative, measuring it weighs macromolecules: the module inverts
`Pi` to a molar mass (osmometry), and also reports concentration, osmolarity, tonicity of one
solution against another, and the minimum reverse-osmosis pressure. The tests check seawater
and blood, the `i`-scaling, the osmometry round-trip (60 kg/mol protein), and the saline
isotonic point.

## Fick diffusion: spreading as the root of time

A random walk carries concentration downhill by Fick's laws. `diffusion.py`:

```
$ python examples/diffusion_demo.py examples/output

  distance          diffusion time      (D = 1e-9 m^2/s)
  10 um (cell)      100.0 ms
  1 mm  (tissue)    1000 s
  1 m   (room)      32 yr

  sigma = sqrt(2 D t):  t=1 s -> 0.045 mm   t=10000 s -> 4.47 mm
```

Fick's first law `J = -D dC/dx` plus conservation gives the diffusion equation
`dC/dt = D d^2C/dx^2`. A point release stays a Gaussian whose rms width grows as
`sqrt(2 D t)` -- the diffusive `sqrt(t)`, never the ballistic `t` of directed motion -- and a
step interface relaxes through an error-function profile (how a doped junction or a quenched
front smooths). Since the time to cross a length scales as `L^2/D`, diffusion is quick across
a cell (~0.1 s) but takes ~30 years across a room, the reason microscopic life leans on it
while large systems need flow. Stokes-Einstein `D = k_B T/(6 pi eta r)` ties the coefficient
to temperature and drag (~2e-10 m^2/s for a nm sphere in water). The tests verify the
`sqrt(t)` spreading, conservation of the released amount, the erfc interface value, the
`L^2/D` scaling, and the Stokes-Einstein size trend.

## The Peclet number: carried vs spreading

Whether transport is dominated by the flow or by diffusion is one ratio. `peclet.py`:

```
$ python examples/peclet_demo.py examples/output

  system                  U (m/s)    L (m)        Pe      regime
  inside a cell           1.0e-07   1.0e-05   1.0e-03    diffusion
  blood in a capillary    5.0e-04   8.0e-06   4.0e+00    advection
  stream / small river    5.0e-01   1.0e+00   5.0e+08    advection

  water: Pr=7  Sc=1000  Le=144      air: Pr=0.70
```

The Peclet number `Pe = U L / D` compares advection to diffusion: below 1 (a cell, a still
cup) a substance spreads faster than it is carried; above 1 (a river, an artery) the flow
sweeps it along in thin plumes and boundary layers. It factors as `Pe = Re*Pr` for heat and
`Re*Sc` for mass, where the Prandtl number `Pr = nu/alpha` (~7 for water, ~0.7 for air) and
Schmidt number `Sc = nu/D` (~1000 for an aqueous solute) are pure fluid properties setting
the relative thickness of the velocity, thermal and concentration boundary layers; their
ratio is the Lewis number `Le = alpha/D`. The crossover `Pe = 1` sits at `L = D/U`. The
tests reproduce water's and air's Prandtl numbers, the aqueous Schmidt number, the
`Le = Sc/Pr` identity, the `Pe = Re*Sc` factoring, and the crossover length.

## Convective heat transfer: Newton's law of cooling

Once you know the Nusselt number, you know how fast a fluid carries heat away. `convection.py`:

```
$ python examples/convection_demo.py examples/output

  1 cm aluminium cube cooling from 100 C in 20 C air:
  regime          h (W/m^2K)    Biot        tau     t to 30 C
  still air                8   0.0002   8.4 min      17.5 min
  breeze / fan            40   0.0010   1.7 min       3.5 min
  forced water          2000   0.0488     2.0 s        4.2 s
```

Convection off a surface obeys Newton's law `q = h (T_s - T_inf)`, and the coefficient `h`
comes from the Nusselt number `Nu = h L / k` -- the ratio of convective to conductive
transport across the boundary layer. Standard correlations supply Nu: Dittus-Boelter
`Nu = 0.023 Re^0.8 Pr^n` for turbulent pipe flow, `Nu = 0.664 Re^0.5 Pr^(1/3)` for a laminar
plate. A lumped object then cools exponentially with time constant `tau = rho c_p V/(h A)`,
valid when the Biot number `Bi = h L/k_solid` stays below ~0.1 (the interior is nearly
isothermal). So an aluminium block sheds heat in an hour in still air but in seconds under
forced water -- same law, hundred-fold `h`. The tests check the Dittus-Boelter and flat-plate
correlations, the `Re^0.8`/`sqrt(Re)` scalings, the lumped/non-lumped Biot split, and the
`1/e`-per-tau exponential decay.

## The Stefan problem: a freezing front as the root of time

A melting or freezing interface moves at a rate set by latent heat. `stefan.py`:

```
$ python examples/stefan_demo.py examples/output

  frost                 St    lambda   ice @ 1 day   ice @ 1 wk
  light frost (-5 C)   0.031   0.125      7.8 cm       20.7 cm
  hard frost (-15 C)   0.094   0.214     13.4 cm       35.6 cm
  arctic (-40 C)       0.251   0.341     21.4 cm       56.7 cm
```

The front position obeys `X(t) = 2 lambda sqrt(alpha t)`, where the latent heat released at
the interface must conduct out through the ice already formed, so the front slows as it
deepens. The growth coefficient `lambda` solves the transcendental Stefan condition
`lambda e^(lambda^2) erf(lambda) = St/sqrt(pi)` (by bisection here), with the Stefan number
`St = c_p dT/L` weighing the sensible heat available against the latent heat needed -- small
`St` (latent-dominated) gives a slow front, `lambda ~ sqrt(St/2)`. This reproduces Stefan's
classic ice result (~10 cm after a day of hard frost) and the `depth^2` time law that makes
thin ice form in hours and the next foot take weeks; the same physics crusts a cooling lava
lake. The tests check the Stefan number, the small-`St` limit, the Stefan condition itself,
the `sqrt(t)` advance, the `depth^2` scaling, and the latent-heat dependence.

## The capillary length: surface tension vs gravity

Why a dewdrop is round and a puddle is flat comes down to one length. `capillary.py`:

```
$ python examples/capillary_demo.py examples/output

  liquid            gamma (N/m)    rho    l_c (mm)
  water                 0.0728     998       2.73
  mercury               0.4870   13534       1.92

  drop / feature      size      Bo       regime
  raindrop          2.00 mm    0.54   round (tension)
  coin of water    10.00 mm   13.44   flat  (gravity)
```

Surface tension pulls a blob toward a sphere; gravity flattens anything taller than the
capillary length `l_c = sqrt(gamma/(rho g))`, about 2.7 mm for water. The Bond (Eotvos)
number `Bo = rho g L^2/gamma = (L/l_c)^2` says which wins -- below 1 drops stay round, above
1 they puddle out to a film no deeper than ~`2 l_c`. A drop moving through another fluid adds
inertia via the Weber number `We = rho v^2 L/gamma` and breaks up once `We` exceeds ~12 (a
2 mm water drop at ~0.66 m/s), the physics behind rain fragmentation and spray atomization,
while a thin jet pinches into drops spaced ~9 radii apart by the Rayleigh-Plateau
instability. The tests reproduce water's and mercury's capillary lengths, the `Bo = (L/l_c)^2`
crossover, the Weber breakup threshold, the `2 l_c` puddle cap, and the Rayleigh-Plateau
spacing.

## The Froude number: racing your own waves

Whether a flow is tranquil or shooting -- and how fast a hull can go -- is one ratio.
`froude.py`:

```
$ python examples/froude_demo.py examples/output

  flow                U (m/s)   h (m)     Fr        regime
  lazy river             0.50   2.000   0.11    subcritical
  below a spillway       6.00   0.200   4.28    supercritical

  hull speed: 7 m day-sailer -> 3.34 m/s (6.5 kn)   Kelvin wedge 19.47 deg
```

A surface disturbance travels at the shallow-water wave speed `sqrt(g h)`, and the Froude
number `Fr = U/sqrt(g h)` compares the flow to it. Below 1 (subcritical, tranquil) ripples
outrun the current and travel upstream; above 1 (supercritical, shooting) the water beats its
own waves and a sudden slowing throws up a **hydraulic jump** -- the turbulent step below a
weir, whose conjugate depth follows Belanger's `h2/h1 = 1/2(sqrt(1+8 Fr1^2)-1)` and conserves
momentum flux. For a ship the hull Froude number `U/sqrt(g L)` governs wave-making drag and
walls a displacement hull near `Fr ~ 0.4` (the `1.34 sqrt(L_ft)` knots rule), while the
Kelvin wake wedge holds a fixed 19.47-degree half-angle at any speed. The tests verify the
wave speed, the regime classification, the critical depth, the hull-speed wall, the Belanger
jump with momentum conservation, and the Kelvin angle.

## The Mach cone: the geometry of going supersonic

When a source outruns its own sound, the geometry is fixed by one relation. `mach_cone.py`:

```
$ python examples/mach_cone_demo.py examples/output

  Mach   cone half-angle   boom lag @ 12 km
   1.0        90.00 deg          0.0 s
   2.0        30.00 deg         35.2 s
   5.0        11.54 deg         39.9 s

  Prandtl-Glauert: M=0.8 -> x1.67 lift    Prandtl-Meyer: M=2 -> nu=26.4 deg
```

The Mach cone's half-angle obeys `sin(mu) = 1/M` -- 90 degrees at Mach 1, tightening to 30 at
Mach 2 and 11.5 at Mach 5 -- and that cone is the shock front heard on the ground as a sonic
boom, landing `H/tan(mu)` behind the overhead point and arriving `H sqrt(M^2-1)/(M c)` after
the aircraft has already passed (approaching the vertical sound-travel time `H/c` as `M`
grows). Below Mach 1 the Prandtl-Glauert factor `1/sqrt(1-M^2)` tracks how thin-airfoil lift
stiffens toward the sound barrier; above it the Prandtl-Meyer function `nu(M)` gives the angle
a supersonic flow turns through in an expansion fan. The tests reproduce the 90/30/11.5-degree
cone angles, the boom offset and delay, the Prandtl-Glauert divergence, and the Prandtl-Meyer
angle (26.4 degrees at Mach 2).

## The de Laval nozzle: making exhaust supersonic

A converging-diverging throat is the only way to push gas past Mach 1. `nozzle.py`:

```
$ python examples/nozzle_demo.py examples/output

  area ratio A_e/A*   exit Mach    P_e/P0
              1.0        1.00      0.5283   (choked throat)
              4.0        2.94      0.0298
             25.0        5.00      0.0019

  chamber 5 MPa / 3000 K, A* = 10 cm^2 -> 3.69 kg/s, exit M 5, exhaust 2241 m/s
```

Subsonic flow accelerates as area shrinks, but a supersonic stream accelerates as area
*grows*, so the flow must reach Mach 1 exactly at a minimum-area throat and then expand
through a widening bell. The isentropic relations give the stagnation ratios from the local
Mach number (`T0/T = 1 + (gamma-1)/2 M^2`, etc.), and the area-Mach relation `A/A*` fixes the
wall shape -- minimal at the throat, larger on both sides. Once the back-pressure ratio drops
below the critical `~0.528` (air), the throat **chokes**: mass flow saturates at
`mdot = A* P0 sqrt(gamma/(R T0)) (2/(gamma+1))^((gamma+1)/(2(gamma-1)))` and the exit Mach
number depends only on the bell's area ratio. The tests reproduce the M=1 isentropic ratios,
the `0.528` choking ratio, the `A/A*` values at Mach 2 and 3, the sub/supersonic inversion of
the area-Mach relation, and the linear scaling of choked flow with chamber pressure and
throat area.

## The Blasius boundary layer

No-slip makes a thin sheared film cling to a plate; Blasius solved its shape exactly.
`blasius.py`:

```
$ python examples/blasius_demo.py examples/output

  x (m)      Re_x    delta (mm)     c_f       (air, U = 10 m/s)
   0.01      6667      0.61      0.00813
   0.10     66667      1.94      0.00257
   0.75    500000      5.30      0.00094   (transition to turbulence)
```

The 99%-thickness grows as `delta = 5.0 x/sqrt(Re_x)` -- millimetres over the front of a wing
-- with the displacement thickness `1.721 x/sqrt(Re_x)` (the outward push on the outer flow)
and momentum thickness `0.664 x/sqrt(Re_x)` tracking it. The wall shear gives a local
skin-friction coefficient `c_f = 0.664/sqrt(Re_x)`, heaviest at the sharp leading edge, and
integrating over a plate of length L gives the drag coefficient `C_D = 1.328/sqrt(Re_L)` --
exactly twice the trailing-edge `c_f`. The layer stays laminar until `Re_x ~ 5e5`, where it
trips to turbulence. The tests verify the `sqrt(x)` growth, the fixed thickness ratios, the
`c_f` and `C_D` laws, the `U^1.5` drag scaling, and the transition distance.

## The Strouhal number: von Karman vortex streets

A blunt body in a steady flow sheds a rhythmic wake. `strouhal.py`:

```
$ python examples/strouhal_demo.py examples/output

  body               d       wind    shed freq
  telephone wire     5 mm    10 m/s   400.0 Hz   (aeolian hum)
  factory chimney    3 m     12 m/s   0.8 Hz

  Roshko: Re=300 -> St 0.197   Re=1e4 -> St 0.212
```

Above a modest Reynolds number a cylinder sheds vortices alternately from each side -- a
staggered von Karman street -- at a frequency `f = St U/d` with the Strouhal number `St ~ 0.2`
nearly constant from `Re ~ 300` to `2e5` (Roshko's `St = 0.212(1 - 21.2/Re)` captures its
slow rise). So the pitch scales linearly with wind speed: a 5 mm wire in a 10 m/s wind sings
at 400 Hz (the aeolian tone), and the wake vortices trail about five diameters apart. When
the shedding frequency crosses a structure's natural frequency the flow **locks in** and the
alternating side-force can build destructive vortex-induced vibration -- the reason tall
chimneys wear helical strakes and why the Tacoma-era lesson still matters. The tests verify
the `f = St U/d` scaling, the Strouhal inversion, the Roshko rise, the aeolian pitch, the
lock-in speed, and the `~5d` vortex spacing.

## Weighing clusters: the virial mass and dark matter

A cluster's own motion weighs it, and the answer launched the dark-matter problem.
`cluster_mass.py`:

```
$ python examples/cluster_mass_demo.py examples/output

  cluster    sigma_los   R (Mpc)    M (Msun)     M/L
  Virgo        700 km/s     1.5      2.6e15       ~850
  Coma        1000 km/s     1.5      5.2e15      ~1050
```

The virial theorem gives `M = alpha sigma^2 R / G` from the velocity dispersion `sigma` and
size `R`; since only the line-of-sight dispersion is measurable, `sigma^2 = 3 sigma_los^2` for
an isotropic system. This is exactly Zwicky's 1933 Coma calculation: galaxies moving at
~1000 km/s across ~1.5 Mpc need a dynamical mass of order `10^15` solar masses -- about a
hundred times the mass of the visible stars. The resulting mass-to-light ratio climbs from a
few for a stellar population to hundreds for a cluster, the first evidence for dark matter,
and the module also reports the escape velocity, the (sub-Hubble) crossing time that confirms
the system is relaxed, and the dark-matter fraction. The tests reproduce Coma's mass and M/L,
the `sigma^2`/`R`/`alpha` scalings, the `3x` line-of-sight factor, and the ~0.95 dark-matter
fraction.

## The Sersic profile: the shape of a galaxy's light

A galaxy's brightness fades from the centre in one regular family. `sersic.py`:

```
$ python examples/sersic_demo.py examples/output

  profile                 b_n    I(0.1 R_e)/I_e   R(90% light)/R_e
  n=1 exponential disk    1.677       4.5              2.32
  n=4 de Vaucouleurs      7.669      28.7              5.55
```

The Sersic law `I(R) = I_e exp{-b_n[(R/R_e)^(1/n)-1]}` describes surface brightness with a
single index `n`: `n=1` is the exponential disk of a spiral (scale length `R_e/1.678`) and
`n=4` the de Vaucouleurs law of a giant elliptical -- a bright cusped core with enormous faint
wings. The constant `b_n ~ 2n - 1/3 + 0.009876/n` is set by the half-light definition, so
exactly half the luminosity sits inside the effective radius `R_e` for every `n`. Integrating
the profile gives the total luminosity in closed form, `L = I_e R_e^2 2 pi n e^{b_n}
Gamma(2n)/b_n^{2n}`, and the enclosed-light fraction follows from the incomplete gamma
function. The tests verify `b_1 ~ 1.678` and `b_4 ~ 7.669`, the `I(R_e)=I_e` normalization,
the exponential `n=1` limit, the half-light property, and that the numeric integral matches
the closed-form luminosity.

## The Grashof number: heat that stirs its own wind

Natural convection is driven by buoyancy alone -- no fan. `grashof.py`:

```
$ python examples/grashof_demo.py examples/output

  height L      Gr         Ra       regime     h (W/m^2K)
   0.05 m    3.6e5      2.6e5     laminar        6.91
   1.00 m    2.9e9      2.1e9     turbulent      3.31
```

The Grashof number `Gr = g beta dT L^3/nu^2` weighs the buoyant drive (a warm surface makes
the fluid lighter, so it rises) against the viscosity that damps it -- the natural-convection
analogue of the Reynolds number. Heat transfer correlates against the Rayleigh number
`Ra = Gr Pr`: for a vertical plate `Nu = 0.59 Ra^(1/4)` laminar and `0.10 Ra^(1/3)` turbulent
past `Ra ~ 1e9`, and `h = Nu k/L` gives the gentle few `W/(m^2 K)` of a radiator warming a
still room -- an order of magnitude below forced convection. The ratio `Gr/Re^2` says which
wins: `>>1` buoyancy (natural), `<<1` forced. The tests reproduce the `L^3`/`dT`/`1/nu^2`
scalings, the `Ra = Gr Pr` relation, the laminar/turbulent branch switch, the few-`W/(m^2 K)`
coefficient, and the natural-vs-forced crossover.

## The Womersley number: why blood flow lags the pulse

A pulsating pressure does not make a pulsating parabola. `womersley.py`:

```
$ python examples/womersley_demo.py examples/output

  vessel        radius   alpha   phase lag   regime
  aorta         11 mm    14.7    86 deg      plug (inertial)
  arteriole     0.15 mm  0.20     0 deg      quasi-steady
```

The Womersley number `alpha = R sqrt(omega/nu)` compares the heartbeat frequency to the rate
viscosity diffuses momentum across a vessel (`alpha = R/delta`, the viscous penetration depth
`delta = sqrt(nu/omega)`). Small `alpha` (< ~1: arterioles, capillaries) is quasi-steady --
the instantaneous profile is the in-phase Poiseuille parabola. Large `alpha` (> ~10: the
aorta at `alpha ~ 15`) has too much core inertia to follow the forcing, so the flow lags the
pressure gradient by up to 90 degrees and flattens into a blunt plug with the shear confined
to a thin oscillating wall layer. The pressure pulse itself travels far faster than the blood,
at the Moens-Korteweg speed `c = sqrt(E h/(2 rho R))` (~6-10 m/s, rising as arteries stiffen).
The tests reproduce the aorta's `alpha ~ 15` and a capillary's `alpha << 1`, the
`R`/`sqrt(omega)` scalings, `alpha = R/delta`, the phase-lag trend to 90 degrees, the
`R^4` Poiseuille flow, and the pulse-wave speed.

## The Marangoni effect: flow along a tension gradient

A gradient in surface tension drags the fluid itself. `marangoni.py`:

```
$ python examples/marangoni_demo.py examples/output

  layer L      Ma        onset?      flow U
  0.05 mm      5.4e2      convects    1.50 m/s

  thin film, Earth   Bo_d = 1.4e-4 -> Marangoni      thick pool, Earth  Bo_d = 1.4 -> buoyancy
```

When surface tension varies along a surface -- from a temperature or composition difference
-- fluid is pulled from low-tension toward high-tension regions and hauls the bulk with it:
the tears of wine climbing a glass, pepper fleeing a soap drop, thermocapillary stirring in a
weld pool. The Marangoni number `Ma = |dgamma/dT| dT L/(mu alpha)` weighs that drive against
viscous and thermal diffusion, and a heated layer breaks into Benard-Marangoni cells above
`Ma ~ 80`. Whether surface tension or buoyancy dominates a heated layer is set by the dynamic
Bond number `Bo_d = Ra/Ma = rho g beta L^2/|dgamma/dT|`: thin films and microgravity are
Marangoni-driven, thick pools on the ground buoyancy-driven. The tests verify the
`dT`/`L`/`1/(mu alpha)` scalings, the `Ma ~ 80` onset, the surface stress direction, the
`L^2` Bond-number scaling, the microgravity limit, and the flow-speed trend.

## Kutta-Joukowski: lift is circulation

A wing flies because the flow around it swirls. `kutta_joukowski.py`:

```
$ python examples/kutta_joukowski_demo.py examples/output

  angle of attack   c_l (thin)          light aircraft: 20 m^2, c_l 0.5, 50 m/s -> 15.3 kN
  2 deg             0.219               tennis topspin: 3000 rpm -> 4.3 N side force
  5 deg             0.548
```

The Kutta-Joukowski theorem gives lift per span `L' = rho U Gamma` -- density times speed
times circulation -- for any 2-D shape. An airfoil sets its own circulation through the Kutta
condition (the flow must leave the sharp trailing edge smoothly), giving the thin-airfoil
lift-slope `c_l = 2 pi alpha` (~0.11 per degree). The same theorem is the **Magnus effect**: a
spinning ball drags a boundary layer around, circulation `Gamma = 2 pi r^2 omega`, and curves
sideways -- the topspin dip, the football bend. Lift is not free: the circulation trails
vortices, so a finite wing pays induced drag `c_di = c_l^2/(pi AR e)` that falls with aspect
ratio, the reason gliders and albatrosses have long thin wings. The tests verify the
`rho U Gamma` lift, the `2 pi` slope, the circulation/coefficient consistency, the `U^2`
scaling, the Magnus force, and the induced-drag `c_l^2`/aspect-ratio trends.

## The Knudsen number: when a gas stops being a fluid

Fluid dynamics assumes a continuum; the Knudsen number says when that holds. `knudsen.py`:

```
$ python examples/knudsen_demo.py examples/output

  system              L        P         Kn        regime
  airliner wing       3 m      1e5 Pa    2e-8       continuum
  MEMS microchannel   1 um     1e5 Pa    7e-2       slip
  nanopore filter     5 nm     1e5 Pa    13         free molecular
```

The mean free path `lambda = k_B T/(sqrt(2) pi d^2 P)` is ~68 nm for air at sea level, and
`Kn = lambda/L` sorts every gas flow: continuum (`Kn < 0.01`, ordinary Navier-Stokes with
no-slip walls), slip (`0.01-0.1`), transitional (`0.1-10`), and free molecular (`Kn > 10`,
molecules fly wall-to-wall). Everything we touch is a perfect fluid because `lambda` is
minuscule -- but shrink `L` to a microchip cooling channel or a nanopore, or thin the air to
orbital altitude, and `Kn` climbs past 1, so the gas slips at walls, thermal creep sets in,
and drag must be computed molecule by molecule. The tests reproduce air's ~68 nm mean free
path, the `T`/`1/P` scalings, the four regime thresholds, the continuum breakdown in vacuum,
the pressure/size inversions, and the ~468 m/s mean molecular speed.

## The Richardson number: shear vs stratification

A stratified fluid resists overturning until shear overwhelms it. `richardson.py`:

```
$ python examples/richardson_demo.py examples/output

  layer                N (rad/s)  shear    Ri      state
  ocean thermocline    0.010      0.005    4.00    layered (stable)
  jet-stream shear     0.012      0.030    0.16    KH billows
```

The gradient Richardson number `Ri = N^2/(du/dz)^2` weighs the buoyant restoring stiffness
`N^2` (the Brunt-Vaisala frequency squared) against the squared velocity shear. The
Miles-Howard theorem guarantees stability wherever `Ri > 1/4`; below that threshold the
Kelvin-Helmholtz instability grows, curling the interface into the "cat's-eye" billows seen
in cloud edges and river surfaces. It sets clear-air turbulence that jolts aircraft, mixing
(or its absence) in the ocean thermocline and atmospheric inversions, and entrainment at a
fog top; the bulk form `Ri_b = g (drho/rho) L/U^2` is the finite-difference version. The tests
verify the `N^2`/shear ratio, the `1/4` threshold, the bulk-Ri density/velocity trends, the
critical shear that brings `Ri` to `1/4`, and the stratification sign from a density gradient.

## The Kolmogorov cascade: turbulence shredding into heat

Turbulence carries energy down a cascade of ever-smaller eddies. `kolmogorov.py`:

```
$ python examples/kolmogorov_demo.py examples/output

  flow               Re       eta        tau_eta    L/eta
  stirred coffee     5e3      0.084 mm   7.1e-3 s   5.9e2
  atmosphere (km)    6.7e8    0.241 mm   3.9e-3 s   4.1e6
```

Big eddies break into smaller ones until viscosity smears the smallest into heat. In the
inertial range between, the statistics depend only on the dissipation rate `epsilon`, giving
Kolmogorov's spectrum `E(k) = C epsilon^(2/3) k^(-5/3)` -- the -5/3 law measured everywhere
from wind tunnels to the ocean to interstellar gas. The cascade ends at the Kolmogorov scale
`eta = (nu^3/epsilon)^(1/4)`, where the eddy Reynolds number drops to 1 (with time
`(nu/epsilon)^(1/2)` and velocity `(nu epsilon)^(1/4)`), and the span `L/eta ~ Re^(3/4)` sets
why a weather-scale flow holds millions of eddy sizes and costs `~Re^(9/4)` grid points to
simulate in 3-D. The tests verify `epsilon = u^3/L`, the microscales and their unit eddy
Reynolds number, the -5/3 spectral slope, the `l^(2/3)` eddy-turnover scaling, and the
`Re^(3/4)` scale separation.

## The Casimir effect: pushed together by empty space

Two plates in a vacuum are drawn together by the quantum zero-point field. `casimir.py`:

```
$ python examples/casimir_demo.py examples/output

  gap d      pressure       force on 1 cm^2
  100 nm     1.30e+01 Pa    1300 uN
  1 um       1.30e-03 Pa    0.13 uN

  1 atmosphere of Casimir pressure at d = 10.6 nm
```

Every electromagnetic mode carries a zero-point energy `hbar omega/2`. Between two conducting
plates only the modes whose wavelengths fit in the gap survive, so the vacuum inside holds
fewer modes than outside and the plates are pressed together by `P = pi^2 hbar c/(240 d^4)`,
with energy per area `-pi^2 hbar c/(720 d^3)`. Predicted by Casimir in 1948 and measured to a
few percent in 1997, the `d^-4` law makes the force utterly negligible at macroscopic gaps
but crushing below ~100 nm -- reaching one atmosphere by ~10 nm -- where it causes stiction in
micro-electromechanical systems. The tests reproduce the ~13 Pa pressure at 100 nm, the
`d^-4` and `d^-3` scalings, the pressure-as-energy-gradient relation, the ~10 nm one-
atmosphere gap, and Casimir's dominance over gravity between thin plates.

## The Hall effect: weighing carriers with a magnet

A current in a magnetic field builds a sideways voltage that reads out the charge carriers.
`hall_effect.py`:

```
$ python examples/hall_effect_demo.py examples/output

  material              n (1/m^3)   carrier      V_H       |R_H|
  copper (electrons)    8.5e28      electrons    0.001 uV  7.34e-11
  n-Si (doped)          1e22        electrons    6.24 mV   6.24e-04
```

The Lorentz force pushes moving charges sideways until the transverse **Hall voltage**
`V_H = I B/(n q t)` balances them. Its magnitude gives the carrier density `n`, and its
*sign* reveals whether the carriers are electrons or positive **holes** -- the result that
classical free-electron theory could not explain and that underlies all semiconductor doping.
The Hall coefficient `R_H = 1/(n q)` packages it, and with the conductivity it separates
density from mobility (`mu = |R_H| sigma`), while the Hall angle `arctan(mu B)` measures how
far the field tilts the current. Sparse-carrier semiconductors give millivolt signals versus
microvolts in a metal. The tests reproduce copper's tiny electron Hall voltage and `|R_H|`,
the `I`/`B`/`1/t` scalings, the density inversion, the electron/hole sign, the mobility, and
the Hall-angle limits.

## Wiedemann-Franz: good conductors of charge and heat

The same electrons carry both currents, so their conductivities are locked. `wiedemann_franz.py`:

```
$ python examples/wiedemann_franz_demo.py examples/output

  metal      sigma (S/m)   kappa pred   kappa meas   L_eff/L
  copper     5.96e7        437          401          0.92
  gold       4.10e7        300          318          1.06
```

Dividing the electronic thermal conductivity by the electrical conductivity leaves only
fundamental constants: `kappa/(sigma T) = L = pi^2 k_B^2/(3 e^2) = 2.44e-8 W ohm/K^2`, the
Lorenz number. Each electron carries a charge `e` and a thermal energy `~k_B T` and the same
scattering limits both currents, so the material-specific mean free path and carrier density
cancel. This lets you read a metal's thermal conductivity off an easy resistance measurement
(copper's ~400 W/(m K) from its `sigma`), and its breakdown is diagnostic -- a Lorenz number
well below `L` signals heat and charge decoupling ("strange metals"), while a huge effective
`L` (an insulator conducting heat by phonons, not electrons) means the electronic law does
not apply. The tests reproduce the Lorenz number, copper's thermal conductivity, the
`sigma`/`T` scalings, the conductivity inversion, and the obey/violate classification for
metals, suppressed-`kappa` cases, and phonon insulators.

## Bragg diffraction: reading a crystal with X-rays

Crystals flash bright reflections where scattered waves add in phase. `bragg.py`:

```
$ python examples/bragg_demo.py examples/output

  (hkl)      d (nm)   1st-order angle   max order    (Cu K-alpha on silicon)
  (1,1,1)    0.3135   14.23 deg         4
  (2,2,0)    0.1920   23.66 deg         2
```

Bragg's law `n lambda = 2 d sin(theta)` says a crystal reflects strongly only when the extra
path between planes spaced `d` apart is a whole number of wavelengths. Because that demands a
wavelength comparable to the ~0.1-0.5 nm atomic spacing, X-rays (and neutrons, electrons) are
the natural probes, and inverting the pattern of spots gives the atomic structure -- the
foundation of crystallography from table salt to DNA to proteins. For a cubic lattice each
Miller plane `(hkl)` has spacing `a/sqrt(h^2+k^2+l^2)` and its own family of angles, while
`sin(theta) <= 1` caps the observable orders at `2d/lambda` (so a wavelength longer than `2d`
diffracts from nothing, and visible light cannot resolve atoms). The tests reproduce the
Si(111) Cu-K-alpha angle, the law itself, the order and spacing trends, the cubic Miller
spacings, the spacing/wavelength inversions, and the maximum-order cutoff.

## The diffraction limit: every aperture's resolution floor

Waves refuse to focus to a point, so every instrument has a resolution floor.
`diffraction_limit.py`:

```
$ python examples/diffraction_limit_demo.py examples/output

  instrument         aperture   resolution      microscope Abbe d = lambda/2NA:
  human eye          2 mm       1.2 arcmin        light  NA 1.4:  196 nm
  Hubble             2.4 m      0.058 arcsec      electron 4 pm:  100 pm (atoms)
```

An aperture of diameter `D` spreads a wave into an Airy disk of angular radius
`theta = 1.22 lambda/D` -- the Rayleigh criterion -- so two point sources closer than that
merge into one. Bigger apertures resolve finer detail (Hubble's 2.4 m reaches ~0.05 arcsec),
long wavelengths demand huge ones (why radio dishes are enormous despite coarse resolution),
and a microscope bottoms out at Abbe's `lambda/(2 NA) ~ 200 nm` for visible light -- which is
why electron microscopes, riding picometre de Broglie wavelengths, image atoms. A diffraction
grating turns the same physics into a spectrometer of resolving power `R = m N`. The tests
reproduce Hubble's and the human eye's resolution, the aperture and wavelength trends, the
~200 nm Abbe limit (and picometre electron limit), the grating orders, and the `m N`
resolving power splitting the sodium doublet.

## Snell's law: bending, trapping, and reflecting light

Light bends when it changes speed at a boundary, and can be trapped entirely. `snell.py`:

```
$ python examples/snell_demo.py examples/output

  medium     n       critical angle   Brewster (from air)
  water      1.333   48.6 deg         53.1 deg
  diamond    2.417   24.4 deg         67.5 deg
```

Snell's law `n1 sin(theta1) = n2 sin(theta2)` bends a ray toward the normal entering a denser
medium and away from it leaving one -- until, past the critical angle `arcsin(n2/n1)`, the
refracted ray is impossible and the light is **totally internally reflected**. That lossless
mirror guides light down an optical fibre for thousands of kilometres and, with diamond's tiny
24-degree critical angle, produces the sparkle of repeated internal bounces. At Brewster's
angle `arctan(n2/n1)` the reflected beam is perfectly polarized (polarizing sunglasses), and a
fibre's numerical aperture `sqrt(n_core^2 - n_clad^2)` sets its acceptance cone. The tests
reproduce the air-water bend, water's 48.6-degree and diamond's 24-degree critical angles, the
TIR onset and refraction raise, Brewster's angle, and the fibre NA and acceptance angle.

## Thin-film interference: bubble colours and lens coatings

A film only nanometres thick paints itself in colour. `thin_film.py`:

```
$ python examples/thin_film_demo.py examples/output

  thickness   2 n t     bright lambda   colour     (soap film, n = 1.33)
  80 nm       213 nm    426 nm          violet
  110 nm      293 nm    585 nm          yellow

  MgF2 AR coating for glass at 550 nm: 100 nm thick, ideal index sqrt(1.52) = 1.233
```

Light reflects off both surfaces of a transparent film and the two waves interfere by the
round-trip optical path `2 n t`, so thickness sets colour -- the sheen of a soap bubble or oil
slick. A half-wave phase flip on the denser-medium reflection decides which colours brighten
or cancel, and drives a soap film black just before it bursts (`2 n t -> 0` is destructive at
every wavelength). Engineered as a quarter-wave layer `t = lambda/(4n)` of index
`sqrt(n_substrate)`, the interference cancels reflection instead -- the anti-glare coating on
every lens (MgF2, ~100 nm). Newton's rings are the same fringes in the air gap under a lens,
with dark-ring radii `sqrt(m lambda R)`. The tests reproduce the 100 nm soap film's green
reflection, the destructive condition, the MgF2 quarter-wave thickness and ideal index, and
the `sqrt(m)` Newton's-ring spacing.

## Malus's law: dialling light down with polarizers

A polarizer passes light by the square of a cosine. `malus.py`:

```
$ python examples/malus_demo.py examples/output

  angle   transmission        crossed (90 deg): 0.000
  45 deg  0.500               + 45-deg middle:  0.125 (I0/8)
  90 deg  0.000               100-step stack:   0.976
```

Malus's law `I = I0 cos^2(theta)` gives the intensity of linearly polarized light through a
polarizer at angle `theta` to its axis. Unpolarized light loses exactly half through any one
polarizer, and two crossed at 90 degrees pass nothing -- yet inserting a third at 45 degrees
between them rescues `I0/8`, restoring light where the outer pair alone gave darkness. A stack
of many slightly rotated polarizers drags the polarization around while passing nearly all the
light (throughput `(cos^2(theta/N))^N -> 1`), an optical quantum Zeno effect, and wave plates
rotate polarization losslessly by retarding one component (`2 pi delta_n t/lambda`: `pi` is a
half-wave plate, `pi/2` a quarter-wave). The tests reproduce the cos^2 values, the
half-through-one and crossed-to-zero results, the `I0/8` three-polarizer rescue, the rotating-
stack throughput, the crossed-polarizer extinction, and the half/quarter-wave retardances.

## Cherenkov radiation: the blue glow of going too fast

A charged particle outrunning light-in-medium sheds a glowing shock cone. `cherenkov.py`:

```
$ python examples/cherenkov_demo.py examples/output

  medium            beta_thr   gamma_thr   max cone
  water (n=1.33)    0.750      1.51        41.4 deg
  aerogel (n=1.05)  0.952      3.28        17.8 deg
```

Light travels at `c/n` in a medium, so a particle with `beta > 1/n` is superluminal there and
drags an electromagnetic shock cone behind it -- the optical analogue of a sonic boom, the
eerie blue of a reactor pool. The cone half-angle obeys `cos(theta) = 1/(n beta)` exactly like
a Mach cone, opening from the threshold toward a maximum `arccos(1/n)` (~41 deg in water) as
`beta -> 1`, and the Frank-Tamm photon yield rises as `sin^2(theta)`. Because the angle
encodes the velocity, ring-imaging Cherenkov detectors read it to identify particles and
neutrino observatories (IceCube, Super-Kamiokande) watch for the faint rings. The tests
reproduce water's 0.75 threshold and 41-degree maximum cone, the threshold Lorentz factor, the
emission test, the cone opening with speed, the velocity-from-cone inversion, and the photon-
yield trend, plus aerogel's high 0.95 threshold.

## The Zeeman effect: splitting lines with a magnetic field

A magnetic field shifts an atom's levels and splits its spectral lines. `zeeman.py`:

```
$ python examples/zeeman_demo.py examples/output

  field    normal split   at 500 nm       Lande g:  2S1/2 = 2.00
  0.1 T    1.40 GHz       1.17 pm                   2P1/2 = 0.67
  1.0 T    14.00 GHz      11.67 pm                  2P3/2 = 1.33
```

Each sublevel shifts by `delta_E = g_J m_J mu_B B`, with the Bohr magneton
`mu_B = e hbar/2 m_e`. The **normal** Zeeman effect (spin cancels, `g = 1`) splits a line into
the clean Lorentz triplet shifted by `mu_B B/h = 14 GHz` per tesla, exactly as classical
physics predicted. The **anomalous** effect uses the Lande g-factor
`1 + [J(J+1)+S(S+1)-L(L+1)]/2J(J+1)` (running 1 for pure orbital to 2 for pure spin), giving
more lines, unevenly spaced -- a puzzle whose resolution required electron spin. Reading the
splitting backwards gives the field (`B = h delta_nu/mu_B`), which is how solar magnetograms
map sunspots. The tests reproduce the Bohr magneton, the 14 GHz/T normal shift, the sodium
D-line g-factors (2/3, 4/3, 2), the sign-flipping sublevel shifts, and the field inversion.

## Rabi oscillations: a two-level atom flopping

A driven two-level system cycles coherently between its states. `rabi.py`:

```
$ python examples/rabi_demo.py examples/output

  pi pulse (X gate): 500 ns    pi/2 pulse: 250 ns   (Omega = 2pi x 1 MHz)

  detuning   gen. Rabi   peak P_e
  0.0 MHz    1.00 MHz    1.000
  1.0 MHz    1.41 MHz    0.500
```

A near-resonant field flops the atom at the Rabi frequency `Omega = dE/hbar`. On resonance
`P_e(t) = sin^2(Omega t/2)` swings the full 0-to-1, so a **pi pulse** inverts the population (a
qubit X gate) and a **pi/2 pulse** builds an equal superposition. Detuned by `delta` the
oscillation speeds up to the generalized Rabi frequency `sqrt(Omega^2 + delta^2)` but only
reaches `Omega^2/(Omega^2 + delta^2)` -- a Lorentzian resonance of width `Omega` (power
broadening). These are the elementary gates of atomic clocks and quantum computers. The tests
verify the ground start, the pi/2pi/pi-2 pulse populations, the `d E/hbar` Rabi frequency, the
generalized Rabi speed-up, the Lorentzian peak (half-max at `delta = Omega`), the
never-fully-inverts-off-resonance bound, and the pulse-time scaling.

## The Franck-Hertz experiment: energy levels in a current

Electrons through mercury vapour reveal quantized atomic energy. `franck_hertz.py`:

```
$ python examples/franck_hertz_demo.py examples/output

  dip 1:  4.9 V     dip 3: 14.7 V     dip spacing = 4.9 V
  dip 2:  9.8 V     dip 4: 19.6 V     emission: 253 nm (UV)
```

Ramping the accelerating voltage, the collected current climbs then drops sharply every
4.9 V. Electrons collide *elastically* with the atoms until they gain the excitation energy,
then dump exactly that quantum in an *inelastic* collision and are left too slow to reach the
collector -- so the current falls. The evenly spaced dips (`V_n = n * E_ex/e`) are direct
proof that atomic energy is quantized, the 1914 confirmation of the Bohr atom, and a fast
electron can excite the atom several times on one crossing. The excited atom relaxes by
emitting a photon at `lambda = h c/E_ex`, mercury's 254 nm ultraviolet line. The tests
reproduce the 4.9 V spacing, the multiple dips and their offset, the excitation count and
residual energy versus voltage, and the 254 nm emission.

## Moseley's law: ordering the elements by X-ray colour

Characteristic X-rays fingerprint an element by its nuclear charge. `moseley.py`:

```
$ python examples/moseley_demo.py examples/output

  element   Z    K-alpha energy   wavelength     line at 6.4 keV -> Z=26 (Fe)
  Cu        29   8.00 keV         0.155 nm       line at 8.0 keV -> Z=29 (Cu)
  Mo        42   17.15 keV        0.072 nm
```

Moseley found the square root of the K-alpha frequency rises linearly with atomic number,
`sqrt(f) = a(Z-1)`, equivalently the K-alpha energy is a screened-hydrogenic
`13.6 (3/4)(Z-1)^2 eV` (the n=2->1 transition seen by a nearly-unscreened charge `Z-1`). This
ordered the elements by nuclear charge rather than atomic weight, exposed the gaps where
technetium and promethium had to sit, and proved `Z` is the true atomic serial number.
Inverting the relation identifies an element from a measured line -- still how XRF guns and
electron microprobes read a sample's composition. The tests reproduce copper's 8 keV K-alpha
and 0.154 nm wavelength, the linear `sqrt(f)`-vs-`Z` law, the `(Z-1)^2` scaling, molybdenum's
17 keV line, the element identification, and the general K-beta/L-series transitions.

## The Stark effect: electric fields on atoms

The electric counterpart of Zeeman splitting. `stark.py`:

```
$ python examples/stark_demo.py examples/output

  n=4: 7 lines  [-4763, -3175, -1588, 0, +1588, +3175, +4763] ueV @ 5 MV/m

  n     binding (eV)   ionizing field
  1     13.61          3.21e10 V/m
  30    0.015          3.97e4 V/m
```

Hydrogen's degenerate levels give the **linear** Stark effect -- shift `(3/2) n k q E a0`,
proportional to the field, fanning level `n` into `2n-1` equally spaced components -- because
the states mix into a permanent electric dipole. Atoms without one shift **quadratically**,
`-1/2 alpha E^2`, always lowering the energy as the field induces a dipole. Crank the field up
and it field-ionizes the atom; since binding energy scales as `1/n^2`, the ionizing field
scales as `1/n^4`, so a Rydberg atom (`n=30`) ionizes in ~40 kV/m versus ground hydrogen's
~3e10 V/m -- the basis of Rydberg-atom field and single-microwave-photon detectors. The tests
verify the linear shift's field/`k` proportionality and symmetric `2n-1` pattern, the negative
`E^2` quadratic shift, the induced dipole, and the `1/n^4` ionization threshold (ground H
~3e10 V/m, Rydberg tiny).

## The Aharonov-Bohm effect: a phase from an untouched field

A field the particle never touches still shifts its quantum phase. `aharonov_bohm.py`:

```
$ python examples/aharonov_bohm_demo.py examples/output

  flux / Phi_0   phase (rad)   fringe shift     Phi_0 = h/e = 4.14e-15 Wb
  0.50           3.142         0.50             h/2e (Cooper pair) = 2.07e-15 Wb
  1.00           6.283         0.00
```

Steer a charged particle around a solenoid whose field is confined entirely inside -- zero on
the path -- and the interference fringes still shift by `delta_phi = q Phi/hbar`, set purely by
the enclosed flux. The particle responds to the vector potential, not the field, proof that
the potentials are physically real in quantum mechanics. The phase is periodic in the flux
quantum `Phi_0 = h/q` (the smaller `h/2e` for Cooper pairs), which quantizes flux through a
superconducting ring and runs SQUID magnetometers: one flux quantum through a 1 mm^2 loop
needs only ~2 nT, letting them sense fields a billion times weaker than Earth's. The tests
reproduce the `h/e` and `h/2e` quanta, the `2 pi`-per-quantum phase, the linear-in-flux phase,
the periodic fringe shift, the Cooper-pair doubling, and the tiny per-quantum field.

## The Josephson junction: a supercurrent that defines the volt

Cooper pairs tunnel a barrier with no voltage, and the effect defines the volt.
`josephson.py`:

```
$ python examples/josephson_demo.py examples/output

  voltage   Josephson freq       Shapiro steps @ 70 GHz:
  100 uV    48.36 GHz            step 1:  144.75 uV
  1000 uV   483.60 GHz           step 2:  289.50 uV
```

The DC Josephson effect is a zero-voltage supercurrent `I = I_c sin(phi)` set only by the
phase difference across the junction; the AC effect winds that phase under a DC voltage so the
current oscillates at `f = 2eV/h = 483.6 GHz` per millivolt -- an exact voltage-to-frequency
link through only `e` and `h`. Irradiate the junction and it locks onto quantized **Shapiro
steps** `V_n = n h f/2e`, each an exact, constants-only voltage; this is how the SI volt is
defined and how quantum voltmeters achieve parts-per-billion accuracy. The Josephson constant
`K_J = 2e/h` and coupling energy `E_J = hbar I_c/2e` (the qubit/oscillator scale) round it out.
The tests verify the `sin(phi)` current and its critical bound, the `K_J` constant, the
483.6 GHz/mV conversion and its inverse, the evenly spaced Shapiro steps, the `2 pi f` phase
rate, and the coupling energy.

## The quantum Hall effect: resistance from pure constants

Cooled and strongly magnetized, a 2D electron gas quantizes its resistance. `quantum_hall.py`:

```
$ python examples/quantum_hall_demo.py examples/output

  filling nu   Hall resistance      B (T)   Landau spacing   degeneracy
  1            25812.8 ohm          10      1.16 meV         2.4e15 /m^2
  2            12906.4 ohm
```

The Hall resistance locks onto plateaus `R_xy = R_K/nu` with the von Klitzing constant
`R_K = h/e^2 = 25812.807 ohm` -- set by fundamental constants alone, independent of the
sample. The electron energies collapse into Landau levels spaced by `hbar eB/m`, each holding
`eB/h` states per unit area; when `nu` of them are filled the bulk is insulating and `nu`
chiral edge channels each carry a conductance quantum `e^2/h`, giving the quantized `R_xy`.
Reproducible to parts per billion in any device, it now defines the SI ohm -- the resistance
counterpart to the Josephson volt. The tests reproduce `R_K` and the `R_K/nu` plateaus, the
conductance-resistance inverse and its quantization, the cyclotron frequency and Landau
spacing/degeneracy, and the filling factor from density and field.

## BCS superconductivity: the gap that kills resistance

Cooper pairs and an energy gap explain zero resistance. `bcs.py`:

```
$ python examples/bcs_demo.py examples/output

  material    T_c (K)   gap (meV)   pair-break      lambda 0.3 -> Tc 9.2 K
  aluminium   1.2       0.183       0.09 THz        lambda 0.5 -> Tc 35 K
  niobium     9.3       1.414       0.68 THz
```

Below `T_c` a phonon-mediated attraction binds electrons into Cooper pairs that condense into
a single coherent state carrying current without resistance. The theory's core is an energy
gap `Delta` at the Fermi surface -- breaking a pair costs `2 Delta`, so nothing scatters the
condensate. BCS predicts the universal ratio `2 Delta(0)/(k_B T_c) = 3.53` for every
weak-coupling superconductor, a gap closing as `sqrt(1-T/Tc)` toward `T_c`, and
`k_B T_c = 1.13 hbar wD exp(-1/lambda)` from the Debye energy and electron-phonon coupling.
Because `wD ~ 1/sqrt(M)`, `T_c ~ M^(-1/2)` -- the isotope effect that proved phonons do the
pairing. The tests reproduce the 3.53 ratio and Al/Nb gaps, the gap-Tc inversion, the
`sqrt(1-T/Tc)` closing, the exponential `T_c(lambda)`, the isotope shift, and the sub-THz
pair-breaking frequency.

## London & Meissner: expelling the magnetic field

A superconductor pushes field out of itself, and the length scale sets its type.
`london.py`:

```
$ python examples/london_demo.py examples/output

  material      lambda (nm)  xi (nm)   kappa    type
  aluminium     16           1600      0.01     I
  Nb-Ti         300          4         75.0     II    (vortex flux 2.07e-15 Wb)
```

The Meissner effect is active field expulsion, not just frozen flux, and it is why a magnet
levitates over a superconductor. The London equations give the field decaying into the surface
as `B(x) = B0 exp(-x/lambda_L)` over the penetration depth
`lambda_L = sqrt(m/(mu0 n_s q^2))` -- tens of nanometres, so thin films never fully expel the
field. The Ginzburg-Landau parameter `kappa = lambda_L/xi` (penetration depth over coherence
length) classifies the material: type I (`kappa < 1/sqrt2`) expels field until it abruptly
goes normal, while type II (`kappa > 1/sqrt2`) admits field as quantized flux vortices (each
`h/2e`) between two critical fields -- which is how Nb-Ti and high-Tc magnets tolerate the
enormous fields of MRI and fusion. The tests reproduce the ~20-60 nm penetration depth, the
`1/sqrt(n)` scaling, the `exp(-x/lambda)` screening, the type I/II boundary at `1/sqrt2`, the
vortex flux quantum, and the critical-field ratio growing with kappa.

## Mean-field ferromagnetism: order from disorder

Spins align spontaneously below the Curie point. `ising_mft.py`:

```
$ python examples/ising_mft_demo.py examples/output

  T / T_c   magnetization   phase          chi above T_c:
  0.50      0.958           ferromagnet    1.05 T_c -> 3.33
  1.00      0.010           paramagnet     2.00 T_c -> 0.17
```

Weiss mean-field theory of the Ising model solves the self-consistent
`m = tanh((z J m + B)/T)`, where each spin feels the average alignment of its `z` neighbours.
Above the Curie temperature `T_c = z J` the only zero-field solution is `m = 0` (a disordered
paramagnet); below it a nonzero magnetization appears spontaneously -- symmetry breaking with
no applied field. Near `T_c` it vanishes as `(1 - T/Tc)^(1/2)`, the mean-field critical
exponent `beta = 1/2`, while the zero-field susceptibility diverges as the Curie-Weiss law
`chi ~ 1/(T - T_c)` -- the signatures of a second-order phase transition. The tests reproduce
`T_c = z J`, the `m=0` paramagnet above and `m->1` ferromagnet below, the field-induced
magnetization, the `beta=1/2` scaling, and the diverging Curie-Weiss susceptibility.

## Percolation: the sudden onset of connectivity

Random occupation crosses a sharp connectivity threshold. `percolation.py`:

```
$ python examples/percolation_demo.py examples/output

  p      spans?   largest cluster       p_c ~ 0.5927 (2D square site)
  0.50   0 %      3 %
  0.59   60 %     30 %
  0.65   100 %    50 %
```

Occupy each lattice site with probability `p` and ask whether a connected path spans the
system. Below the threshold `p_c` (~0.59 for a 2D square lattice) the occupied sites are
isolated islands; above it a single cluster abruptly spans the whole lattice, and the
largest-cluster fraction jumps from near zero to order one -- a geometric phase transition.
Clusters are found by union-find and spanning is tested top-to-bottom (a small seeded LCG
keeps runs reproducible without `random`). The same threshold governs forest fires, oil in
porous rock, disease on a contact network, and current through a random resistor grid. The
tests verify the occupation fraction, empty/full/stripe spanning, the largest-cluster growth
with `p`, and the spanning probability sharpening from near 0 below `p_c` to near 1 above.

## Polya's random walk: home, or lost forever?

Whether a lattice walk returns to the origin depends only on dimension. `polya.py`:

```
$ python examples/polya_demo.py examples/output

  dim   return prob   escape prob   exp. visits   class
  2     1.0000        0.0000        inf           recurrent
  3     0.3405        0.6595        1.516         transient
```

Polya proved a random walk is recurrent (returns with probability 1, visiting every site
infinitely often) in 1D and 2D, but transient (escapes to infinity with nonzero probability)
in 3D and above -- a ~0.34 chance of ever returning in 3D. The knife-edge is exactly two
dimensions, because the probability of being back at the origin decays as `n^(-d/2)`, whose
sum over time diverges (recurrent) only for `d <= 2`. "A drunk man finds his way home, but a
drunk bird may get lost forever." The tests reproduce the certain return in 1D/2D and ~0.34
in 3D, the infinite/finite expected visits, the escape-plus-return sum, the `sqrt(n)` rms
displacement, and a direct simulation (97% of 1D walks return, far fewer in 3D).

## Langevin paramagnetism: moments vs thermal chaos

Independent magnetic moments align against thermal randomization. `langevin_para.py`:

```
$ python examples/langevin_para_demo.py examples/output

  B (T)   T (K)     x        aligned      Curie: chi = C/T
  1       300       0.011    0.4 %        T=300 -> 1.7e-25
  10      4         8.4      88.1 %       T=1   -> 5.2e-23
```

Each moment feels an alignment energy `-mu.B` while temperature randomizes it, and averaging
over the Boltzmann distribution gives the Langevin function `m/mu = L(x) = coth(x) - 1/x` with
`x = mu B/(k_B T)`. Weak field or high temperature is the linear regime `L(x) ~ x/3`, so the
susceptibility follows Curie's law `chi = n mu^2/(3 k_B T) ~ 1/T` -- the paramagnet's
fingerprint; strong field or low temperature saturates every moment at `L = 1`. The tests
reproduce `L(0)=0`, the `x/3` slope and `L(1) ~ 0.313`, saturation to 1, the `1/T` Curie
susceptibility and constant, the small-field linear response, and the field-for-saturation
inversion.

## Buffon's needle: estimating pi by dropping sticks

pi falls out of a purely mechanical experiment. `buffon.py`:

```
$ python examples/buffon_demo.py examples/output

  drops N     crossings   pi estimate   error
  1000        645         3.10078       0.041
  1000000     636951      3.13996       0.0016
```

Drop a needle of length `L <= d` on a floor ruled with lines spacing `d` apart: it crosses a
line with probability `2 L/(pi d)`. So counting crossings estimates pi -- `pi ~ 2 L N/(d C)`
for `N` drops and `C` crossings -- the first problem in geometric probability (Buffon, 1777),
and pi emerges with no measurement of pi entering anywhere, purely from the random position and
angle. The convergence is the slow Monte Carlo `1/sqrt(N)`: 1% accuracy needs ~10000 drops and
0.1% about a million. The tests reproduce the `2/pi` crossing probability, the pi-estimate
inversion, the `1/error^2` needle count, the `1/sqrt(N)` error scaling, and a seeded
simulation converging to pi within a few percent.

## Metropolis Monte Carlo: sampling the Ising transition

Systems too large to sum exactly are sampled by biased random walks. `metropolis.py`:

```
$ python examples/metropolis_demo.py examples/output

  T (J/kB)   avg |m|   phase          exact Onsager T_c = 2.269 J/k_B
  1.00       0.999     ordered
  2.60       0.543     ordered
  3.50       0.084     disordered
```

The Metropolis rule accepts a proposed spin flip with probability `min(1, exp(-dE/T))` --
always downhill, Boltzmann-weighted uphill -- which satisfies detailed balance, so the chain
of configurations it visits is distributed as `exp(-E/T)` and thermal averages are just walk
averages. Run on the 2D Ising ferromagnet it reproduces the true phase transition mean-field
theory only approximates: spins order below the exact Onsager `T_c = 2J/(k_B ln(1+sqrt2)) ~
2.269 J/k_B` and disorder above it, with genuine critical fluctuations (domains at every scale
near `T_c`). The tests verify the accept-all-downhill / Boltzmann-uphill rule, the ground-state
energy and magnetization, the exact `T_c`, and a full run ordered below and disordered above
the transition.

## The logistic map: period doubling into chaos

One line iterates its way into chaos. `logistic_map.py`:

```
$ python examples/logistic_map_demo.py examples/output

  r      period   Lyapunov   regime       Feigenbaum delta = 4.669
  2.50   1        -0.693     period-1
  3.50   4        -0.873     period-4
  3.90   many     +0.492     chaos
```

The map `x' = r x(1-x)` settles to one value for `1 < r < 3`, splits into a 2-cycle at `r=3`,
then 4, 8, 16, ... in a period-doubling cascade accumulating at `r ~ 3.5699`, the onset of
chaos -- aperiodic and sensitive to initial conditions -- broken by periodic windows (the
famous period-3 near 3.83). The bifurcation spacings shrink by the universal **Feigenbaum
constant** `4.669`, the same for any smooth unimodal map, and the Lyapunov exponent
`<ln|r(1-2x)|>` is negative in the periodic regime and positive in chaos (reaching `ln 2` at
`r=4`). The tests reproduce the `1-1/r` fixed point and its stability, the period 1/2/4 and
period-3 window, the negative-to-positive Lyapunov crossover, and the Feigenbaum constant.

## The Henon map: a strange attractor in two lines

A two-line map holds a full fractal attractor. `henon.py`:

```
$ python examples/henon_demo.py examples/output

  fixed points: (+0.631, +0.189), (-1.131, -0.339)
  area contraction |det J| = |b| = 0.30       Lyapunov = 0.421 (chaos)
```

Henon's map `x' = 1 - a x^2 + y, y' = b x` at `a=1.4, b=0.3` is the canonical low-dimensional
strange attractor: the iterates never settle and never repeat, tracing a fractal of nested
arcs that resolve, on zoom, into a Cantor set of ever-finer strands. It is dissipative -- each
step contracts area by the constant Jacobian `|det J| = |b|` -- yet chaotic, with a positive
largest Lyapunov exponent (~0.42 nat/iteration) so nearby points separate exponentially. The
squeeze-in-area-while-stretching-along-the-unstable-direction is what collapses a blob onto a
fractal of dimension ~1.26. The tests reproduce the map formula, the bounded spanning
attractor, the `|b|` area contraction, the two fixed points (verified as fixed), the ~0.42
classic Lyapunov exponent, and a non-chaotic small-`a` case.

## The Lorenz attractor: the butterfly effect

Three equations for toy weather that founded chaos theory. `lorenz.py`:

```
$ python examples/lorenz_demo.py examples/output

  fixed points: origin, (+/-8.485, +/-8.485, 27)
  div F = -13.667 (dissipative)   Lyapunov = 0.917   error 10x every ~2.5 t
```

Lorenz's system `dx/dt = sigma(y-x)`, `dy/dt = x(rho-z)-y`, `dz/dt = xy - beta z` at
`sigma=10, beta=8/3, rho=28` never repeats: the trajectory winds around two spiral lobes,
jumping between them unpredictably, tracing the butterfly-shaped strange attractor -- the
first for a continuous flow. It is dissipative (phase-space volume shrinks at the constant
rate `-(sigma+1+beta)`, collapsing onto the zero-volume fractal) yet chaotic: two starts a
millionth apart diverge to opposite wings, with a positive largest Lyapunov exponent (~0.9,
so prediction error grows tenfold every ~2.5 time units) -- the reason weather is
unforecastable beyond ~two weeks. Integrated here with RK4. The tests verify the fixed points
(origin plus the two convection points for `rho>1`), the `-(sigma+1+beta)` volume contraction,
the bounded two-lobe attractor, the ~0.9 Lyapunov exponent, and the decay to the origin for
`rho<1`.

## The double pendulum: chaos you can hang from a nail

The simplest chaotic machine: two rods and gravity. `double_pendulum.py`:

```
$ python examples/double_pendulum_demo.py examples/output

  energy conserved to ~1e-6 over a fine short run
  two starts 1e-4 rad apart: separation 1e-4 -> 3e-3 over a few seconds (and growing)
```

Hang one pendulum off another and the coupled equations of motion (from the Lagrangian) have
no closed-form solution -- they must be integrated numerically (RK4). The motion is fully
deterministic yet chaotic: above a threshold energy two nearly identical releases diverge
exponentially, flailing into completely different configurations within seconds -- sensitive
dependence on initial conditions in a system you can build from string. Two things stay clean
and the tests check them: the total mechanical energy is conserved along the exact motion (so
a well-resolved RK4 run holds it to a part in a million; the slow long-time drift is the
expected non-symplectic numerical effect), and a hair's difference in the start grows far
beyond its initial size. The tests verify the straight-down equilibrium, energy conservation
at a fine step, the bob positions, and the chaotic divergence outpacing a near-linear start.

## The Mandelbrot set: infinite detail from z -> z^2 + c

One quadratic rule generates an infinitely detailed fractal. `mandelbrot.py`:

```
$ python examples/mandelbrot_demo.py examples/output

  c = 0.00  -> IN SET      c = 0.35 -> escapes at 8
  c = 0.25  -> IN SET      c = 1.00 -> escapes at 3
  the set fills ~24% of its bounding box
```

Iterating `z -> z^2 + c` from `z=0`, the Mandelbrot set is the `c` whose orbit stays bounded.
Because `|z|>2` guarantees escape, the escape time colours the classic images and paints the
filaments just outside the set. The main cardioid (the body) and the period-2 bulb at `c=-1`
have exact tests that skip iteration; the boundary carries tiny copies of the whole set at
every scale, and the real slice `[-2, 1/4]` maps onto the logistic map's period-doubling route
to chaos. The tests reproduce the `z^2+c` step, the in-set interior points (0, -1, -0.5) and
fast-escaping exterior ones, escape time growing toward the boundary, the exact cardioid and
period-2-bulb membership, and the partial escaped-fraction of the bounding box.

## The Van der Pol oscillator: a self-sustaining rhythm

A pendulum that pumps itself onto a fixed rhythm. `van_der_pol.py`:

```
$ python examples/van_der_pol_demo.py examples/output

  mu    amplitude   period    character
  0.3   2.00        6.4       near-sinusoidal
  5.0   2.02        11.6      relaxation
```

The equation `x'' - mu(1-x^2)x' + x = 0` has damping that is *negative* at small amplitude
(feeding energy in) and *positive* at large (draining it), so from almost any start the system
settles onto the same closed loop -- a **limit cycle** of amplitude ~2 that forgets its initial
conditions, unlike a linear oscillator. It is the canonical model of a self-regulated rhythm:
heartbeats, firing neurons, the circadian clock, a bowed string. Small `mu` gives
near-sinusoidal oscillation at frequency ~1; large `mu` gives relaxation oscillation -- long
slow charges broken by fast jumps, period ~`1.614 mu`. Integrated with RK4. The tests verify
the vector field, amplitude growth from a tiny start, the ~2 limit-cycle amplitude for several
`mu`, convergence of large and small starts onto the same cycle, the `mu>0` self-sustaining
condition (and decay to rest for `mu<0`), and the small-`mu` ~2 pi period growing at large mu.

## The Duffing oscillator: a spring that bends the rules

A cubic spring term produces bistability and a bent resonance. `duffing.py`:

```
$ python examples/duffing_demo.py examples/output

  alpha  beta  regime        minima          backbone (hardening):
  -1.0   1.0   double-well   -1.00, +1.00    A=0 -> 1.00, A=1.5 -> 1.64
  1.0   -0.5   softening     0
```

The equation `x'' + delta x' + alpha x + beta x^3 = gamma cos(omega t)` adds a cubic stiffness
to a spring. Its unforced potential `V = (1/2) alpha x^2 + (1/4) beta x^4` is a single well for
`alpha>0` but a **double well** for `alpha<0, beta>0` (minima at `+/-sqrt(-alpha/beta)`) -- a
buckled beam or bistable switch, into one side of which a damped mass rolls. Driven, the
resonance peak bends with amplitude along the backbone `sqrt(alpha + 3/4 beta A^2)`, so the
response is multi-valued and jumps between branches as the drive frequency sweeps (hysteresis);
push harder and the forced Duffing goes chaotic. The tests verify the potential and its
`+/-1` double-well minima, the hardening/softening/double-well/linear regimes, the vector
field, a damped mass settling into a well, and the backbone shifting up (hardening) or down
(softening) with amplitude.

## The Kuramoto model: oscillators falling into sync

Coupled oscillators pull themselves into step -- a phase transition to collective order.
`kuramoto.py`:

```
$ python examples/kuramoto_demo.py examples/output

  K     order r   state             (N=50, freq spread +/-1)
  0.5   0.13      incoherent
  2.0   0.95      synchronized
```

Each oscillator obeys `dtheta_i/dt = omega_i + (K/N) sum sin(theta_j - theta_i)`, and the
synchrony is the order parameter `r = |(1/N) sum e^{i theta}|` -- 0 for scattered phases, 1
when all lock. There is a sharp critical coupling `K_c = 2/(pi g(0))` (`2 gamma` for a
Lorentzian spread, `sigma sqrt(8/pi)` for a Gaussian): below it the oscillators drift
independently and `r ~ 0`; above it a synchronized cluster spontaneously forms and `r` climbs
toward 1. It is the canonical model of emergent order -- fireflies flashing in unison,
pacemaker cells, applause locking into rhythm, power-grid generators. The tests verify `r=1`
for aligned and `r=0` for evenly spread / antiphase, the natural-frequency drift at zero
coupling, the Lorentzian and Gaussian `K_c`, and a simulation going from incoherent (weak `K`)
to synchronized (strong `K`).

## The Abelian sandpile: self-organized criticality

A pile of sand tunes itself to the edge of chaos. `sandpile.py`:

```
$ python examples/sandpile_demo.py examples/output

  central stack: 1000 grains -> 18182 topples
  8000 random drops: mean avalanche 16.8, max 528 (heavy-tailed)
```

Grains drop one at a time; a site with 4 grains topples, sending one to each of its 4
neighbours, and grains falling off the edge are lost. A single grain can trigger a chain
reaction from nothing to a system-spanning cascade. With no parameter tuning the pile drives
itself to a critical state where avalanche sizes follow a power law -- the founding model of
**self-organized criticality** (Bak-Tang-Wiesenfeld, 1987), a candidate for the scale-free
statistics of earthquakes, forest fires, and neuronal avalanches. The toppling is Abelian: the
final stable configuration and total topple count are independent of the order in which
unstable sites relax. The tests verify the single-topple rule, grain conservation (minus edge
loss), the Abelian order-independence, a symmetric relaxed central stack, and heavy-tailed
avalanche sizes (max far above the mean).

## Elementary cellular automata: complexity from 8 bits

The simplest computers: a row of cells updated from their neighbours. `cellular_automaton.py`:

```
$ python examples/cellular_automaton_demo.py examples/output

  rule    table       class / note
  90      01011010    Sierpinski fractal (XOR of neighbours)
  30      00011110    chaotic (once Mathematica's RNG)
  110     01101110    complex, Turing-complete
```

Each cell's next value depends only on itself and its two neighbours, so with 3 binary inputs
there are `2^8 = 256` rules, numbered by their 8-bit output tables (Wolfram's convention). From
that trivial definition come all four Wolfram classes: die to uniform (rule 0), fractal stripes
(rule 90 draws the Sierpinski triangle by `left XOR right`), chaos indistinguishable from random
(rule 30), and localized interacting structures (rule 110, proven Turing-complete -- a universal
computer from an 8-bit lookup). The tests verify the rule-table decoding, rule 90 as XOR and its
power-of-two Sierpinski populations, rule 0/255 dying/filling, rule 30's aperiodic non-dying
evolution, and rule 110 sustaining bounded activity.

## Conway's Game of Life: a universe from four rules

The most famous cellular automaton, complexity from B3/S23. `game_of_life.py`:

```
$ python examples/game_of_life_demo.py examples/output

  block   : still life (unchanged)
  blinker : period-2 oscillator
  glider  : spaceship, moves (1,1) every 4 generations
```

On a 2D grid a live cell survives with 2 or 3 live neighbours and a dead cell is born with
exactly 3 -- the whole rule. From it emerge **still lifes** (the block never changes),
**oscillators** (the blinker flips between horizontal and vertical every 2 steps), and
**spaceships** (the glider keeps its shape while translating one cell diagonally every 4
generations). Because gliders can be generated by guns and collided to build logic gates, Life
is Turing-complete -- a computer can run inside it. The tests verify neighbour counting, the
loneliness/crowding/birth rules, the stable block, the period-2 blinker (horizontal to
vertical), and the glider translating diagonally by (1,1) per 4 generations (and 4 cells in 16).

## Reaction-diffusion: Turing's spots and stripes

Patterns from chemistry alone, no blueprint. `reaction_diffusion.py`:

```
$ python examples/reaction_diffusion_demo.py examples/output

  step    total v   contrast
  0       ~16       small (seed)
  1500    grows     0.3+ (spots formed)
```

Turing's 1952 idea: a slowly diffusing self-promoting **activator** and a fast-diffusing
**inhibitor** make a uniform mixture unstable, so it settles into standing spots or stripes
with no template. The Gray-Scott model `du/dt = Du lap(u) - u v^2 + F(1-u)`,
`dv/dt = Dv lap(v) + u v^2 - (F+k)v` (with `Du > Dv` and the autocatalytic `u v^2` reaction)
produces spots, stripes, mazes, self-replicating blobs, or travelling waves depending on the
feed `F` and kill `k` rates -- a working model of morphogenesis (leopard spots, seashell
ridges). Integrated with an explicit Euler step and a 5-point Laplacian. The tests verify the
uniform-field zero Laplacian, the bare-substrate steady state (no pattern), a seed introducing
autocatalyst, a bounded pattern with real spatial contrast growing from the seed, and shape
preservation.

## Boids: flocking from three local rules

A flock with no leader. `boids.py`:

```
$ python examples/boids_demo.py examples/output

      step    polarization    mean spacing
         0           0.146            7.90
        30           0.282            9.04
        80           0.337            8.28
       200           0.727            8.54
```

Craig Reynolds' 1987 model: each "boid" steers by three rules that use only its nearby
neighbours -- **separation** (avoid crowding), **alignment** (match average heading), and
**cohesion** (steer toward average position). Sum those into an acceleration each step, limit
speed, and a swarm of identical agents produces lifelike murmurations from a random scatter:
the alignment order parameter (polarization, the magnitude of the mean unit heading) climbs
from near 0 toward 1 while separation keeps the birds apart, all bottom-up with no flock-level
rule anywhere. It is the model behind starling murmurations, sardine bait balls, and the
crowds and creatures in films and games. The tests verify the random start is unaligned,
parallel headings give polarization 1 and antiparallel 0, separation pushes crowded boids
apart, alignment steers toward the neighbour heading, the step caps speed and wraps the
toroidal box, and a full evolve raises polarization while holding the birds spaced.

## Diffusion-limited aggregation: a fractal from random walkers

Sparse, feathery growth from nothing but a random walk. `dla.py`:

```
$ python examples/dla_demo.py examples/output

   particles    R_gyration   fractal D
         200         10.81       1.530
         600         21.94       1.587
        1200         31.81       1.609
```

Release a particle far from a seed and let it random-walk (Brownian motion) until it touches
the cluster, where it sticks forever; repeat. What grows (Witten & Sander, 1981) is not a blob
but a self-similar fractal, because a wanderer almost always brushes an outer tip long before
it can diffuse into an interior fjord -- the tips screen the inside and grow faster still. The
mass inside radius `r` scales as `N(r) ~ r^D` with `D ~ 1.71` in the plane, not 2: the branches
leave most of the plane empty. This module grows an on-lattice cluster (walkers launched on a
circle, killed if they stray too far, stuck when they step next to an occupied cell) and
measures `D` by the mass-radius scaling and the radius of gyration. The tests verify the
cluster is connected, deterministic per seed, that a filled disc scales as `D ~ 2` while the
DLA cluster comes out sparser near `1.71`, and the mass-within-radius and bounds helpers. The
same instability draws mineral dendrites, electrodeposits, viscous fingers, lightning, and soot.

## Benford's law: why leading digits are not uniform

The first digit of real-world numbers is not one-in-nine. `benford.py`:

```
$ python examples/benford_demo.py examples/output

   digit   Benford   Fibonacci   uniform
       1     0.301       0.301     0.177
       2     0.176       0.177     0.159
       ...
       9     0.046       0.045     0.092

  Fibonacci:  chi2 =   0.17   -> follows Benford
  uniform:    chi2 = 168.7    -> does NOT
```

Count the leading digit of river areas, physical constants, stock prices, or file sizes and 1
leads about 30% of the time while 9 barely reaches 4.6%, following `P(d) = log10(1 + 1/d)`
(Newcomb 1881, Benford 1938). The reason is scale invariance: a quantity spanning many orders
of magnitude is effectively uniform in its logarithm, and a uniform log-mantissa maps to this
logarithmic digit law -- the only distribution invariant under a change of units. This module
extracts leading digits (from a fractional-log10 so arbitrarily large integers like `300!`
never overflow), gives the first- and general-position digit probabilities, and scores a
dataset with a chi-square statistic and total-variation distance. The tests confirm the
probabilities sum to 1 and decrease from 1 to 9, that Fibonacci numbers, powers of 2, and
factorials pass while a uniform-digit control fails, and the huge-integer and edge cases.
Departures from the law flag fabricated ledgers and election tallies, which is why forensic
auditors test for it.

## The coupon collector: how long to collect the whole set

The wait for the last few items dominates. `coupon_collector.py`:

```
$ python examples/coupon_collector_demo.py examples/output

      n      E[T]   n ln n+..   std dev  sim mean
      6     14.70       14.71      6.24     14.59
     50    224.96      224.96     61.95    224.84
    100    518.74      518.74    125.82    517.80
```

Each box holds one of n equally likely coupons; how many boxes to collect all n? Once you hold
k, a fresh box is new with probability `(n-k)/n`, so the next new coupon is a geometric wait of
mean `n/(n-k)`, and summing gives `E[T] = n H_n ~ n ln n`. The last coupon alone averages n
boxes -- the tail dominates. This module gives the expected time, its variance (bounded by
`pi^2 n^2/6`), the completion probability `P(T <= t)` by inclusion-exclusion, the generalized
"collect any k of n" and "collect m copies of each" expectations, and the concentration tail
bound `P(T > n ln n + c n) <= e^{-c}`. The tests check the exact small cases (`E[T]` for n=1,2,
a die), recover `E[T]` by summing the CDF, verify the tail bound holds, and match a seeded
Monte-Carlo simulation to within a few percent. The same `n ln n` law sets cache warmup, random
test-coverage, and how many samples see every category at least once.

## The secretary problem: optimal stopping and the 1/e rule

The best strategy is to look, then leap. `secretary.py`:

```
$ python examples/secretary_demo.py examples/output

       n  r* (cutoff)  look frac   P(win)      sim
      10            4     0.3000   0.3987   0.4045
     100           38     0.3700   0.3710   0.3768
    1000          369     0.3680   0.3682   0.3643
```

Interview n candidates one at a time in random order, accept or reject on the spot, and you
only care about the single best. The optimal policy is a cutoff: reject the first r-1 (a look
phase), then take the first later candidate who beats all seen so far. The win probability
`P(r) = (r-1)/n * sum_{i=r}^{n} 1/(i-1)` is maximized near `r ~ n/e`, and as n grows both the
optimal look-fraction and the win probability tend to `1/e ~ 0.368` -- look at 37% of the
field, then leap at the next record, and you land the very best about 37% of the time no matter
how large n is. This module gives the exact win probability for any cutoff, the optimal cutoff,
the 1/e asymptotics, and the expected number of candidates seen, all checked against exact
small cases (n=3 gives r=2 and P=1/2) and a seeded Monte-Carlo run. The same optimal-stopping
law governs flat-hunting, parking, and online auctions.

## The birthday problem: coincidences are more common than they feel

Twenty-three people, better-than-even odds. `birthday.py`:

```
$ python examples/birthday_demo.py examples/output

   people k     exact   Poisson      sim
         10    0.1169    0.1160   0.1222
         23    0.5073    0.5000   0.5147
         57    0.9901    0.9874   0.9880
         70    0.9992    0.9987   0.9992
```

How many people before two share a birthday with better-than-even odds? Only 23, because k
people make `k(k-1)/2` pairs and it is the pair count, growing like `k^2`, that drives
collisions. The probability all k are distinct is `prod (365-i)/365`, so a collision passes 1/2
at 23 and 99.9% by 70. In general a collision becomes likely once `k ~ 1.177 sqrt(d)` -- a
square-root law that sizes hash tables and UUID spaces and sets the birthday attack: a b-bit
hash collides after `~2^(b/2)` tries, not `2^b`, which is why collision resistance needs twice
the bits of preimage resistance. This module gives the exact collision and distinct
probabilities, the smallest group for a target probability, the median and expected
first-collision counts, the Poisson approximation, and the hash-attack cost, all checked
against exact values (23, 57, 70; the pigeonhole certainty at 366) and a seeded Monte-Carlo run.

## Gambler's ruin: the walk that ends at a wall

A hair of edge decides everything. `gamblers_ruin.py`:

```
$ python examples/gamblers_ruin_demo.py examples/output

       p     i      ruin  sim ruin    duration   sim dur
    0.50    50    0.5000    0.5098      2500.0    2464.0
    0.49    50    0.8808    0.8818      1904.1    1867.9
    0.60    50    0.0000    0.0000       250.0     249.0
```

Start with i dollars, bet $1 a round with win probability p, and stop at broke (0) or a target
N -- a random walk with two absorbing walls. In a fair game the ruin chance is exactly `1 - i/N`
(your stake as a fraction of the table) and the game lasts `i(N-i)` rounds. Shift the odds a
hair to p=0.49 and, starting at the halfway mark, the ruin chance leaps from 50% to 88%;
against an infinitely rich house any `p <= 1/2` is ruin with certainty -- the origin of "the
house always wins". This module gives the exact ruin and reach-target probabilities and the
expected duration for fair and biased games (`(r^i-r^N)/(1-r^N)` with `r=(1-p)/p`), the
infinite-house limit `(q/p)^i`, and a seeded Monte-Carlo sampler. The tests verify the fair
`1-i/N` and `i(N-i)` laws, the biased small cases, that the finite ruin approaches the
infinite-house limit as N grows, and agreement with simulation. The same absorbing-walk math
models allele fixation in a finite population and sequential hypothesis tests.

## Parrondo's paradox: two losing games that together win

Losing plus losing equals winning. `parrondo.py`:

```
$ python examples/parrondo_demo.py examples/output

        game   drift/round   sim drift
     A alone      -0.01000    -0.00681
     B alone      -0.00870    -0.00763
   50/50 mix       0.01570     0.01846
```

Two gambling games, each a sure loser played alone, can be alternated -- or chosen at random
each round -- to make your capital drift UP. Game A is a slightly-losing flat coin; game B
flips a terrible coin (win 1/10) whenever your capital is a multiple of 3 and a good one (win
3/4) otherwise, and loses because the walk gets stuck visiting the bad state too often. Mixing
in game A reshuffles that state occupancy so the good coin comes up more, and the combined
drift -- the stationary average of `(2p_s - 1)` over the capital-mod-3 Markov chain -- turns
positive. This module computes each game's per-state win probabilities, the exact stationary
distribution by power iteration, the resulting drift, and a seeded Monte-Carlo trajectory. The
tests verify the stationary distribution is a genuine fixed point, that game B over-visits the
bad state, that A and B lose while the mix wins (even at eps=0, where both are exactly fair),
and agreement with simulation. The same flashing-ratchet mechanism drives molecular motors,
pumping directed motion from noise.

## The Galton board: coin flips converge to a bell curve

The central limit theorem, made physical. `galton.py`:

```
$ python examples/galton_demo.py examples/output

  binomial -> Gaussian distance shrinks like 1/sqrt(rows):
    rows =    8:  TV distance = 0.0139
    rows =   32:  TV distance = 0.0037
    rows =  128:  TV distance = 0.0009
```

Galton's bean machine is a board of n staggered peg rows; a bead bounces left or right with
probability 1/2 at each row and lands in one of n+1 slots. Its slot is the number of
right-bounces in n coin flips, so the slot occupancy is the binomial `C(n,k) p^k (1-p)^(n-k)`
-- and because it is a sum of n independent steps, the central limit theorem makes the
histogram converge to a Gaussian of mean `np` and variance `np(1-p)`. This module gives the
exact binomial slot probabilities and their moments, the CLT normal approximation, the
total-variation distance between them (shrinking like `1/sqrt(n)`), and a seeded Monte-Carlo
bead drop. The tests verify the fair board is symmetric with center slot `C(10,5)/2^10`, the
moments are `np` and `np(1-p)`, the normal peak height is `1/(sigma sqrt(2 pi))`, the distance
follows the `1/sqrt(n)` rate, and the simulated histogram matches the binomial. No bead is
steered, yet thousands pile into a smooth bell curve; bias the pegs and the pile slides to np.

## The Monty Hall problem: why switching doors wins

Switch, and the odds double. `monty_hall.py`:

```
$ python examples/monty_hall_demo.py examples/output

                strategy   theory      sim
          stay (classic)    0.333    0.333
        switch (classic)    0.667    0.667
    switch (random host)    0.500    0.498
```

A car hides behind one of three doors; you pick one, and the host -- who knows where the car is
-- opens a different door revealing a goat and offers a switch. Switching wins 2/3, staying only
1/3, because your first pick is right just 1/3 of the time and the host's forced reveal
concentrates the whole remaining 2/3 onto the other closed door. The paradox lives in the host's
knowledge: if he opened a door blindly and it happened to show a goat, switching would only be
1/2. Generalized to N doors with the host opening all but one other, switching wins `(N-1)/N` --
99% at 100 doors. This module gives the exact stay and switch probabilities for the classic and
generalized game and the informed-vs-random-host comparison, all checked against a seeded
Monte-Carlo play.

## Bayes and the base-rate fallacy: a positive test can still mean healthy

Ninety-nine percent accurate, and still probably wrong. `bayes_test.py`:

```
$ python examples/bayes_test_demo.py examples/output

  prevalence           = 0.100%
  sensitivity          = 99%   specificity = 99%
  P(sick | positive)   = 9.0%   <- not 99%!   (Monte-Carlo: 9.1%)
  P(sick | 2 positives)= 90.7%
  break-even prevalence= 1.00%  (a positive is 50-50 here)
```

A disease affects 1 in 1000; a 99%-sensitive, 99%-specific test comes back positive. The chance
you are actually sick is about 9%, not 99%, because the rare base rate makes false positives
swamp the true ones: among 100,000 people, 99 true positives are buried under 999 false ones. A
positive becomes more-likely-than-not only once the prevalence passes `(1-spec)/(sens+1-spec) =
1%`, and two independent positives push the posterior above 90%. This module computes the
positive and negative predictive values via Bayes' theorem, the likelihood ratios, the odds-form
posterior for chaining retests, and the even-odds prevalence, all checked against exact
hand-counts (99/1098) and a seeded Monte-Carlo cohort. This base-rate fallacy underlies medical
screening, spam filters, and security profiling.

## Shannon entropy and Huffman coding: the limit of lossless compression

The fewest bits per symbol, and the code that reaches them. `shannon.py`:

```
$ python examples/shannon_demo.py examples/output

   symbol    prob    codeword  bits
        e    0.27          10     2
        t    0.20          00     2
        z    0.05        0110     4

  entropy  H = 2.6318 bits/symbol
  Huffman  L = 2.6500 bits/symbol   (fixed-length would need 3)
  bound: H <= L < H+1   OK    efficiency H/L = 99.3%
```

Shannon's entropy `H = -sum p log2 p` is the fewest bits, on average, to record one symbol, and
his source-coding theorem says no lossless code can beat it: the average codeword length obeys
`L >= H`, with a code always existing at `L < H+1`. Entropy is maximal (`log2 n`) for a uniform
source and zero when one symbol is certain. Huffman's algorithm repeatedly merges the two
least-likely symbols to build the optimal prefix code, landing in the `[H, H+1)` band and
obeying the Kraft inequality `sum 2^-len <= 1`. This module computes entropy, builds the Huffman
code, measures its efficiency, and round-trips encode/decode losslessly. The tests verify the
entropy of known sources, the Shannon bound, that a dyadic source is coded at 100% efficiency,
and the prefix-free/Kraft/round-trip properties -- the Huffman stage that sits inside ZIP, JPEG,
and MP3.

## The Kelly criterion: how much to bet to grow fastest

Bet the edge over the odds, no more. `kelly.py`:

```
$ python examples/kelly_demo.py examples/output

    fraction    growth   vs f*
        0.10    0.0150    0.75
        0.20    0.0201    1.00  <- Kelly
        0.40   -0.0024   -0.12
        0.50   -0.0340   -1.69
```

Given an edge -- a bet paying b-to-1 that wins with probability p above break-even -- Kelly's
rule maximizes long-run compound growth by staking a fixed fraction `f* = p - (1-p)/b`, the edge
over the odds. The growth rate `g(f) = p ln(1+bf) + (1-p) ln(1-f)` is concave and peaks at f*;
betting less is safe but slow, and betting past `2f*` drives the growth rate negative -- you go
broke despite a winning edge. This module gives the optimal fraction, the growth rate at any
fraction, the break-even (zero-growth) fraction, the doubling time, and a seeded Monte-Carlo of
the compounding bankroll. The tests verify f* for even-money and b-to-1 odds, that a grid search
confirms f* is the growth argmax, that overbetting loses, that half-Kelly keeps ~3/4 of the
growth, and that simulation grows fastest exactly at f*. Kelly derived it from Shannon's channel
capacity, and it sizes positions in quantitative finance.

## Hamming codes: correcting a bit error from the syndrome

Detect and fix a flipped bit with a handful of parity checks. `hamming.py`:

```
$ python examples/hamming_demo.py examples/output

   flipped pos            received  syndrome  corrected?
             1[1, 1, 1, 0, 0, 1, 1]         1         yes
             5[0, 1, 1, 0, 1, 1, 1]         5         yes
             7[0, 1, 1, 0, 0, 1, 0]         7         yes
```

Bits flip on a noisy channel, and a plain parity bit can only detect a single error. Hamming's
1950 codes correct it: parity bits at the power-of-two positions cover each data bit with a
unique combination of checks, so the pattern of failed checks -- the syndrome -- reads out, in
binary, the exact position of the flipped bit. The classic Hamming(7,4) carries 4 data bits in
7; `Hamming(2^m-1, 2^m-1-m)` needs only m parity bits, so overhead shrinks as blocks grow.
Every Hamming code has minimum distance 3, and one extra overall parity bit gives SECDED
(single-error-correct, double-error-detect), the scheme in ECC memory. This module encodes,
computes the syndrome, decodes with correction, and does SECDED -- and the tests prove
correctness *exhaustively*, correcting every single-bit error in every codeword of the (7,4) and
(15,11) codes.

## RSA: public-key cryptography from the hardness of factoring

Two strangers, a secret channel, no shared key. `rsa.py`:

```
$ python examples/rsa_demo.py examples/output

  message   m = 42424242
  encrypt   c = m^e mod n = 8483966817973133348...
  decrypt   m'= c^d mod n = 42424242   -> round-trip OK
  sign      s = m^d mod n = 2694638152342582462...
  verify    s^e mod n = m ? True   (tampered: False)
```

RSA rests on one asymmetry: multiplying two large primes into `n = pq` is easy, but factoring n
back apart is astronomically hard. Pick primes p, q, take `phi = (p-1)(q-1)`, a public exponent
e coprime to phi, and the private `d = e^-1 mod phi`; then `c = m^e mod n` encrypts and
`m = c^d mod n` decrypts, undoing each other because `ed = 1 mod phi` by Euler's theorem. The
same keys sign (encrypt a message with the private key, verify with the public). This pure-stdlib
educational reference implements Miller-Rabin primality (which catches the Carmichael numbers
that fool the Fermat test), the extended Euclidean modular inverse, fast square-and-multiply
exponentiation, key generation, and full encrypt/decrypt and sign/verify round-trips, including
byte-string chunking. The number theory behind TLS and SSH -- real deployments add OAEP/PSS
padding, but the core is exactly this.

## Diffie-Hellman: agreeing on a secret in the open

A shared key over a wiretapped line, no prior meeting. `diffie_hellman.py`:

```
$ python examples/diffie_hellman_demo.py examples/output

              secret    sends g^secret mod p                  computes
     Alice     12345                   30204          B^a mod p = 9050
       Bob     54321                   20462          A^b mod p = 9050

  shared secret g^(ab) mod p = 9050   (both match: True)
```

Fix a prime p and a generator g. Alice sends `g^a mod p`, Bob sends `g^b mod p`, and each raises
what they received to their own secret, both landing on `g^(ab) mod p` while the wire carried
only `g^a` and `g^b`. Stealing the secret means recovering a from `g^a mod p` -- the
discrete-logarithm problem, easy to state and (for large p) astronomically hard. This module
generates safe-prime parameters (`p = 2q+1`), finds a generator, runs the exchange, and includes
a baby-step/giant-step discrete-log solver whose `sqrt(p)` cost dwarfs the parties' `log(p)` work
-- the gap that keeps the secret safe. The tests verify generators cover every residue, both
sides derive the same secret, the parameters are genuine safe primes, and BSGS cracks the small
exchange (feasible only because p is tiny). Without authentication a man-in-the-middle can
intercept, which is why real protocols sign the exchanged values.

## CRC: catching transmission errors with polynomial division

The checksum on every Ethernet frame and ZIP file. `crc.py`:

```
$ python examples/crc_demo.py examples/output

  check string '123456789':
             CRC-8 = 0xF4
      CRC-16-CCITT = 0x29B1
            CRC-32 = 0xCBF43926
```

A cyclic redundancy check treats the message as a polynomial over GF(2) (arithmetic mod 2,
addition = XOR), divides by a fixed generator, and appends the remainder; the receiver divides
again and a nonzero remainder flags corruption. A degree-r generator guarantees detection of
every single-bit error, every burst shorter than r+1 bits, and misses a random error only with
probability `~2^-r` -- 1 in 4 billion for CRC-32, computed with nothing but shifts and XORs.
This module does bit-at-a-time polynomial division for CRC-8, CRC-16-CCITT, and CRC-32,
appends and verifies frames, and demonstrates the detection guarantees. The tests reproduce the
published `123456789` check values, match `zlib.crc32` exactly across many inputs, and confirm
every single-bit error and short burst is caught. The error-*detection* companion to the Hamming
code, which *corrects*.

## LZ77: compression by pointing back at what you have seen

The engine inside ZIP, gzip, and PNG. `lz77.py`:

```
$ python examples/lz77_demo.py examples/output

                        data  bytes   ratio
                    'ab' x 50    100   20.00
           English text x 20    400   14.81
         pseudo-random bytes    400    1.00
```

Scanning the data, whenever the next bytes have appeared recently LZ77 emits a back-reference --
a `(distance, length)` pair meaning "copy length bytes from distance back" -- instead of
repeating them; only genuinely new bytes are stored literally. A sliding window holds the recent
history, and the decompressor replays the tokens from its own growing output, so an overlapping
copy (length > distance) expands a whole run from a single token. Repetitive data compresses
enormously while random data cannot shrink at all -- Shannon's entropy limit showing through.
This module implements the sliding-window encoder and the replay decoder, measures the
compression ratio and token counts, and guarantees a lossless round-trip (verified on empty,
single-byte, text, binary, and long-run inputs). LZ77 plus Huffman together are DEFLATE, the
heart of gzip.

## Bloom filters: membership in a fraction of the space

Ask "have I seen this?" without storing what you have seen. `bloom.py`:

```
$ python examples/bloom_demo.py examples/output

  1000 items at 1% target: 9586 bits (9.6 bits/item), 7 hash functions
  after inserting 1000: false negatives = 0 (guaranteed 0)
  observed false-positive rate = 0.0103,  theory = 0.0100
```

A Bloom filter answers membership with a bit array and a few hash functions, in a tiny fraction
of the memory the items would take. The trade is one-sided: it may say "possibly present" for
something never added (a false positive) but NEVER says "absent" for something you did add -- no
false negatives. Add an item by setting its k bits; test by checking all k are set. After n
items in m bits the false-positive rate is `(1 - e^{-kn/m})^k`, minimized at `k = (m/n) ln 2`,
needing only `~1.44 log2(1/p)` bits per item regardless of item size. This module uses double
hashing (two FNV-1a hashes combined to simulate k) and computes the optimal parameters; the
tests confirm zero false negatives, an observed false-positive rate matching theory, that the
optimal k is a genuine minimum, and membership for ints, bytes, and tuples. Web caches, spell
checkers, and databases use one as a fast pre-filter.

## HyperLogLog: counting distinct items in kilobytes

Estimate a billion distinct items in ~1.5 KB. `hyperloglog.py`:

```
$ python examples/hyperloglog_demo.py examples/output

   true distinct    estimate   rel error
             100          98     -0.0183
           10000        9760     -0.0240
          500000      496746     -0.0065
```

Exact distinct-counting means storing every item; HyperLogLog estimates the cardinality to a
percent or two in fixed tiny memory. Hash each item; the longest run of leading zeros seen hints
at the count (k zeros suggests ~2^k items). To tame the noise, the first p bits pick one of
`m = 2^p` registers each holding its max leading-zero rank, and the harmonic mean across
registers gives `E = alpha_m m^2 / sum 2^-M[j]` with relative error `~1.04/sqrt(m)`. Small counts
get a linear-counting correction, and two sketches merge by register-wise max, so counts are
trivially distributed. This module uses a splitmix64-finalized hash (plain FNV-1a leaves too many
registers unused), the small/large-range corrections, and the merge; the tests confirm the
estimate lands within a few standard errors across four orders of magnitude, duplicates do not
inflate the count, and a merge recovers the union. Redis, Presto, and BigQuery all ship it.

## Fenwick trees: running sums that update in log time

Prefix sums and point updates, both O(log n). `fenwick.py`:

```
$ python examples/fenwick_demo.py examples/output

  values : [3, 1, 4, 1, 5, 9, 2, 6]
  prefix sums : [3, 4, 8, 9, 14, 23, 25, 31]
  total = 31,  range_sum[2,6) = 19
```

A plain array gives instant updates but O(n) prefix sums; a prefix-sum array the reverse.
Fenwick's binary indexed tree does both in `O(log n)` using the binary structure of the indices:
node i stores the partial sum of the range ending at i whose length is its lowest set bit
`i & -i`. A prefix sum strips the low bit each step (`i -= i & -i`), an update adds it
(`i += i & -i`), so each walk touches only one node per set bit -- in a single array of n
integers, no pointers. Range sums come by subtraction, and because cumulative sums are monotone
you can binary-search the tree for the smallest index whose prefix reaches a target, an
`O(log n)` "select" for weighted sampling and rank queries. This module implements build, update,
prefix/range sums, set, and the cumulative search; every operation is cross-checked against a
brute-force array over thousands of mixed updates and queries. It powers competitive-programming
range queries, database index statistics, and streaming quantiles.

## Union-Find: connectivity in near-constant time, and Kruskal's MST

Merge groups and query connectivity in effectively O(1). `union_find.py`:

```
$ python examples/union_find_demo.py examples/output

  Kruskal's MST (add cheapest edge that joins two components):
   weight      edge
        5   0--3
        5   2--4
  total spanning-tree weight = 39  (6 edges for 7 nodes)
```

Given a stream of "these two are connected" facts, Union-Find answers whether any two items
share a group, running m operations in `O(m alpha(n))` time -- the inverse Ackermann `alpha(n)`
is at most 4 for any conceivable n, so effectively constant. Each set is a tree with a
representative root; union by rank hangs the shorter tree under the taller, and path compression
repoints every node visited during a find straight at the root, keeping trees flat. A cycle is
exactly two endpoints already in the same set, which makes Union-Find the whole of Kruskal's
minimum-spanning-tree algorithm. This module implements find with path compression, union by
rank, component counting, and Kruskal's MST; the tests cross-check component counts against a
brute-force flood fill and confirm the MST is minimal, cycle-free, and spanning. It also drives
image segmentation, percolation, and account-merging.

## Dijkstra's algorithm: shortest paths, greedily

Cheapest route from a source to every node. `dijkstra.py`:

```
$ python examples/dijkstra_demo.py examples/output

   node  distance  Bellman-Ford
      0         0             0
      4         7             7
  shortest 0 -> 4: [0, 2, 1, 3, 4], cost 7
```

Given nonnegative edge weights, Dijkstra keeps a tentative distance to every node, repeatedly
settles the unsettled node with the smallest distance, and relaxes its edges; once settled, a
node's distance is final -- which holds precisely because no nonnegative detour can improve a
shorter path. With a binary-heap priority queue it runs in `O((V+E) log V)`. This module builds
a weighted graph, runs Dijkstra (returning distances and a predecessor tree for path
reconstruction), reconstructs the actual path, and includes a from-scratch binary min-heap (no
`heapq`) and a Bellman-Ford implementation used to verify every distance. The tests check the
heap sorts, the classic shortest path, unreachable nodes, directed one-way edges, and agreement
with Bellman-Ford across random graphs. Add a goal heuristic and it becomes A*, the workhorse of
map and game routing -- shown here solving a grid maze.

## k-d trees: fast nearest-neighbour search in space

Which point is closest? in O(log n), not O(n). `kdtree.py`:

```
$ python examples/kdtree_demo.py examples/output

  400 points in 2D, tree height 9 (~log2 n = 8.6)
  nearest to (50.0, 50.0): 49.2,47.8  (brute force agrees: True)
  5 nearest agree with brute force: True
  within radius 15: 32 points  (brute force agrees: True)
```

A k-d tree organizes points by recursive median splitting -- the root splits on x, the next
level on y, then z, cycling axes -- so a nearest-neighbour query descends to its leaf and then
unwinds, crossing a splitting plane into the far subtree only when that hyper-rectangle could
hold something nearer. Whole branches are pruned, giving `O(log n)` typical queries; the same
descent-and-prune serves k-nearest-neighbours and radius search. This module builds a balanced
tree by recursive median splitting and does exact nearest, k-nearest, and radius queries, each
cross-checked against a brute-force scan in 2D and 3D (including duplicate points and
query-on-a-point edge cases). It powers k-NN classification, particle neighbour lists, and
map/geographic search.

## Boyer-Moore: the string search that skips ahead

The find in your editor, matching right-to-left. `boyer_moore.py`:

```
$ python examples/boyer_moore_demo.py examples/output

                 pattern  BM comps    naive  speedup
                     fox      1079     2818     2.6x
                lazy dog       781     3113     4.0x
    the quick brown fox       1559     4352     2.8x
```

The naive substring scan checks every position in `O(n*m)`; Boyer-Moore matches the pattern
right-to-left and, on a mismatch, jumps forward by more than one position -- often nearly the
whole pattern. The bad-character rule shifts so the last occurrence of the mismatched text
character lines up (or past it entirely if absent), and the good-suffix rule realigns an
already-matched suffix without undoing confirmed matches; taking the larger shift keeps the
search safe and sublinear -- most characters are never examined. This module builds both tables
and finds first, all, and overlapping occurrences, proven correct *exhaustively* against a naive
search over 5000 random text/pattern pairs plus repetitive and DNA-like cases.

## A* search: Dijkstra with a sense of direction

Optimal paths, but explore far less. `astar.py`:

```
$ python examples/astar_demo.py examples/output

              path cost  nodes expanded
          A*         49             300
    Dijkstra         49             360
  same optimal cost: True   A* expanded 17% fewer nodes
```

Dijkstra explores outward in every direction equally; A* keeps the optimality guarantee but adds
a heuristic `h(n)` estimating the distance still to go, ordering its frontier by
`f(n) = g(n) + h(n)` -- known cost plus the guess ahead -- so it pushes toward the goal. If the
heuristic never overestimates the true remaining cost (admissible), the path is still guaranteed
optimal; with `h = 0` it becomes Dijkstra exactly. On a grid the Manhattan distance is
admissible for 4-directional movement, octile for 8-directional. This module runs A* on a
weighted grid with obstacles, returns the path and expanded-node set, and the tests verify --
across 39 random grids -- that A* finds the same optimal cost as Dijkstra while expanding no
more nodes. It is the standard for game and robot navigation.

## Topological sort: ordering tasks so prerequisites come first

The order to build, install, or schedule. `toposort.py`:

```
$ python examples/toposort_demo.py examples/output

  Kahn:  fetch -> compile -> link -> test -> assets -> package -> deploy
  critical path: design -> backend -> integrate -> qa -> release  =  22 days
```

A directed acyclic graph encodes dependencies -- an edge `u -> v` means u must precede v -- and
a topological order arranges the nodes so every edge points forward. It exists exactly when the
graph has no cycle. Kahn's algorithm repeatedly emits a node with no remaining incoming edges
(BFS on in-degrees); the DFS method reverses finish-times and detects a cycle as a back-edge to
a node still on the recursion stack. With a duration on each task, the longest path through the
DAG is the critical path -- the minimum time to finish everything (PERT scheduling). This module
implements both sorts, cycle detection, and the critical path, and the tests verify -- over
hundreds of random DAGs -- that every order respects all edges, that cycles are caught, and that
the critical path is correct. It runs package managers, build systems, and spreadsheet
recompute.

## Levenshtein edit distance: how far apart are two strings

The measure behind "did you mean...?". `levenshtein.py`:

```
$ python examples/levenshtein_demo.py examples/output

  'kitten' -> 'sitting' : distance 3, similarity 0.57
  spell-check 'recieve': receive (Levenshtein 2, Damerau 1 -- one adjacent swap)
```

The edit distance is the fewest single-character edits -- insert, delete, substitute -- that
turn one string into another. Dynamic programming fills a table where `d[i][j]` is the distance
between the first i and first j characters, each cell the cheapest of a match, substitution,
insertion, or deletion, in `O(mn)`; backtracing the choices recovers the actual alignment, not
just the count. The distance is a true metric -- symmetric, zero only for equal strings, and
triangle-inequality-respecting. This module computes the distance, a memory-lean two-row variant
(`O(min(m,n))` space), the alignment operations (verified to reproduce the target with exactly
`distance` edits), a normalized similarity ratio, and the Damerau variant that counts an
adjacent-character transposition as one edit -- the commonest typo. The tests check known
distances, the metric axioms over 500 random triples, and the alignment on 300 random pairs. It
powers spell-checkers, fuzzy search, diff tools, and DNA sequence alignment.

## The 0/1 knapsack: packing the most value under a weight limit

Maximize value under a hard cap. `knapsack.py`:

```
$ python examples/knapsack_demo.py examples/output

        item  weight  value
      camera       2      6  <- take
    gold bar       5     12  <- take
       phone       1      5  <- take
  optimal value 23 using 8 kg  (brute force agrees: True)
```

Items each have a weight and a value; which subset maximizes value without exceeding a capacity
W, taking each item whole? Brute force checks 2^n subsets, but dynamic programming solves it in
pseudo-polynomial `O(nW)`: `best[i][w]` is the most value from the first i items within capacity
w, each cell the better of skipping item i or taking it (freeing `w - weight_i`). Backtracing
recovers which items to take; a rolling array (iterating capacity downward) cuts memory to
`O(W)`. The related subset-sum (hit an exact target) and unbounded knapsack (unlimited copies,
iterate capacity upward) are the same table. This module solves all four, reconstructs the chosen
items, and is verified *exhaustively* against a brute-force subset search over 1000 random
knapsacks and 500 subset-sums. It models budget allocation, cargo loading, and portfolio
selection under a hard cap.

## Longest common subsequence: the engine behind diff

The longest shared in-order thread. `lcs.py`:

```
$ python examples/lcs_demo.py examples/output

  a = ABCBDAB,  b = BDCAB
  LCS = 'BCAB' (length 4), indel edit distance 4
  diff: keep lines, - deletions, + insertions -> reproduces v2
```

A subsequence keeps some elements in order but may skip others; the LCS of two sequences is the
longest ordering appearing in both. It is what `diff`, git, and `patch` are built on -- the
complement of the LCS is exactly the lines to add or delete, so a bigger LCS means a smaller
diff. The dynamic program fills `L[i][j] = L[i-1][j-1]+1` on a match, else `max(L[i-1][j],
L[i][j-1])`, in `O(mn)`; backtracing recovers an actual longest subsequence, and turning the walk
into keep/delete/insert steps yields the diff. The LCS length also gives the insert/delete edit
distance `m + n - 2*LCS`. This module computes the length, one subsequence, the diff edit-script
(verified to reproduce the target), and the indel distance, all checked *exhaustively* against a
brute-force subsequence search over 1000 random pairs. It runs version control and bioinformatics
sequence comparison.

## Quickselect: the k-th smallest without sorting

Order statistics in O(n). `quickselect.py`:

```
$ python examples/quickselect_demo.py examples/output

         n   quickselect   sort ~n log n   ratio
      1024          3491           10240    2.9x
     16384         61484          229376    3.7x
```

Finding the median, a percentile, or a top-k threshold looks like it needs a full `O(n log n)`
sort -- it does not. Quickselect partitions the array around a pivot and recurses only into the
side containing rank k, discarding half the data each step for `O(n)` expected time. A bad pivot
would degrade it to `O(n^2)`, so the median-of-medians algorithm (medians of groups of five,
recursively) picks a pivot provably better than 30% and worse than 30% of the data, guaranteeing
worst-case linear time -- the classic proof that selection beats sorting. This module implements
quickselect with a randomized pivot and with median-of-medians, plus median, k-th
smallest/largest, and percentile wrappers, each verified against a full sort over 1000 random
arrays (with varied pivots and duplicates).

## Aho-Corasick: finding many patterns in one pass

Match a whole dictionary in one scan. `aho_corasick.py`:

```
$ python examples/aho_corasick_demo.py examples/output

  patterns ['he', 'she', 'his', 'hers'] in 'ushers':
    'he' at [2]
    'she' at [1]
    'hers' at [2]
```

A spam filter or virus scanner must match hundreds of patterns at once; a single-pattern search
per pattern costs `O(n * patterns)`. Aho-Corasick finds every occurrence of every pattern in a
single left-to-right scan, in `O(n + total pattern length + matches)` -- independent of the
pattern count. It builds a trie of the patterns, then adds failure links (fall back to the
longest proper suffix that is also a pattern prefix, so no character is re-examined -- KMP
generalized to many patterns) and output links (report every pattern ending at the current
state, catching overlapping and nested matches). Built once by breadth-first traversal, then one
text pass reports all matches. This module builds the automaton and finds all matches (with
positions), verified *exhaustively* against a brute-force per-pattern search over 1000 random
multi-pattern cases. It powers virus scanners, spam filters, and DNA motif search.

## Floyd-Warshall: shortest paths between every pair

All-pairs shortest paths in one O(V^3) pass. `floyd_warshall.py`:

```
$ python examples/floyd_warshall_demo.py examples/output

  shortest 0 -> 4: [0, 2, 5, 4]  (cost 20)
  agrees with all-pairs Dijkstra: True
  negative cycle detected: True
```

Dijkstra gives shortest paths from one source; for a routing table or a road-network distance
matrix you need them between all pairs. Floyd-Warshall does it in one dynamic program that also
handles negative edge weights (which Dijkstra cannot) and detects negative cycles. It allows
ever-larger sets of intermediate nodes: `dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])`
as k runs over every node -- three nested loops, no priority queue. Recording the next hop per
pair reconstructs the routes; a negative value on the diagonal means a negative cycle; and
swapping min for boolean OR gives the transitive closure. This module computes the distance
matrix, paths, cycle detection, and closure, verified against running Dijkstra from every source
on 300 random graphs. Used for routing tables and network distance matrices.

## Misra-Gries: frequent items of a stream in tiny memory

Heavy hitters with only k-1 counters. `misra_gries.py`:

```
$ python examples/misra_gries_demo.py examples/output

  distinct items: 455 (exact counting would need 455 counters)
  Misra-Gries uses only k-1 = 3
  heavy hitters (verified): {'A': 1736}   matches exact: True
```

Which items appear more than `n/k` times, when a counter per distinct value is impossible for
billions of distinct items? The Misra-Gries summary finds every such heavy hitter with only
`k-1` counters in one pass, by a generalized vote: increment a tracked item, start tracking a
new one if a slot is free, else decrement every counter (the incoming item cancels one of each).
Any item over `n/k` is guaranteed to survive (no false negatives); a cheap second pass counts the
survivors exactly to drop the false positives, and the carried counts underestimate by at most
`n/k`. The special case `k=2` is the Boyer-Moore majority vote -- the strict-majority element
with a single counter. This module builds the summary, verifies candidates, and does majority
vote, all checked *exhaustively* against exact counting over thousands of random streams. Used
for network traffic monitors, trending queries, and word-frequency counting.

## Reservoir sampling: a uniform sample from an endless stream

A uniform k-sample in one pass, unknown length. `reservoir.py`:

```
$ python examples/reservoir_demo.py examples/output

  sample 4 of 12, 60000 times: each picked ~20000, chi-square 0.78 -> uniform
  weighted (1:3:12): sampled 6.0% / 18.8% / 75.2% vs weight 6.2% / 18.8% / 75.0%
```

Keep a uniform random sample of k items from a stream whose length you do not know and cannot
store. Vitter's Algorithm R does it in one pass with `O(k)` memory: fill the reservoir with the
first k items, then keep the i-th with probability `k/i`, evicting a random existing one -- and
every item ever seen ends up in the sample with probability exactly `k/n`, whatever n turns out
to be (k=1 is the classic "random line from a huge file"). The weighted Efraimidis-Spirakis
variant gives each item a key `u^(1/w)` and keeps the largest, sampling in proportion to weight.
This module implements both plus a streaming reservoir object, and the tests verify the
uniformity with a chi-square test over tens of thousands of runs. It powers log sampling, A/B
bucketing, and random selection from data too big to hold.

## Count-Min sketch: frequency estimates in sublinear memory

Every item's count in a few kilobytes. `count_min.py`:

```
$ python examples/count_min_demo.py examples/output

      item    true  estimate  error
         A    5956      5956      0
         B    3565      3567      2
  never underestimates: 0 of 2815 items fell below the truth
```

How often has each item appeared, when a counter per distinct key is impossible? The Count-Min
sketch estimates every item's count from a fixed `d x w` grid with d hash functions: add by
incrementing one counter per row, query by taking the MINIMUM of the d counters -- since
collisions only inflate a counter, the smallest is the tightest overestimate and the true count
is never above it. With width `e/epsilon` and depth `ln(1/delta)` the estimate exceeds the truth
by more than `epsilon * total` with probability at most `delta`, so a few kilobytes track a
stream of any size, and two sketches merge by element-wise addition (counting is distributed).
This module builds the sketch, queries, merges, and finds heavy hitters, and the tests verify it
*never* underestimates and stays within the error bound across many skewed streams. Powers
network flow monitors, query optimizers, and n-gram frequency tables.

## The alias method: O(1) sampling from a weighted die

Weighted draws in constant time. `alias_method.py`:

```
$ python examples/alias_method_demo.py examples/output

     outcome  weight   target  sampled
      common      12    0.571    0.569
        rare       1    0.048    0.048
  chi-square: 0.09 -> matches
```

To draw outcome i with probability `p_i`, the obvious way binary-searches a cumulative
distribution at `O(log n)` per draw. Walker's alias method does it in `O(1)` per draw after
`O(n)` setup. It chops the distribution into n equal-area columns, each holding at most two
outcomes -- a main and an alias -- by repeatedly pairing an under-full outcome with an over-full
one until every column has area 1. A draw is one integer roll (pick a column) plus one float
flip (main outcome or its alias): two operations, no search, however many outcomes there are,
and the long-run frequencies exactly equal the weights. This module builds the table by Vose's
algorithm and samples from it, verified against the target weights with chi-square tests over
tens of thousands of draws. The standard for loot tables, particle spawning, and any hot loop
sampling one categorical distribution.

## Fisher-Yates: the only correct way to shuffle

Every permutation equally likely. `fisher_yates.py`:

```
$ python examples/fisher_yates_demo.py examples/output

  Uniformity over all 24 permutations of 4 items, 60000 trials:
    Fisher-Yates: chi-square 4.9    -> uniform
    naive swap:   chi-square 1777.2 -> BIASED
```

Shuffling looks trivial and almost everyone gets it wrong: the naive "swap each position with a
random position anywhere" makes `n^n` equally likely swap sequences but only `n!` permutations,
and since `n^n` is not divisible by `n!` some orderings come up more often. Fisher-Yates fixes it
by shrinking the range -- to place position i, swap it with a random position in `[i, n)`, only
the unshuffled tail -- so each of the `n!` permutations results from exactly one choice sequence
and every ordering is equally likely. The same sweep gives a partial-shuffle k-sample (uniform
without replacement), and restricting swaps to strictly earlier positions (Sattolo) yields a
uniform random single cycle. This module implements all of these and demonstrates the bias by
enumerating every permutation over tens of thousands of trials: Fisher-Yates is flat (chi-square
~5), the naive shuffle measurably lumpy (chi-square ~1800).

## Box-Muller: turning uniform randomness into a bell curve

Uniforms in, Gaussians out. `box_muller.py`:

```
$ python examples/box_muller_demo.py examples/output

  mean +0.0028, variance 1.0033, skewness +0.0028, kurtosis 2.9973
  within 1 sigma 0.6821, 2 sigma 0.9546, 3 sigma 0.9971
```

Random generators give uniform values, but almost every simulation -- Brownian motion,
Monte-Carlo finance, noise models -- needs Gaussians. The Box-Muller transform converts a pair
of uniforms into two independent standard normals: `z0 = sqrt(-2 ln u1) cos(2 pi u2)`, `z1 =
sqrt(-2 ln u1) sin(2 pi u2)`. It is exact -- the polar change of variables onto the 2D Gaussian,
whose radius has `r^2` exponentially distributed and whose angle is uniform. Marsaglia's polar
method rejects to the unit disc and reuses its coordinates, skipping the trig; scaling by sigma
and shifting by mu gives any `N(mu, sigma^2)`. This module implements both, and the tests verify
the output's mean, variance, skewness (~0), kurtosis (~3), and the 68-95-99.7 rule over large
samples. Every simulation that needs noise starts here.

## Rejection sampling: drawing from any density you can evaluate

Darts under a curve. `rejection_sampling.py`:

```
$ python examples/rejection_sampling_demo.py examples/output

  bimodal density, 100000 samples; acceptance 0.307 (theory 0.307)
  envelope M=1.01 -> 0.307,  M=2 -> 0.155,  M=4 -> 0.078
```

You can compute a density `f(x)` but cannot sample it directly. Rejection sampling draws a
candidate from a simpler proposal `g` you can sample and accepts it with probability
`f(x)/(M g(x))`, where M bounds `f <= M g`. Geometrically you throw darts uniformly under the
envelope `M g` and keep those below `f` -- the kept points are distributed exactly as f, even
when f is known only up to a constant (which is why it underlies Bayesian computation). The
acceptance rate is the area ratio `1/M`, so a loose envelope or a high dimension wastes darts.
This module does box rejection sampling on an interval and general rejection sampling with an
arbitrary proposal, and the tests verify the sampled moments, the acceptance-rate theory, and a
histogram chi-square against the target (Gaussian, triangular, unnormalized, and beta shapes).

## Welford's algorithm: mean and variance in one stable pass

Running statistics that never lose precision. `welford.py`:

```
$ python examples/welford_demo.py examples/output

        offset  true var     Welford           naive
         1e+09       2.0      2.0000          0.0000
         1e+12       2.0      2.0000 -134217728.0000
```

The textbook variance `E[x^2] - E[x]^2` is a numerical disaster: it subtracts two large,
nearly-equal numbers, so on offset data (temperatures near 1e6, timestamps, prices) catastrophic
cancellation can even return a NEGATIVE variance. Welford's algorithm updates a running mean and
the sum of squared deviations as each datum arrives -- `delta = x - mean; mean += delta/n; M2 +=
delta*(x - new_mean)` -- never forming those giant intermediates, so it is both online (no need
to store the data) and numerically stable. Terriberry's extension carries M3 and M4 for skewness
and kurtosis, and two accumulators merge by combining counts, means, and M2 with a correction --
so statistics over shards combine in parallel, exactly. This module provides the accumulator and
merge, verified against a two-pass computation over 200 random datasets and shown staying exact
(variance 2.0) at a 1e12 offset where the naive formula returns -134 million.

## Kahan summation: adding floats without losing the small ones

Bounded error over any number of terms. `kahan.py`:

```
$ python examples/kahan_demo.py examples/output

           n     naive error   Kahan error    pairwise
     1000000        1.33e-11      0.00e+00    2.33e-15
    10000000        1.61e-10      0.00e+00    1.40e-15
```

Add a million small numbers naively and the answer drifts: once the total is large, each tiny
addend has fewer mantissa bits to land in, so the error grows with n. Kahan's compensated
summation carries a correction term for the bits lost on the previous addition -- `y = x - c;
t = sum + y; c = (t - sum) - y` -- keeping the error bounded, as if the sum were done in twice
the precision. Neumaier's variant also handles catastrophic cancellation (`[1, 1e100, 1, -1e100]`
sums to 2, not 0), and pairwise summation gives `O(log n)` error growth with no correction term.
This module implements all of these plus a compensated dot product and running-mean accumulator,
verified against Python's exact `math.fsum` on ill-conditioned inputs: Kahan drives the
million-term error from 1.3e-11 to zero. It matters for long dot products, running averages, and
any accumulation over millions of terms.

## Horner's method: evaluating a polynomial the fast, stable way

Nested multiplication, and roots for free. `horner.py`:

```
$ python examples/horner_demo.py examples/output

  p(x) = 2x^3 - 6x^2 + 2x - 1,  p(3) = 5
  synthetic division by (x - 3): quotient [2, 0, 2], remainder 5 = p(3)
  roots of x^3 - 6x^2 + 11x - 6: [1.0, 2.0, 3.0]
```

Evaluating a polynomial by separate powers costs ~2n multiplications and sums terms of wildly
different sizes. Horner rewrites it as nested multiplication -- `p(x) = (...(a_n x + a_{n-1})x +
...)x + a_0` -- in exactly n mults and n adds with far better rounding. The same sweep IS
synthetic division: the intermediate values are the quotient of dividing by `(x - r)` and the
final value is the remainder `p(r)` (the Remainder Theorem); a second sweep gives `p'(r)` for
free, which makes Horner the engine of Newton's method for polynomial roots. This module
evaluates by Horner, does synthetic division and derivatives, and finds real roots by Newton
refinement plus deflation, verified against direct power-sum evaluation over 2000 random
polynomials and by substituting the roots back.

## Bracketing root-finders: bisection, secant, false position, Brent

Guaranteed convergence when you have a bracket. `rootfind.py`:

```
$ python examples/rootfind_demo.py examples/output

            method              root  iters       error
         bisection    1.414213562372     41     6.7e-13
             Brent    1.414213562373     26     0.0e+00
            secant    1.414213562373      8     2.2e-16
```

Newton is fast but can diverge; when you have a bracket `[a, b]` where f changes sign, these
methods guarantee convergence. Bisection halves the interval each step (foolproof, linear -- one
bit per iteration); the secant method fits a line through the last two points (superlinear, order
~1.618, but not guaranteed); false position keeps the secant inside the bracket (safe and faster
than bisection); and Brent combines bisection's safety with inverse quadratic interpolation's
speed, falling back to bisection when the fast step misbehaves -- the default root-finder in most
numerical libraries. This module implements all four with a shared bracket interface plus a
sign-change scanner, verified against roots of polynomials (1, 2, 3) and transcendentals
(cos x = x, x = e^-x) and checked to agree.

## Numerical quadrature: integrating what algebra cannot

Definite integrals with no closed form. `quadrature.py`:

```
$ python examples/quadrature_demo.py examples/output

       n     trapezoid   ratio       Simpson   ratio
      64      4.85e-03    4.0x      1.56e-06   16.0x
     256      3.03e-04    4.0x      6.08e-09   16.0x
  Romberg error 1.8e-15, Gauss-Legendre 1.2e-14
```

Most integrals have no closed form, so you approximate the area under f by sampling it. The
trapezoid rule joins samples with lines (error `O(h^2)`); Simpson fits parabolas (`O(h^4)`,
exact for cubics); Romberg applies Richardson extrapolation to a ladder of halved-step trapezoid
estimates, cancelling error terms two orders at a time to reach machine precision in a handful
of levels; adaptive Simpson subdivides only where the function is hard; and Gauss-Legendre
places n nodes optimally to integrate polynomials of degree `2n-1` exactly. This module
implements all five and verifies them against integrals with known values (polynomials, exp,
trig, the Gaussian bell), confirming the convergence orders -- the trapezoid error quarters and
Simpson's sixteenths each time the step is halved. It powers physics simulations, option
pricing, and Bayesian evidence.

## Cubic spline interpolation: a smooth curve through every point

Smooth through the data, no wild oscillation. `spline.py`:

```
$ python examples/spline_demo.py examples/output

       x      true    spline   polynomial
    0.90    0.0471    0.0476       1.5787
    0.95    0.0424    0.0429       1.9236
```

A single high-degree polynomial through many points oscillates wildly between them (Runge's
phenomenon). A cubic spline instead fits a separate cubic to each interval and stitches them
`C^2` -- value, slope, and curvature all match at every join -- giving the smoothest
interpolant, the shape a flexible draftsman's ruler takes. The construction reduces to solving
for the knot second derivatives, a tridiagonal system solved in `O(n)` by the Thomas algorithm;
the natural spline sets zero end curvature, the clamped spline fixes end slopes. This module
builds and evaluates the spline and its derivatives, verified to pass through every knot, be
`C^2`, reproduce cubics exactly, and stay near the true Runge curve where a single polynomial
explodes to ~1.9. It is the interpolation behind fonts, animation, and CAD.

## The Fast Fourier Transform: O(n log n) instead of O(n^2)

The frequencies in a signal, computed the fast way. `fft.py`:

```
$ python examples/fft_demo.py examples/output

  signal = cos(2pi*4t) + 0.5 cos(2pi*12t)  ->  peaks at 4 Hz and 12 Hz (ratio 2:1)
           n           FFT               DFT   speedup
     1048576    20,971,520 1,099,511,627,776   52,428x
```

The discrete Fourier transform turns n samples into their n frequency components -- the recipe
behind audio and image compression, spectrum analysis, and fast polynomial multiplication.
Computed directly it costs `O(n^2)`; the Cooley-Tukey FFT does the same transform in `O(n log n)`
by splitting the samples into even- and odd-indexed halves, transforming each recursively, and
combining them with twiddle-factor butterflies -- collapsing a million-sample transform from
`10^12` operations to `~2x10^7`, about 50,000x faster. The same butterfly runs backwards for the
inverse, and the convolution theorem turns an `O(n^2)` convolution into three FFTs. This module
implements the radix-2 FFT, inverse, a naive DFT check, and FFT convolution using only Python's
built-in complex numbers, verified to match the DFT, round-trip exactly, recover known
frequencies, and satisfy Parseval's identity. Often called the most important numerical algorithm
of the 20th century.

## Gaussian elimination and LU decomposition: solving A x = b

The most-solved problem in computation. `linsolve.py`:

```
$ python examples/linsolve_demo.py examples/output

  solution x = [2.0, 3.0, -1.0]   residual ||Ax-b|| = 4.4e-16
  P A = L U (permutation [1, 2, 0]); det = -1
  reuse the factorization: b=[1,0,0] -> x=[4,-2,5], b=[0,5,5] -> x=[10,-5,15]
```

`A x = b` -- n equations in n unknowns -- underlies circuit analysis, structural mechanics,
least squares, and the linearized step of every nonlinear solver. Gaussian elimination
row-reduces to triangular form and back-substitutes in `O(n^3)`, and done once it factors
`A = L U`, after which each new right-hand side is solved in `O(n^2)` by two triangular sweeps.
Partial pivoting swaps in the largest pivot at each step to stay numerically stable, recorded as
a permutation P so `P A = L U`. The determinant is the product of U's diagonal times the
permutation sign, and the inverse comes from solving against each unit column. This module builds
the factorization, solves systems, and computes determinants and inverses, verified by residuals
and `P A = L U` over hundreds of random systems.

## QR decomposition and least squares: fitting more data than parameters

Orthogonalize, then fit. `qr.py`:

```
$ python examples/qr_demo.py examples/output

  Q^T Q = I: True,  Q R = A: True
  least-squares line fit to 25 noisy points: y = 1.724 x + 3.178  (true 1.7, 3.0)
  quadratic fit (exact data): 3.00 x^2 + -2.00 x + 5.00
```

Any matrix A (m >= n) factors as `A = Q R` with Q orthonormal (`Q^T Q = I`) and R
upper-triangular. It is the workhorse of overdetermined systems: fitting a model to more data
points than parameters, solving `min ||A x - b||` by `R x = Q^T b` is far more stable than the
normal equations `A^T A x = A^T b` (which square the condition number). The construction is the
Gram-Schmidt process -- subtract from each column the components along the earlier orthonormal
directions, then normalize -- in its modified form, which subtracts each projection immediately
to stay orthogonal under rounding. QR also drives eigenvalue iteration and orthogonal
regression. This module builds the thin QR, solves least-squares and square systems, and
verifies `Q^T Q = I`, `Q R = A`, and that the residual is orthogonal to the column space.

## Power iteration: eigenvalues without the characteristic polynomial

Eigenvalues by repeated multiplication. `eigen.py`:

```
$ python examples/eigen_demo.py examples/output

  dominant eigenvalue = 4.745285 (converged in a few iterations)
  full spectrum by deflation: [4.7453, 3.1773, 1.8227, 0.2547]
  sum of eigenvalues = 10.0000 (trace = 10) -- they match
```

An eigenvector is a direction a matrix only stretches: `A v = lambda v`. They govern vibration
modes, Markov stationary distributions, PCA axes, and PageRank -- and for anything beyond 2x2
the characteristic polynomial is a poor way to find them. Power iteration is the simplest
alternative: start from a vector and repeatedly multiply by A and normalize; each multiply
amplifies the largest-|eigenvalue| direction most, so the vector converges to the dominant
eigenvector and the Rayleigh quotient `v^T A v / v^T v` gives its eigenvalue. Shifted inverse
iteration power-iterates `(A - mu I)^-1` to target the eigenvalue nearest mu, and deflation peels
off found eigenpairs to recover a symmetric matrix's whole spectrum. This module implements all
three, verified by `A v = lambda v`, the trace and determinant identities, and analytic cases.

## Conjugate gradient: huge sparse SPD systems without a factorization

Iterative solving with mat-vecs only. `conjugate_gradient.py`:

```
$ python examples/conjugate_gradient_demo.py examples/output

  20x20 SPD system: CG converged in 12 iterations (<= n = 20), residual 2.0e-09
  steepest descent: 22 iterations -- CG's A-conjugate directions win
```

For a symmetric positive-definite A, solving `A x = b` by LU costs `O(n^3)` and stores the whole
factorization -- impossible at millions of rows (finite-element meshes, image operators, graph
Laplacians). Conjugate gradient solves it with nothing but matrix-vector products, so a sparse A
costs `O(nnz)` per step and `O(n)` memory. It minimizes the energy `(1/2)x^T A x - b^T x`,
choosing each search direction A-conjugate to all previous ones so it never undoes earlier
progress -- converging in at most n steps exactly, far fewer in practice at a rate set by
`sqrt(kappa)`, which is why preconditioning (here the Jacobi diagonal) is the whole game. This
module implements CG and preconditioned CG, verified against a dense LU solve, the `<= n` step
guarantee, and the monotone residual decay, and shown beating steepest descent's zig-zag.

## SVD and PCA: the axes a matrix acts along

Every matrix's most informative factorization. `svd.py`:

```
$ python examples/svd_demo.py examples/output

  rank 2, spectral norm 6.49, condition number 1.88
  low-rank error: rank 1 -> 0.084, rank 2 -> 0.062, rank 6 -> 0.000
  PCA: component 1 [0.881, 0.474] explains 96.1% of variance
```

Every matrix A factors as `A = U S V^T`: orthonormal input directions V, orthonormal outputs U,
and non-negative singular values S saying how much A stretches each -- geometrically A sends the
unit sphere to an ellipsoid whose semi-axes are the singular values. It is the most informative
factorization: the rank, 2-norm, condition number, best low-rank approximation (Eckart-Young,
the basis of image compression), and pseudo-inverse all read off it. Here it is built via the
symmetric eigendecomposition of `A^T A` (reusing the power-iteration eigensolver). Principal
component analysis is SVD of mean-centred data: the top singular vectors are the directions of
greatest variance. This module computes the thin SVD, low-rank reconstruction, and PCA with
explained variance, verified by `A = U S V^T`, orthonormality, and the variance ordering.

## k-means clustering: finding groups in unlabelled data

Partition points into k groups. `kmeans.py`:

```
$ python examples/kmeans_demo.py examples/output

     k     inertia  silhouette
     3      1105.1       0.652
     4       102.3       0.851   <- elbow + peak silhouette at the true k
     5        89.9       0.729
```

Given points and a number k, k-means partitions them so each belongs to the nearest centroid,
minimizing the total within-cluster squared distance (the inertia). Lloyd's algorithm alternates
two steps -- assign each point to its nearest centroid, then move each centroid to its members'
mean -- each of which can only lower the inertia, so it converges (to a local minimum). k-means++
seeding spreads the initial centres by picking each with probability proportional to its squared
distance from the nearest chosen one, and a few restarts keep the best. The inertia elbow and the
silhouette score both flag the natural cluster count. This module implements Lloyd's with random
and k-means++ initialization, multi-restart selection, and the silhouette, verified to recover
well-separated blobs (centroids on the true centres, sizes 30/30/30) with monotone inertia.

## Linear and logistic regression: fitting a line and a decision boundary

The base of supervised learning. `regression.py`:

```
$ python examples/regression_demo.py examples/output

  linear: y = 1.44 x + 2.45 (true 1.5, 2.0), R^2 = 0.78
  logistic: accuracy 100%, boundary x = 5.23, log-loss 0.58 -> 0.03 over training
```

Linear regression fits `y = w.x + b` by minimizing squared error -- solved in closed form by QR
least squares (avoiding the normal equations' condition-number squaring) or by gradient descent
for large data. Logistic regression predicts a probability `sigmoid(w.x + b)` for binary labels,
fit by gradient descent on the convex log-loss, and its decision boundary `w.x + b = 0` is a
separating hyperplane. Both are linear models; logistic just squashes through the sigmoid to stay
a probability, and an L2 penalty shrinks the weights for generalization. This module fits linear
regression by both QR and gradient descent, logistic by gradient descent with optional L2, and
reports R^2 for regression and accuracy / log-loss for classification, verified against exact
fits and separable data.

## Decision trees: classification by the best yes/no questions

The most interpretable model and the base learner of forests. `decision_tree.py`:

```
$ python examples/decision_tree_demo.py examples/output

  90 points, 3 classes, training accuracy 100%, depth 2, 3 leaves
    if x[0] <= 5.11:
      if x[1] <= 4.95: predict 0  else: predict 2
    else: predict 1
  feature importances: x = 0.50, y = 0.50
  max_depth=1: 66.7%   max_depth=2: 100%   (a cap trades fit for simplicity)
```

CART grows a tree greedily: at each node it tries every feature and every threshold and keeps the
split that most reduces the impurity of the children, where Gini impurity is `1 - sum p^2` (the
chance two random draws differ) and entropy is `-sum p log2 p` (bits of surprise), both zero for a
pure node. Recursing carves the feature space into axis-aligned rectangles, each a leaf that votes
the majority class; a max-depth or minimum-node-size cap fights the overfitting a fully grown tree
invites, and the root-to-leaf path reads as a plain if/else rule with no feature scaling needed.
This module builds the classifier with Gini or entropy, exposes the learned rules and feature
importances, and is verified on separable blobs, a train/test split that generalizes, and a known
single split. Trees are the building block of random forests and gradient boosting, the workhorses
of tabular machine learning.

## Random forests: a committee of decorrelated trees

A single tree overfits; a forest averages many that disagree. `random_forest.py`:

```
$ python examples/random_forest_demo.py examples/output

  single tree (unlimited): train 100%, test 70%, 64 leaves (memorized noise)
  random forest (41 trees): train 95%, test 85%, out-of-bag 81%
  importances: x 0.30, y 0.34, noise1..3 ~0.11 each (real features dominate)
```

A random forest weakens each tree deliberately and makes them disagree, so their errors cancel
while their signal adds. Two randomizations decorrelate them: **bagging** trains each tree on a
bootstrap sample (n rows drawn with replacement, ~63% distinct), and **feature subsampling** lets
each split see only a random `sqrt(d)` subset of features so no single strong feature dominates
every tree. Prediction is a majority vote. Because ~37% of rows are out-of-bag for each tree,
voting each row over only its out-of-bag trees gives a free, honest validation estimate -- no
held-out set needed. This module builds a bagged forest of the CART learner with per-node feature
sampling, majority-vote prediction, out-of-bag scoring, and averaged feature importances, verified
to beat an overfit single tree on a noisy problem with its OOB estimate tracking true test error.
Forests are the strong, low-tuning default for tabular data.

## Gaussian mixtures & EM: soft, probabilistic clustering

k-means assigns hard; a mixture assigns probabilities. `gmm.py`:

```
$ python examples/gmm_demo.py examples/output

  360 points, 3 clusters of unequal spread; converged in 25 iterations
  recovered: (2.06,2.01) sd 0.47, (7.81,7.90) sd ~1.4, (1.89,8.04) sd 0.80
  log-likelihood monotone: -2065.9 -> -1259.0
  BIC: k=1 3623, k=2 2776, k=3 2600 (min), k=4 2613  -> selects true k=3
```

A Gaussian mixture models the data as drawn from k Gaussians and asks, for each point, the
probability it came from each -- a soft assignment that lets clusters differ in size, weight, and
spread. Expectation-Maximization fits it: the **E-step** computes each point's responsibility
(posterior over components) with parameters fixed, the **M-step** re-estimates each component as
the responsibility-weighted mean, variance, and weight. Each round provably cannot decrease the
log-likelihood -- that monotone climb is the correctness check -- and EM converges to a local
optimum, so it is run from several inits. This module fits a diagonal-covariance mixture with
log-sum-exp numerics, gives soft responsibilities and hard labels, and reports AIC/BIC for choosing
k, verified to recover known parameters, climb the log-likelihood every iteration, and let BIC pick
the true number of components.

## Hidden Markov models: decoding sequences with hidden state

See the emissions, infer the hidden path. `hmm.py`:

```
$ python examples/hmm_demo.py examples/output

  260 die rolls; hidden state fair (0) or loaded (1)
  forward log-likelihood -442.96;  Viterbi decode accuracy 76.9%
  forward-backward: 24 rolls flagged loaded with >90% confidence
  Baum-Welch relearns from rolls alone: P(six|loaded) 0.67 (true 0.60), monotone LL
```

A hidden Markov model describes a sequence you can see (emissions) generated by a chain of states
you cannot (the hidden path). Three questions, three exact dynamic-programming answers over the
trellis: the **forward** algorithm sums the probability of every consistent path in `O(T k^2)` to
evaluate the sequence (a naive sum is `O(k^T)`); **Viterbi** is the same recursion with max for sum
plus backpointers, giving the single most-likely hidden path; and **Baum-Welch** is EM --
forward-backward gives the posterior of each state and transition at each step, and those soft
counts re-estimate the matrices, the likelihood provably climbing each round. It all runs in log
space with log-sum-exp so long sequences never underflow. This module implements forward, Viterbi,
forward-backward posteriors, and Baum-Welch training, verified that Viterbi recovers a planted
path, forward and backward agree on the likelihood, and Baum-Welch relearns a known loaded-die
model with monotone log-likelihood.

## The Kalman filter: optimal tracking of a hidden state

Fuse a motion model with a noisy sensor. `kalman.py`:

```
$ python examples/kalman_demo.py examples/output

  80 steps, position sensor noise sd = 6.0
  RMSE vs truth:  raw 6.40,  Kalman filter 3.52 (45% better),  RTS smoother 2.18 (66% better)
  estimate variance collapses 24.0 -> steady 8.60 (measurement variance 36)
  velocity never measured, only inferred: final 3.64 (truth ~3.60)
```

Where an HMM tracks a discrete hidden state, the Kalman filter tracks a continuous one -- position,
velocity, a trajectory -- optimally for a linear-Gaussian system, and it is the math behind GPS and
sensor fusion. The world is `x <- F x + process noise`, `z = H x + measurement noise`; the filter
carries a Gaussian belief and alternates **predict** (push through the dynamics, uncertainty grows)
and **update** (fold in a measurement weighted by the Kalman gain `K = P H' (H P H' + R)^-1`,
uncertainty shrinks). The fused estimate beats either model or sensor alone -- variance provably
below the sensor's -- and a backward RTS smoother, using future data, beats the causal filter. This
module implements the multivariate filter and smoother with self-contained matrix helpers, verified
on constant-velocity tracking: error and variance fall below the raw measurements', a steady-state
gain is reached, a perfect sensor is trusted exactly and a useless one ignored, and the smoother
improves on the filter.

## PageRank: ranking a graph by its random walk

The algorithm that launched Google. `pagerank.py`:

```
$ python examples/pagerank_demo.py examples/output

  8 pages, 13 links, damping 0.85
     home 0.3313 (in-links 5)   blog 0.2137   shop 0.1463   about 0.1126 ...
  scores sum to 1.000000 (a probability distribution)
  personalized (teleport = shop): shop 0.1463 -> 0.2736, blog 0.2137 -> 0.1629
  power iteration converges geometrically at rate ~ damping
```

PageRank scores every node of a directed graph by one recursive idea -- a node is important if
important nodes link to it -- formalized as a random surfer who with probability `d` follows a
random out-link and with probability `1-d` teleports to a uniformly random page. The score is the
fraction of time spent on each page: the stationary distribution of that Markov chain, equivalently
the dominant eigenvector of the Google matrix `G = d M + (1-d)/N 11'`. The teleport makes `G`
strictly positive, so Perron-Frobenius guarantees a unique positive stationary vector and power
iteration converges geometrically at rate `d`. Dangling nodes would leak probability, so their mass
is redistributed by teleport, and it is all done sparsely without forming the dense NxN matrix. This
module computes PageRank by sparse power iteration with damping and correct dangling handling, plus
the personalized variant, verified against the analytic stationary distribution of small chains,
ring symmetry, and the fixed-point property.

## LU & Cholesky: factoring a matrix to solve, invert, and take determinants

Factor once, reuse for solves, determinants, and inverses. `lu.py`:

```
$ python examples/lu_demo.py examples/output

  P A = L U (partial pivoting): pivot order [1,0,2], sign -1, max|PA-LU| = 0
  det(A) = sign * prod(diag U) = -16.0
  factor once, solve many:  A x = [5,-2,9] -> [1, 1, 2]
  Cholesky A = L L' (SPD): max|LL'-A| = 0; attempting it IS the positive-definiteness test
    SPD -> True,  indefinite [[1,2],[2,1]] -> False
```

LU decomposition writes any square matrix as `P A = L U` -- row-swap permutation (partial pivoting
for stability), unit-lower-triangular `L`, upper-triangular `U` -- by Gaussian elimination in
`O(n^3)` once; afterwards each solve is two `O(n^2)` triangular sweeps, the determinant is the
signed product of `U`'s diagonal, and the inverse is `n` solves. For a symmetric positive-definite
matrix, **Cholesky** `A = L L'` is the smaller, stabler special case: half the work, no pivoting,
and it succeeds iff the matrix is positive definite -- so attempting it is the standard SPD test,
the backbone of least squares and Kalman filters. This module implements LU with partial pivoting,
Cholesky, triangular and general solves, determinant, and inverse, verified by reconstructing
`P A = L U` and `A = L L'`, cross-checking determinants against cofactors, confirming Cholesky
rejects non-positive-definite matrices, and round-tripping `A A^-1 = I`.

## Gaussian process regression: prediction with honest error bars

Fit a distribution over functions, get a calibrated error bar for free. `gaussian_process.py`:

```
$ python examples/gaussian_process_demo.py examples/output

  8 noisy observations of a smooth function, with a gap in the middle
  length scale by max marginal likelihood: l = 0.7 (interior peak, not boundary)
  2-sigma band ~0.16 at the data, ~1.99 in the middle gap
  100% of the true curve lies inside the 2-sigma band (well-calibrated)
```

Where linear regression fits fixed coefficients, a GP fits a distribution over functions and
returns an error bar that widens where there is no data. It assumes any finite set of function
values is jointly Gaussian with covariance set by a kernel -- the RBF kernel
`k(x,x') = sigma^2 exp(-||x-x'||^2 / 2 l^2)` encoding "smooth, length scale l". Conditioning on the
observations gives the posterior in closed form: `mean = k*' (K + sigma_n^2 I)^-1 y` and
`var = k(x*,x*) - k*' (K + sigma_n^2 I)^-1 k*`, the single solve done by a Cholesky factorization of
the SPD matrix `(K + noise)` and reused for the log marginal likelihood that scores
hyperparameters. Noise-free, the posterior interpolates the data exactly with zero variance there;
far from data the variance rises back to the prior. This module builds GP regression with the RBF
kernel, posterior mean and variance, and marginal-likelihood length-scale selection, verified to
interpolate noise-free data exactly, grow uncertainty away from data, recover a known smooth
function, and peak the marginal likelihood near the true length scale. Built on the Cholesky solver.

## Bayesian optimization: minimizing an expensive black box

Spend each costly evaluation where the expected payoff is highest. `bayes_opt.py`:

```
$ python examples/bayes_opt_demo.py examples/output

  1-D multimodal target, 20 evaluations: found x=5.136 f=-1.8990 (true x=5.146 f=-1.8996)
  gap to optimum 0.0006
  2-D Branin (global min 0.398): BO mean 0.446, random search mean 3.339
```

Grid or random search wastes budget on uninteresting regions. Bayesian optimization builds a cheap
GP surrogate of the objective and spends each expensive evaluation where an acquisition function
says the expected payoff is highest, balancing exploitation (sample where the surrogate predicts a
low value) against exploration (sample where it is uncertain). The classic acquisition is **Expected
Improvement**: with current best `f_best` and posterior `(mu, sigma)`,
`EI = (f_best - mu) Phi(z) + sigma phi(z)`, `z = (f_best - mu)/sigma` -- zero at observed points,
large where the surrogate is both promising and unsure, so maximizing it trades the two off
automatically. This module implements EI and the full loop over a bounded domain, verified to locate
the minima of a 1-D multimodal function (within 0.001 of optimum in 20 evaluations) and the 2-D
Branin function, and to beat random search at equal budget. Built on the Gaussian-process regressor.

## DBSCAN: density clustering of arbitrary shapes

Find clusters by density, discover k automatically, flag outliers as noise. `dbscan.py`:

```
$ python examples/dbscan_demo.py examples/output

  232 points (two interlocking moons + 12 outliers), eps 0.22, min_pts 5
  clusters discovered (no k given): 2;  points flagged as noise: 4
  classification: 224 core, 4 border, 4 noise
  k-distance graph elbow marks a good eps
```

k-means and Gaussian mixtures need you to pick `k` and assume blobby clusters; DBSCAN assumes
neither. It finds clusters as connected regions of high point density, so interlocking moons or
concentric rings are recovered whole where k-means would slice them. A **core** point has at least
`min_pts` neighbours within `eps`, a **border** point is within `eps` of a core but not itself
core, and everything else is **noise**; a cluster grows by flood-filling through core-to-core
neighbourhoods. This module implements DBSCAN with the core/border/noise classification and a
k-distance helper for choosing `eps`, verified to separate two half-moons that k-means cannot, flag
sparse outliers as noise, discover the cluster count on its own, and degenerate sensibly at extreme
parameters.

## Hierarchical clustering: a tree of nested groupings

One run yields the whole family of clusterings. `hierarchical.py`:

```
$ python examples/hierarchical_demo.py examples/output

  24 points, Ward linkage, 23 merges; heights monotone 0.113 -> 18.230
  largest jump in merge height suggests 3 clusters; cut -> sizes {0:8, 1:8, 2:8}
  long chain cut in 2:  single [5,19] (chains),  complete [10,14] (compact)
```

Agglomerative clustering starts with every point its own cluster and repeatedly merges the two
closest, recording each merge and its distance -- the dendrogram. Cut it at any height to read off a
flat clustering, so `k` comes from where you cut (often the biggest gap in merge heights) rather
than being fixed up front. The **linkage** sets what "closest" means: single (nearest points,
chains along filaments), complete (farthest points, compact), average (UPGMA), and Ward (least
increase in within-cluster variance, tight and spherical). Merge heights are monotone for these, so
the tree has no crossings. This module builds the full dendrogram by the Lance-Williams update and
cuts it into `k` clusters, verified to recover well-separated blobs with every linkage, climb merge
heights monotonically, and show single-linkage chaining where complete linkage stays compact.

## Naive Bayes: fast probabilistic classification

A crude independence assumption, a strong baseline. `naive_bayes.py`:

```
$ python examples/naive_bayes_demo.py examples/output

  Gaussian NB, 3 classes: training accuracy 100%, per-class mean/var/prior recovered
  posterior matches exact analytic Bayes to 1e-6
  Multinomial spam filter: log-odds free +2.54, money +1.48, meeting -2.51
  [free, money, offer] -> spam (P=1.00);  [meeting, report, project] -> ham (P=0.00)
```

Naive Bayes applies Bayes' theorem assuming features are conditionally independent given the class,
collapsing a joint distribution into a product of per-feature terms -- training is one counting pass,
prediction a sum of logs: pick the class maximizing `log P(c) + sum_i log P(x_i | c)`. **Gaussian**
NB models each continuous feature as a per-class Normal; **multinomial** NB models word counts with
Laplace add-alpha smoothing so an unseen word never zeroes the product (the classic spam filter).
This module implements both entirely in log space, verified that the Gaussian model separates blobs
and matches a hand-computed posterior exactly, that the multinomial model classifies documents and
its smoothing prevents zero probabilities, and that predicted class-probabilities are normalized.

## k-nearest-neighbours: lazy, instance-based learning

No model, just the data and a vote. `knn.py`:

```
$ python examples/knn_demo.py examples/output

  120 points, 2 classes, 15% label noise
  train accuracy: k=1 100% (memorizes), k>=3 ~88%
  leave-one-out CV picks k=15 (k=1 overfits noise, huge k oversmooths)
  regression on noisy sine, k=5: R^2 uniform 0.95, distance-weighted 1.00
```

k-NN builds no model: to predict a point it finds the `k` nearest training points and lets them
vote (classification) or averages their values (regression), all at query time. The number of
neighbours `k` trades variance for bias -- k=1 fits every point exactly (jagged, noise-fitting),
large k smooths -- and the vote can be uniform or distance-weighted (`weight = 1/distance`). It
compares raw coordinates, so standardization is included; a brute-force search is `O(n)` per query
while a k-d tree cuts it to `O(log n)` in low dimensions. This module implements k-NN classification
and regression with both weightings and leave-one-out cross-validation, verified that 1-NN memorizes
the training labels, recovers separable classes and a smooth regression target, that
distance-weighting follows the closest neighbour, that LOO selects k>1 under label noise, and that
its neighbours agree with the k-d tree.

## Gradient boosting: shallow trees that correct each other

Grow trees in sequence, each fixing the last's mistakes. `gradient_boosting.py`:

```
$ python examples/gradient_boosting_demo.py examples/output

  regression on noisy sin(x)+0.3x: R^2 0.9997, MSE 0.0003
  training loss falls monotonically: 0.79 (1 tree) -> 0.0003 (120 trees)
  single depth-3 tree MSE 0.083 -> boosting is 252x better
  circular-boundary classification: accuracy 99.5%, log-loss 0.11
```

Where a random forest averages independent deep trees, gradient boosting grows trees in sequence,
each correcting the errors of the last -- gradient descent in function space. Start with a constant,
then repeatedly fit a small tree to the negative gradient of the loss (the residual `y - F(x)` for
squared error, `y - sigmoid(F(x))` for logistic) and add a shrunken step of it. The number of
trees adds capacity, the **learning rate** shrinks each tree's contribution (small rates need more
trees but generalize better -- shrinkage is regularization), and tree depth caps feature
interactions. This module implements boosting for squared-error regression and log-loss binary
classification over self-contained regression trees with staged predictions, verified that training
loss decreases monotonically, the ensemble beats a single tree by ~250x on a noisy target, a smaller
learning rate needs more trees, and it separates a circular class boundary. The method that wins
most tabular-data competitions.

## Spectral clustering: cutting a graph by its Laplacian

Cluster non-convex shapes by an eigen-embedding, then k-means. `spectral_clustering.py`:

```
$ python examples/spectral_clustering_demo.py examples/output

  44 points on two concentric rings
  spectral clustering separates the rings: True
  plain k-means separates the rings:       False (centroids can't wrap a ring)
  Laplacian's smallest eigenvalues: [0.0, 0.0007]  -> two clusters
  3 far-apart blobs -> 3 graph components, 3 zero eigenvalues
```

k-means splits space by distance to a centroid, so it fails on concentric rings or interlocking
moons. Spectral clustering works on a graph instead: build a Gaussian affinity matrix, form the
normalized Laplacian `L = I - D^-1/2 W D^-1/2`, take the eigenvectors of its `k` smallest
eigenvalues (the number near zero equals the number of connected components), embed each point by
its coordinates there, and run k-means -- where the tangled shapes become tight, separable blobs.
This module builds the affinity graph and Laplacians, extracts the low eigenvectors by reusing a
symmetric eigensolver on `cI - L` (turning smallest into largest), and clusters the embedding,
verified to separate concentric rings and two moons that k-means cannot, and that the Laplacian's
zero-eigenvalue multiplicity counts the graph's connected components.

## Particle filters: nonlinear, non-Gaussian tracking

Track a cloud of samples where the Kalman filter's single Gaussian breaks. `particle_filter.py`:

```
$ python examples/particle_filter_demo.py examples/output

  nonlinear motion x <- x + 0.3 sin(x) + 0.5, noisy position sensor
  mean abs error vs truth:  raw 0.754,  particle filter 0.293 (61% better)
  effective sample size (of 500): adaptive min 212, no-resampling min 1.1 (collapses)
  more particles -> lower error
```

The Kalman filter is optimal only for linear-Gaussian systems; a particle filter drops that,
representing the belief as a cloud of weighted samples propagated through the true dynamics
(sequential Monte Carlo). Each step **predicts** (push particles through the motion model plus
noise), **weights** (by the measurement likelihood), and **resamples** (draw a new equal-weight set
in proportion to the weights). Resampling is the crux -- without it a few particles hoard the weight
(degeneracy) and the cloud stops representing the posterior; the effective sample size `1/sum(w^2)`
measures that, and we resample only when it drops below `N/2` using low-variance systematic
resampling. This module implements a generic bootstrap filter with adaptive resampling, verified on
a nonlinear tracking problem: its estimate beats the raw sensor by ~60%, resampling keeps the
effective sample size high where a weight-only filter collapses to a single particle, and more
particles reduce the error.

## Simulated annealing: escaping local minima by cooling

Sometimes move uphill, less often as you cool. `simulated_annealing.py`:

```
$ python examples/simulated_annealing_demo.py examples/output

  multimodal 1-D: SA found x=-0.761 (true global -0.760), greedy trapped elsewhere
  TSP, 25 cities: nearest-neighbour 53.49 -> annealed 45.61 (15% shorter)
  tour length cools 148 -> 45.6; acceptance rate high early, ~0 once cold
```

Greedy search only moves downhill and gets trapped; simulated annealing escapes by taking uphill
moves with probability `exp(-delta/T)`. High `T` (early) accepts almost anything and roams out of
local basins; as `T` cools only improving moves survive. The cooling schedule is the key knob --
too fast quenches into a poor minimum, slowly (geometric `T <- alpha T`) approaches the global
optimum. This module implements generic annealing over any state plus a travelling-salesman solver
with 2-opt segment-reversal moves, verified to find the global minimum of a multimodal function
greedy descent misses, converge a square TSP tour to its exact optimal perimeter, beat
nearest-neighbour greedy on random tours, and shrink its acceptance rate as it cools.

## Genetic algorithms: optimization by simulated evolution

Evolve a population; let good solutions breed. `genetic_algorithm.py`:

```
$ python examples/genetic_algorithm_demo.py examples/output

  OneMax (60-bit): best 60/60 (all ones), mean fitness 29.8 -> 58.5
  bumpy real function: GA found x=-0.761 (true global -0.760)
  0/1 knapsack (cap 20): value 28 at weight 20
```

Where simulated annealing perturbs a single state, a GA evolves a whole population. Each generation
**selects** parents biased toward fitness (tournament: best of k random individuals), applies
**crossover** to splice two parents' genes, **mutates** for new variation, and keeps the best few
via **elitism** so the best-so-far never regresses. Population-based, it explores many basins at
once and needs no gradients -- strong on rugged, discrete, or black-box landscapes. This module
implements a generic GA over binary and real-valued genomes plus a 0/1 knapsack solver, verified to
solve OneMax to all-ones, maximize a multimodal real function, match the brute-force knapsack
optimum, keep the best fitness monotone under elitism, and beat random search at equal budget.

## Particle swarm optimization: a flock homing on the optimum

A swarm of solutions with velocity, memory, and shared knowledge. `particle_swarm.py`:

```
$ python examples/particle_swarm_demo.py examples/output

  Sphere / Rastrigin / Rosenbrock: all solved to cost ~0 (Rosenbrock at (1,1))
  PSO vs random search (12000 evals): Rastrigin PSO 0.00000 vs random 0.00166
  decaying inertia converges faster than fixed
```

A swarm of candidate solutions flies through the search space, each pulled by three terms:
**inertia** (coast on the old velocity `w*v`, exploring), **cognitive** (`c1*r1*(pbest - x)`, toward
its own best), and **social** (`c2*r2*(gbest - x)`, toward the swarm's best), then `x <- x + v`.
High inertia explores, low inertia exploits, so `w` is often decayed. No gradients, just local
rules balancing exploration and convergence. This module implements PSO over a bounded box with
velocity clamping and linearly-decaying inertia, verified to find the global minimum of the Sphere,
Rastrigin, and Rosenbrock benchmarks, drive the global best down monotonically, beat random search
at equal budget, and converge faster when inertia decays.

## Reed-Solomon codes: recovering data from errors

The error correction behind QR codes, CDs, and deep-space probes. `reed_solomon.py`:

```
$ python examples/reed_solomon_demo.py examples/output

  message 'REED-SOLOMON' (12 bytes) + 8 parity -> corrects up to t=4 errors
  corrupt 4 bytes (a scratch): recovered 'REED-SOLOMON' exactly
  5 errors (one past the limit): correctly flagged as uncorrectable
```

Reed-Solomon treats a message as the coefficients of a polynomial over GF(256) and appends `2t`
parity symbols so the codeword is divisible by a fixed generator. Corruption breaks that
divisibility in a way that pinpoints both where the errors are and what they should have been --
correcting up to `t` byte-errors per block however they are distributed. The field is GF(2^8): XOR
addition, multiplication mod `0x11d`, so multiplication becomes log-table addition. Decoding is the
classic pipeline: **syndromes** (evaluate at the code roots), **Berlekamp-Massey** (the
error-locator polynomial), a **Chien** search (its roots = error positions), and **Forney**'s
formula (the magnitudes). This module implements GF(256) arithmetic, encoding, and full syndrome
decoding, verified that a clean codeword is unchanged, that up to `t` corrupted bytes anywhere
(including in the parity) are corrected exactly across dozens of random trials, and that one error
past the limit is flagged rather than mis-corrected.

## Mutual information: measuring dependence between variables

How much one variable tells you about another. `mutual_information.py`:

```
$ python examples/mutual_information_demo.py examples/output

  binary channel Y=X flipped w.p. f: I falls 1.000 bit (f=0) -> 0 (f=0.5), matches 1-H(f)
  Y = X^2 (deterministic): correlation -0.002, mutual information 1.52 bits
  feature info gain: informative 0.505, weak 0.070, noise 0.000 bits
  KL(D([0.9,0.1] || fair)) = 0.531 bits
```

Mutual information `I(X;Y) = H(X) + H(Y) - H(X,Y)` measures the shared information -- zero exactly
when X and Y are independent, `H(X)` for a deterministic copy, and blind to nothing (it catches
nonlinear dependence that correlation reports as ~0). The KL divergence `D(p||q) = sum p log2(p/q)`
is the extra bits to code p-samples with a q-code, and information gain (the decision-tree split
criterion) is exactly `I(feature; label)`. This module estimates entropies and mutual information
from samples or a joint distribution with KL divergence and normalized MI, verified that
independent variables have zero MI, a copy has maximal `I = H`, the entropy identities hold, KL is
nonnegative and zero only for equal distributions, and against hand-computed values.

## LRU & LFU caches: O(1) eviction policies

Which item to evict when the cache is full. `lru_cache.py`:

```
$ python examples/lru_cache_demo.py examples/output

  LRU trace (cap 3): put A,B,C; get A; put D (evicts B); put E (evicts A)
  hit rate by workload:  uniform LRU 4.5% LFU 5.0%
                         skewed  LRU 22.7% LFU 31.5%  (LFU wins on hot keys)
                         looping LRU 0.0% LFU 0.0%   (classic LRU pathology)
```

LRU evicts the item untouched longest (temporal locality); LFU the least-accessed (popularity). The
craft is O(1): LRU uses a hash map for lookup plus a recency-ordered doubly-linked list so
touch-and-promote and tail-eviction are constant time; LFU groups keys into frequency buckets so
increments and min-frequency eviction are O(1) amortized. This module implements both with
hit/miss statistics, verified that LRU evicts in true least-recently-used order (checked against a
brute-force reference over 60 random workloads), that touching an item spares it, that LFU evicts
the least-frequent breaking ties by recency, that capacity is never exceeded, and that a skewed
hot-key workload gives LFU a higher hit rate.

## Tries: the prefix tree behind autocomplete

Store strings as a tree of characters; O(length) lookup. `trie.py`:

```
$ python examples/trie_demo.py examples/output

  autocomplete 'the': ['the', 'their', 'them', 'there', 'they']
  by frequency: the (9x), to (8x), there (5x), their (4x), they (3x)
  longest stored prefix of 'thereafter' = 'there'
  suffix index of 'abracadabra': 'abra' at [0, 7], 'bra' at [1, 8]
```

A trie stores strings as a tree where each edge is a character and each root-to-node path spells a
prefix, so words sharing a prefix share its path and every operation is O(length of the key),
independent of how many keys are stored. That per-character walk powers autocomplete, longest-prefix
matching (IP routers), and spell-check. Deletion unmarks a word and prunes now-childless
non-terminal nodes; building a trie over all suffixes of a text turns it into a substring index.
This module implements insert, search, prefix membership, autocomplete (alphabetical or
frequency-ranked), deletion with pruning, longest-prefix matching, and a suffix-trie substring
index, verified that it distinguishes a stored word from a mere prefix, autocompletes exactly the
words under a prefix, deletes without disturbing siblings or shared prefixes, and finds substrings
and their positions.

## Comparison sorts: the classic four and their trade-offs

Same O(n log n) floor, four different trades. `sorting.py`:

```
$ python examples/sorting_demo.py examples/output

  n comparisons:  insertion O(n^2) grows ~4x per doubling; merge/quick/heap ~2x
  stability: insertion & merge stable; quick & heap reorder equal keys
  heap doubles as a priority queue: pop by priority -> fire, bug, email, meeting, lunch
```

Insertion sort is O(n^2) but fast on nearly-sorted data (the base case big sorts fall back to);
merge sort is O(n log n) always and stable but needs O(n) scratch; quick sort is usually fastest and
in-place but O(n^2) on adversarial input unless the pivot is good (here median-of-three plus an
insertion cutoff); heap sort is O(n log n) worst-case AND in-place, built on a binary heap that
doubles as a priority queue. This module implements all four plus the heap with a comparison counter
and key function, verified that every sort matches Python's built-in on random, sorted, reverse, and
duplicate-heavy inputs, that merge and insertion are stable while quick and heap are not, that
comparison counts scale as O(n log n) for the good sorts and O(n^2) for insertion, that
median-of-three keeps quick sort fast on its sorted/reverse adversaries, and that the heap is a
correct priority queue.

## Newton's method in n dimensions: solving nonlinear systems

The multivariate root-finder: `J(x) delta = -F(x)`. `newton_nd.py`:

```
$ python examples/newton_nd_demo.py examples/output

  x^2+y^2=4, y=x -> (sqrt2, sqrt2) in 5 steps; residual 2.9e-2 -> 5.3e-5 -> 1.7e-10 (quadratic)
  arctan from x0=5: plain Newton diverges, damped converges to 0
  3-variable system (sum 6, sq-sum 14, product 6) -> (1, 2, 3)
```

In n dimensions the derivative becomes the Jacobian and the division becomes solving a linear
system `J(x) delta = -F(x)`, then `x <- x + delta`. Each step linearizes at the current point and
jumps to that model's root; near a solution the error squares each iteration (quadratic
convergence). When the analytic Jacobian is unavailable it is approximated by finite differences;
because plain Newton can overshoot far from a root, a damped line search backtracks until the
residual decreases; and a Broyden quasi-Newton mode updates a Jacobian approximation instead of
recomputing it. This module implements all three, each solving the linear step via LU with partial
pivoting, verified on a circle-line intersection and the Rosenbrock stationary point, that
convergence is quadratic, that the finite-difference Jacobian matches an analytic one, that damping
rescues a start where plain Newton diverges, and that Broyden converges too. Built on the LU solver.

## Differential evolution: optimization by vector differences

Mutate by scaled differences between population members. `differential_evolution.py`:

```
$ python examples/differential_evolution_demo.py examples/output

  Sphere / Rastrigin / Rosenbrock: all driven to cost ~0 (Rosenbrock at (1,1))
  DE vs random (8000 evals): Rastrigin DE 0.000000 vs random 0.938405
  scales to 20-D; best cost is monotone (greedy selection)
```

Where genetic algorithms mutate bits and particle swarms track velocities, DE mutates by adding a
scaled difference between members: donor `v = a + F*(b - c)`. The population's own spread sets the
step size -- large while dispersed, small as it converges, with no schedule to tune. A binomial
crossover builds a trial from the donor (probability CR) and the target, and greedy selection keeps
whichever is better, so the best never worsens. This module implements DE/rand/1/bin over a bounded
box with bound reflection and a random-search baseline, verified that it finds the global minimum of
the Sphere, Rastrigin, and Rosenbrock benchmarks, that the best cost is monotone, that it beats
random search at equal budget, that the solution stays in bounds, and that it scales to 10-20
dimensions.

## Nelder-Mead: derivative-free optimization by a crawling simplex

Minimize using only function values. `nelder_mead.py`:

```
$ python examples/nelder_mead_demo.py examples/output

  Sphere / Rosenbrock / Beale: all to ~1e-21 with NO gradient (261 evals for Rosenbrock)
  Rosenbrock value: 5.2 -> 1.9 -> 0.18 -> 1e-12 as the simplex crawls the valley
  evals grow with dimension: 2-D 144, 8-D 1143, 16-D 2848
```

Nelder-Mead needs no Jacobian or gradient -- it maintains a simplex of `n+1` points that tumbles
downhill using function values alone. Each step reflects the worst vertex through the centroid of
the rest, then expands into a promising direction, contracts back when it overshoots, or shrinks
the whole simplex toward the best vertex. It crawls like an amoeba, stretching down valleys and
through the curved Rosenbrock banana -- the default when all you can do is evaluate the function.
This module implements the standard algorithm with the classic coefficients, value- and size-based
convergence, and restarts, verified that it finds the minimum of the Sphere, Rosenbrock, and Beale
benchmarks from several starts with no gradient, that the best vertex improves monotonically, that
it even minimizes a non-smooth objective, and that restarting refines the result.

## HITS: hubs and authorities

Two scores per node -- the source everyone cites vs the best list of links. `hits.py`:

```
$ python examples/hits_demo.py examples/output

  top hub: portal (a link directory);  top authority: wiki (what everyone cites)
  hubs and authorities fall on different nodes; PageRank collapses them into one score
  authority vector = dominant eigenvector of A'A, hub vector of AA'
```

HITS splits importance into two mutually-recursive scores: an authority is pointed to by good hubs,
a hub points to good authorities. Iterating `authority(p) = sum of hub scores linking to p` and
`hub(p) = sum of authority scores p links to` (normalized) converges to a fixed point -- equivalently
the dominant eigenvectors of `A'A` (authorities) and `AA'` (hubs). Unlike PageRank's single score,
HITS keeps the two roles distinct, so a curated link list and the cited source rank differently.
This module computes HITS by power iteration with the eigenvector check, verified that a
hub-and-spoke graph scores the linker as the hub and the targets as authorities, that scores
converge and are unit-normalized, that a pure authority has zero hub score and vice versa, and that
the results match the dominant eigenvectors of `A'A` and `AA'`.

## Classical MDS: a map from a table of distances

Distances in, coordinates out. `mds.py`:

```
$ python examples/mds_demo.py examples/output

  6 cities, input is only the distance matrix -> recovered layout, stress 2.4e-23
  eigenvalue scree: 2 positive, rest ~0 -> intrinsic dimension 2
  Procrustes-aligned reconstruction matches the true map to ~1e-12
```

Classical (Torgerson) MDS reconstructs coordinates from pairwise distances by **double centering**:
`B = -1/2 J D2 J` turns the squared-distance matrix into the centered Gram matrix `B = X X'`, and
eigendecomposing `B = V L V'` gives `X = V L^{1/2}` -- the top eigenvectors scaled by the square
roots of their eigenvalues, whose sizes report how much shape each dimension carries. The map is
unique up to rotation, reflection, and translation. This module builds the squared-distance and
double-centered matrices, extracts the embedding by eigendecomposition, reports the eigenvalue
spectrum, and Procrustes-aligns a reconstruction to a known map, verified that it recovers a square,
a line, and random point sets so their reconstructed distances match, that a flat configuration has
exactly two positive eigenvalues, and that the stress is essentially zero for Euclidean inputs.
Built on the eigen module.

## Skip lists: a probabilistic ordered dictionary

Balanced-tree bounds without rotations, using coin flips. `skiplist.py`:

```
$ python examples/skiplist_demo.py examples/output

  express lanes: L0 has every key, L1 ~half, L2 ~quarter, ...
  search 21 in 3 forward hops (a level-0 scan would take 7)
  level distribution over 4000 keys: 49% L0, 26% L1, 13% L2, 6% L3 (geometric)
```

A skip list is an ordered linked list with express lanes: each node is promoted to the next level
with probability p (~1/2), so the lanes thin out geometrically and a search drops down from the top,
skipping far along each level, in O(log n) expected hops -- the same bound as a balanced tree but
with random splices instead of rotations. Insert picks a random height and links in; delete unlinks;
no rebalancing. This module implements a skip-list ordered map with insert, search, delete, ordered
iteration, range queries, and min/max, verified against a brute-force sorted dictionary over 4000
random operations (every search, deletion, and traversal agrees), that keys iterate sorted, that
duplicates update rather than duplicate, that range queries return exactly the in-range keys, and
that the level distribution is geometric as designed.

## Ant colony optimization: pheromone trails for the TSP

Simulated ants converge on short tours. `ant_colony.py`:

```
$ python examples/ant_colony_demo.py examples/output

  22 cities: nearest-neighbour 52.94 -> ant colony 42.85 (19% shorter)
  best length falls 57.2 -> 42.85 as pheromone accumulates
  79% of all pheromone ends up on the best tour's edges (converged)
  alpha/beta: pheromone-only 76.1, greedy-only 44.8, combined 42.85
```

Each iteration a swarm of ants each builds a tour, choosing the next city with probability
proportional to `tau^alpha * eta^beta` (pheromone `tau` = learned memory, `eta = 1/distance` = greedy
heuristic). Then pheromone evaporates and each ant deposits an amount inversely proportional to its
tour length, so shorter tours reinforce their edges and the colony converges -- swarm intelligence,
no central plan. This module implements ant system for the symmetric TSP with the standard
transition rule, evaporation, length-weighted deposit, and elitist reinforcement, verified that it
recovers the optimal perimeter of a square and a circle's polygon, beats the nearest-neighbour
greedy tour on random cities, drives the best length down monotonically, and concentrates pheromone
on short edges.

## AVL trees: a self-balancing binary search tree

Deterministic O(log n) via rotations. `avl_tree.py`:

```
$ python examples/avl_tree_demo.py examples/output

  sorted inserts: n=4095 -> AVL height 12 (log2=12); a naive BST would be 4095 (a chain)
  four rotation cases (LL/RR/LR/RL) all rebalance to root 2
  5000 random ops: balance invariant held throughout, matches a reference dict
```

An AVL tree keeps every node height-balanced (subtree heights differ by at most 1); after each
insert or delete it rotates wherever the balance factor exceeds the bound -- a constant-time pointer
rewiring. Four cases cover every imbalance (LL/RR single, LR/RL double). The strict invariant makes
it the most rigidly balanced classic BST, so lookups are fast; where a skip list balances
probabilistically, AVL does so deterministically. This module implements an AVL ordered map with
insert, delete, search, ordered traversal, range queries, and min/max, verified against a
brute-force sorted dictionary over 5000 random operations, that the balance invariant holds
throughout, that the height stays O(log n) even for sorted insertions (where a naive BST would be
linear), and that all four rotation cases trigger.

## Segment trees with lazy propagation: range query and range update

Both in O(log n), where a Fenwick tree can only do point updates. `segment_tree.py`:

```
$ python examples/segment_tree_demo.py examples/output

  sum[2..5]=19; after +10 to [2..5] sum[2..5]=59; after +100 to [0..7] sum=871
  min/max over sub-ranges; range-add shifts them
  full-array update visits ~2 log n nodes (65536 -> ~34), not n
```

Each node stores the aggregate of a contiguous segment; a query descends only into the O(log n)
nodes whose segments tile the range. Range updates use **lazy propagation**: a node records a
pending update as a lazy tag, applies it to itself, and pushes it to children only when a later
operation visits them -- so a full-array add touches ~2 log n nodes, not n. This module implements a
segment tree parameterized by the aggregate (sum, min, or max) with lazy range-add updates, point
updates, and range queries, verified against a brute-force array over 3000 random mixed operations
for all three aggregates, that overlapping range-adds accumulate, that point updates match a plain
list, and that it handles single-element and full-array edge ranges.

## Convex hull: the tightest polygon enclosing points

The rubber-band boundary of a point set. `convex_hull.py`:

```
$ python examples/convex_hull_demo.py examples/output

  60 random points -> hull of 11 vertices, convex & CCW, all points inside
  area 8310, perimeter 345, diameter 123.9 (farthest pair)
  interior points dropped: square+centre 5->4, collinear 4->2, triangle+inside 4->3
```

Andrew's monotone chain builds the hull in `O(n log n)`: sort the points, then sweep left-to-right
for the lower hull and right-to-left for the upper, keeping only left turns (positive cross product)
and popping any vertex that would turn right. From the hull come the enclosed area (shoelace),
perimeter, point-in-hull (orientation tests against each edge), and the diameter (farthest pair,
which always lies on the hull). This module implements all of these, verified that a square's hull
is its four corners with interior points dropped, that collinear and duplicate points are handled,
that the hull is convex and counter-clockwise with area matching an independent shoelace, that every
input point lies inside it, and that the diameter is the true farthest pair.

## Closest pair of points: divide and conquer beats O(n^2)

The two nearest points in O(n log n). `closest_pair.py`:

```
$ python examples/closest_pair_demo.py examples/output

  80 points -> closest pair at distance 0.94, brute-force agrees
  n=4096: brute does ~8.4M comparisons, divide-and-conquer ~49K (~170x saving)
  matches brute force exactly across n = 10 .. 1000
```

Sort by x, split into halves, recurse, take the smaller distance `d`, then check only the pairs in
the width-2d strip around the split line -- where, sorted by y, each point can beat `d` against at
most a constant number of neighbours (a `d x 2d` rectangle holds only so many points that are all
`>= d` apart). Linear merge, so `T(n) = 2T(n/2) + O(n) = O(n log n)`. This module implements the
divide-and-conquer closest pair with the strip merge plus the brute-force reference, verified that
the two agree exactly on random sets of many sizes, that it finds a planted near-coincident pair,
handles duplicate points (distance 0), collinear and grid inputs, small `n`, and survives a crowded
strip that would trap a naive merge.

## Line-segment intersection: do two segments cross, and where?

The geometric atom, via the orientation predicate. `segment_intersection.py`:

```
$ python examples/segment_intersection_demo.py examples/output

  X-crossing -> (2,2); T-touch -> (2,0); collinear overlap -> intersect, no single point
  parallel/disjoint -> no intersection
  square/pentagon/arrow simple; figure-eight self-intersecting; 5x5 grid = 25 crossings
```

Two segments properly cross when each straddles the other's line -- decided from the sign of a cross
product (`orient(a,b,c)`: +1 CCW, -1 CW, 0 collinear), with no slopes so vertical segments are no
trouble. Zero orientations mean collinear/touching, settled by a bounding-box check; the crossing
point comes from the 2x2 parametric system. From this atom the module builds the simple-polygon test
(no non-adjacent edges cross -- the precondition for area and point-in-polygon) and pairwise
intersection counting, verified on crossing, touching, collinear-overlap, parallel, and disjoint
segments, that the intersection point is correct and lies on both segments, that a convex polygon is
simple while a figure-eight is not, and that a 5x5 grid has exactly 25 crossings.

## Point-in-polygon: ray casting vs winding number

Is a point inside an arbitrary polygon? `point_in_polygon.py`:

```
$ python examples/point_in_polygon_demo.py examples/output

  concave arrow, 7x7 grid: ray casting and winding number agree on all 49 points
  pentagram centre (doubly wound): ray/even-odd = OUTSIDE, winding/nonzero = INSIDE (count 2)
```

Ray casting counts edge crossings of a ray to infinity (odd = inside, even-odd rule); the winding
number sums the signed turns the polygon makes around the point (nonzero = inside). They agree on
any simple polygon -- including concave and star shapes a convex side-test can't handle -- but on a
self-overlapping polygon they split: even-odd cancels a doubly-wrapped region to "outside" while
winding keeps it "inside". This module implements both with a half-open edge convention (so a ray
grazing a vertex counts once) and an explicit boundary test, plus signed area (orientation) and
centroid, verified that the methods agree on convex/concave/star polygons and across a grid, that
boundary points are detected, and -- the textbook case -- that a pentagram's doubly-wound centre is
classified differently by the two rules.

## Polygon clipping: intersecting a polygon with a window

Keep only what's inside the viewport. `polygon_clip.py`:

```
$ python examples/polygon_clip_demo.py examples/output

  7-vertex concave subject clipped to [2,8]^2: 82% of the area retained
  same subject to triangle / diamond / tiny-box windows, right areas
  fully inside -> unchanged; fully outside -> empty
```

Sutherland-Hodgman clips the subject polygon against each edge of the convex clip polygon in turn,
feeding the output of one edge into the next; whatever survives all edges is the intersection. Per
edge it keeps inside vertices and adds the crossing point wherever an edge leaves or enters --
"inside" decided by the cross-product orientation predicate. `O(n*k)` for an n-vertex subject and
k-edge window; the clip must be convex, the subject may be concave. This module implements clipping
against an arbitrary convex clip polygon plus a rectangle convenience, verified that a polygon fully
inside is unchanged, one fully outside clips to empty, two overlapping squares clip to their
analytic 2x2 overlap, a square clips to a triangular or diamond window at the right area, a concave
subject stays within the window bounds, and the clipped area never exceeds the original.

## Marching squares: contour lines from a scalar field

The algorithm behind every contour plot. `marching_squares.py`:

```
$ python examples/marching_squares_demo.py examples/output

  f = x^2 + y^2 contoured at 1,4,9,16 -> circles r=1,2,3,4; lengths match 2*pi*r to 3 dp
  Gaussian terrain iso-lines at several heights, closed loops with no loose ends
```

At each grid cell the four corners are above or below the chosen level, giving a 4-bit case index
(16 possibilities) that selects which cell edges the contour crosses; linear interpolation between
corner values places each crossing exactly where the field equals the level, so the curve is smooth.
The two ambiguous saddle cases are resolved by the cell-center average. This module builds the
16-case lookup, extracts contour segments from a grid or a sampled function, and sums contour
length, verified that a radial field contours to circles of the right radius and circumference, a
linear ramp gives straight contours, a level outside the field range yields nothing, a diagonal
saddle gives two segments, and a closed blob's contour forms closed loops with no loose ends.

## Bresenham rasterization: lines and circles from integer math

Draw pixels with integer arithmetic only. `bresenham.py`:

```
$ python examples/bresenham_demo.py examples/output

  lines in every octant: connected, endpoints exact, max error < 0.5 (sub-pixel)
  midpoint circle r=8: 44 pixels, max radius deviation 0.38, 8-fold symmetric
  circle pixel count tracks 2*pi*r
```

Bresenham draws a line tracking an integer error term -- how far the true line has drifted from the
current pixel -- stepping the minor axis and correcting whenever it crosses a threshold; no floating
point, no division, no rounding. All eight octants are handled uniformly with absolute deltas and
step signs; the companion midpoint circle draws one octant and mirrors it eight ways. This module
implements line drawing, the midpoint circle, and a filled disk, verified that a line hits both
endpoints, is 8-connected, stays within half a pixel of the true line, is symmetric under reversal,
and handles every octant, and that a circle's pixels lie within half a pixel of the true radius with
8-fold symmetry and a count tracking the circumference.

## Flood fill: paint bucket, region growing, connected components

Which cells are reachable through a same-valued region? `flood_fill.py`:

```
$ python examples/flood_fill_demo.py examples/output

  paint-bucket inside a wall: 8 cells filled, barrier respected
  queue / stack / scanline strategies fill identically
  checkerboard: 9 components under 4-connectivity, 2 under 8; diagonal chain 1 vs 3
```

From a seed, flood fill spreads to same-valued neighbours until it hits a boundary. Three strategies:
a queue/stack fill (BFS/DFS, one visit per cell), a scanline fill (paint whole horizontal runs,
seed only the rows above and below -- the optimization for big flat regions), and connected-component
labeling (fill from every unlabeled cell). Connectivity is a parameter -- 4-connected (orthogonal)
or 8-connected (with diagonals) -- which changes what counts as one region. This module implements
all three fills (4/8-connected) plus component labeling and sizing, verified that the strategies
produce identical results, that fills respect barriers and the grid edge, that 8-connectivity merges
diagonal regions 4-connectivity separates, that a bounded region leaves the rest untouched, and that
a checkerboard has 9 components under 4-connectivity but 2 under 8.

## Bezier curves: control-point curves via de Casteljau

Shaped by control points, evaluated by repeated interpolation. `bezier.py`:

```
$ python examples/bezier_demo.py examples/output

  quadratic B(0.5)=(2,2), hits its endpoints, arc length 5.92
  subdivision at 0.5 splits into two cubics reproducing the curve
  degree elevation rewrites a quadratic as a cubic with the same shape
```

A Bezier curve starts at its first control point, ends at its last, and is tugged toward the ones
between. De Casteljau's algorithm evaluates `B(t)` by repeatedly taking pairwise linear
interpolations of the control points until one point remains; keeping the triangle's outer edges
splits the curve in two (subdivision). This module implements de Casteljau evaluation, the
derivative (tangents), subdivision, degree elevation, and arc length by adaptive subdivision,
verified that the curve hits its first and last control points, that de Casteljau matches the
Bernstein sum, that a linear curve is exactly the straight segment, that the curve stays within its
control points' convex hull, that subdivision reproduces the original and degree elevation preserves
the shape, and that a linear curve's arc length is the endpoint distance.

## Burrows-Wheeler transform: the heart of bzip2

Permute the bytes so they compress, reversibly. `bwt.py`:

```
$ python examples/bwt_demo.py examples/output

  BWT(banana) = 'annb$aa', reversible; DNA-like text becomes ~15x runnier
  full pipeline on 100 repetitive chars -> 5 RLE pairs (20x fewer tokens), lossless
  worked: banana -> BWT 'annb$aa' -> MTF [1,3,0,3,3,3,0] -> RLE -> banana
```

BWT compresses nothing on its own -- it is a reversible permutation -- but by sorting all rotations
and taking the last column it clusters same-context bytes into runs. The bzip2 pipeline then chains
**move-to-front** (recode each byte as its index in a running alphabet, so a run becomes zeros) and
**run-length** encoding (collapse runs into value/count pairs); the inverse runs them backwards, and
BWT is undone by the LF-mapping. This module implements BWT and its inverse (sentinel-terminated so
any input works), move-to-front, run-length coding, and the full pipeline, verified that BWT
round-trips any string, that it increases the mean run length on structured text, that MTF and RLE
round-trip, that the pipeline is lossless across 50 random strings, and that it yields far fewer
tokens than the input on repetitive data.

## Arithmetic coding: entropy compression past the Huffman limit

Encode the whole message as one number; beat whole-bit codewords. `arithmetic_coding.py`:

```
$ python examples/arithmetic_coding_demo.py examples/output

  very skewed (90/7/3): entropy 0.557, arithmetic 0.560, Huffman 1.100 -> AC 49% smaller
  arithmetic coding hugs the entropy across all distributions; all round-trip losslessly
  extreme skew (995/5): under 0.1 bits/symbol
```

Huffman assigns each symbol a whole number of bits, wasting up to nearly a bit; arithmetic coding
encodes the entire message as a single number in `[0,1)`, narrowing an interval by each symbol's
probability, so a symbol costing 0.15 bits adds only 0.15 bits. It gets within a fraction of a bit
of the Shannon entropy for any distribution -- the entropy coder inside JPEG and H.264. This module
uses integer range coding with renormalization (emit settled top bits, with an underflow counter
for the straddle-the-middle case), verified that encode/decode round-trips arbitrary messages (60
random included), that the code length approaches the entropy, that it beats Huffman's
one-bit-per-symbol floor on skewed data (~49% smaller at 90% skew), and that it handles
single-symbol and uniform alphabets.

## Sequence alignment: global and local dynamic programming

Line up two sequences by gaps to maximize matches. `sequence_alignment.py`:

```
$ python examples/sequence_alignment_demo.py examples/output

  global GCATGCU / GATTACA: score 0, 67% identity, end-to-end with gaps
  motif GATTACA in different flanks: global score -2, LOCAL isolates it at 100% identity
  gap penalty steers whether a gap or a mismatch is used
```

Both algorithms fill a DP matrix where cell (i,j) is the best score aligning the first i and j
symbols via align / gap-in-a / gap-in-b. **Needleman-Wunsch** aligns end to end (global): borders are
cumulative gap penalties, the answer is the corner cell. **Smith-Waterman** finds the best-matching
subsequences (local): scores floor at zero so a bad stretch resets, and the answer traces back from
the maximum cell -- ideal for a conserved motif inside dissimilar sequences. This module implements
both with configurable match/mismatch/gap scoring and traceback to the gapped strings, verified that
identical sequences align perfectly, the global score matches recomputing it from the alignment, a
local alignment isolates an embedded motif the global one drags flanks into, the gap penalty steers
gaps vs mismatches, and the score is symmetric.

## Linear-time string matching: KMP, Z-algorithm, Manacher

Find patterns and palindromes without rescanning. `string_matching.py`:

```
$ python examples/string_matching_demo.py examples/output

  prefix function of 'ababaca': [0,0,1,2,3,0,1]
  'ababaca' in a text: KMP, Z-algorithm, and brute force all agree ([2, 10])
  'aa' in 'aaaaa': 4 overlapping matches; longest palindrome of 'banana' -> 'anana'
```

The naive pattern search is O(n*m); these do it in O(n). **KMP** builds the prefix function (longest
proper prefix that is also a suffix at each position) so a mismatch shifts the pattern by more than
one without rescanning. The **Z-algorithm** computes each position's longest prefix-match; run on
pattern + separator + text it reads off the matches. **Manacher** finds the longest palindromic
substring in O(n) via mirror reuse. This module implements the prefix function and KMP search, the
Z-array and Z-based search, and Manacher's palindrome, verified that KMP and Z find exactly the same
overlapping occurrences as brute force across 400 random strings, that the prefix function matches
its definition, and that Manacher's palindrome matches brute-force length across 100 strings and the
known cases.

## Suffix arrays: a compact string index

A suffix tree's power in one length-n array. `suffix_array.py`:

```
$ python examples/suffix_array_demo.py examples/output

  suffix array + LCP of 'mississippi'; 'issi' occurs at [1, 4] by binary search
  longest repeated substring = 'issi' (the largest LCP value)
  longest common substring of 'dogandcat' & 'thecatsat' = 'cat'
```

A suffix array is the sorted order of all suffixes, stored as start indices, so substring search is
a binary search in O(m log n) and every occurrence is a contiguous run. It is built by prefix
doubling (sort by the first 1, 2, 4, ... characters using previous ranks, O(n log^2 n)); the LCP
array (longest common prefix of adjacent sorted suffixes) is built in O(n) by Kasai, and its largest
value is the longest repeated substring. This module builds the suffix array, the LCP array,
substring search, and the longest repeated and common substrings, verified that the suffix array is
the true sorted order (checked against a brute-force sort over 150 strings), that search finds
exactly the same occurrences as a scan, that the LCP array matches the direct prefix computation,
and that the longest repeated and common substrings match brute force.

## Quadtrees: recursive spatial partition for region queries

Index the plane; answer "which points are in here?" fast. `quadtree.py`:

```
$ python examples/quadtree_demo.py examples/output

  180 points, depth 6 (a dense corner cluster subdivides deeper)
  rectangle and circle range queries match a brute-force scan
  nearest to (50,50) found; out-of-bounds points rejected
```

A quadtree splits a square into four quadrants whenever a node exceeds its capacity, so empty
regions stay shallow and crowded ones subdivide deeply. A range query visits only the cells whose
square overlaps the query, pruning whole branches -- the basis of collision broad-phase and the
Barnes-Hut n-body approximation. Where a k-d tree splits on one coordinate at a time, a quadtree
splits on both at once. This module implements insert (with a max-depth guard so coincident points
don't subdivide forever), rectangle and circular range queries, and nearest-neighbour, verified that
rectangle and radius queries return exactly the same points as a linear scan across 80 random
queries, that nearest matches brute force, that out-of-bounds points are rejected, that a dense
cluster subdivides deeply while spread data stays shallow, and that duplicate points are all stored.

## Maximum flow: augmenting paths, the min-cut theorem, and matching

How much can flow from a source to a sink through capacitated pipes? `max_flow.py`:

```
$ python examples/max_flow_demo.py examples/output

  A 6-node network (source 0, sink 5):
    maximum flow: 23
    minimum cut edges: [(1, 3), (4, 3), (4, 5)]
    min-cut capacity: 23  == max flow: True
  Bipartite matching by reduction: 8 assignments -> maximum matching of 4
```

Ford-Fulkerson pushes flow along augmenting paths and keeps a residual graph where every used edge
gains a reverse edge, so later paths can cancel and reroute earlier flow -- the trick that makes a
greedy-looking method reach the true optimum. Edmonds-Karp always takes the shortest augmenting path
(BFS) for O(V E^2) time, independent of the capacities. The max-flow min-cut theorem says the maximum
flow equals the smallest total capacity of edges whose removal severs source from sink, and after the
flow saturates, the vertices still reachable from the source in the residual graph reveal exactly
those bottleneck edges. Bipartite matching reduces to flow: a super-source into every left vertex, a
super-sink from every right, unit capacities, and the max flow is the matching size. Verified that it
hits the textbook flow of 23, that the min-cut capacity equals the flow, that conservation and
capacity constraints hold, that residual rerouting beats the greedy trap, and that matching-by-flow
equals a direct augmenting-path matching over 40 random graphs.

## De Bruijn sequences: every window exactly once, via Eulerian circuits

The shortest cyclic string containing every length-n window. `de_bruijn.py`:

```
$ python examples/de_bruijn_demo.py examples/output

  B(2,3) = 01011100   (length 8 = 2^3; all 8 binary triples appear once)
  B(10,4): 10000 digits contain all 10000 four-digit PINs, each once
  B(2,6):  a 64-position absolute rotary encoder track
```

A De Bruijn sequence B(k, n) packs all k**n length-n strings into one cyclic string of length k**n
with zero waste -- the maths behind rotary encoders, PIN brute-forcing, card tricks, and genome
assembly. Build the De Bruijn graph (vertices = length-(n-1) strings, edges = length-n windows) and
find an Eulerian circuit; every vertex has equal in- and out-degree so one exists, and Hierholzer's
algorithm splices detour cycles to use every edge once in linear time. This module builds B(k,n) for
any parameters, offers the greedy 'Ford' construction, and a general Eulerian path/circuit finder for
directed multigraphs, verified that every window appears exactly once, that lengths are exactly k**n,
that B(2,3) matches the classic example, that the greedy sequence is valid, and that the Eulerian
finder recovers a circuit using each edge once and rejects graphs where none exists.

## Rotating calipers: diameter, width, and the minimum-area bounding box

Pinch the convex hull between rotating parallel lines; extremal measurements fall out. `rotating_calipers.py`:

```
$ python examples/rotating_calipers_demo.py examples/output

  40 points, hull has 8 vertices
  diameter (farthest pair): 277.31
  width (thinnest slab): 66.70
  minimum-area bounding box: area 18455, rotated box saves 51.3% vs axis-aligned
```

As a pair of parallel supporting lines rotates around the hull, the vertices they touch enumerate
every antipodal pair; the diameter is always among them, so one O(h) sweep finds the farthest pair.
The thinnest slab of parallel lines gives the width, and by the Freeman-Shapira theorem the
minimum-area enclosing rectangle has a side flush with a hull edge, so trying each edge orientation
yields it. This module computes the hull, diameter, width, and minimum-area/perimeter rectangle,
verified against brute force: the calipers diameter equals the O(n^2) farthest pair, the width equals
the brute minimum over hull directions, and the minimum rectangle contains every point and never
loses to the axis-aligned box, across many random and structured sets plus exact values on squares
and triangles.

## Ternary search trees: a trie's prefix power at a BST's space

The string map between a trie and a BST. `ternary_search_tree.py`:

```
$ python examples/ternary_search_tree_demo.py examples/output

  autocomplete 'ca'  -> car, card, care, cat, cats
  longest_prefix_of 'doghouse' -> 'dog'
  wildcard 'c.r'     -> car        (. matches any one letter)
  wildcard '..t'     -> cat, dot
```

Each node holds one character and three children: left/right for the BST of alternatives at this
position, and a middle link that advances to the next character (spelling keys, like a trie). So a
TST keeps the prefix structure and sorted traversal of a trie while spending BST-like space, with no
wasted k-way arrays for large alphabets. A single three-way comparison drives insert, lookup, prefix
completion, and -- the trick hash maps can't do cheaply -- '.'-wildcard partial-match search, which
recurses into all three children at a wildcard position. This module implements insert with values,
lookup, deletion, sorted iteration, autocomplete, longest-prefix-of, and wildcard search, verified
against a plain dict and brute force over hundreds of random prefix, wildcard, and deletion queries.

## Delaunay triangulation and the Voronoi diagram: two faces of proximity

The dual structures of "who is near what". `delaunay.py`:

```
$ python examples/delaunay_demo.py examples/output

  24 sites -> 37 Delaunay triangles (Euler 2n-2-h), 51 finite Voronoi edges
  empty-circumcircle property holds: True
  every site's nearest neighbour is a Delaunay edge: True
```

The Voronoi diagram cuts the plane into nearest-site cells; its straight-line dual, the Delaunay
triangulation, connects sites whose cells touch and is characterized by the empty-circumcircle
property (no site lies inside any triangle's circumcircle), which makes it the triangulation that
maximizes the minimum angle. Built by Bowyer-Watson incremental insertion -- delete every triangle
whose circumcircle contains the new point, re-triangulate the cavity -- with the in-circle predicate
evaluated in exact rational arithmetic so the mesh is always valid. The Voronoi vertices are the
triangle circumcentres. Verified that every triangle has an empty circumcircle, that the triangle
count obeys Euler's 2n-2-h, that the triangle areas exactly fill the convex hull, that Voronoi
vertices are equidistant from their three sites, and that each site's nearest neighbour is a Delaunay
edge.

## The simplex method: linear programming by walking polytope vertices

Maximize a linear objective under linear constraints by sliding vertex to vertex. `simplex.py`:

```
$ python examples/simplex_demo.py examples/output

  maximize 3x + 5y  s.t. x<=4, 2y<=12, 3x+2y<=18  -> profit 36 at (2, 6)
  LP duality: primal 36 == dual 36
  diet problem min 2x+3y s.t. x+y>=10, x+3y>=18   -> cost 24 at (6, 4)
  detects unbounded and infeasible LPs
```

The feasible region of a linear program is a convex polytope, and an optimum (if one exists) sits at
a vertex. Simplex starts at a vertex and slides along improving edges until none remain. It pivots a
tableau -- slack variables make inequalities equalities, an entering variable is chosen by reduced
cost and a leaving one by the minimum-ratio test -- with Bland's rule (smallest index) to prevent
cycling, and a two-phase method with artificial variables to start from >= and = constraints. This
module solves LPs in general form and reports optimal/unbounded/infeasible, verified against
hand-solved textbook LPs, a brute-force vertex enumerator over 60 random LPs, the LP-duality theorem,
and degenerate/unbounded/infeasible instances.

## MinHash and LSH: estimating set similarity at scale

Compress sets into short signatures whose agreement equals their Jaccard similarity. `minhash.py`:

```
$ python examples/minhash_demo.py examples/output

  A-C: estimate 0.980  true 0.975   (near-duplicate sentences)
  A-D: estimate 0.005  true 0.013   (unrelated)
  error shrinks like 1/sqrt(k): k=8 -> 0.130, k=512 -> 0.018
  LSH candidate pairs: (A,B) (A,C) (B,C) (D,E)  -- exactly the two clusters
```

MinHash (Broder's near-duplicate web-page trick) makes the probability that two signatures agree in
any position equal to the Jaccard similarity of the underlying sets, so the fraction of matching
positions is an unbiased estimate from k small integers, with error like 1/sqrt(k). Locality-sensitive
hashing splits signatures into bands and hashes each band, so similar items collide in a hash table
and near-duplicate search becomes a few lookups instead of all-pairs comparison. This module
implements MinHash signatures with universal hashing, the Jaccard estimator, and banded LSH, verified
that the estimate converges to the true Jaccard as k grows, that identical sets estimate 1 and
disjoint ~0, and that LSH recalls every high-Jaccard pair while filtering dissimilar ones.

## CMA-ES: covariance matrix adaptation for black-box optimization

The de-facto standard derivative-free optimizer for hard continuous problems. `cma_es.py`:

```
$ python examples/cma_es_demo.py examples/output

  sphere (4D)               : fx = 6.4e-13 in 952 evals
  Rosenbrock (2D banana)    : fx = 7.9e-13, x ~ (1.000, 1.000)
  ellipsoid (3D, cond 1e6)  : fx = 5.1e-13   (covariance adaptation beats the conditioning)
  random search same budget : 2.7e+00        (CMA-ES wins by ~12 orders)
```

CMA-ES samples candidates from a multivariate normal and, each generation, moves the mean to the
best samples, adapts the step size from the length of a cumulative path (lengthen if progress is
consistent, shorten if it doubles back), and bends the full covariance matrix toward recent progress
so the search ellipsoid learns the landscape's curvature -- a second-order-like method with no
derivatives. This module implements a faithful (mu/mu_w, lambda)-CMA-ES with the standard strategy
parameters and a self-contained Jacobi eigensolver for the covariance decomposition, verified that it
converges to the global optimum of the sphere, Rosenbrock, ill-conditioned ellipsoid, and shifted
problems to near machine precision, beats random search by many orders under an equal budget, handles
a rotated anisotropic bowl, and is fully reproducible from a seed.

## L-BFGS: limited-memory quasi-Newton optimization

Newton-like convergence on smooth problems, with O(m n) memory and no stored matrix. `lbfgs.py`:

```
$ python examples/lbfgs_demo.py examples/output

  Rosenbrock (2D): fx = 2.7e-17 at (1.00000, 1.00000) in 34 iterations
  ill-conditioned quadratic (cond 1000): L-BFGS 1.3e-18  vs gradient descent 2.19
  logistic regression: fitted [1.55, -1.77, 0.66] vs true [1.5, -2.0, 0.5], acc 0.85
```

L-BFGS approximates the action of the inverse Hessian from the last m pairs of (step,
gradient-change) vectors through the two-loop recursion -- no matrix stored, O(m n) per step -- and a
Wolfe line search keeps the curvature pairs positive-definite. This module implements it with an
automatic finite-difference gradient fallback, verified on the quadratic bowl, Rosenbrock (2D/4D), a
shifted optimum, an ill-conditioned quadratic where it beats gradient descent by eighteen orders of
magnitude, and a logistic-regression fit that recovers the generating weights, with finite-difference
gradients matching the analytic ones.

## t-SNE: nonlinear dimensionality reduction that preserves neighborhoods

A 2-D map of high-dimensional data where clusters leap out. `tsne.py`:

```
$ python examples/tsne_demo.py examples/output

  80 points in 8 dimensions, 4 true clusters
  KL divergence: 1.639 -> 0.095 over training
  trustworthiness (k=8): 0.989   (1.0 = perfect neighbor preservation)
  2-D separation: inter-cluster distance 28x the intra-cluster distance
```

t-SNE matches per-point Gaussian neighbor probabilities in high-D (with each point's bandwidth tuned
by binary search to a target perplexity) to a heavy-tailed Student-t kernel in 2-D, minimizing
KL(P||Q) by gradient descent with momentum and early exaggeration. The heavy tail cures the crowding
problem, letting clusters breathe apart. This module implements perplexity calibration, the symmetric
joint P, the Student-t affinities, and the KL-gradient descent, verified that P is a valid symmetric
distribution, that the perplexity search hits its target, that KL falls over training, that separated
clusters map to separated 2-D groups, and that trustworthiness is high.

## Wavelet trees: rank, select, and quantile over a sequence

A succinct structure answering a whole family of queries in O(log sigma). `wavelet_tree.py`:

```
$ python examples/wavelet_tree_demo.py examples/output

  rank(5, 15)   = 3     (occurrences of 5 in the first 15 positions)
  select(5, 2)  = 9     (position of the 3rd occurrence of 5)
  quantile([4,11), k=3) = 5   (4th-smallest value in that range)
  range_count([0,15), 3..5) = 7   (values in 3..5 across the range)
```

A wavelet tree recursively splits the alphabet at its midpoint, storing one bit per element per level
(upper half = 1, lower half = 0), so a value is encoded by its root-to-leaf bit path. Every query
walks the O(log sigma)-deep tree using bit-rank to map an index into the right child, yielding rank,
select, quantile (k-th smallest in a range -- a range median generalization), and range-count -- in
essentially the space of the sequence. This module implements all five, verified exhaustively against
brute force (access, rank, select, quantile, and range-count) across hundreds of random sequences and
queries.

## Fibonacci heaps: O(1) amortized decrease-key

The priority queue that improves Dijkstra to O(m + n log n). `fibonacci_heap.py`:

```
$ python examples/fibonacci_heap_demo.py examples/output

  extract-min order is sorted; decrease-key 90 -> 5 makes 5 the new min
  merge two heaps in O(1) (root-list concatenation)
  max root degree vs n: n=1000 -> degree 9 (bound 14.4); n=5000 -> 12 (bound 17.7)
  Dijkstra from node 0: [0, 7, 9, 20, 20, 11]
```

A Fibonacci heap stays lazy: insert and merge just splice into a circular root list (O(1)), and
decrease-key cuts a node to the roots, cascading upward via mark bits. Only extract-min consolidates
equal-degree trees, which keeps the tree count logarithmic and decrease-key O(1) amortized. This
module implements the full heap with node handles plus a Dijkstra built on it, verified against a
binary heap over a 2000-operation random stream, that a drained heap yields sorted order, that merge
preserves elements, that the max root degree stays within the log_phi(n) bound, and that Dijkstra
matches a binary-heap Dijkstra across 40 random graphs.

## Treaps: balanced search trees by randomization

Balance for free from random priorities, plus split/merge and order statistics. `treap.py`:

```
$ python examples/treap_demo.py examples/output

  order statistics: 4th-smallest = 40, rank of 65 = 7
  split at 50 -> [10,20,25,30,40] and [50,60,65,70,80]; merge reassembles
  sorted-insert height: n=50000 -> treap 39  vs  2log2(n)=31  vs  plain BST 49999
```

A treap gives each key a random priority and stays a BST on keys and a heap on priorities, so its
shape equals a BST from a random insertion order -- balanced with high probability. Split and merge
(which AVL/red-black trees don't expose) make it ideal for slicing ordered sequences, and subtree-size
augmentation gives O(log n) select and rank. This module implements insert/delete/select/rank/split/
merge, verified against a sorted list over a 3000-operation random stream, with select/rank matching a
sorted array, split/merge round-tripping, and the height staying near 2 log2(n) even under adversarial
sorted insertion.

## Splay trees: self-adjusting search trees that keep hot keys near the root

No balance invariant, yet O(log n) amortized -- plus the working-set property. `splay_tree.py`:

```
$ python examples/splay_tree_demo.py examples/output

  access 40 -> root is now 40   (every access splays the key to the root)
  skewed access (n=4000): avg access depth drops from 16.5 to 8.7 as it concentrates
  a balanced tree pays log2(n) ~ 12 every time, regardless of skew
```

A splay tree rotates every touched node to the root in zig-zig / zig-zag pairs that halve the depth
of everything on the access path, so recently used keys stay near the root (the working-set property)
and the tree is statically optimal to within a constant on any access sequence. This module implements
insert/delete/membership/find-min/max/predecessor/successor with splaying on every access, verified
against a sorted set over a 3000-operation stream (BST and parent invariants hold, pred/succ match a
sorted array, the accessed key is always at the root, and a hot access set drives the average depth
well below log2(n)).

## Van Emde Boas trees: integer sets with O(log log u) successor

For integer keys from a bounded universe, exponentially faster than a BST. `van_emde_boas.py`:

```
$ python examples/van_emde_boas_demo.py examples/output

  successor(21) = 33, predecessor(33) = 21   (universe u=64)
  recursion depth: u=2^32 -> 5 steps, u=2^64 -> 6 steps (a BST would be 32, 64 deep)
  in a 2^24 universe with 5 keys: successor(123456) = 5000000
```

A van Emde Boas tree splits each key into a high half (cluster) and low half (position), recursing on
the square root of the universe, with a summary structure marking non-empty clusters. Storing each
node's min/max directly and not recursing on the min caps the work at one recursive call per level:
T(u) = T(sqrt u) + O(1) = O(log log u). This module implements insert/delete/membership/min/max/
successor/predecessor, verified against a reference sorted set over a 4000-operation stream with
successor and predecessor matching a linear scan at every point in the universe.

## Sparse tables and binary lifting: O(1) range minimum, O(log n) LCA

Constant-time range minimum on a static array, plus lowest common ancestor on a tree. `sparse_table.py`:

```
$ python examples/sparse_table_demo.py examples/output

  min[3,9) = 1, max[3,9) = 9   (each query is two table lookups)
  LCA(7, 8) = 0, distance = 6   (binary lifting on a 9-node tree)
  LCA(3, 4) = 1, distance = 2
```

A sparse table works because min is idempotent: any range is covered by two overlapping power-of-two
blocks, so a query is min(table[l][k], table[r-2^k][k]) in O(1) after O(n log n) preprocessing. The
same doubling gives binary-lifting LCA: store each node's 2^k-th ancestor, lift the deeper node to
the other's depth, then jump both up by shrinking powers until their parents meet -- O(log n), with
tree distance for free. This module implements a generic sparse table (min/max/gcd), RMQ, and LCA,
verified against brute force: the table matches a scan over every subrange, and LCA/distance match a
naive ancestor walk and BFS over many random trees.

## Pollard's rho: factoring integers in sqrt(p) steps

Recover the prime factors that RSA relies on staying hidden. `pollard_rho.py`:

```
$ python examples/pollard_rho_demo.py examples/output

  600851475143 = 71 * 839 * 1471 * 6857
  1000036000099 = 1000003 * 1000033   (a 13-digit semiprime, instantly)
  10000004400000259 = 100000007 * 100000037   (a 17-digit semiprime)
  phi(720720) = 138240, divisors = 240
```

Pollard's rho iterates x -> x^2 + c (mod n); by the birthday paradox the sequence cycles after about
sqrt(p) steps modulo a prime factor p, so gcd(|x_i - x_j|, n) exposes p in expected O(n^{1/4}) time
and O(1) space -- versus the p of trial division. This module implements deterministic Miller-Rabin,
Brent's Pollard rho, Pollard p-1, full recursive factorization, and Euler totient / divisor count
from the factors, verified against brute-force trial division and a sieve: the product of the factors
equals the input, every factor is prime, both primes of random semiprimes are recovered, and totient
and divisor counts match brute-force enumeration.

## Perlin noise: smooth gradient fields for procedural generation

Random-looking but continuous fields -- the basis of procedural terrain and clouds. `perlin.py`:

```
$ python examples/perlin_demo.py examples/output

  noise2(3,5) = 0 exactly (zero at every integer lattice point)
  noise2(3.5, 5.5) = 0.3535 (smooth between)
  100x100 grid: range [-0.86, 0.86], mean -0.008
  fractal Brownian motion layers octaves for detail at every scale
```

Perlin noise assigns each integer lattice point a pseudo-random gradient (from a hashed permutation
table, so it is reproducible and infinite) and interpolates the corner dot products with a smoothstep
fade 6t^5 - 15t^4 + 10t^3, giving a differentiable field that passes through zero on the grid.
Fractal Brownian motion sums octaves at doubling frequency and halving amplitude for natural detail.
This module implements 1-D and 2-D noise and fBm, verified that the noise is exactly zero at lattice
points, stays bounded, is deterministic and continuous (a 1e-3 step moves output by under 0.003), has
near-zero mean, and that more fBm octaves add high-frequency detail.

## Wave function collapse: procedural generation by constraint propagation

Coherent tile maps from local adjacency rules. `wave_function_collapse.py`:

```
$ python examples/wave_function_collapse_demo.py examples/output

  coastline map (land never touches sea; coast always between): 0 violations
  forced checkerboard rules -> a perfect alternating 2-coloring
  every one of 20 seeds satisfies all adjacency rules
```

Each cell starts as a superposition of all tiles; WFC repeatedly collapses the lowest-entropy
(most-constrained) cell to one tile by weight, then propagates -- neighbours lose any option the new
choice forbids, cascading until the grid is arc-consistent. A cell with no options left is a
contradiction and triggers a restart. The per-direction adjacency rules are the whole specification.
This module implements tiled WFC with weighted collapse, lowest-entropy observation, full
propagation, and contradiction restart, verified that every generated grid strictly satisfies the
rules across 20 seeds, that a seed reproduces its grid, that an over-constrained rule set is reported
unsatisfiable, and that a forcing rule set yields exactly its unique tiling.

## The Hungarian algorithm: optimal assignment in O(n^3)

The exact minimum-cost one-to-one assignment, faster than trying all n! permutations. `hungarian.py`:

```
$ python examples/hungarian_demo.py examples/output

  optimal worker->job assignment: 32 hours total
  brute force over all 4! permutations: 32 (matches)
  greedy heuristic: 34 (2 hours worse than optimal)
```

The Hungarian algorithm exploits that subtracting a constant from a full row or column leaves the
optimal assignment unchanged, so it reduces the matrix to expose zeros, selects n independent zeros
(one per row and column), and when fewer exist covers them with a minimum set of lines and shifts the
smallest uncovered value to create new zeros -- until the optimum appears, in O(n^3). This module
implements the potential/augmenting-path form for rectangular matrices (min or max), verified against
brute-force permutation search over 60 random matrices and up to n=8, the row/column reduction
invariant, and rectangular padding.

## Dynamic time warping: aligning time series that vary in speed

Match signals that trace the same shape at different, varying speeds. `dtw.py`:

```
$ python examples/dtw_demo.py examples/output

  two speed-varying signals: DTW distance 2.73 vs Euclidean 17.90 (6.6x smaller)
  stretch invariance: DTW(base, 3x-stretched-base) = 0.0 exactly
  Sakoe-Chiba band=5 restricts the warp: 3.52 >= unconstrained 2.73
```

DTW is dynamic programming over a cost grid where cell (i,j) is the local distance plus the cheapest
of three predecessors (match/insert/delete); the corner is the DTW distance and backtracking recovers
the monotone warping path. A Sakoe-Chiba band confines the path near the diagonal. This module
implements distance, path recovery, an optional band, and a multi-dimensional variant, verified
against an independent DP and known properties: identical series score zero, DTW is symmetric and
non-negative, invariant to time stretching, crushes the Euclidean distance on shifted signals, and
the recovered path is monotone with unit steps whose summed cost equals the distance.

## The P-square algorithm: streaming quantiles in constant memory

Estimate p50/p95/p99 of an endless stream without storing any samples. `p2_quantile.py`:

```
$ python examples/p2_quantile_demo.py examples/output

  200000 latency samples (heavy-tailed):
    p50  49.96 vs exact 49.97   (0.02%)
    p95 134.39 vs exact 134.51  (0.09%)
    p99 212.30 vs exact 212.05  (0.12%)
  memory: 20 floats vs 1562 KB to store and sort every sample
```

P-square tracks five markers (min, max, the target quantile, and two midpoints), each with a height
and a desired position that grows linearly with the sample count; each new value nudges the markers
toward their targets via parabolic interpolation, falling back to linear if the parabola breaks the
ordering. The middle marker is the estimate. This module implements the single-quantile estimator and
a multi-quantile histogram, verified against exact quantiles on uniform, normal, and exponential
streams (errors near 0.01%), with exact min/max markers and a constant-stream sanity check.

## Tarjan's strongly connected components and the condensation DAG

Collapse a directed graph's cycles into a DAG in one DFS. `tarjan_scc.py`:

```
$ python examples/tarjan_scc_demo.py examples/output

  8 vertices -> SCCs {5,6,7}, {3,4}, {0,1,2} (reverse topological order)
  condensation DAG: {0,1,2} -> {3,4} -> {5,6,7}, acyclic
  original graph has a cycle: True; condensation has a cycle: False
```

Tarjan's algorithm gives each vertex a discovery index and a low-link (smallest index reachable via
one back-edge from its subtree); a vertex whose low-link equals its index roots an SCC, and the DFS
stack above it is the component -- all in O(V+E). Shrinking each SCC to a node gives the condensation,
always a DAG. This module implements it iteratively (no recursion-depth limit), plus the condensation,
a Kahn topological sort, and a cycle test, verified against brute-force mutual reachability over 100
random graphs, that the condensation is always acyclic, that SCCs come in reverse topological order,
and on a 5000-cycle that the iterative DFS survives.

## 2-SAT: satisfying two-literal clauses in linear time

NP-complete SAT becomes linear when every clause has two literals. `two_sat.py`:

```
$ python examples/two_sat_demo.py examples/output

  (F1 v F2)(~F1 v F3)(~F2 v ~F3) -> SATISFIABLE: F1=T, F2=F, F3=T
  (a v b)(a v ~b)(~a v b)(~a v ~b) -> UNSATISFIABLE
  verdict matches brute force over 300 random formulas
```

Each clause (a v b) becomes two implications (~a -> b), (~b -> a), forming an implication graph. The
formula is satisfiable iff no variable shares a strongly connected component with its own negation
(otherwise x -> ~x -> x is a contradiction), checkable by Tarjan's SCC in O(V+E); a satisfying
assignment reads off the SCC order since the condensation is a DAG. This module (built on the Tarjan
SCC above) adds clauses/implications, tests satisfiability, and extracts an assignment, verified
against brute force over all 2^n assignments across 300 random formulas plus the canonical
unsatisfiable cases.

## Dormand-Prince RK45: adaptive-step ODE integration

Solve ODEs to a requested accuracy, big steps in calm regions, tiny in fast ones. `rk45.py`:

```
$ python examples/rk45_demo.py examples/output

  y' = -y: y(5) error 1.8e-9 in 35 adaptive steps
  harmonic oscillator: energy drift 2.5e-9 over 10 periods
  tol 1e-4 -> 8 steps / err 1.6e-5;  tol 1e-10 -> 83 steps / err 2e-11
  Van der Pol: 9x bigger steps in smooth stretches than at the switch-backs
```

Dormand-Prince pairs a 5th- and a 4th-order Runge-Kutta formula sharing their stage evaluations, so
their difference estimates the local error, accepting/rejecting each step and rescaling it by
(tol/error)^(1/5) (FSAL reuse makes it seven stages, one recycled). This module solves scalar and
vector ODEs forward or backward with optional sampling at requested times, verified against
closed-form solutions (decay/growth, harmonic energy conservation, logistic, 2-frequency oscillator),
with tighter tolerances shrinking the error and faster dynamics demanding more steps.

## CORDIC: trigonometry and logarithms with only shifts and adds

Transcendental functions with no multiplier -- the algorithm inside early calculators. `cordic.py`:

```
$ python examples/cordic_demo.py examples/output

  cos(1.0472) = 0.5000000000, sin = 0.8660254038 (vs math, ~1e-12)
  atan2(1,1) = pi/4; hypot(3,4) = 5
  exp(2) = 7.389056099; ln(10) = 2.302585093; sqrt(50) = 7.071067812
```

CORDIC rotates a point by ever-smaller angles whose tangents are powers of two, so each rotation is a
bit shift, not a multiply. Circular mode gives cos/sin (and atan2/hypot by vectoring); hyperbolic mode
gives exp/ln/sqrt. This module implements all of them with precomputed angle and gain tables using
only shifts and additions, verified against the math library across their ranges: cos/sin to ~1e-9
over [-2pi,2pi], atan2 in all quadrants, and exp/ln/sqrt to high precision with range reduction.

## Savitzky-Golay: peak-preserving smoothing and noisy differentiation

Smooth noisy data without flattening its peaks, and differentiate it stably. `savitzky_golay.py`:

```
$ python examples/savitzky_golay_demo.py examples/output

  noisy MSE 0.0048 -> moving avg 0.0028 -> Savitzky-Golay 0.0008
  peak height (true 1.45): SG 1.43, moving avg 1.26 (MA flattens it)
  smoothed derivative finds the peaks near 60 and 130
```

Savitzky-Golay fits a low-degree polynomial to a sliding window by least squares and takes the centre
value -- for evenly spaced points this reduces to fixed convolution coefficients, so it is one
convolution. The polynomial follows a peak's curvature, so it smooths without blunting features, and
the derivative of the fit gives a stable estimate of the signal's slope. This module computes
coefficients for any odd window/degree/derivative and applies the filter with edge handling, verified
that a polynomial of degree <= the filter degree passes unchanged, the derivative mode recovers the
analytic derivative, smoothing cuts the MSE to the clean signal, and it beats a moving average at
preserving a Gaussian peak (0.996 vs 0.842).

## LZW: adaptive dictionary compression

Build the codebook on the fly, no dictionary transmitted -- the GIF/compress algorithm. `lzw.py`:

```
$ python examples/lzw_demo.py examples/output

  "TOBEORNOTTOBEORTOBEORNOT" (24 bytes) -> 16 codes, round-trips exactly
  ratio vs repetition: 1x -> 1.00, 10x -> 0.40, 100x -> 0.14
  repetitive 2700 bytes: 0.047; random 2700 bytes: 0.983 (incompressible)
```

LZW starts with every byte as a code, then replaces repeated substrings with single codes, building
its dictionary as it reads; the decoder rebuilds the identical dictionary one step behind, resolving
the KwKwK self-reference (a code for the entry about to be built) by the previous-string-plus-its-own
-first-char rule. This module implements byte-oriented compression/decompression with an optional
GIF-style capped code width, verified by exhaustive round-tripping over 300 random strings, repetitive
data, all-same/all-distinct/empty inputs, the KwKwK case, and the capped-width dictionary reset.

## Convolutional codes and the Viterbi decoder: error correction for noisy channels

Protect a bit stream so channel errors can be undone -- the code that carried Voyager. `convolutional_code.py`:

```
$ python examples/convolutional_code_demo.py examples/output

  message 1011001011 -> encoded (24 bits); 2 bit errors injected -> decoded exactly
  decode success at 5% channel error: coded 0.95 vs uncoded 0.41 (coding gain)
```

A convolutional code feeds the message through a shift register, emitting XOR combinations that
spread each bit across several outputs. The Viterbi algorithm decodes by dynamic programming on the
trellis -- keeping one survivor path per register state (least accumulated Hamming distance) and
tracing back the maximum-likelihood sequence in linear time. This module implements a rate-1/n encoder
for arbitrary generator polynomials and a Viterbi decoder with zero-tail termination, verified that a
clean channel decodes exactly, every single-bit error is corrected, Viterbi matches a brute-force
minimum-distance search over 60 noisy trials, and the classic (7,5) code shows a clear coding gain.

## Ear-clipping: triangulating any simple polygon

Split a concave polygon into n-2 triangles that tile its interior. `ear_clipping.py`:

```
$ python examples/ear_clipping_demo.py examples/output

  star:    10 vertices -> 8 triangles, area 4.7023 == triangle sum
  L-shape:  6 vertices -> 4 triangles, area 5.0000 == triangle sum
  arrow:    7 vertices -> 5 triangles, area 8.0000 == triangle sum
```

The two-ears theorem guarantees a convex vertex whose diagonal stays inside and whose triangle holds
no other vertex; ear clipping snips it and repeats, yielding n-2 triangles in O(n^2). This module
triangulates convex or concave polygons in either winding order, verified against exact references:
the triangle count is always n-2, the areas sum exactly to the polygon's (shoelace), every triangle
centroid lies inside, and stars, L-shapes, arrows, and a deeply non-convex comb triangulate correctly.

## Multi-layer perceptron and backpropagation

A neural net learning what no linear model can. `mlp.py`:

```
$ python examples/mlp_demo.py examples/output

  XOR: [0,0]->0.004 [0,1]->0.991 [1,0]->0.991 [1,1]->0.011 (loss 0.13 -> 4e-5)
  a linear model gets only 3/4 right
  circular decision boundary: accuracy 0.993
```

The MLP stacks nonlinear layers (a universal approximator with one hidden layer) and learns by
backpropagation -- the chain rule run backward, so the whole gradient costs one backward pass. This
module implements a feedforward net with sigmoid/tanh/ReLU activations, full backprop, and momentum
mini-batch SGD, verified that its analytic gradients match finite differences to 8e-11 (the
definitive backprop test), that it learns XOR (which a linear model can't), fits a nonlinear
regression, and separates a circular decision boundary to 99%.

## Markov decision processes: value iteration and policy iteration

Optimal control under uncertainty -- the foundation of reinforcement learning. `mdp.py`:

```
$ python examples/mdp_demo.py examples/output

  5x4 gridworld, 10% slip: value iteration 26 sweeps, policy iteration 3 rounds
  same policy and values, Bellman residual 1e-11
  optimal policy routes around obstacles toward the goal
```

Value iteration applies the Bellman optimality backup until the value function stops changing (a
contraction converging geometrically); policy iteration alternates exact evaluation with greedy
improvement, converging in a few rounds. This module implements both for a finite MDP plus a
stochastic gridworld builder, verified that the two agree on the optimal policy and values, that the
result satisfies Bellman optimality (zero residual), that stochastic slip is handled, and that the
error contracts by exactly the discount factor each sweep.

## Multi-armed bandits: the exploration-exploitation tradeoff

Learn which of several unknown options is best while paying to find out. `bandit.py`:

```
$ python examples/bandit_demo.py examples/output

  5 arms, best p=0.75, over 3000 rounds:
    random         regret 758  best-arm 21%
    epsilon-greedy regret  92  best-arm 90%
    UCB1           regret 151  best-arm 78%
    Thompson       regret  36  best-arm 94%
```

Regret is the reward lost by not always pulling the best arm. This module implements epsilon-greedy,
UCB1 (optimism: mean + sqrt(2 ln t / n), logarithmic regret with no tuning), and Thompson sampling
(Beta-posterior probability-matching), verified that every learning policy beats random, UCB1 and
Thompson achieve sublinear regret, all identify the best arm as the most-pulled, and UCB1's regret
grows logarithmically rather than linearly.

## Q-learning: model-free reinforcement learning from experience

Learn the optimal policy from raw experience, no transition model. `q_learning.py`:

```
$ python examples/q_learning_demo.py examples/output

  6x5 gridworld, agent sees only (s, a, r, s') samples -- never the model
  learned policy reaches the goal in the optimal 9 steps (matches value iteration)
  learned agent 5 steps vs random walker 57 steps
```

Q-learning nudges Q(s,a) toward r + gamma max_a' Q(s',a') after each step (the temporal-difference
update), learning optimal action-values off-policy while exploring epsilon-greedily; SARSA is the
on-policy cousin. This module implements both over a step-based gridworld, verified against value
iteration: the learned greedy policy matches the optimal one and reaches the goal in the optimal
number of steps (4x3 and 6x6 worlds), the learned Q-values approach the MDP optimum, and a learned
agent beats a random walker by an order of magnitude.

## Reverse-mode automatic differentiation

Exact gradients through any expression, the engine behind autograd. `autodiff.py`:

```
$ python examples/autodiff_demo.py examples/output

  f = sin(xy) + e^(z^2)*x - y/z: autodiff gradient == finite diff to 2e-10 (exact)
  trained a*sin(bx)+c: learned a=2.000 b=1.500 c=0.500 (true 2.0/1.5/0.5), loss -> 0
```

A Value records the operation and parents that produced it, so an expression becomes a computation
graph; the backward pass walks it in reverse topological order applying each node's local derivative,
yielding the whole gradient in one sweep. This module implements +, -, *, /, **, and
exp/log/sin/cos/tanh/relu/sqrt with backward rules, verified that gradients match finite differences
and symbolic derivatives, accumulate correctly through graph diamonds, drive gradient descent to the
analytic optimum, and train a model to recover its true parameters exactly.

## Shamir's secret sharing: any k of n pieces reconstruct the secret

Split a secret so no one holds it but any k together recover it. `shamir.py`:

```
$ python examples/shamir_demo.py examples/output

  secret 1234567890 split (3,5): any 3 shares reconstruct it exactly
  just 2 shares -> a wrong value (reveals nothing)
  byte secret b'attack at dawn' round-trips from 2 of 4 shares
```

The secret is the constant term of a random degree-(k-1) polynomial over GF(p); each share is a point
(x, f(x)), and any k points interpolate f(0) = the secret while k-1 leave every secret equally likely
(information-theoretic security). Reconstruction is Lagrange interpolation at x=0 with modular
inverses. This module implements (k,n) splitting of integer or byte-string secrets over a 256-bit
prime and reconstruction, verified that any k shares reconstruct exactly, every k-subset agrees, no
k-1 subset recovers it, and the k=1 and k=n boundaries work.

## Merkle trees: compact proofs of membership in a dataset

Prove an item is in a dataset with O(log n) hashes -- the structure behind blockchains and Git. `merkle.py`:

```
$ python examples/merkle_demo.py examples/output

  8 transaction blocks -> one Merkle root
  inclusion proof for block 3: 3 sibling hashes, verifies against the root
  altered block / forged proof / wrong root all rejected
  1,000,000 blocks -> a 20-hash (640-byte) proof, not a 1M-block download
```

The leaves are block hashes, each parent hashes its two children, and the root fingerprints the whole
ordered set. An inclusion proof is the sibling hash at each level from the leaf to the root; a verifier
who trusts only the root recomputes the path and checks it lands there. This module builds a tree
(SHA-256 with domain-separated leaf/node prefixes), generates and verifies proofs, checked that valid
proofs verify, tampering with the block/proof/root all fail, the proof length is logarithmic, and
changing or reordering any block changes the root.

## The traveling salesman problem: exact Held-Karp and 2-opt

The shortest tour of every city -- solved exactly for small n, near-optimally for large. `tsp.py`:

```
$ python examples/tsp_demo.py examples/output

  11 cities: nearest-neighbour 371 -> Held-Karp optimum 349 (exact)
  60 cities: nearest-neighbour 778 -> 2-opt 684 (12% shorter, crossings removed)
```

Held-Karp is the exact dynamic program: shortest paths over every (subset, last-city) pair, O(n^2 2^n)
instead of O(n!). 2-opt is local search: reverse the segment between two edges whenever it shortens
the tour, removing the self-crossings a good tour never has. This module implements both over an
arbitrary distance matrix, verified that Held-Karp matches brute-force permutation search, 2-opt never
worsens a tour and averages within ~1% of the Held-Karp optimum, every tour is a valid permutation,
and the Euclidean helper satisfies the triangle inequality.

## The Poisson equation by relaxation: fields, potentials, and steady heat

Solve laplacian(u) = f on a grid by sweeping neighbour averages. `poisson.py`:

```
$ python examples/poisson_demo.py examples/output

  40x40 heated plate (hot left 100, cold right 0): center 48.7, harmonic
  convergence: Jacobi 5261 sweeps, Gauss-Seidel 2733, SOR 158 (30x faster)
  point charge in a grounded box: symmetric monotone potential, residual 9e-9
```

Laplace's equation makes every interior point the average of its neighbours (a harmonic field with no
interior extrema); relaxation sweeps the grid averaging until it settles. Jacobi uses old values,
Gauss-Seidel the freshly updated ones, and SOR overshoots each correction by omega in (1,2) to
converge an order of magnitude faster. This module solves 2-D Poisson/Laplace with Dirichlet
boundaries by all three, verified that linear boundary data reproduces the exact harmonic solution,
the mean-value property and maximum principle hold, a separable analytic solution is matched to grid
accuracy, and SOR beats Jacobi by ~30x.

## Manacher's algorithm: every palindrome in linear time

The longest palindromic substring in O(n), not O(n^2). `manacher.py`:

```
$ python examples/manacher_demo.py examples/output

  'racecar'     -> longest 'racecar', 10 palindromic substrings
  'mississippi' -> longest 'ississi'
  radius profile of 'abacabadabacaba': 1 2 1 4 1 2 1 8 1 2 1 4 1 2 1
```

Manacher computes the palindrome radius at every center, reusing the mirror symmetry of already-found
palindromes so each character is touched a constant number of times. Separators make every palindrome
odd-length with one center. This module returns the radii, the longest palindromic substring, and the
count of all palindromic substrings, verified against brute force over 500 random strings that the
longest and the count both match, with edge cases (empty, single, all-same, a 2000-char worst case)
handled.

## Stoer-Wagner: the global minimum cut of a weighted graph

The cheapest way to split a graph in two, without max-flow. `stoer_wagner.py`:

```
$ python examples/stoer_wagner_demo.py examples/output

  classic 8-vertex graph: global min cut 4 (brute force confirms)
  two dense clusters joined by 2 weak links: min cut 2, splits the clusters exactly
```

Stoer-Wagner runs minimum-cut phases: grow a set by repeatedly adding the most tightly-connected
vertex, the last vertex added gives a provable s-t min cut, then merge those two vertices and repeat.
After V-1 phases the smallest cut-of-the-phase is the global minimum, in O(V^3) with no augmenting
paths. This module returns the cut weight and partition, verified against brute force over all vertex
bipartitions of small graphs, on known graphs (bridge, cycle, K_n), that parallel edges are summed,
and that a disconnected graph gives a zero cut.

## Chebyshev approximation: near-optimal fits that dodge the Runge phenomenon

Polynomial approximation that converges where naive interpolation explodes. `chebyshev.py`:

```
$ python examples/chebyshev_demo.py examples/output

  exp(x) on [-1,1]: degree 16 -> max error 2e-15 (machine precision)
  Runge's function 1/(1+25x^2), degree 24:
    Chebyshev error 0.007  vs  equispaced error 257 (the Runge blowup)
```

Sampling at Chebyshev points (clustered at the interval ends) instead of equally-spaced ones makes
polynomial interpolation converge for every continuous function, near-optimally. The function is
expanded in Chebyshev polynomials with coefficients from the node samples and evaluated by Clenshaw
recurrence; the equal-ripple extrema spread the error evenly (equioscillation). This module builds
Chebyshev interpolants on any interval, verified that smooth functions reach near machine precision,
error shrinks geometrically with degree, Runge's function stays bounded where equispaced blows up, and
a low-degree polynomial is recovered exactly.

## Gibbs sampling: drawing a joint distribution via its conditionals

Sample an intractable joint by resampling one coordinate at a time. `gibbs.py`:

```
$ python examples/gibbs_demo.py examples/output

  correlated bivariate Gaussian: sampled mean/cov match target, correlation 0.735 exact
  4-D Gaussian: full mean and covariance recovered
```

Gibbs sampling cycles through the variables, drawing each from its conditional given the rest -- a
Markov chain whose stationary distribution is the target joint, and a Metropolis-Hastings special
case where every proposal is accepted. For a multivariate Gaussian the conditionals are 1-D Gaussians
from the precision matrix. This module implements it for bivariate/multivariate Gaussians and generic
user conditionals, verified that the sampled mean, covariance, and correlation converge to the
target, a 3-D full covariance is recovered, and a discrete conditional sampler reproduces a known
joint.

## Louvain community detection: the natural clusters in a network

Find a network's communities by maximizing modularity. `louvain.py`:

```
$ python examples/louvain_demo.py examples/output

  20 nodes, 4 planted groups + bridges -> Louvain finds 4 communities, modularity 0.66
  every planted group recovered exactly; beats all-in-one (0.0) and singletons (-0.05)
```

Louvain alternates local moving (each node joins the community that most raises modularity) with
aggregation (collapse communities into super-nodes and recurse), each phase only increasing
modularity. This module implements it on weighted undirected graphs, verified that the reported
modularity matches a direct computation, planted communities (cliques joined by sparse bridges) are
recovered exactly, a complete graph gives low modularity, the partition beats both trivial ones, and
weighted edges are respected.

## Matrix-chain multiplication: the optimal order by dynamic programming

The cheapest parenthesization of a matrix product, in O(n^3) not exponential. `matrix_chain.py`:

```
$ python examples/matrix_chain_demo.py examples/output

  CLRS chain: ((A1(A2A3))((A4A5)A6)) -> 15,125 ops vs 40,500 left-to-right (63% saved)
  skewed chain [50,5,100,5,100,5]: 6,375 vs 100,000 (94% saved)
```

The best way to multiply matrices i..j splits at some k with both halves optimal, so
m[i][j] = min_k m[i][k] + m[k+1][j] + p_{i-1} p_k p_j, filled by increasing chain length; recording
each split reconstructs the parenthesization. This module computes the minimum cost and optimal order,
verified against brute-force search over all Catalan-many orderings for short chains, that the
reconstructed order achieves the cost, that it beats left-to-right on skewed dimensions, and on the
CLRS instance (15125).

## Longest increasing subsequence: patience sorting in O(n log n)

The longest strictly-increasing run in a sequence, found by dealing solitaire. `lis.py`:

```
$ python examples/lis_demo.py examples/output

  [3,1,4,1,5,9,2,6,5,3,5,8,9,7,9] -> LIS length 6
  patience piles: 6 piles = LIS length; strict 4 vs non-decreasing 6 on a tied sequence
  longest decreasing subsequence: [9,6,5,3]
```

Patience sorting places each number on the leftmost pile whose top is >= it (binary search) or starts
a new pile; the pile count equals the LIS length, and back-pointers reconstruct the actual
subsequence. This module computes the length and a witness with strict/non-decreasing/decreasing
variants, verified against brute-force O(2^n) search and the O(n^2) DP that the length is optimal and
the returned subsequence is genuine.

## Continued fractions: the best rational approximations of a real

Why 355/113 is such a good approximation of pi. `continued_fraction.py`:

```
$ python examples/continued_fraction_demo.py examples/output

  pi = [3; 7, 15, 1, 292, ...]; convergents 3, 22/7, 333/106, 355/113 (error 2.7e-7)
  golden ratio = [1; 1, 1, 1, ...]; sqrt(2) = [1; 2, 2, 2, ...]
  best rational for pi with denominator <= 113 is exactly 355/113
```

Truncating a continued fraction gives the convergents, the best rational approximations (no smaller
denominator gets closer). A large partial quotient makes the preceding convergent exceptional -- pi's
292 is why 355/113 is accurate to seven digits. This module computes the expansion of a real or exact
fraction, the convergents, and the best rational within a denominator bound, verified that pi's
convergents are the famous ones, a finite expansion recovers its rational, each convergent is best
for its denominator, and the golden ratio and sqrt(2) have their known patterns.

## The Chinese Remainder Theorem: a number from its remainders

Reconstruct a number from its remainders modulo coprime moduli. `crt.py`:

```
$ python examples/crt_demo.py examples/output

  Sunzi: x = 2 mod 3, 3 mod 5, 2 mod 7 -> x = 23 (mod 105)
  reconstruct 8675309 from residues mod [101,103,107,109,113] -> exact
  non-coprime: x = 3 mod 4 and 5 mod 6 -> 11 mod 12; 1 mod 2 and 0 mod 4 -> contradiction
```

For pairwise-coprime moduli CRT builds the unique solution as sum(r_i * M_i * (M_i^-1 mod m_i)) mod M,
with modular inverses from the extended Euclidean algorithm; a generalized version merges non-coprime
congruences and detects contradictions. This module implements extended Euclid, modular inverse, and
both CRTs, verified against brute force that the solution satisfies every congruence and is the
smallest non-negative one, the Bezout identity is correct over 300 random pairs, and non-coprime
systems are solved when consistent and rejected when not.

## Tonelli-Shanks: square roots modulo a prime

Solve x^2 = n (mod p) -- the operation that decompresses elliptic-curve points. `tonelli_shanks.py`:

```
$ python examples/tonelli_shanks_demo.py examples/output

  sqrt(10) mod 13 = (6, 7); sqrt(2) mod 7 = (3, 4); sqrt(5) mod 7 = none (non-residue)
  sqrt(123456) mod 1000033 = (450092, 549941)
  mod 13: 6 residues, 6 non-residues (Legendre symbol splits them)
```

Euler's criterion (the Legendre symbol n^((p-1)/2)) decides existence; for p = 3 mod 4 the root is
n^((p+1)/4), otherwise Tonelli-Shanks iteratively corrects a candidate using a quadratic non-residue.
This module implements the Legendre symbol, residue test, and modular square root, verified against
brute force that the root squares back to n, a root exists iff n is a genuine square (checked over
every residue of 40 primes), both roots are r and p-r, and the Legendre symbol matches a residue
count, up to million-scale primes.

## Baby-step giant-step: the discrete logarithm in O(sqrt(n))

Solve g^x = h (mod m) by meet-in-the-middle -- and see why crypto needs huge groups. `discrete_log.py`:

```
$ python examples/discrete_log_demo.py examples/output

  3^x = 13 mod 17 -> x = 4
  toy Diffie-Hellman (p=7919): eavesdropper recovers Alice's secret 5555 and the shared key
  work: p=100003 -> BSGS ~634 steps vs brute force 100002
```

Writing x = i*N + j with N = ceil(sqrt(n)), BSGS precomputes the baby steps g^j in a hash table and
takes giant steps h*(g^-N)^i, looking each up -- both loops run sqrt(n) times. This module implements
it modulo a prime plus a multiplicative-order helper, verified against brute force that the returned x
satisfies g^x = h, existence and validity agree over dozens of primes, no-solution cases are reported,
and a toy Diffie-Hellman exchange is broken by recovering the secret exponent.

## Welzl's algorithm: the smallest enclosing circle

The tightest circle containing every point, in expected linear time. `welzl.py`:

```
$ python examples/welzl_demo.py examples/output

  50 points -> smallest circle radius 170.56, pinned by 2 boundary points
  8% smaller radius than the bounding-box circumscribed circle
```

The smallest enclosing circle is determined by at most three boundary points; Welzl processes points
in random order, and when one falls outside the current circle it must lie on the new circle's
boundary, so the circle is rebuilt with it fixed there. This module implements the iterative
move-to-front variant, verified against brute force that every point lies inside, the radius matches
an O(n^4) all-triples minimum over 60 random sets, shrinking the radius excludes a point (minimality),
and known cases hold (diameter, circumscribed circle, points on a circle) up to 1000-point sets.

## Cuckoo filters: approximate membership with deletion

Bloom-like membership that Bloom filters can't match: it deletes. `cuckoo_filter.py`:

```
$ python examples/cuckoo_filter_demo.py examples/output

  2000 items, no false negatives; false-positive rate 0.00002 at 16 fingerprint bits
  deletion: 'user-500' present -> deleted -> absent, others unaffected
  FP rate vs bits: 4->0.067, 8->0.0008, 16->~0
```

Each item stores a small fingerprint in one of two candidate buckets, the second found by XORing the
first with hash(fingerprint), so either bucket recovers the other from the fingerprint alone; a full
bucket evicts a resident to its alternate (the cuckoo kick). This module implements add/contains/delete,
verified that it never reports a false negative, the false-positive rate shrinks with fingerprint
size, deletion removes an item while leaving others, deleting one of several duplicates leaves the
rest, and it packs to a ~95% load factor.

## Rollback disjoint-set union: undoable connectivity

Union-find you can undo -- for offline dynamic connectivity. `dsu_rollback.py`:

```
$ python examples/dsu_rollback_demo.py examples/output

  8 nodes, add edges (count drops 8->...->2), roll back 3 edges (count jumps to 5)
  full rollback returns to 8 singletons; connectivity restored exactly
```

Path compression makes union-find fast but un-undoable; using only union by rank, each union changes
O(1) state recorded on a stack, so a snapshot is a stack length and rollback replays it in reverse.
This module implements union/find/component-count/snapshot/rollback, verified against a brute-force
recompute that connectivity always matches a fresh union-find over the live edges (50 graphs), rolling
back restores connectivity and count exactly, nested snapshots and full rollback work, and redundant
unions roll back cleanly.

## Weighted interval scheduling: the most valuable non-overlapping jobs

Pick the highest-value non-conflicting jobs -- where greedy fails and DP wins. `interval_scheduling.py`:

```
$ python examples/interval_scheduling_demo.py examples/output

  8 booking requests -> optimal value 60 (jobs A, C, E)
  count-maximizing greedy: 4 jobs, value 50; highest-value-first greedy: 40
  only the DP guarantees the optimum
```

Sort by finish time; opt(i) = max(skip job i, weight_i + opt(p(i))) where p(i) is the latest job
finishing before i starts (binary search), O(n log n). This module implements the weighted DP with
the chosen jobs and the greedy count-maximizer, verified against brute force over all 2^n subsets that
the DP finds the true maximum, the returned jobs are non-overlapping and sum to it, the greedy count
matches the maximum independent set, and equal weights reduce the DP to the greedy count.

## Coin change: fewest coins, ways to make change, and the greedy trap

Minimum coins and the number of ways -- where greedy fails and DP is exact. `coin_change.py`:

```
$ python examples/coin_change_demo.py examples/output

  US 63c -> 6 coins; the greedy trap: {1,3,4} make 6 -> DP 2 (3+3), greedy 3 (4+1+1)
  {1,15,25} make 30 -> DP 2 (15+15), greedy 6 (25 + five 1s)
  5 with {1,2,5}: 4 combinations, 9 ordered sequences
```

The minimum-coins DP is min[a] = 1 + min over coins c of min[a-c] with reconstructed coins; the
counting DP puts coins in the outer loop for order-independent combinations, the amount outer for
ordered sequences. This module computes minimum coins, makeability, and both counts, verified against
brute force over 150 random instances that the minimum is truly minimal, unmakeable amounts are
detected, the combination count matches enumeration, and greedy-defeating denomination sets are
handled correctly.

## Combinatorial ranking: objects as integers

Every finite combinatorial object can be bijected to an integer. `combinatorial_rank.py`:

```
$ python examples/combinatorial_rank_demo.py examples/output

  permutation rank 12 <-> [2, 0, 1, 3]   (Lehmer / factorial number system)
  3-subset rank 5 of {0..5} <-> [0, 2, 4] (combinatorial number system)
  trillionth permutation of 15: jumps straight there, no enumeration
  Gray code 0..7 -> 0,1,3,2,6,7,5,4      (consecutive codes differ in one bit)
```

Permutations rank via the Lehmer code (each digit counts remaining smaller elements, times descending
factorials); k-subsets rank via the combinatorial number system, a mixed-radix in binomial
coefficients; Gray code ranks bit-strings so consecutive ranks flip exactly one bit (n XOR n>>1). All
exact integer arithmetic, so unranking a trillion-index permutation is instant. Verified against brute
force: rank-then-unrank is the identity, ranks are a contiguous 0..N-1 bijection with no gaps or
collisions, permutation ranks match itertools' lexicographic order, combination ranks match
itertools.combinations, and consecutive Gray codes differ in exactly one bit. This lets you store a
permutation as one integer, draw a uniformly random one, or split an enumeration by rank range without
ever materializing the full set.

## Bridges and articulation points: single points of failure

The structural weak points of a network, found in one DFS pass. `bridges.py`:

```
$ python examples/bridges_demo.py examples/output

  3 clusters joined by thin links -> bridges: 2--3, 5--6, 8--9
  articulation points: [2, 3, 5, 6, 8]
  2-edge-connected components: {0,1,2} {3,4,5} {6,7,8} {9}
```

A BRIDGE is an edge whose removal disconnects the graph; an ARTICULATION POINT is a vertex whose
removal does. Tarjan finds all of them in one O(V+E) DFS using disc[u] (discovery time) and low[u]
(smallest discovery time reachable from u's subtree via one back edge): a tree edge (u,v) is a bridge
iff low[v] > disc[u], and a non-root u is a cut vertex iff some child has low[v] >= disc[u]. Parallel
edges are skipped by edge-id so a doubled link is never falsely flagged; the DFS is iterative so a
5000-deep path doesn't overflow the stack. Verified against the brute-force definition on 400 random
graphs -- an edge is a bridge iff deleting it raises the component count, a vertex is a cut vertex iff
deleting it does; trees have every edge a bridge, cycles and K5 have neither.

## Maximum bipartite matching: pairing two sides

Pair as many workers with jobs as possible, in O(E*sqrt(V)). `bipartite_matching.py`:

```
$ python examples/bipartite_matching_demo.py examples/output

  5 workers, 5 jobs, some qualified for some -> maximum placement: 5 of 5
  Ada->backend  Ben->data  Cam->ops  Dee->frontend  Eli->design
  Konig minimum cover size = 5 = matching size; Hall: perfect placement True
```

A MATCHING is a set of edges sharing no vertex; the maximum one pairs the most. Hopcroft-Karp augments
many vertex-disjoint shortest paths per BFS phase (O(sqrt(V)) phases, O(E*sqrt(V)) total). The result
ties to Konig's theorem (max matching == minimum vertex cover, recovered from alternating-reachability)
and Hall's theorem (a left-perfect matching exists iff every subset S of the left has at least |S|
neighbours). Verified against brute force on 500 random graphs, against Konig (the cover equals the
matching and touches every edge), and against Hall (perfect-left iff the subset condition), plus a
2000x2000 sparse instance. Distinct from `hungarian.py`, which minimises weighted assignment cost;
this maximises the unweighted count on an arbitrary bipartite graph.

## Minimum-cost maximum flow: cheapest way to push the most

Ship the maximum flow through a network at the least total cost. `min_cost_flow.py`:

```
$ python examples/min_cost_flow_demo.py examples/output

  factory-to-store network -> maximum shippable 7 units, min total cost 22
  favours the cheap factB->whY->hub route; whX saturated at 4/4
  same engine as assignment: 3 workers -> 3 jobs, min cost 9
```

Each edge has a capacity and a per-unit cost. SUCCESSIVE SHORTEST PATHS repeatedly pushes flow along
the cheapest source-to-sink path in the residual graph (each edge of cost c gaining a reverse arc of
cost -c), so the running cost stays minimal; when no path remains the flow is both maximum and
cheapest. The negative reverse-arc costs rule out plain Dijkstra, so this uses SPFA (queue-based
Bellman-Ford). It generalises plain max-flow (zero costs) and the assignment problem (unit-capacity
bipartite network). Verified against independent references: the flow value equals the Edmonds-Karp
max-flow (200 nets), the cost is minimal by brute force over all integer flows (120 tiny nets), and as
a bipartite assignment its optimum matches the Hungarian algorithm and the best over all permutations.

## Sprague-Grundy: every impartial game is secretly Nim

Compute who wins any impartial game by reducing it to a Nim heap. `sprague_grundy.py`:

```
$ python examples/sprague_grundy_demo.py examples/output

  Nim (3,4,5): Grundy 2 -> WIN; the only winning move is to (1,4,5)
  subtraction {1,2,3}: Grundy(n) = n mod 4  (losing heaps are multiples of 4)
  Kayles rows 0..12: 0,1,2,3,1,4,3,2,1,4,2,6,4  (the famous irregular nimbers)
```

The Sprague-Grundy theorem: every position of an impartial game (two players, identical moves, last to
move wins) is equivalent to a Nim heap whose size is the position's GRUNDY NUMBER, computed by the mex
rule g = mex{ g(reachable) }; the position is a loss for the mover iff g == 0. Independent subgames
compose by XOR (the Nim-sum), so the winning move is the one making the total Nim-sum zero. Verified
against a brute-force minimax oracle (Grundy==0 iff the position is a theoretical loss, on Nim,
subtraction, and Kayles), against Nim's XOR-of-heaps rule, the periodicity of subtraction-game nimbers,
and the published Kayles sequence.

## Fast Walsh-Hadamard transform: the FFT for XOR

Convolve two sequences over bitwise XOR (or OR, or AND) in O(n log n). `walsh_hadamard.py`:

```
$ python examples/walsh_hadamard_demo.py examples/output

  two 3-bit distributions A,B (each sum 12) combined bitwise:
  A XOR B = [10,11,13,14,26,25,23,22]   (spreads mass evenly)
  A OR  B pushes toward all-ones; A AND B pulls toward zero
  every result conserves total mass 12*12 = 144
```

The FFT does cyclic convolution (indices added mod n); the FWHT does XOR convolution -- (a*b)[k] =
sum over i XOR j == k of a[i]*b[j] -- via a butterfly (x,y)->(x+y,x-y) at each of log2(n) stages, then
pointwise multiply, then inverse. The Hadamard matrix diagonalises the XOR group algebra, so the same
convolution theorem applies. OR and AND convolutions use the sum-over-subsets (zeta) and superset-sum
transforms with their Mobius inverses. Everything is integer-exact. Verified against the brute-force
O(n^2) definition of all three convolutions on hundreds of random arrays, plus round-trip, linearity,
commutativity, the delta identity, and a 2^16-point transform.

## DPLL: deciding Boolean satisfiability

Decide whether a CNF formula can be made true, the engine of every SAT solver. `dpll.py`:

```
$ python examples/dpll_demo.py examples/output

  (x1 v x2 v ~x3) ^ (~x1 v x3) ^ ... -> SATISFIABLE, model x1=T x2=T x3=T
  pigeonhole: 3 pigeons/2 holes, 4/3, 5/4 ... all UNSAT (proven impossible)
  unit propagation: one forced literal cascades x1->x2->...->x5
```

DPLL is depth-first assignment with two pruning rules: UNIT PROPAGATION (a clause with one unassigned
literal forces it, cascading; an all-false clause is a conflict) and PURE LITERAL elimination (a
variable of single polarity is fixed for free). After propagation it branches on a variable, trying
true then false. Verified against a brute-force truth-table oracle on 600 random formulas (SAT exactly
when some assignment works, every model genuinely satisfies all clauses) and on the pigeonhole
principle -- n+1 pigeons into n holes is proven UNSAT without enumerating the 2^n space. Distinct from
`two_sat.py`, which handles only the polynomial 2-literal case; this decides general (NP-complete) SAT.

## Berlekamp-Massey: the recurrence hidden in a sequence

Recover the shortest linear recurrence behind a sequence. `berlekamp_massey.py`:

```
$ python examples/berlekamp_massey_demo.py examples/output

  Fibonacci -> s[n]=s[n-1]+s[n-2]; Pell -> 2s[n-1]+s[n-2]; powers of 2 -> 2s[n-1]
  LFSR keystream (length 5): complexity climbs 3->4->5, cracked at 2L=10 bits
```

Given the first terms, Berlekamp-Massey finds the minimal recurrence s[n] = c1*s[n-1]+...+cL*s[n-L] in
O(n^2), scanning left to right and correcting the current recurrence by a scaled shift of the best
previous failed one whenever the predicted term is wrong. The length L is the LINEAR COMPLEXITY. Works
over the exact rationals (Fibonacci, Tribonacci, Pell -- no rounding) and over GF(2) for LFSRs.
Verified by round-trip (the recovered recurrence regenerates the input and is never longer than the
true generator, 300 random recurrences), against a brute-force minimality search, and on GF(2)
m-sequences whose complexity is recovered from just 2L bits -- the classic reason a raw LFSR keystream
is cryptographically broken.

## Suffix automaton: one tiny machine, every substring

The smallest automaton recognising every substring, built online in O(n). `suffix_automaton.py`:

```
$ python examples/suffix_automaton_demo.py examples/output

  'abracadabra' (11 chars) -> 12 states (bound 2n-1 = 21)
  distinct substrings: 54 (vs 66 raw); 'abra' occurs 2x, 'a' 5x
  longest repeated substring: 'abra'; LCS with 'cadabraxyz': 'cadabra'
```

At most 2n-1 states and 3n-4 transitions encode all O(n^2) substrings; every path from the start spells
a distinct one. Built online in amortised O(n): each state is a class of substrings sharing an endpos
set, states form a tree under suffix links, and appending a character follows those links and clones a
state when a transition conflicts. Distinct substrings = sum of len[v]-len[link[v]]. Verified against
brute force -- the distinct count matches the set of all substrings, membership and occurrence counts
match direct scanning, the longest common substring matches an O(n*m) DP -- on hundreds of random
strings, with the state count within the 2n bound.

## Lyndon words: the primes of strings

Factorise any string uniquely into non-increasing Lyndon words in O(n). `lyndon.py`:

```
$ python examples/lyndon_demo.py examples/output

  banana = b | an | an | a; bbababaab = b | b | ab | ab | aab
  least rotation of 'cabab' -> 'ababc'; 'bca' -> 'abc'
  Lyndon words over {a,b} up to length 4 concatenate to De Bruijn B(2,4)
```

A Lyndon word is strictly smaller than all its rotations; the Chen-Fox-Lyndon theorem factorises every
string uniquely into a non-increasing sequence of them. Duval's algorithm does it in O(n) time and O(1)
space with a two-pointer scan (extend the period on a tie, restart on a larger character, emit Lyndon
words on a smaller one); run over the doubled string it gives the least rotation (necklace
canonicalisation / Booth's problem). The FKM algorithm generates all Lyndon words up to a length, and
concatenating those whose length divides n builds the De Bruijn sequence. Verified against brute force:
factors are Lyndon, non-increasing, and concatenate back; membership matches the rotation definition;
the least rotation matches an exhaustive scan; generated words match a brute filter and the Mobius
necklace-counting formula.

## Eertree: every distinct palindrome in linear space

Store all distinct palindromic substrings of a string in O(n). `eertree.py`:

```
$ python examples/eertree_demo.py examples/output

  'abacabadabacaba' (15 chars) -> 15 distinct palindromes (RICH: hits the n bound)
  by length: a,b,c,d | aba,aca,ada | bacab,badab | abacaba,abadaba | ...
  most frequent: 'a' x8, 'b' x4, 'aba' x4
```

A string has at most n distinct palindromic substrings; the eertree (palindromic tree, Rubinchik 2014)
stores them all in O(n) space and time. It is the palindrome analogue of the suffix automaton: two
roots (lengths -1 and 0), one node per distinct palindrome with a suffix link to its longest proper
palindromic suffix, edges where c from v gives c+v+c. Building is online -- each character walks suffix
links to the longest extendable palindromic suffix, amortised O(n). Verified against brute force: the
distinct count and per-length set match all O(n^2) substrings filtered for the palindrome property,
occurrence counts match direct scanning, the classical <= n bound holds, and the online build equals
batch construction.

## Li Chao tree: the lower envelope of a bundle of lines

Query the min (or max) of a growing set of lines at any x in O(log range). `li_chao.py`:

```
$ python examples/li_chao_demo.py examples/output

  6 lines inserted in arbitrary order -> lower envelope across x:
  x=-12 min -18 (y=2x+6); x=0 min 2 (y=x+2); x=12 min 8 (y=0x+8)
  matches brute-force min over all lines at every integer x
```

Keep a set of lines y = m*x + b and ask for the minimum at a given x; the answer as x sweeps is the
lower envelope (a convex piecewise-linear curve) -- the query behind the convex hull trick that turns
an O(n^2) DP with linear transition costs into O(n log n). The Li Chao tree is a segment tree over the
x-domain where each node owns the line minimal at its midpoint; insertion keeps the lower line and
pushes the other into the half where it might still win (O(log range)), and handles lines in arbitrary
order with interleaved queries -- unlike the monotonic-stack hull trick. Verified against brute force:
min and max queries match the true optimum over all lines on hundreds of random sets and points,
including interleaved insert/query and a convex-hull-trick DP matching its O(n^2) reference.

## Mo's algorithm: batch range queries by reordering

Answer many offline range queries in O((n+q)vn) by ordering them cleverly. `mo_algorithm.py`:

```
$ python examples/mo_algorithm_demo.py examples/output

  20-element array, 8 range queries -> distinct counts + power sums, all match brute
  block size sqrt(20)=4; Mo's order cuts pointer travel from 99 to 56 steps
```

Maintain a current window [l,r] and a running answer, moving the endpoints one element at a time (each
an O(1) add/remove) to morph one query into the next. Sorting queries into sqrt(n)-blocks by left
endpoint, then by right endpoint snaking per block, bounds the total pointer travel at O((n+q)vn). The
problem supplies only a cheap add/remove: a frequency table + nonzero-count for distinct values, an
incremental sum(f^2) for the power sum. Inherently offline (queries permuted). Verified against brute
force: every distinct-count and power-sum answer matches direct recomputation over the subrange, on
hundreds of random arrays and query batches plus a 2000x2000 stress, answers returned in original order.

## Chu-Liu/Edmonds: the directed minimum spanning tree

The cheapest one-way broadcast tree from a root, where Kruskal and Prim don't apply. `arborescence.py`:

```
$ python examples/arborescence_demo.py examples/output

  root 0 broadcasting to 5 nodes; greedy cheapest-incoming traps 1,2,3 in a cycle
  Chu-Liu/Edmonds contracts + reweights it -> min arborescence cost 26
  verified against brute force over all arborescences
```

A minimum spanning ARBORESCENCE gives every non-root vertex exactly one incoming edge, all reachable
from the root, at least total cost. Greedy "cheapest incoming edge" can form a cycle; Chu-Liu/Edmonds
contracts each such cycle into a super-vertex, reweights entering edges by what they'd save, recurses,
then expands -- breaking each cycle at the vertex entered from outside. Verified against brute force
(enumerate one incoming edge per vertex, keep valid arborescences, confirm the minimum) on 500 random
graphs, plus trees, unreachable vertices, multi-edges, and nested cycles; returned edges always form a
genuine spanning arborescence.

## Steiner tree: cheapest network through junctions

Connect a chosen set of terminals at least cost, routing through optional junctions. `steiner_tree.py`:

```
$ python examples/steiner_tree_demo.py examples/output

  4 corner terminals: perimeter-only tree costs 30, routing through the hub costs 12
  Dreyfus-Wagner dp[S][v] = cheapest tree connecting subset S and reaching v
```

Unlike the MST (which spans all vertices), the Steiner tree connects only the terminals, free to route
through Steiner points when cheaper. NP-hard in general, but exact for small terminal counts via the
Dreyfus-Wagner bitmask DP: MERGE two disjoint terminal sub-trees at a shared root, GROW along shortest
paths (a Dijkstra sweep per subset), answer min over v of dp[full][v], in O(3^k n + 2^k n^2). Verified
against brute force (enumerate every subset of Steiner points, take the induced MST) on 400 random
graphs, against the MST when all vertices are terminals, and on cases where a Steiner point strictly
helps.

## AHU tree isomorphism: same shape, in linear time

Decide whether two trees are the same shape via a canonical form. `tree_isomorphism.py`:

```
$ python examples/tree_isomorphism_demo.py examples/output

  rooted canonical form of a 6-node tree: ((()())(()))
  A vs B (A relabelled): isomorphic; A vs C (a path): not isomorphic
  centers found by peeling leaves; unrooted trees rooted there for canonicity
```

Each leaf is '()', each internal node sorts its children's strings and wraps them -- so the encoding
ignores child order and is equal iff rooted trees match. Unrooted trees are rooted at their center (the
1 or 2 vertices at the middle of the longest path, found by peeling leaves), which is
isomorphism-invariant. General graph isomorphism has no known polynomial algorithm; trees fall in
linear time. Verified against brute force (some vertex permutation maps one edge set onto the other) on
hundreds of random trees -- relabelled copies always isomorphic, same-size different-shape never -- with
canonical-form equality coinciding exactly with isomorphism.

## Eulerian trails: every edge once, from Konigsberg to Hierholzer

Walk every edge exactly once, when the degrees allow. `eulerian.py`:

```
$ python examples/eulerian_demo.py examples/output

  triangle -> circuit 0->1->2->0; path 0-1-2-3 -> path
  Seven Bridges of Konigsberg -> no Eulerian trail (all 4 land masses odd)
  bowtie -> circuit 0->1->2->0->3->4->0
```

An Eulerian circuit exists iff (undirected) the graph is connected with all even degrees, or (directed)
every vertex is balanced and weakly connected; a path allows exactly two odd vertices (undirected) or
one +1/one -1 imbalance (directed). Hierholzer's algorithm builds the trail in linear time: walk until
stuck, then splice detours from vertices with unused edges. Handles undirected and directed
multigraphs. Verified: the existence predicate matches the degree/connectivity definition and the trail
uses every edge exactly once with real adjacent steps, on hundreds of random graphs -- including
Konigsberg (correctly impossible) and multigraphs.

## Yen's algorithm: the K best alternative routes

Find the K shortest loopless paths, not just the single best. `yen_ksp.py`:

```
$ python examples/yen_ksp_demo.py examples/output

  4 cheapest A->F routes: 13 (A-C-B-D-E-F), 14, 14, 15
  matches brute-force enumeration of all simple paths
```

Yen's algorithm finds the K shortest simple (no repeated vertex) source-to-target paths in cost order.
The first is plain Dijkstra; each next is found by taking a prefix of the previous path, banning the
edges already used out of the spur node (forcing a different continuation) and the earlier nodes (to
stay loopless), then Dijkstra-ing from the spur to the target -- the cheapest such candidate becomes the
next path. Verified against brute force (enumerate every simple path, sort by cost, take the first K)
on hundreds of random graphs: paths are loopless, valid, distinct, non-decreasing in cost, and exactly
the K cheapest.

## Karger's algorithm: minimum cut by random contraction

Find a graph's weakest point by luck and repetition. `karger.py`:

```
$ python examples/karger_demo.py examples/output

  two heavy triangles + two light bridges -> exact min cut 2
  63/100 single contraction trials happen to hit it; repeated Karger + Karger-Stein: 2
```

Contract a random edge (merging its endpoints), repeat until two super-vertices remain -- the edges
between them are a cut, the minimum one with probability >= 2/(n(n-1)) per run, so O(n^2 log n) runs
make failure vanish. Karger-Stein contracts to ~n/sqrt(2), recurses twice, and keeps the better, for
O(n^2 log^3 n). Weighted graphs pick edges proportional to weight. Being Monte Carlo, it is checked
statistically: a contraction cut is always valid (never below the true minimum) and, given enough
trials, equals the exact Stoer-Wagner value -- confirmed on hundreds of random weighted graphs, with a
fixed seed for reproducibility.

## Bron-Kerbosch: every fully-connected group

Enumerate all maximal cliques of a graph. `bron_kerbosch.py`:

```
$ python examples/bron_kerbosch_demo.py examples/output

  7-person friend network -> 4 maximal cliques: {Ana,Bo,Cy},{Bo,Cy,Di},{Di,Ed,Fi},{Fi,Gu}
  largest fully-connected group: {Ana,Bo,Cy}; matches brute-force subset check
```

Recursive backtracking over R (clique so far), P (candidates), X (already-used); when P and X are empty
R is maximal. PIVOTING branches only on non-neighbours of a pivot in P union X (pruning redundant
branches), and a degeneracy ordering of the outer loop bounds the work to O(d*n*3^(d/3)). Verified
against brute force (a set is a maximal clique iff it is a clique no outside vertex is adjacent to all
of) on hundreds of random graphs, with pivoting and degeneracy variants agreeing, the Moon-Moser graph
yielding its 3^k cliques, and a 60-vertex sparse instance validated.

## Dinic's algorithm: max flow by blocking flows

Maximum flow, faster than Edmonds-Karp by augmenting many paths per phase. `dinic.py`:

```
$ python examples/dinic_demo.py examples/output

  6-node pipe network -> max flow 19 (Edmonds-Karp agrees)
  min cut {S,b}, cut edges S->a (10) + b->d (9) = 19 = max flow
```

Each phase runs a BFS to build the level graph (level[v] = shortest edge-distance from the source),
then a DFS pushes a blocking flow along level-increasing edges, saturating an edge per path; an
iteration pointer skips dead-end edges so a phase costs O(V*E), and only O(V) phases run. Recovering the
source's residual-reachable set gives the min cut. Verified against an independent Edmonds-Karp
reference and the max-flow/min-cut theorem on hundreds of random networks (plus the CLRS graph and a
200-node instance), with a bipartite-matching reduction matching an augmenting-path reference.

## Graph coloring: conflict-free scheduling

Color a graph so neighbours differ, with as few colors as possible. `graph_coloring.py`:

```
$ python examples/graph_coloring_demo.py examples/output

  7 exams, 9 conflicts -> greedy 4, DSATUR 4, exact chromatic number 4
  optimal schedule: slot1 Chem/Econ/Art, slot2 Bio/CS, slot3 Math, slot4 Phys
```

Greedy colors vertices in order (smallest unused color) -- fast but order-dependent. DSATUR colors the
most-saturated vertex next (adjacent to the most distinct colors so far), near-optimal in practice.
Exact chromatic number runs branch-and-bound over increasing k with clique lower and DSATUR upper
bounds. Verified against brute force -- the exact number matches an exhaustive search over all
k-colorings, every coloring is proper, greedy/DSATUR never beat the true minimum, and known values hold
(even cycles 2, odd 3, K_n needs n) -- on hundreds of random graphs.

## k-core decomposition: a network's dense heart

Peel a graph into its densely-connected core and onion layers. `k_core.py`:

```
$ python examples/k_core_demo.py examples/output

  10-node network -> coreness [3,3,3,3,2,2,1,1,1,1], degeneracy 3
  shells: 1-shell {leaves}, 2-shell {middle}, 3-shell {dense K4 core}
```

The k-core is the largest subgraph where every vertex has >= k neighbours inside it; a vertex's
coreness is the largest such k, and the max coreness is the degeneracy. Smallest-last peeling (remove a
minimum-degree vertex; its degree when removed is its coreness) computes all corenesses in O(V+E). The
k-shell (coreness exactly k) gives the network's onion layers. Verified against the brute-force
definition (iterate "remove all degree-<k vertices until stable") on hundreds of random graphs:
coreness matches, k-cores are nested and each vertex has >= k neighbours inside, the degeneracy
ordering has <= degeneracy later-neighbours per vertex, and shells partition the graph.

## Prufer sequences: labeled trees as integer strings

Encode any labeled tree as a length-(n-2) string and back. `prufer.py`:

```
$ python examples/prufer_demo.py examples/output

  star (hub 0) -> [0,0,0,0]; path 0..5 -> [1,2,3,4]; caterpillar -> [1,2,1,2]
  Cayley: n^(n-2) labeled trees, confirmed by enumeration through n=7
```

Encoding strips leaves smallest-first, appending each removed leaf's neighbour; decoding inverts it with
a heap of current leaves; both O(n log n). A vertex appears (degree - 1) times, so leaves never appear.
Since every length-(n-2) string over {0..n-1} is a valid sequence and each gives a distinct tree,
counting trees reduces to counting strings -- Cayley's n^(n-2) formula. Verified against brute force:
encode-then-decode is the identity on hundreds of random trees, every sequence decodes to a valid tree,
the degree reading matches, and Cayley's count is confirmed by exhaustive enumeration through n=7.

## Gale-Shapley: stable matching by deferred acceptance

Pair two ranked groups so no couple wants to defect. `gale_shapley.py`:

```
$ python examples/gale_shapley_demo.py examples/output

  4 applicants <-> 4 schools -> stable matching, unique here (every school gets its #1)
  Ada->Yale, Bo->UCLA, Cy->NYU, Di->MIT; no blocking pair
```

While a proposer is free he proposes to his next-favourite reviewer; she holds her best offer so far
and rejects the rest. Reviewers only trade up and proposers work down their lists, so it ends in <= n^2
proposals with a stable result -- proposer-optimal (best partner each proposer gets in any stable
matching) and reviewer-pessimal; swapping roles gives the reviewer-optimal matching. Verified against
brute force: no blocking pair, everyone matched, no proposer beats his partner in any enumerated stable
matching, the reviewer-optimal variant is reviewer-optimal, and unique-stable instances make the two
agree -- on hundreds of random profiles plus a 200x200 instance.

## Degree sequences: which connection wish-lists are real?

Decide whether a list of degrees is realizable, and build a witness. `degree_sequence.py`:

```
$ python examples/degree_sequence_demo.py examples/output

  (2,2,2) triangle, (4,1,1,1,1) star, K4 -> graphic; (3,3,3,1), odd-sum (1,1,1) -> not
  Havel-Hakimi (3,3,2,2): [3,3,2,2] -> [2,1,1] -> [0,0]; witness graph built
```

Havel-Hakimi removes the largest degree d, subtracts 1 from the next d degrees, and recurses (graphic
iff it reaches all zeros); Erdos-Gallai tests prefix-sum <= k(k-1) + sum min(d_i,k) for every k. The two
always agree. realize() builds a simple witness graph with exactly the requested degrees. Verified
against brute force (search all simple graphs for the exact degree sequence) on hundreds of random
sequences: the criteria agree with each other and with true realizability, the witness is simple with
the right degrees, and the handshake even-sum and d-regular (n*d even) rules hold.

## Matrix-Tree theorem: counting spanning trees with a determinant

Count a graph's spanning trees via one Laplacian cofactor. `matrix_tree.py`:

```
$ python examples/matrix_tree_demo.py examples/output

  K3->3, C4->4, K4->16, diamond->8, P5->1; Cayley K_n = n^(n-2) through n=7
  weighted triangle (2,3,5) -> ab+bc+ca = 31
```

Form the Laplacian L = D - A, delete any one row and its column, take the determinant -- that integer
is the exact spanning-tree count (Kirchhoff, 1847). It generalises Cayley's n^(n-2) and, for weighted
graphs, gives the sum over spanning trees of edge-weight products. Computed with exact Fraction
arithmetic (no float rounding). Verified against brute force (enumerate every size-(n-1) edge subset) on
hundreds of random graphs, against Cayley for complete graphs, and on known values (cycle C_n -> n, tree
-> 1, disconnected -> 0), with the cofactor identical whichever row/column is deleted.

## Hirschberg: optimal alignment in linear space

Compute an optimal sequence alignment in O(min(m,n)) memory. `hirschberg.py`:

```
$ python examples/hirschberg_demo.py examples/output

  AGGTCACGTA vs AGCTACGCA -> score 4 (matches full NW), LCS AGCACGA
  full matrix 110 cells vs Hirschberg's 2 rows = 20; split A at mid, recurse
```

Needleman-Wunsch's O(m*n) memory breaks before its time does; Hirschberg gets the same optimal
alignment in linear space by divide-and-conquer -- split the first sequence at its midpoint, find where
the optimal alignment crosses the second (via a forward and a backward score profile, each two rows),
and recurse on the halves. Verified against a full-matrix Needleman-Wunsch reference on hundreds of
random pairs: the score matches, the alignment degaps to the originals and achieves the optimum under
several scoring schemes, and the LCS matches a standard DP -- including a 3000x3000 pair that is
memory-prohibitive for the full matrix.

## Centroid decomposition: log-deep divide-and-conquer on a tree

Recursively split a tree at its most balanced vertex. `centroid_decomposition.py`:

```
$ python examples/centroid_decomposition_demo.py examples/output

  10-vertex tree -> centroid tree depth 3 (~log2 10); root centroid = vertex 2
  pairs within distance k: 1->9, 2->22, 3->34, 4->42, 5->45; all match brute
```

A centroid's removal leaves pieces of at most half the size, so the centroid tree is O(log n) deep and
every path crosses the centroid of the smallest level containing both endpoints. Counting pairs at
distance <= k: at each centroid gather distances to all vertices, two-pointer count pairs summing to
<= k, subtract per-branch pairs that don't cross it -- O(n log^2 n) versus naive O(n^2). Verified against
all-pairs-BFS brute force for every k on hundreds of random trees, with the centroid tree confirmed a
valid single-rooted O(log n)-deep tree (a 63-vertex path decomposes to depth <= 6) plus a 400-vertex
instance.

## Meet in the middle: halving a brute-force exponent

Solve subset-sum for huge values in O(2^(n/2)) instead of O(2^n). `meet_in_middle.py`:

```
$ python examples/meet_in_middle_demo.py examples/output

  8 items -> subset sums to 50/100 yes, 137/3 no; max under 90 = 87; count(sum=50) = 4
  billion-scale values reachable; brute 2^8=256 vs 2^4+2^4=32 sums
```

Split the items into two halves, enumerate all 2^(n/2) subset-sums of each, then combine: for existence
store the left sums in a set and look up target minus each right sum; for closest-sum or
max-under-capacity sort one half and binary-search the other; for counting, tally with a dictionary.
Works on values far too large for a pseudo-polynomial DP. Verified against brute enumeration of all 2^n
subsets (existence, max-under, closest, count) on hundreds of random instances, plus billion-scale
values and an n=30 instance beyond brute force's reach.

## 2D Fenwick tree: dynamic rectangle sums on a grid

Point updates and rectangle sums on a grid, each in O(log R log C). `fenwick_2d.py`:

```
$ python examples/fenwick_2d_demo.py examples/output

  4x5 grid -> rectangle (1,1)-(2,3) sum 37; add 100 to (2,2) -> re-query 137
  all rectangle queries match a brute grid after interleaved updates
```

A Fenwick tree of Fenwick trees: updating a cell walks both indices up by adding the lowest set bit,
a prefix-rectangle query walks them down by subtracting it, and an arbitrary rectangle is four prefix
sums by inclusion-exclusion. Unlike a static prefix-sum table (O(R*C) to rebuild per update), it stays
fast when cells and queries both change constantly. Verified against a brute 2D prefix-sum reference:
every rectangle query matches after arbitrary interleaved updates on hundreds of random grids, plus a
200x200 grid with 2000 updates.

## Linear sieve: primes, totient, and Mobius in O(N)

Sieve primes and multiplicative functions in true linear time. `linear_sieve.py`:

```
$ python examples/linear_sieve_demo.py examples/output

  25 primes up to 100; 360 = 2^3 * 3^2 * 5 via the SPF table (O(log n))
  phi and mu verified vs brute to 3000; both divisor identities hold
```

Every composite is struck exactly once, by its smallest prime factor (the inner loop breaks the moment
that prime divides i), making the sieve O(N) and yielding the SPF table, Euler totient phi, and Mobius
mu in the same pass. Verified against independent brute force -- trial-division primality/factorization,
coprime-counting phi, squarefree/prime-count mu -- across the whole range to 3000, plus the identities
sum of phi(d) over divisors of n = n and sum of mu(d) = [n==1], and phi's multiplicativity.

## Weighted union-find: relative offsets and contradictions

Solve 'x - y = d' difference constraints incrementally, catching contradictions. `weighted_dsu.py`:

```
$ python examples/weighted_dsu_demo.py examples/output

  A-B=3, B-C=-5, A-C=-2 accepted; A-D=10 REJECTED (true offset is 2)
  parity variant catches an odd cycle: 0!=1, 1!=2, 0!=2 -> contradiction
```

Each element carries a potential relative to its set's root; find() accumulates edge weights while
compressing, union() links two roots with the weight satisfying the relation or, if already connected,
checks consistency and rejects contradictions. A mod-2 variant gives the same-or-different / bipartite
structure. Verified against a brute reference that re-derives all pairwise offsets by BFS over the
accepted-constraint graph: accept/reject decisions and reported differences match, offsets are
symmetric and transitive, ground-truth-derived constraints are always accepted, and the parity variant
matches a mod-2 reference -- on hundreds of random constraint sequences.

## Durand-Kerner: all polynomial roots at once

Find every root of a polynomial simultaneously, real and complex. `durand_kerner.py`:

```
$ python examples/durand_kerner_demo.py examples/output

  x^3-6x^2+11x-6 -> 1,2,3; x^2+1 -> +-i; x^3+1 -> -1, 0.5+-0.866i
  built from {2+3i, 2-3i, -1, 0.5} -> recovered exactly; residuals ~1e-16
```

The Weierstrass iteration r_i <- r_i - p(r_i) / prod_{j!=i}(r_i - r_j) refines all n guesses together
from spread complex seeds -- no bracketing, no derivative, no deflation -- converging quadratically to
every root, including complex ones the real bracketing methods can't see. Verified by construction
(build a polynomial from known roots and recover them), residual (p at each root ~0), and Vieta's
formulas (the roots' symmetric functions reproduce the coefficients) on hundreds of random polynomials.

## Tanh-sinh quadrature: integrating through singularities

Integrate functions that blow up at the endpoints, where classical rules fail. `tanh_sinh.py`:

```
$ python examples/tanh_sinh_demo.py examples/output

  1/sqrt(x) -> 2, ln(1/x) -> 1, 1/sqrt(1-x^2) -> pi/2, 1/sqrt(x(1-x)) -> pi
  tanh-sinh ~1e-8 on these; naive Simpson loses 2-3 digits
```

The substitution x = tanh((pi/2) sinh t) makes the transformed integrand decay double-exponentially
and clusters abscissae exponentially toward the endpoints (never reaching them), so a plain trapezoid
rule in t converges with the correct-digit count roughly doubling per step halving -- taming integrable
endpoint singularities. Verified against closed forms: smooth integrands to machine precision, the
singular integrals of 1/sqrt(x), ln(1/x), 1/sqrt(1-x^2), and both-ends 1/sqrt(x(1-x))=pi, plus a
fine-Simpson reference on smooth cases.

## Number-theoretic transform: exact integer convolution

The FFT in modular arithmetic -- exact, no rounding. `ntt.py`:

```
$ python examples/ntt_demo.py examples/output

  (1+2x+3x^2+4x^3)(5+6x+7x^2) = [5,16,34,52,45,28] exactly (matches schoolbook)
  12345 * 6789 and (10^100-1)^2 computed exactly by digit convolution
```

Replaces the complex root of unity e^(2 pi i / n) with a primitive n-th root modulo the NTT-friendly
prime p = 998244353 = 119*2^23+1 (root 3), so every butterfly is an exact integer operation.
Convolution is transform, pointwise multiply, inverse -- exactly mod p. Powers big-integer
multiplication (convolve digit arrays, then carry). Verified against schoolbook O(n^2) convolution and
Python's exact bignum multiplication -- identical on hundreds of inputs including 500-digit numbers --
plus round-trip and agreement with the complex FFT convolution.

## Karatsuba & Toom-Cook: fast multiplication by fewer sub-products

Multiply big numbers below O(n^2) by trading multiplications for additions. `karatsuba.py`:

```
$ python examples/karatsuba_demo.py examples/output

  12345678 * 87654321 and (10^300-1)^2 computed exactly by both methods
  schoolbook O(n^2), Karatsuba O(n^1.585), Toom-3 O(n^1.465)
```

Karatsuba splits each number in two and gets the middle cross-term from ONE product (x1+x0)(y1+y0)
minus the two already known -- 3 sub-multiplications instead of 4. Toom-3 splits in three, evaluates the
part-polynomials at 5 points, and interpolates -- 5 instead of 9. Verified against Python's exact bignum
multiply and an independent schoolbook limb reference on hundreds of random inputs (including
1000-digit numbers, negatives, edge cases), with the polynomial version matched to direct convolution.

## Strassen: sub-cubic matrix multiplication

Multiply matrices with seven block products instead of eight. `strassen.py`:

```
$ python examples/strassen_demo.py examples/output

  16x16 random matrices: Strassen matches schoolbook; exponent log2(7) ~ 2.807
  n=1024: schoolbook 1.07B mults vs Strassen 282M
```

Split each matrix into four n/2 blocks; the four output blocks assemble from seven block products
(M1..M7 of block sums/differences) recombined by additions, so recursively the exponent drops from 3 to
log2(7). Below a cutoff it falls back to plain multiplication; non-power-of-two and rectangular shapes
are zero-padded. Verified against the schoolbook O(n^3) product -- identical on hundreds of random
matrices of assorted shapes and sizes including deep recursion, floats, and large-integer matrices --
plus identity, associativity, and known products.

## Dancing Links: Algorithm X for exact cover

Solve exact-cover puzzles (Sudoku, N-queens, tilings) with Knuth's pointer dance. `dancing_links.py`:

```
$ python examples/dancing_links_demo.py examples/output

  Knuth's example -> unique cover {B, D, F}; N-queens n=1..8: 1,0,0,2,10,4,40,92
```

The cover matrix is a toroidal doubly-linked list; covering a column splices it out with
x.L.R = x.R; x.R.L = x.L and uncovering restores it with the exact inverse -- no allocation during the
search. Algorithm X covers the column with the fewest options, tries each, recurses, and uncovers on
backtrack. Verified against an independent brute-force subset search (identical solution sets on
hundreds of random instances), the known N-queens counts, and domino-tiling counts.

## WalkSAT: satisfiability by randomized local search

Solve satisfiable SAT instances fast by flipping variables to repair conflicts. `walksat.py`:

```
$ python examples/walksat_demo.py examples/output

  40-var 160-clause 3-SAT (ratio 4.0): DPLL says SAT, WalkSAT finds a verified model
  conflict count falls 20 -> 0 as flips repair unsatisfied clauses
```

Start from a random assignment; while a clause is unsatisfied, pick a random unsatisfied clause and
flip one of its variables -- greedily (fewest clauses broken) most of the time, randomly sometimes to
escape local minima -- with bounded flips and restarts. Incomplete (can't prove UNSAT) but fast on
large satisfiable instances where DPLL's tree explodes. Verified against the complete DPLL solver: on
hundreds of random satisfiable formulas WalkSAT returns a genuine model, and it never falsely claims to
solve an unsatisfiable one.

## Householder QR: orthogonal factorization by reflections

Factor A = Q R stably via reflections, and solve least squares. `householder_qr.py`:

```
$ python examples/householder_qr_demo.py examples/output

  classic [[12,-51,4],...] -> R diag (-14,-175,35); QR-A error ~1e-14
  least-squares line fit to 8 noisy points: y = 1.03 + 1.99 x (data near 1+2x)
```

Applies reflections H = I - 2 v v^T / (v^T v), each mirroring a column onto a coordinate axis to zero
everything below the diagonal; n reflections triangularize A into R and their product is the orthogonal
Q, which stays orthonormal to machine precision where Gram-Schmidt drifts. Verified by reconstruction
(QR = A), orthonormality (Q^T Q = I), R triangularity, and agreement of the least-squares solution with
the normal equations and the repository's Gram-Schmidt QR -- on hundreds of random matrices.

## Jacobi eigenvalues: diagonalising by rotations

Diagonalise a symmetric matrix A = V D V^T via Givens rotations. `jacobi_eigen.py`:

```
$ python examples/jacobi_eigen_demo.py examples/output

  4x4 symmetric -> eigenvalues 6.84, 2.27, 1.08, -2.20; reconstruction error ~1e-15
  off-diagonal norm falls 5.29 -> 2.6e-16 over 19 rotations
```

Each rotation in the (p,q) plane, at the angle with cot(2 theta) = (a_qq-a_pp)/(2 a_pq), zeros the
largest off-diagonal entry; the off-diagonal mass strictly decreases every step, driving A to diagonal
form while the accumulated rotations build the eigenvectors. Verified by reconstruction (V D V^T = A),
orthonormality, the eigen-equation A v = lambda v, the trace and determinant identities, and agreement
with the repository's power-iteration eigenvalues -- on hundreds of random symmetric matrices.

## Kitamasa: the N-th recurrence term in log time

Find the enormous N-th term of a linear recurrence in O(k^2 log n). `kitamasa.py`:

```
$ python examples/kitamasa_demo.py examples/output

  Fibonacci char poly x^2-x-1; F(100) exact = 354224848179261915075
  F(10^18) mod 1e9+7 = 209783453 (impossible to unroll)
```

s[n] is a fixed linear combination of the first k terms whose coefficients are those of x^n reduced
modulo the characteristic polynomial c(x) = x^k - c1 x^(k-1) - ... - ck. Computing x^n mod c(x) by
square-and-multiply (each reduction O(k^2)) then dotting with the initial terms gives s[n] in
O(k^2 log n) -- beating even the O(k^3 log n) companion-matrix power. Works over integers, rationals, or
any modulus. Verified against O(n) unrolling for moderate n, known closed forms, modular
exact-then-reduce, and Cassini's identity at n = 10^15 -- on hundreds of random recurrences.

## Exact rational RREF: linear algebra with certainty

Row-reduce over exact fractions -- rank, null space, and solving with no rounding. `rational_rref.py`:

```
$ python examples/rational_rref_demo.py examples/output

  [[1,2,3],[4,5,6],[7,8,9]] -> RREF, rank 2, null space (1,-2,1)
  systems: unique (4/5,7/5), inconsistent -> none, x+y+z=6 -> infinite
```

Gaussian elimination over Python's Fraction: find a nonzero pivot, scale it to 1, clear its column
elsewhere; pivot columns are basic, the rest free, and the pivot count is the rank. Because every pivot
test is exact, the rank is never misjudged the way a floating-point solver's might be. Solving augments
and reduces, returning a unique solution, "none" for inconsistent systems, or a particular solution
plus a null-space basis for underdetermined ones. Verified against an independent elimination count
(rank), the rank-nullity theorem, A x = 0 for null-space vectors, exact A x = b for solutions, and RREF
idempotence -- on hundreds of random singular and rectangular matrices.

## Rabin-Karp: substring search by rolling hash

Find substrings in O(n+m) average time by hashing windows. `rabin_karp.py`:

```
$ python examples/rabin_karp_demo.py examples/output

  'abra' in 'abracadabra_abracadabra' -> [0, 7, 12, 19]; multi-pattern in one pass
  longest common substring of two sentences -> ' quick brown ' (length 13)
```

Each window is a base-B number mod a 61-bit prime; the rolling hash updates in O(1) (drop the leaving
character, shift, add the arriving one), and only hash-equal windows are compared in full -- so search
is fast and, because every match is verified, always exact. Multi-pattern search groups patterns by
length and scans once; the longest common substring binary-searches the length with hashed windows.
Verified against brute-force search on hundreds of random pairs (and a 20k-char text) and the LCS
against an O(n*m) DP.

## Half-plane intersection: the feasible region of constraints

Intersect linear inequalities into their feasible convex polygon. `half_plane_intersection.py`:

```
$ python examples/half_plane_intersection_demo.py examples/output

  5 constraints -> feasible 5-gon, area 14.25; maximise 3x+2y at vertex (4.5,1.5)
  x<=0 and x>=1 -> infeasible (empty region)
```

Each inequality a x + b y <= c is a half-plane; clipping a large bounding box against them one by one
(Sutherland-Hodgman) carves out their intersection -- a convex polygon, or empty if they conflict. This
is exactly the feasible region of a 2-D linear program, whose optimum sits at a vertex. Verified against
brute force -- a dense grid where a point is feasible iff it satisfies every inequality must match the
computed polygon -- on hundreds of random systems, plus box/triangle/infeasible cases with exact areas.

## Sweep-line segment intersection: all crossings

Find every crossing among n segments without testing all pairs. `bentley_ottmann.py`:

```
$ python examples/bentley_ottmann_demo.py examples/output

  6 segments -> 14 intersecting pairs (matches brute); 20x20 line grid -> 400 crossings
```

A vertical line sweeps left to right, keeping only the segments whose x-range straddles it (the active
set); a new segment is tested only against those, never against segments already ended or not yet
begun, so every x-disjoint pair is pruned. Any crossing pair is active together at some sweep position,
so none is missed. Verified against the brute all-pairs test on thousands of random arrangements, plus
grids, a star of concurrent segments, parallel families, and shared-endpoint cases, with every reported
crossing point confirmed to lie on both segments.

## GJK: convex collision by the Minkowski difference

Detect whether two convex shapes overlap without building their intersection. `gjk.py`:

```
$ python examples/gjk_demo.py examples/output

  overlapping squares, touching corner, pentagon-in-hexagon -> collide; far apart -> not
  GJK verdict matches both SAT and the Minkowski-contains-origin test
```

Convex A and B overlap iff their Minkowski difference A(-)B = {a - b} contains the origin. GJK explores
that difference lazily via a support function (the farthest vertex in a direction), growing a simplex
toward the origin -- deciding overlap in a few iterations regardless of vertex count, the collision core
of physics engines. Verified against two independent references, the Separating Axis Theorem and a
convex-hull Minkowski-origin test, with identical verdicts on hundreds of random polygon pairs, plus
translation-invariance and self-collision.

## Kolmogorov-Smirnov: comparing distributions by their largest gap

Test goodness-of-fit or whether two samples match, distribution-free. `ks_test.py`:

```
$ python examples/ks_test_demo.py examples/output

  300 uniform draws vs Uniform CDF -> D 0.044, p 0.62 (fit accepted)
  uniform vs shifted-uniform -> D 0.415, p 0.0000 (DIFFERENT)
```

The statistic D is the largest vertical gap between the compared CDFs -- for one sample the sample's
empirical CDF vs a reference, for two samples the two empirical CDFs. Under the null D's distribution
is universal (the Kolmogorov distribution), so one critical-value table works for any continuous law.
Verified against the brute ECDF gap on a fine grid (one- and two-sample), the D=0/D=1 extremes and
symmetry, and statistically -- same-distribution samples rarely reject at 5% while clearly different
ones reject with high power over many seeded trials.

## Bootstrap: confidence intervals by resampling

Get a confidence interval for any statistic without a variance formula. `bootstrap.py`:

```
$ python examples/bootstrap_demo.py examples/output

  mean of N(50,8) sample: bootstrap SE 0.74 ~ analytic s/sqrt(n) 0.75; 95% CI brackets 50
  median and std CIs from the same resampling; BCa corrects skew via the jackknife
```

Resample the data with replacement thousands of times, recompute the statistic each time, and the
spread of those replicates is its sampling distribution -- read the CI off its percentiles. The BCa
interval corrects for median bias and skewness (acceleration from the jackknife) for sharper coverage.
Verified against known quantities -- bootstrap SE of the mean matches analytic s/sqrt(n), the jackknife
is exact for the mean, intervals bracket the estimate -- and a 90% interval covers the true mean about
90% of the time over 200 seeded datasets.

## Permutation test: significance by shuffling labels

Test whether two groups differ without assuming any distribution. `permutation_test.py`:

```
$ python examples/permutation_test_demo.py examples/output

  small trial: enumerated 126 permutations -> exact p = 0.0079
  larger trial: 20000 Monte-Carlo shuffles, t-statistic -> p = 0.0227
  paired before/after: 256 sign patterns -> p = 0.0078
```

Under the null that both groups come from the same distribution, the labels are exchangeable, so
every relabelling of the pooled data is equally likely. Compute the statistic, recompute it for each
relabelling, and the fraction at least as extreme is the p-value -- exact when all C(N, n_A)
permutations are enumerated, Monte-Carlo with the (b+1)/(B+1) correction when there are too many. The
statistic is pluggable (mean, median, Welch t) and a paired sign-flip variant covers before/after
designs. Validated three ways -- the Monte-Carlo p matches complete enumeration, tracks the analytic
t-test p on normal data, and rejects at the nominal rate under a true null over many seeded trials.

## LLL: reducing a lattice to a short, near-orthogonal basis

Turn a skewed integer basis into short, nearly-perpendicular vectors spanning the same lattice.
`lll.py`:

```
$ python examples/lll_demo.py examples/output

  original basis [[201,37],[98,18]]  norms^2 [41770, 9928]
  reduced  basis [[-2,-2],[1,-3]]    norms^2 [8, 10]
  integer relation among [6,3,4]: [1,-2,0]  ->  1*6 - 2*3 = 0
```

A lattice is every integer combination of its basis; infinitely many bases (related by
determinant-±1 integer matrices) span the same points. LLL alternates size reduction (Gram-Schmidt
coefficients ≤ 1/2) and Lovász swaps to produce a short, near-orthogonal basis in polynomial time.
All Gram-Schmidt arithmetic is exact over the rationals, so it never mis-swaps on a rounding error.
Validated: the reduction is unimodular (Gram determinant, hence the lattice, is preserved), the
output provably satisfies both LLL conditions, the reduced shortest vector matches a brute-force
search over integer combinations, and known integer relations are recovered exactly.

## Levinson-Durbin: Toeplitz solving and autoregressive models

Solve a symmetric Toeplitz system in O(n^2) and fit autoregressive models. `levinson_durbin.py`:

```
$ python examples/levinson_durbin_demo.py examples/output

  Toeplitz T x = b (first row [4,1,0]), b=[1,2,3] -> x=[0.179, 0.286, 0.679]
  AR(2) fit: recovered [0.735, -0.499] vs true [0.75, -0.50]; error var 0.092
  reflection coeffs all |k|<1 -> stable; 5-step forecast produced
```

A stationary signal has a Toeplitz covariance matrix, so its Yule-Walker equations are Toeplitz and
solvable in O(n^2) instead of O(n^3). The recursion grows the solution one order at a time via
reflection (PARCOR) coefficients, which double as a positive-definiteness / stability test (|k| < 1)
and hand back the prediction-error variance for free. The AR coefficients are what LPC speech coding,
spectral estimation, and one-step forecasting use. Validated: the solver matches a dense
Gaussian-elimination solve to machine precision on random positive-definite Toeplitz systems, and on
data from a known AR(2) process the recovered coefficients match the generating [0.75, -0.5] and the
reported error variance equals the measured residual variance.

## Poisson-disk sampling: blue noise by Bridson's algorithm

Scatter points that look random but never crowd. `poisson_disk.py`:

```
$ python examples/poisson_disk_demo.py examples/output

  Poisson-disk: 301 points, min distance 14.02 (radius 14.0), gap fraction 0.0000
  uniform random (301 points): min distance 0.54  <- clumps badly
```

Uniform random points clump; blue noise keeps a guaranteed minimum spacing while still looking
unstructured, which is what stippling, dithering, vegetation scatter, and anti-aliasing want.
Bridson's algorithm (2007) achieves it in O(n) with a background grid of cells r/sqrt(2) across (at
most one sample per cell) so each candidate is tested only against a constant-size neighbourhood, and
every accepted point lands in the annulus [r, 2r) of an active sample. Validated: the closest pair is
never nearer than r (all-pairs scan), the packing is near-maximal (higher candidate count k drives the
insertable gap to zero), the count sits within disk-packing density bounds, and the n-D sampler holds
the same invariant.

## Low-discrepancy sequences: quasi-Monte-Carlo integration

Integrate faster than random sampling with deterministic space-filling points. `low_discrepancy.py`:

```
$ python examples/low_discrepancy_demo.py examples/output

  star discrepancy of 256 points: Halton 0.0149 vs pseudo-random 0.0474 (3.2x worse)
  pi via quarter disk, N=1000: QMC err 0.0064 vs MC err 0.0944
```

Monte-Carlo error shrinks like 1/sqrt(N) because random points clump; quasi-Monte-Carlo uses a
low-discrepancy sequence whose points fill space evenly, giving error near (log N)^d / N. The van der
Corput sequence reflects an index's base-b digits about the radix point so each point lands in the
largest gap; Halton uses a coprime prime base per axis, Hammersley pins the first coordinate to n/N.
Validated: the radical inverse matches hand-computed values, van der Corput exactly stratifies its
first b^m points, Halton's star discrepancy is several times smaller than pseudo-random and shrinks
with N, and QMC integration of a smooth function beats the average plain-MC error at the same N.

## Thompson's NFA: regex matching without catastrophic backtracking

A regex engine that matches in guaranteed linear time. `thompson_nfa.py`:

```
$ python examples/thompson_nfa_demo.py examples/output

  /(a|b)*c/ matches 'abbac', rejects 'ababba' (agrees with Python re)
  catastrophic /(a*)*b/ on 800 a's: 0.95 ms  (backtracking engine would hang)
```

Most languages' regex matchers backtrack, so `(a*)*b` on a long run of `a` can hang for seconds.
Thompson's 1968 construction never backtracks: it tracks the set of all NFA states the machine could
be in and advances the whole set one character at a time, so matching costs O(n*m) always. The engine
implements a recursive-descent parser (`|`, concatenation, `* + ?`, groups, `.`, `\` escapes), a
Thompson compiler gluing two-state machines with epsilon transitions, and a subset-construction
simulator with a cycle-safe epsilon-closure. Validated against Python's `re` as an oracle: over 1000
seeded random (pattern, string) pairs, `fullmatch` and `search` agree on every case, and the
catastrophic pattern returns in under a millisecond with match time scaling linearly.

## Bluestein's algorithm: the FFT for any length, even a prime

Compute an exact DFT at any length in O(n log n), not just powers of two. `bluestein.py`:

```
$ python examples/bluestein_demo.py examples/output

  length 251 (prime): Bluestein vs direct DFT max error 4.6e-12
  n=1021: Bluestein 8.2 ms vs direct 292 ms -> 36x speedup
```

The Cooley-Tukey FFT is fast only for nicely-factoring lengths; a prime degrades it to O(n^2).
Bluestein rewrites the DFT exponent `n*k = (n^2 + k^2 - (k-n)^2)/2`, turning the transform into a
convolution that a power-of-two FFT does in O(n log n) -- so any length is fast, with no zero-padding
that would change the transform. Implements arbitrary-length DFT, inverse, and linear convolution on
a self-contained radix-2 FFT. Validated against a direct O(n^2) DFT at powers of two, composites, and
primes (7, 13, 101, 251) to machine precision, plus round-trip inversion, linearity, known transforms,
and convolution against a naive reference. Speedup over the direct sum grows from 6x at n=127 to 36x
at n=1021.

## Vantage-point trees: nearest neighbours in any metric space

Nearest-neighbour search when your data have no coordinates. `vp_tree.py`:

```
$ python examples/vp_tree_demo.py examples/output

  2D 8-NN over 1000 points: matches brute force, touches 89 points (8.9%)
  edit-distance NN: 'alocator' -> allocator(1), aligator(2), alternator(4)
```

A k-d tree needs axes to split on; a vantage-point tree (Yianilos 1993) indexes any metric space from
pairwise distances alone. It picks a vantage point, splits the rest at the median distance into an
inside ball and an outside shell, and recurses; queries prune whole subtrees whenever the triangle
inequality proves nothing there can beat the current best. Works for Euclidean/Manhattan points, string
edit distance, angular distance -- any metric. Validated: single- and k-nearest results match a
brute-force scan on hundreds of seeded queries across all those metrics, range queries match a radius
filter exactly, and only ~9% of points are touched on a 1000-point query (the pruning fires). Note that
1 - cosine similarity is not a metric (it breaks the triangle inequality), so the angular distance
`arccos(cos)/pi` is used instead.

## The t-digest: streaming quantiles with sharp tails

Get p50/p90/p99/p999 from one pass over a stream too big to store. `tdigest.py`:

```
$ python examples/tdigest_demo.py examples/output

  500,000 latency samples -> 64 centroids (~7800x smaller)
  p99.99: t-digest 236.8 vs exact 195.0, rank error 0.00008
  8 per-shard digests merge into one: p99 matches the single digest
```

The t-digest (Dunning 2013) summarises a stream as centroids sized by a scale function
`k(q) = (delta/2pi) arcsin(2q-1)` that is compressed at the tails, so the tails get many tiny
high-resolution centroids while the median gets a few coarse ones -- accurate where it matters. The
centroid count stays bounded (64 for 500k values), any quantile or CDF can be queried, and merges are
associative so per-shard digests roll up exactly. Validated against exact sorted quantiles on uniform,
normal, exponential, and skewed streams: estimates within a few percent, deep-tail (p999) rank error
under 0.01, bounded size, and split-then-merge matching a single digest.

## Myers' diff: the shortest edit script behind git

The algorithm that produces every `git diff`. `myers_diff.py`:

```
$ python examples/myers_diff_demo.py examples/output

  char diff 'ABCABBA' -> 'CBABAC': distance 5
  single changed line among 500: distance 2 (Myers' sweet spot)
```

Given two sequences, Myers (1986) finds the shortest edit script -- fewest single-element deletions and
insertions -- turning A into B, the dual of the longest common subsequence. Rather than fill an O(N*M)
DP table, it searches the edit graph (diagonal = free match, right/down = insert/delete) with a BFS over
the edit count D, running in O((N+M)*D), tiny when files are similar. Computes distance, the actual
edit script, the LCS, and a unified diff. Validated: distance equals `len(A)+len(B)-2*LCS` cross-checked
against an independent DP on hundreds of random pairs, applying the reconstructed script to A reproduces
B exactly every time, and the LCS is a genuine subsequence of both with the correct length.

## Push-relabel: maximum flow from the other direction

Max flow via preflow and heights instead of augmenting paths. `push_relabel.py`:

```
$ python examples/push_relabel_demo.py examples/output

  CLRS 6-node network: push-relabel = 23, Dinic = 23 (MATCH)
  min cut {0,1,2,4} capacity 23 = max flow (max-flow min-cut theorem)
```

Push-relabel (Goldberg-Tarjan 1988) maintains a preflow -- nodes may hold excess -- and a height per
node, then PUSHES excess downhill or RELABELS stuck nodes upward until only the sink holds excess.
Uses the FIFO rule (O(V^3)) plus the gap heuristic. Returns flow value, per-edge flow, and the min
cut. Validated against the repo's independent Dinic solver on 300 random networks (flow values agree
every time), plus flow conservation, capacity limits, the max-flow min-cut theorem, and bipartite
matching vs a brute-force augmenting search.

## Butterworth filters: maximally-flat frequency response

Design digital low/high-pass filters with a ripple-free passband. `butterworth.py`:

```
$ python examples/butterworth_demo.py examples/output

  order 8 low-pass fc=0.1: -3.01 dB at cutoff, -55.9 dB one octave up
  denoising a buried sine: RMS error 0.31 -> 0.044 (7.1x cleaner)
```

The Butterworth filter is maximally flat -- no passband ripple, gain falling monotonically. Designed by
the textbook pipeline: analog prototype poles on the unit circle, tangent frequency pre-warp, then the
bilinear transform `s = (1-z^-1)/(1+z^-1)` to get the digital coefficients. Applies by direct-form
recurrence or zero-phase filtfilt. Validated against the defining Butterworth properties: exactly -3 dB
at the cutoff for every order/cutoff, monotone (no-ripple) response, ~6 dB/octave per order of
roll-off, and clean two-tone separation -- all pure stdlib, no scipy.

## The unscented Kalman filter: nonlinear state estimation

Track a hidden state through nonlinear dynamics without Jacobians. `unscented_kalman.py`:

```
$ python examples/unscented_kalman_demo.py examples/output

  projectile from noisy range/bearing: raw RMS 4.49 m -> UKF RMS 1.03 m (4.3x better)
```

The plain Kalman filter is optimal only for linear systems. The UKF (Julier & Uhlmann 1997) handles
nonlinear dynamics via the unscented transform: pick sigma points that capture the state's mean and
covariance, push each through the true nonlinear function, and recover the transformed mean and
covariance -- accurate to second order, no derivatives. Validated the decisive way: on a linear system
it reproduces the repo's own linear Kalman filter step for step (the transform is exact for affine
maps), the covariance stays symmetric positive-definite, and on nonlinear tracking (projectile in
range/bearing, pendulum by angle) the RMS error falls well below the measurement noise.

## The discrete cosine transform: energy compaction behind JPEG

The real-valued transform that powers JPEG and MP3. `dct.py`:

```
$ python examples/dct_demo.py examples/output

  smooth 64-pt signal: top 8 of 64 coeffs hold 99.76% of energy
  keep 12% of coeffs -> reconstruct at 4.9% RMS error
```

The DCT reflects a signal evenly at its ends, giving a real spectrum whose energy piles into the
lowest frequencies -- the compaction that makes lossy compression work. Implements orthonormal
DCT-II/III, a fast O(N log N) route via a radix-2 FFT (Makhoul), and the separable 2D DCT JPEG runs on
8x8 blocks. Validated: fast matches the direct O(N^2) definition to machine precision, the transform
is orthonormal (inverse recovers input, energy preserved by Parseval), the basis is orthonormal, a
smooth signal compacts into a few coefficients while noise does not, and the 2D block transform
inverts exactly.

## Quaternions: gimbal-lock-free 3D rotation

The rotation algebra behind aerospace and graphics. `quaternion.py`:

```
$ python examples/quaternion_demo.py examples/output

  slerp over 170 deg: 34/68/102/136 at t=0.2..0.8 (even)
  lerp: 27/64/106/143 (lags then rushes -- uneven angular velocity)
```

A unit quaternion `(cos(theta/2), sin(theta/2)*axis)` encodes a rotation; composing two is one Hamilton
multiply, rotating a vector is the sandwich `q v q^-1`, and slerp walks the great-circle arc at constant
speed. Four numbers, one unit constraint, no gimbal lock. Provides arithmetic, vector rotation, slerp,
and conversions to/from axis-angle, rotation matrices, and Euler angles. Validated: rotating a vector
equals multiplying by the equivalent rotation matrix, every conversion round-trips (up to q/-q sign),
composition is a homomorphism with an orthonormal det-+1 matrix, and slerp traverses at constant angular
velocity.

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
