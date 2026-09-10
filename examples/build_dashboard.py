"""Build a single self-contained HTML dashboard from every demo in the repo.

Runs each demo, captures its text output, inlines the rendered SVGs, and writes
one index.html you can open locally or serve via GitHub Pages. Zero dependencies.

    python examples/build_dashboard.py [output_dir]   # default: examples/output
"""

import html
import io
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def capture(fn, *args):
    """Run fn(*args), return whatever it printed to stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args)
    return buf.getvalue()


def inline_svg(path):
    with open(path, "r", encoding="utf-8") as f:
        svg = f.read()
    # strip the XML width/height so CSS can make it responsive
    return svg


def section(title, blurb, body_html):
    return f"""
  <section>
    <h2>{html.escape(title)}</h2>
    <p class="blurb">{blurb}</p>
    {body_html}
  </section>"""


def pre(text):
    return f'<pre>{html.escape(text.strip())}</pre>'


def svg_card(path, caption):
    if not os.path.exists(path):
        return f'<div class="card missing">missing: {html.escape(os.path.basename(path))}</div>'
    return (f'<figure class="card">{inline_svg(path)}'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>')


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "output")
    os.makedirs(outdir, exist_ok=True)

    # import demos lazily so their module-level sys.path tweaks are set
    import energy_drift_demo
    import convergence_demo
    import scaling_benchmark
    import adaptive_demo
    import lagrange_demo
    import solar_system_demo
    import precession_demo
    import chaos_demo
    import poincare_demo
    import galaxy_collision_demo
    import hermite_demo
    import gravwave_demo
    import circularization_demo
    import sitnikov_demo
    import virial_demo
    import roche_demo
    import kozai_demo
    import resonance_demo
    import coorbital_demo
    import tisserand_demo
    import lensing_demo
    import rotation_curve_demo
    import schwarzschild_demo
    import friedmann_demo
    import lane_emden_demo
    import distances_demo
    import chandrasekhar_demo
    import tov_demo
    import jeans_demo
    import sedov_demo
    import kerr_demo
    import hawking_demo
    import eddington_demo
    import hohmann_demo
    import gr_time_demo
    import pulsar_demo
    import mond_demo
    import penrose_demo
    import tidal_heating_demo
    import lense_thirring_demo
    import oberth_demo
    import dynamical_friction_demo
    import gw_strain_demo
    import saha_demo
    import cmb_demo
    import bbn_demo
    import cluster_demo
    import roche_lobe_demo
    import degeneracy_demo
    import blackbody_demo
    import main_sequence_demo
    import bondi_demo
    import cosmic_velocities_demo
    import compton_demo
    import synchrotron_demo
    import sz_demo
    import kelvin_helmholtz_demo
    import atmosphere_demo
    import larmor_demo
    import optical_depth_demo
    import exoplanet_demo
    import habitable_zone_demo
    import faber_jackson_demo
    import focusing_demo
    import bremsstrahlung_demo
    import pair_production_demo
    import axial_precession_demo
    import alfven_demo
    import parker_spiral_demo
    import magnetic_braking_demo
    import tidal_locking_demo
    import jeans_escape_demo
    import snow_line_demo
    import poynting_robertson_demo
    import toomre_demo
    import accretion_disk_demo
    import fermi_acceleration_demo
    import opacity_demo
    import brunt_vaisala_demo
    import ram_pressure_demo
    import free_fall_demo
    import shock_jump_demo
    import stromgren_demo
    import relaxation_time_demo
    import parker_wind_demo
    import greenhouse_demo
    import rossby_demo
    import rayleigh_benard_demo
    import terminal_velocity_demo
    import snr_phases_demo
    import magnetic_mirror_demo
    import debye_demo
    import line_broadening_demo
    import curve_of_growth_demo
    import sackur_tetrode_demo
    import maxwell_boltzmann_demo

    import plot_orbits

    fast = "--fast" in sys.argv
    _old_argv = sys.argv

    # Each demo runs ONCE. Demos that render SVGs take an output dir via argv and
    # write their figures as a side effect of the same call we capture text from.
    # Captured text is cached to <name>.txt so a --fast rebuild can reuse it and
    # skip the (sometimes minute-long) simulation entirely.
    def run(name, fn, needs_dir):
        cache = os.path.join(outdir, f"_{name}.txt")
        if fast and os.path.exists(cache):
            with open(cache, encoding="utf-8") as f:
                return f.read()
        if needs_dir:
            sys.argv = [name, outdir]
        txt = capture(fn)
        sys.argv = _old_argv
        with open(cache, "w", encoding="utf-8") as f:
            f.write(txt)
        return txt

    print("fast rebuild (cached)" if fast else "running demos...")
    energy_txt = run("energy_drift_demo", energy_drift_demo.main, False)
    conv_txt = run("convergence_demo", convergence_demo.main, False)
    adapt_txt = run("adaptive_demo", adaptive_demo.main, False)
    hermite_txt = run("hermite_demo", hermite_demo.main, False)
    scale_txt = run("scaling_benchmark", scaling_benchmark.main, False)
    run("plot_orbits", plot_orbits.main, True)  # SVGs only, no text card
    lagr_txt = run("lagrange_demo", lagrange_demo.main, True)
    solar_txt = run("solar_system_demo", solar_system_demo.main, True)
    prec_txt = run("precession_demo", precession_demo.main, True)
    chaos_txt = run("chaos_demo", chaos_demo.main, True)
    poincare_txt = run("poincare_demo", poincare_demo.main, True)
    galaxy_txt = run("galaxy_collision_demo", galaxy_collision_demo.main, True)
    gw_txt = run("gravwave_demo", gravwave_demo.main, True)
    circ_txt = run("circularization_demo", circularization_demo.main, True)
    sitnikov_txt = run("sitnikov_demo", sitnikov_demo.main, True)
    virial_txt = run("virial_demo", virial_demo.main, True)
    roche_txt = run("roche_demo", roche_demo.main, True)
    kozai_txt = run("kozai_demo", kozai_demo.main, True)
    resonance_txt = run("resonance_demo", resonance_demo.main, True)
    coorbital_txt = run("coorbital_demo", coorbital_demo.main, True)
    tisserand_txt = run("tisserand_demo", tisserand_demo.main, True)
    lensing_txt = run("lensing_demo", lensing_demo.main, True)
    rotcurve_txt = run("rotation_curve_demo", rotation_curve_demo.main, True)
    schwarz_txt = run("schwarzschild_demo", schwarzschild_demo.main, True)
    friedmann_txt = run("friedmann_demo", friedmann_demo.main, True)
    laneemden_txt = run("lane_emden_demo", lane_emden_demo.main, True)
    distances_txt = run("distances_demo", distances_demo.main, True)
    chandra_txt = run("chandrasekhar_demo", chandrasekhar_demo.main, True)
    tov_txt = run("tov_demo", tov_demo.main, True)
    jeans_txt = run("jeans_demo", jeans_demo.main, True)
    sedov_txt = run("sedov_demo", sedov_demo.main, True)
    kerr_txt = run("kerr_demo", kerr_demo.main, True)
    hawking_txt = run("hawking_demo", hawking_demo.main, True)
    eddington_txt = run("eddington_demo", eddington_demo.main, True)
    hohmann_txt = run("hohmann_demo", hohmann_demo.main, True)
    grtime_txt = run("gr_time_demo", gr_time_demo.main, True)
    pulsar_txt = run("pulsar_demo", pulsar_demo.main, True)
    mond_txt = run("mond_demo", mond_demo.main, True)
    penrose_txt = run("penrose_demo", penrose_demo.main, True)
    tidalheat_txt = run("tidal_heating_demo", tidal_heating_demo.main, True)
    lt_txt = run("lense_thirring_demo", lense_thirring_demo.main, True)
    oberth_txt = run("oberth_demo", oberth_demo.main, True)
    df_txt = run("dynamical_friction_demo", dynamical_friction_demo.main, True)
    gwstrain_txt = run("gw_strain_demo", gw_strain_demo.main, True)
    saha_txt = run("saha_demo", saha_demo.main, True)
    cmb_txt = run("cmb_demo", cmb_demo.main, True)
    bbn_txt = run("bbn_demo", bbn_demo.main, True)
    cluster_txt = run("cluster_demo", cluster_demo.main, True)
    rochelobe_txt = run("roche_lobe_demo", roche_lobe_demo.main, True)
    degeneracy_txt = run("degeneracy_demo", degeneracy_demo.main, True)
    blackbody_txt = run("blackbody_demo", blackbody_demo.main, True)
    ms_txt = run("main_sequence_demo", main_sequence_demo.main, True)
    bondi_txt = run("bondi_demo", bondi_demo.main, True)
    cosmicv_txt = run("cosmic_velocities_demo", cosmic_velocities_demo.main, True)
    compton_txt = run("compton_demo", compton_demo.main, True)
    synchrotron_txt = run("synchrotron_demo", synchrotron_demo.main, True)
    sz_txt = run("sz_demo", sz_demo.main, True)
    kh_txt = run("kelvin_helmholtz_demo", kelvin_helmholtz_demo.main, True)
    atm_txt = run("atmosphere_demo", atmosphere_demo.main, True)
    larmor_txt = run("larmor_demo", larmor_demo.main, True)
    optdepth_txt = run("optical_depth_demo", optical_depth_demo.main, True)
    exoplanet_txt = run("exoplanet_demo", exoplanet_demo.main, True)
    hz_txt = run("habitable_zone_demo", habitable_zone_demo.main, True)
    fj_txt = run("faber_jackson_demo", faber_jackson_demo.main, True)
    focusing_txt = run("focusing_demo", focusing_demo.main, True)
    brems_txt = run("bremsstrahlung_demo", bremsstrahlung_demo.main, True)
    pairprod_txt = run("pair_production_demo", pair_production_demo.main, True)
    axprec_txt = run("axial_precession_demo", axial_precession_demo.main, True)
    alfven_txt = run("alfven_demo", alfven_demo.main, True)
    parker_txt = run("parker_spiral_demo", parker_spiral_demo.main, True)
    magbrake_txt = run("magnetic_braking_demo", magnetic_braking_demo.main, True)
    tidelock_txt = run("tidal_locking_demo", tidal_locking_demo.main, True)
    jeansesc_txt = run("jeans_escape_demo", jeans_escape_demo.main, True)
    snowline_txt = run("snow_line_demo", snow_line_demo.main, True)
    prdrag_txt = run("poynting_robertson_demo", poynting_robertson_demo.main, True)
    toomre_txt = run("toomre_demo", toomre_demo.main, True)
    accdisk_txt = run("accretion_disk_demo", accretion_disk_demo.main, True)
    fermi_txt = run("fermi_acceleration_demo", fermi_acceleration_demo.main, True)
    opacity_txt = run("opacity_demo", opacity_demo.main, True)
    brunt_txt = run("brunt_vaisala_demo", brunt_vaisala_demo.main, True)
    rampress_txt = run("ram_pressure_demo", ram_pressure_demo.main, True)
    freefall_txt = run("free_fall_demo", free_fall_demo.main, True)
    shock_txt = run("shock_jump_demo", shock_jump_demo.main, True)
    stromgren_txt = run("stromgren_demo", stromgren_demo.main, True)
    relax_txt = run("relaxation_time_demo", relaxation_time_demo.main, True)
    pwind_txt = run("parker_wind_demo", parker_wind_demo.main, True)
    green_txt = run("greenhouse_demo", greenhouse_demo.main, True)
    rossby_txt = run("rossby_demo", rossby_demo.main, True)
    rb_txt = run("rayleigh_benard_demo", rayleigh_benard_demo.main, True)
    termv_txt = run("terminal_velocity_demo", terminal_velocity_demo.main, True)
    snr_txt = run("snr_phases_demo", snr_phases_demo.main, True)
    mirror_txt = run("magnetic_mirror_demo", magnetic_mirror_demo.main, True)
    debye_txt = run("debye_demo", debye_demo.main, True)
    linebroad_txt = run("line_broadening_demo", line_broadening_demo.main, True)
    cog_txt = run("curve_of_growth_demo", curve_of_growth_demo.main, True)
    sackur_txt = run("sackur_tetrode_demo", sackur_tetrode_demo.main, True)
    mb_txt = run("maxwell_boltzmann_demo", maxwell_boltzmann_demo.main, True)

    def out(name):
        return os.path.join(outdir, name)

    sections = [
        section(
            "Energy conservation: symplectic vs RK4",
            "Symplectic integrators keep total energy bounded forever; RK4's "
            "energy walks off in one direction despite higher local accuracy.",
            pre(energy_txt)),
        section(
            "Convergence order (vs the exact Kepler orbit)",
            "Halving the step cuts verlet's error 4x (order 2) and "
            "forest_ruth/rk4's error 16x (order 4) -- measured, not assumed.",
            pre(conv_txt)),
        section(
            "Hermite: 4th order at one force call per step",
            "RK4 and Forest-Ruth reach 4th order at 4 and 3 force calls per step; "
            "the Hermite predictor-corrector reaches it with a single force+jerk "
            "call, so for a fixed force budget it takes more steps and lands far "
            "more accurate. It's the integrator real star-cluster codes use.",
            pre(hermite_txt)),
        section(
            "Adaptive stepping: same accuracy, far less work",
            "Dormand-Prince RK45 spends tiny steps at pericenter and long steps "
            "at apocenter, matching fixed-RK4 accuracy with ~9x fewer force evals.",
            pre(adapt_txt)),
        section(
            "Barnes-Hut scaling",
            "An octree collapses distant bodies to their centre of mass, turning "
            "O(N^2) direct summation into a sub-quadratic force evaluation.",
            pre(scale_txt)),
        section(
            "Orbit gallery",
            "Trajectories integrated and rendered to dependency-free SVG. The "
            "animated versions move the bodies along their paths via SMIL (no JS).",
            '<div class="grid">'
            + svg_card(out("figure_eight_animated.svg"), "figure-eight choreography (animated)")
            + svg_card(out("eccentric.svg"), "eccentric two-body, e=0.7")
            + svg_card(out("pythagorean.svg"), "Burrau pythagorean 3-body")
            + '</div>'),
        section(
            "The real solar system & Kepler's third law",
            "Eight planets from published orbital elements, in AU/years/solar "
            "masses. T^2/a^3 comes out constant -- Kepler's third law, straight "
            "from Newtonian gravity and a symplectic integrator, no fitting.",
            '<div class="grid">'
            + svg_card(out("inner_planets.svg"), "inner solar system (2 Mars years)")
            + f'<div class="card">{pre(solar_txt)}</div>'
            + '</div>'),
        section(
            "Exoplanet detection: transits & radial velocity",
            "Two methods, both simple geometry + Kepler: a transit dims the star by "
            "(R_p/R_star)^2 (Jupiter ~1%, Earth 0.008%), and the star wobbles by a "
            "radial-velocity K (Jupiter 12 m/s, Earth 9 cm/s). Hot Jupiters give "
            "the biggest signals -- which is why they were found first.",
            '<div class="grid">'
            + svg_card(out("exoplanet.svg"), "a transit light-curve dip")
            + f'<div class="card">{pre(exoplanet_txt)}</div>'
            + '</div>'),
        section(
            "The habitable zone: where liquid water survives",
            "A planet's equilibrium temperature T_eq ~ L^1/4 / sqrt(d) sets the band "
            "of orbits where water stays liquid. Earth's T_eq is 255 K (greenhouse "
            "warms it to 288 K). The zone marches out as sqrt(L) -- close in for red "
            "dwarfs, far out for luminous stars.",
            '<div class="grid">'
            + svg_card(out("habitable_zone.svg"), "HZ inner/outer edges vs stellar luminosity")
            + f'<div class="card">{pre(hz_txt)}</div>'
            + '</div>'),
        section(
            "Gravitational focusing & runaway growth",
            "Colliding bodies don't need a direct hit -- gravity bends distant "
            "trajectories in, enhancing the cross-section by 1 + v_esc^2/v_inf^2. "
            "In a cold planetesimal swarm this makes the biggest bodies grow "
            "fastest (runaway growth), seeding planetary embryos.",
            '<div class="grid">'
            + svg_card(out("focusing.svg"), "cross-section enhancement plunging with encounter speed")
            + f'<div class="card">{pre(focusing_txt)}</div>'
            + '</div>'),
        section(
            "Hulse-Taylor binary pulsar (GW before LIGO)",
            "PSR B1913+16's orbit shrinks as it radiates gravitational waves; the "
            "predicted period decay dP/dt = -2.40e-12 s/s matches the measured "
            "value to 99%. Tracking the cumulative shift for decades won the 1993 "
            "Nobel Prize, 22 years before LIGO's direct detection.",
            '<div class="grid">'
            + svg_card(out("pulsar.svg"), "the famous cumulative-periastron-shift parabola")
            + f'<div class="card">{pre(pulsar_txt)}</div>'
            + '</div>'),
        section(
            "Gravitational-wave inspiral (the LIGO chirp)",
            "A 2.5PN radiation-reaction force bleeds orbital energy into "
            "gravitational waves. The binary spirals inward and its frequency "
            "chirps upward -- the same mechanism as GW150914. The energy-loss "
            "rate matches Peters (1964) to a few percent.",
            '<div class="grid">'
            + svg_card(out("gw_inspiral.svg"), "relative orbit spiralling to merger")
            + f'<div class="card">{pre(gw_txt)}</div>'
            + '</div>'),
        section(
            "GW strain: the number LIGO measures",
            "The wave's amplitude h ~ (G M_c/c^2)^{5/3}(pi f/c)^{2/3}/d. For "
            "GW150914 (chirp mass ~28 M_sun, 410 Mpc) that is h ~ 1e-21, moving "
            "LIGO's 4 km arms by ~1e-18 m -- a thousandth of a proton's width.",
            '<div class="grid">'
            + svg_card(out("gw_strain.svg"), "strain vs distance, with GW150914 marked")
            + f'<div class="card">{pre(gwstrain_txt)}</div>'
            + '</div>'),
        section(
            "Gravitational waves circularize binaries (Peters 1964)",
            "The coupled Peters (a, e) equations: as a binary radiates, both its "
            "size and its eccentricity shrink, with e falling faster near merger. "
            "Every orbit -- even a wildly eccentric one -- is nearly circular by "
            "the time it merges.",
            '<div class="grid">'
            + svg_card(out("circularization.svg"), "(a, e) tracks all bending to e=0")
            + f'<div class="card">{pre(circ_txt)}</div>'
            + '</div>'),
        section(
            "Galaxy collision & tidal tails",
            "Two disk galaxies (heavy cores + cold tracer disks) pass close, and "
            "differential gravity draws their disks into bridges and tails -- the "
            "same physics as the Antennae and the Mice. Barnes-Hut forces, ~1000 "
            "bodies. Frames left-to-right in time.",
            '<div class="grid">'
            + svg_card(out("galaxy_t1.svg"), "approach")
            + svg_card(out("galaxy_t2.svg"), "close passage")
            + svg_card(out("galaxy_t4.svg"), "tidal tails")
            + f'<div class="card">{pre(galaxy_txt)}</div>'
            + '</div>'),
        section(
            "Dynamical friction: satellites spiralling in",
            "A massive body moving through a star field pulls a wake behind it "
            "that drags it back (Chandrasekhar friction ~ M^2 rho / v^2). "
            "Satellites and globular clusters spiral into their host on a time "
            "that scales as 1/M -- heavier sinks faster, dragging black holes to centres.",
            '<div class="grid">'
            + svg_card(out("dynamical_friction.svg"), "drag vs speed: zero at rest, peaks, then 1/v^2")
            + f'<div class="card">{pre(df_txt)}</div>'
            + '</div>'),
        section(
            "Sitnikov problem: a clean route to chaos",
            "A massless body on the z-axis through a binary. For a circular "
            "binary the stroboscopic map is smooth nested curves (integrable); "
            "give the binary eccentricity and the inner curves shred into a "
            "chaotic layer -- the system Moser used to prove chaos exists.",
            '<div class="grid">'
            + svg_card(out("sitnikov_circ.svg"), "e=0: nested tori (integrable)")
            + svg_card(out("sitnikov_ecc.svg"), "e=0.3: chaotic layer")
            + f'<div class="card">{pre(sitnikov_txt)}</div>'
            + '</div>'),
        section(
            "Poincare surface-of-section",
            "Many orbits at the SAME Jacobi energy, their y=0 crossings overlaid "
            "on the (x, vx) plane. Smooth closed loops are quasi-periodic KAM "
            "tori; the scattered dust is chaos -- coexisting at one energy.",
            '<div class="grid">'
            + svg_card(out("poincare_section.svg"), "tori and chaotic sea at one energy")
            + f'<div class="card">{pre(poincare_txt)}</div>'
            + '</div>'),
        section(
            "Sedov-Taylor blast wave",
            "A point energy release drives a self-similar shock, R ~ (E t^2/rho)^1/5. "
            "The same law dates supernova remnants (pc-scale, thousands of km/s) "
            "and -- run backwards -- let G. I. Taylor weigh the Trinity bomb from "
            "a movie of the fireball while its yield was still classified.",
            '<div class="grid">'
            + svg_card(out("sedov.svg"), "remnant radius (t^2/5) and decelerating shock speed")
            + f'<div class="card">{pre(sedov_txt)}</div>'
            + '</div>'),
        section(
            "Jeans instability: the birth of a star",
            "A gas cloud collapses when gravity beats pressure. The dispersion "
            "relation omega^2 = c_s^2 k^2 - 4 pi G rho splits into stable sound "
            "waves (short wavelength) and collapsing modes (long wavelength) at "
            "the Jeans length -- the threshold for all star and structure formation.",
            '<div class="grid">'
            + svg_card(out("jeans.svg"), "omega^2 goes negative below k_J: collapse")
            + f'<div class="card">{pre(jeans_txt)}</div>'
            + '</div>'),
        section(
            "Neutron stars & the TOV maximum mass",
            "For a neutron star, gravity is strong enough that Newtonian "
            "hydrostatics fails -- you need the relativistic TOV equation. The "
            "mass-radius curve turns over at a maximum mass (no static star above "
            "it), while the Newtonian version has no limit. GR makes black holes possible.",
            '<div class="grid">'
            + svg_card(out("tov.svg"), "neutron-star mass-radius curve with a maximum mass")
            + f'<div class="card">{pre(tov_txt)}</div>'
            + '</div>'),
        section(
            "Fermi degeneracy pressure",
            "The Pauli principle makes a cold, dense electron gas resist "
            "compression -- the quantum pressure that supports white dwarfs. It "
            "softens from P ~ n^5/3 to P ~ n^4/3 as electrons turn relativistic, "
            "and that softer exponent is the seed of the Chandrasekhar mass.",
            '<div class="grid">'
            + svg_card(out("degeneracy.svg"), "pressure laws vs density with the relativistic transition")
            + f'<div class="card">{pre(degeneracy_txt)}</div>'
            + '</div>'),
        section(
            "The Chandrasekhar mass",
            "White dwarfs are held up by electron degeneracy pressure. Integrating "
            "the full relativistic degenerate equation of state, the mass climbs "
            "toward a hard limit ~1.44 M_sun as the star shrinks -- above it no "
            "white dwarf is stable, the trigger for type-Ia supernovae.",
            '<div class="grid">'
            + svg_card(out("chandrasekhar.svg"), "mass-radius curve approaching the 1.44 M_sun limit")
            + f'<div class="card">{pre(chandra_txt)}</div>'
            + '</div>'),
        section(
            "Stellar structure (Lane-Emden)",
            "A self-gravitating polytropic gas sphere obeys the Lane-Emden "
            "equation. Its density profile and surface radius depend on the "
            "index n: n=1 is the exact sin(xi)/xi, n=3 the Eddington standard "
            "model, n=5 has finite mass but infinite radius.",
            '<div class="grid">'
            + svg_card(out("lane_emden.svg"), "density profiles for several polytropic indices")
            + f'<div class="card">{pre(laneemden_txt)}</div>'
            + '</div>'),
        section(
            "The main sequence & the HR diagram",
            "Mass rules a star's life: L ~ M^3.5, so massive blue stars are "
            "millions of times brighter but burn out in a few Myr, while red "
            "dwarfs live hundreds of Gyr. Luminosity vs temperature traces the "
            "main sequence -- the backbone of the Hertzsprung-Russell diagram.",
            '<div class="grid">'
            + svg_card(out("main_sequence.svg"), "the main sequence on an HR diagram")
            + f'<div class="card">{pre(ms_txt)}</div>'
            + '</div>'),
        section(
            "Kelvin-Helmholtz time: why gravity can't power the Sun",
            "Gravitational contraction (t_KH = G M^2 / R L) could light the Sun for "
            "only ~30 Myr -- far short of Earth's 4.5 Gyr age, the historic proof "
            "that stars need nuclear fusion. It is instead how long a protostar "
            "contracts before fusion ignites.",
            '<div class="grid">'
            + svg_card(out("kelvin_helmholtz.svg"), "Kelvin-Helmholtz vs nuclear timescale by mass")
            + f'<div class="card">{pre(kh_txt)}</div>'
            + '</div>'),
        section(
            "Cosmic distances & the discovery of acceleration",
            "Every cosmological distance is one integral of 1/E(z). A dark-energy "
            "universe puts a given redshift farther away, so distant type-Ia "
            "supernovae look ~0.4 mag fainter -- the 1998 acceleration result. "
            "The angular-diameter distance also turns over near z~1.6.",
            '<div class="grid">'
            + svg_card(out("distances.svg"), "Hubble diagram: LCDM vs decelerating, plus D_A turnover")
            + f'<div class="card">{pre(distances_txt)}</div>'
            + '</div>'),
        section(
            "Expansion of the universe (Friedmann)",
            "The scale factor a(t) under the Friedmann equation: radiation gives "
            "a ~ t^1/2, matter a ~ t^2/3, dark energy exponential growth. A flat "
            "LCDM universe ages to ~0.96/H0 (~13.5 Gyr) and is now entering its "
            "accelerating dark-energy era.",
            '<div class="grid">'
            + svg_card(out("friedmann.svg"), "scale factor for radiation, matter, dark energy, LCDM")
            + f'<div class="card">{pre(friedmann_txt)}</div>'
            + '</div>'),
        section(
            "Cosmic recombination (Saha equation)",
            "The universe went neutral -- releasing the cosmic microwave "
            "background -- at z~1400, T~3700 K, NOT at kT = 13.6 eV (~158000 K). "
            "The ~1.6 billion photons per baryon keep hydrogen ionized far below "
            "its binding energy; the Saha equation pins the transition.",
            '<div class="grid">'
            + svg_card(out("saha.svg"), "ionization fraction plunging to zero at recombination")
            + f'<div class="card">{pre(saha_txt)}</div>'
            + '</div>'),
        section(
            "CMB acoustic scale: the 1-degree spots",
            "The sound horizon at recombination is a fixed ruler; seen across the "
            "distance to last scattering it subtends ~1 degree, putting the first "
            "acoustic peak at multipole l ~ 220. Its position pins the universe's "
            "geometry to flat.",
            '<div class="grid">'
            + svg_card(out("cmb.svg"), "schematic acoustic peaks with the first at l~220")
            + f'<div class="card">{pre(cmb_txt)}</div>'
            + '</div>'),
        section(
            "Blackbody radiation: Planck, Wien, Stefan-Boltzmann",
            "The universal thermal spectrum: hotter bodies peak bluer "
            "(lambda_max T = 2.9 mm K) and radiate as T^4. It sets stellar colors, "
            "the Sun's 500 nm peak, and the 2.725 K CMB's microwave peak.",
            '<div class="grid">'
            + svg_card(out("blackbody.svg"), "Planck spectra: hotter = bluer and brighter")
            + f'<div class="card">{pre(blackbody_txt)}</div>'
            + '</div>'),
        section(
            "Compton & inverse-Compton scattering",
            "Photons trade energy with electrons: Compton down-shifts a photon "
            "(shift = lambda_C(1-cos theta), lambda_C = 2.426 pm), while inverse "
            "Compton off a relativistic electron boosts it by ~gamma^2 -- turning "
            "CMB and starlight into X-rays and gamma-rays.",
            '<div class="grid">'
            + svg_card(out("compton.svg"), "scattered photon energy falling with angle")
            + f'<div class="card">{pre(compton_txt)}</div>'
            + '</div>'),
        section(
            "The Larmor formula: radiation from acceleration",
            "Any accelerating charge radiates, with power P ~ q^2 a^2. "
            "Relativistically a circular accelerator boosts it by gamma^4 and a "
            "linear one by gamma^6. It also dooms the classical atom (an electron "
            "spirals in in ~1.6e-11 s) -- and it's the engine under synchrotron.",
            '<div class="grid">'
            + svg_card(out("larmor.svg"), "power vs gamma: gamma^4 (circular) and gamma^6 (linear)")
            + f'<div class="card">{pre(larmor_txt)}</div>'
            + '</div>'),
        section(
            "Synchrotron radiation: cosmic radio glow",
            "Relativistic electrons spiralling in magnetic fields radiate at a "
            "critical frequency ~ gamma^2 B (GHz radio for gamma~1e4 in microgauss "
            "fields). A power-law electron distribution N(E)~E^-p gives a power-law "
            "spectrum with index (p-1)/2 -- how we read jets and supernova remnants.",
            '<div class="grid">'
            + svg_card(out("synchrotron.svg"), "critical frequency climbing with electron energy")
            + f'<div class="card">{pre(synchrotron_txt)}</div>'
            + '</div>'),
        section(
            "Optical depth: where a star's surface is",
            "Light is attenuated as exp(-tau) crossing matter. A star has no solid "
            "surface -- its photosphere is the layer where the inward optical depth "
            "reaches tau ~ 2/3, the depth photons escape from and that sets the "
            "effective temperature.",
            '<div class="grid">'
            + svg_card(out("optical_depth.svg"), "transmitted fraction falling as exp(-tau)")
            + f'<div class="card">{pre(optdepth_txt)}</div>'
            + '</div>'),
        section(
            "Big Bang nucleosynthesis: the primordial 25% helium",
            "In the first minutes the neutron/proton ratio freezes at ~1/6, decays "
            "to ~1/7, and nearly all surviving neutrons lock into helium-4, giving "
            "Y_p ~ 0.25. That quarter-helium, seen everywhere, is a triumph of the "
            "hot Big Bang.",
            '<div class="grid">'
            + svg_card(out("bbn.svg"), "n/p ratio freezing out vs temperature")
            + f'<div class="card">{pre(bbn_txt)}</div>'
            + '</div>'),
        section(
            "The Eddington luminosity & black-hole growth",
            "Radiation pressure caps how bright -- and how fast-growing -- an "
            "accreting object can be: L_Edd = 4 pi G M m_p c / sigma_T, linear in "
            "mass. The e-folding growth time is ~45 Myr, so building a billion-"
            "solar-mass quasar from a seed takes ~0.8 Gyr.",
            '<div class="grid">'
            + svg_card(out("eddington.svg"), "Eddington-limited exponential growth to a quasar")
            + f'<div class="card">{pre(eddington_txt)}</div>'
            + '</div>'),
        section(
            "Bondi accretion: feeding on ambient gas",
            "Spherical accretion onto a compact object: Mdot ~ M^2 rho / c_s^3. It "
            "runs away with mass and is far stronger in cold gas -- a black hole in "
            "a molecular cloud eats millions of times faster than one in hot "
            "coronal gas. Compared to Eddington, it says whether growth is supply- or radiation-limited.",
            '<div class="grid">'
            + svg_card(out("bondi.svg"), "accretion rate plunging with gas temperature")
            + f'<div class="card">{pre(bondi_txt)}</div>'
            + '</div>'),
        section(
            "Hawking radiation & black-hole thermodynamics",
            "Quantum effects at the horizon give a black hole a temperature "
            "T ~ 1/M and an entropy = 1/4 its area in Planck units. Big holes are "
            "colder and live longer (t_evap ~ M^3); a ~1.7e11 kg primordial hole "
            "evaporates in a Hubble time, while a solar-mass one is ~60 nK and eternal.",
            '<div class="grid">'
            + svg_card(out("hawking.svg"), "temperature and evaporation time vs black-hole mass")
            + f'<div class="card">{pre(hawking_txt)}</div>'
            + '</div>'),
        section(
            "Kerr black holes: spin & frame-dragging",
            "A rotating black hole drags spacetime around it. Its horizon shrinks "
            "with spin, an ergosphere appears outside it, and the ISCO splits: "
            "prograde orbits reach down toward 1M at extremal spin while "
            "retrograde ones recede to 9M -- how black-hole spins are measured.",
            '<div class="grid">'
            + svg_card(out("kerr.svg"), "ISCO vs spin (prograde/retrograde) + horizon & ergosphere")
            + f'<div class="card">{pre(kerr_txt)}</div>'
            + '</div>'),
        section(
            "Penrose process: mining spin energy",
            "Inside the ergosphere a fragment can carry negative energy, so the "
            "escaping piece leaves with more than it entered -- energy mined from "
            "the hole's spin. Up to 29% of an extremal hole's mass-energy is "
            "extractable, and removing it only grows the horizon area (area theorem).",
            '<div class="grid">'
            + svg_card(out("penrose.svg"), "extractable rotational-energy fraction vs spin")
            + f'<div class="card">{pre(penrose_txt)}</div>'
            + '</div>'),
        section(
            "Schwarzschild black-hole orbits",
            "Strong-field GR from the effective potential V=(1-2M/r)(1+L^2/r^2): "
            "the ISCO at 6M, the photon sphere at 3M, bound orbits that precess "
            "tens of degrees per orbit, and low-angular-momentum orbits that "
            "plunge through the horizon.",
            '<div class="grid">'
            + svg_card(out("schwarzschild.svg"), "precessing (teal) and plunging (pink) geodesics")
            + f'<div class="card">{pre(schwarz_txt)}</div>'
            + '</div>'),
        section(
            "Gravitational time: redshift, GPS & Shapiro delay",
            "Clocks run slower deeper in gravity and light lags near mass. "
            "Pound-Rebka measured the redshift on a tower; GPS satellites gain "
            "~38 us/day (correct it or navigation fails); the Shapiro radar delay "
            "past the Sun is the tightest Solar-System test of GR.",
            '<div class="grid">'
            + svg_card(out("gr_time.svg"), "Shapiro delay diverging as the ray grazes the Sun")
            + f'<div class="card">{pre(grtime_txt)}</div>'
            + '</div>'),
        section(
            "Frame-dragging & geodetic precession (Gravity Probe B)",
            "An orbiting gyroscope precesses two ways: geodetic (from spatial "
            "curvature, ~6600 mas/yr) and frame-dragging (from Earth's rotation "
            "twisting spacetime, ~40 mas/yr). Gravity Probe B measured both -- the "
            "frame-dragging term is ~180x smaller and took near-perfect gyros to see.",
            '<div class="grid">'
            + svg_card(out("lense_thirring.svg"), "both precession rates vs orbit radius")
            + f'<div class="card">{pre(lt_txt)}</div>'
            + '</div>'),
        section(
            "Galaxy rotation curves & dark matter",
            "A visible disk alone gives a Keplerian decline (v ~ r^-1/2) past its "
            "edge; real galaxies stay flat. Adding an NFW dark halo, whose "
            "enclosed mass keeps growing as ~r, flattens the curve -- the classic "
            "evidence for dark matter.",
            '<div class="grid">'
            + svg_card(out("rotation_curve.svg"), "visible declines (red), disk+halo stays flat (blue)")
            + f'<div class="card">{pre(rotcurve_txt)}</div>'
            + '</div>'),
        section(
            "Faber-Jackson: elliptical galaxy scaling",
            "Elliptical galaxies obey L ~ sigma^4 -- luminosity from the random "
            "stellar velocity dispersion, following from the virial theorem plus a "
            "roughly constant mass-to-light ratio. A line width gives the "
            "luminosity, hence the distance -- the elliptical twin of Tully-Fisher.",
            '<div class="grid">'
            + svg_card(out("faber_jackson.svg"), "L climbing as the fourth power of sigma")
            + f'<div class="card">{pre(fj_txt)}</div>'
            + '</div>'),
        section(
            "MOND: flat curves without dark matter",
            "The rival to the dark halo: instead of adding unseen mass, MOND "
            "modifies gravity below a0 ~ 1.2e-10 m/s^2. A bare baryonic mass then "
            "has a naturally flat rotation curve and obeys the tight baryonic "
            "Tully-Fisher law v^4 = G M a0 -- MOND's sharpest prediction.",
            '<div class="grid">'
            + svg_card(out("mond.svg"), "MOND (flat) vs Newton on visible mass (declining)")
            + f'<div class="card">{pre(mond_txt)}</div>'
            + '</div>'),
        section(
            "Gravitational lensing",
            "Mass bends light by 4GM/(c^2 b) -- twice the Newtonian value, the "
            "1919 eclipse result (1.75 arcsec at the Sun's limb). A point-mass "
            "lens splits a source into two images, an Einstein ring at perfect "
            "alignment, and the symmetric microlensing brightening used to find exoplanets.",
            '<div class="grid">'
            + svg_card(out("lensing.svg"), "microlensing light curve + Einstein ring and images")
            + f'<div class="card">{pre(lensing_txt)}</div>'
            + '</div>'),
        section(
            "Tisserand parameter & gravity assists",
            "Across a planetary flyby a small body's semi-major axis and "
            "eccentricity change a lot, but the Tisserand parameter "
            "T = a_p/a + 2 sqrt(a/a_p (1-e^2)) cos i barely moves -- how Tisserand "
            "recognized comets Jupiter had reshaped, and what bounds a gravity assist.",
            '<div class="grid">'
            + svg_card(out("tisserand.svg"), "a & e jump at the flyby; Tisserand stays flat")
            + f'<div class="card">{pre(tisserand_txt)}</div>'
            + '</div>'),
        section(
            "Coorbital orbits: tadpoles & horseshoes",
            "A body sharing a planet's orbit librates in the rotating frame: a "
            "tadpole loops one Lagrange point (Jupiter's Trojans); a horseshoe "
            "wraps around L3 enclosing both L4 and L5, turning back before it "
            "reaches the planet (Saturn's Janus & Epimetheus, Earth's Cruithne).",
            '<div class="grid">'
            + svg_card(out("coorbital.svg"), "tadpole (teal) and horseshoe (grey) in the rotating frame")
            + f'<div class="card">{pre(coorbital_txt)}</div>'
            + '</div>'),
        section(
            "Mean-motion resonance",
            "Two planets at the 2:1 spacing lock into resonance: the resonant "
            "argument phi librates in a bounded band instead of circulating. "
            "This is what carves the Kirkwood gaps and binds the Laplace "
            "resonance of Io-Europa-Ganymede.",
            '<div class="grid">'
            + svg_card(out("resonance.svg"), "phi librates when locked (red), circulates when free (grey)")
            + f'<div class="card">{pre(resonance_txt)}</div>'
            + '</div>'),
        section(
            "Kozai-Lidov cycles",
            "In a hierarchical triple the inner orbit trades eccentricity for "
            "inclination and back, conserving sqrt(1-e^2) cos i. Above a critical "
            "inclination (~39.2 deg) the eccentricity is driven to large values -- "
            "the mechanism behind hot-Jupiter migration and compact-binary mergers.",
            '<div class="grid">'
            + svg_card(out("kozai.svg"), "e (red) and inclination (blue) oscillating out of phase")
            + f'<div class="card">{pre(kozai_txt)}</div>'
            + '</div>'),
        section(
            "Roche limit & tidal disruption",
            "A rubble-pile satellite survives only outside the Roche limit; "
            "inside it, the tide overwhelms self-gravity and tears it into a "
            "stream -- how planetary rings and Shoemaker-Levy 9's fragment chain "
            "formed. The surviving bound fraction drops sharply across the limit.",
            '<div class="grid">'
            + svg_card(out("roche_disruption.svg"), "satellite shredding into a tidal stream")
            + f'<div class="card">{pre(roche_txt)}</div>'
            + '</div>'),
        section(
            "Tidal heating: why Io erupts",
            "A moon on an eccentric orbit is flexed by the varying tide and "
            "dissipates the energy as heat: dE/dt ~ (k2/Q) e^2 R^5 / a^{15/2}. "
            "For Io this is ~1e14 W (40x Earth's heat flux), and the eccentricity "
            "is forced by the Laplace resonance -- resonance and volcanoes linked.",
            '<div class="grid">'
            + svg_card(out("tidal_heating.svg"), "heating vs eccentricity, with Io marked")
            + f'<div class="card">{pre(tidalheat_txt)}</div>'
            + '</div>'),
        section(
            "Roche lobes: binary mass transfer",
            "Each star in a binary owns a Roche lobe meeting its companion's at L1. "
            "When a star fills its lobe, gas pours through L1 onto the companion. "
            "From a lighter donor the orbit widens (stable transfer); from a heavier "
            "one it runs away -- the physics of X-ray binaries and type-Ia progenitors.",
            '<div class="grid">'
            + svg_card(out("roche_lobe.svg"), "Eggleton lobe radius vs mass ratio, with the stability line")
            + f'<div class="card">{pre(rochelobe_txt)}</div>'
            + '</div>'),
        section(
            "Virial theorem & violent relaxation",
            "For a bound gravitational system 2T + U = 0. A Plummer sphere sits "
            "at 2T/U = -1; a cold, sub-virial cluster collapses, overshoots, and "
            "relaxes to the same value -- forgetting its initial state through "
            "violent relaxation.",
            '<div class="grid">'
            + svg_card(out("virial.svg"), "running 2T/U converging on -1")
            + f'<div class="card">{pre(virial_txt)}</div>'
            + '</div>'),
        section(
            "Galaxy clusters: virial temperature & X-rays",
            "The virial theorem applied to a cluster's gas: falling into a "
            "10^15-solar-mass well heats it to a few keV (~10^8 K), radiating "
            "X-rays. Because kT ~ M^{2/3}, an X-ray temperature weighs the "
            "cluster's total (mostly dark) mass.",
            '<div class="grid">'
            + svg_card(out("cluster.svg"), "the kT ~ M^2/3 mass-temperature relation")
            + f'<div class="card">{pre(cluster_txt)}</div>'
            + '</div>'),
        section(
            "Sunyaev-Zeldovich effect: clusters shadowing the CMB",
            "The same hot cluster gas inverse-Compton scatters CMB photons, "
            "imprinting a Compton-y distortion: a cold spot dT/T = -2y in the "
            "radio. It is redshift-independent, so SZ surveys find clusters clear "
            "across the universe.",
            '<div class="grid">'
            + svg_card(out("sz.svg"), "the CMB temperature decrement vs Compton y")
            + f'<div class="card">{pre(sz_txt)}</div>'
            + '</div>'),
        section(
            "Bremsstrahlung: the X-rays of cluster gas",
            "Free electrons braking in ion fields radiate free-free X-rays with "
            "emissivity ~ n^2 sqrt(T). The cooling time ~ sqrt(T)/n drops below a "
            "Hubble time in dense cluster cores (cooling flows) but never in the "
            "outskirts -- the emission whose CMB imprint is the SZ effect.",
            '<div class="grid">'
            + svg_card(out("bremsstrahlung.svg"), "cooling time crossing the Hubble threshold")
            + f'<div class="card">{pre(brems_txt)}</div>'
            + '</div>'),
        section(
            "Pair production & the gamma-ray horizon",
            "Two photons colliding turn into an electron-positron pair once "
            "E1 E2 (1 - cos theta) >= 2(m_e c^2)^2, so two 511 keV gammas just "
            "pair head-on. A single high-energy gamma pair-produces off a soft "
            "background photon above (m_e c^2)^2 / E_bg -- TeV gammas from blazars "
            "are eaten by starlight/IR, PeV gammas by the CMB. The universe is "
            "opaque to gamma rays beyond a horizon that shrinks as energy rises.",
            '<div class="grid">'
            + svg_card(out("pair_production.svg"), "threshold gamma energy vs background photon energy")
            + f'<div class="card">{pre(pairprod_txt)}</div>'
            + '</div>'),
        section(
            "Precession of the equinoxes",
            "Earth's equatorial bulge, tilted 23.4 deg to the ecliptic, feels an "
            "uneven Sun+Moon pull that torques the spin axis into a 26,000-year "
            "cone -- like a leaning gyroscope. Torque scales as M/r^3, so the "
            "nearby Moon beats the Sun ~2.2 to 1; the sum is ~50.3 arcsec/yr. "
            "That is why Polaris is only a temporary North Star and why zodiac "
            "dates have slipped a whole sign since antiquity.",
            '<div class="grid">'
            + svg_card(out("axial_precession.svg"), "the circle the celestial pole traces over a Great Year")
            + f'<div class="card">{pre(axprec_txt)}</div>'
            + '</div>'),
        section(
            "Alfven waves & the magnetized solar wind",
            "A magnetic field threading a plasma behaves like a set of tensioned "
            "strings: pluck the field lines and they spring back at the Alfven "
            "speed v_A = B / sqrt(mu0 rho). The plasma beta = p_gas/p_mag says who "
            "is in charge -- beta << 1 in the field-dominated corona, beta > 1 in "
            "gas-dominated interiors. The solar wind starts sub-Alfvenic (the Sun's "
            "field co-rotates and brakes it), then crosses the Alfven surface near "
            "~15 R_sun and coasts out decoupled from the Sun's spin.",
            '<div class="grid">'
            + svg_card(out("alfven.svg"), "wind speed overtaking the Alfven speed at the Alfven surface")
            + f'<div class="card">{pre(alfven_txt)}</div>'
            + '</div>'),
        section(
            "The Parker spiral",
            "The solar wind drags the Sun's magnetic field radially outward while "
            "its footpoints stay rooted in a Sun that rotates every ~25 days -- a "
            "rotating sprinkler. Each parcel flies straight out, but the field line "
            "traces an Archimedean spiral: nearly radial near the Sun, bent ~45 deg "
            "at Earth (the garden-hose angle), nearly azimuthal by Jupiter. It is "
            "why western-limb flares connect best to Earth along the spiral.",
            '<div class="grid">'
            + svg_card(out("parker_spiral.svg"), "field lines spiralling out through the ecliptic")
            + f'<div class="card">{pre(parker_txt)}</div>'
            + '</div>'),
        section(
            "Magnetic braking & gyrochronology",
            "That same magnetized wind is a brake. Plasma stays locked to the field "
            "out to the Alfven radius (~15 R_sun), so it corotates on a long lever "
            "arm and bleeds angular momentum -- fast rotators brake hardest, so "
            "stellar spins converge onto one sequence. Skumanich's law P ~ t^(1/2) "
            "then turns a measured rotation period into an age: the Sun's 25-day "
            "spin reads 4.6 Gyr, a 3-day Pleiad reads ~60 Myr.",
            '<div class="grid">'
            + svg_card(out("magnetic_braking.svg"), "the Skumanich age-period sequence, Sun and clusters marked")
            + f'<div class="card">{pre(magbrake_txt)}</div>'
            + '</div>'),
        section(
            "Tidal locking",
            "Internal friction drags a body's tidal bulge out of line with its "
            "primary; the misaligned bulge feels a torque that despins it toward "
            "synchronous rotation. The locking time goes as a^6, so close-in moons "
            "lock in a geological blink (Phobos, Io) while distant ones never do -- "
            "the Moon locked to Earth long ago, but the Earth needs far longer than "
            "the universe is old to lock back to the Moon's weaker tide.",
            '<div class="grid">'
            + svg_card(out("tidal_locking.svg"), "the a^6 locking time crossing the age of the solar system")
            + f'<div class="card">{pre(tidelock_txt)}</div>'
            + '</div>'),
        section(
            "Jeans escape & the cosmic shoreline",
            "At the exobase, molecules faster than escape speed leave for good. "
            "Light, hot gases have a fatter Maxwell-Boltzmann tail, so the escape "
            "parameter lambda = v_esc^2/v_th^2 decides who keeps an atmosphere: "
            "Earth holds N2/O2/CO2 but loses H2 and He, the hot low-gravity Moon "
            "holds almost nothing, cold Titan clings even to nitrogen, and Jupiter "
            "keeps everything. Escape speed vs temperature is a cosmic shoreline.",
            '<div class="grid">'
            + svg_card(out("jeans_escape.svg"), "worlds and gases sorted by the v_esc = 6 v_th retention line")
            + f'<div class="card">{pre(jeansesc_txt)}</div>'
            + '</div>'),
        section(
            "The snow line",
            "A protoplanetary disk cools with distance as T ~ r^(-1/2), so ice "
            "condenses only beyond the point where it drops past ~160 K -- the snow "
            "line, at ~3 AU for the young Sun. Inside, water is vapour and planets "
            "grow small and dry; outside, ice roughly triples the solid density and "
            "lets giant cores grow fast enough to grab gas. Water, CO2 and CO each "
            "have their own frost line, sorting the disk by composition.",
            '<div class="grid">'
            + svg_card(out("snow_line.svg"), "the disk temperature profile with frost lines and planets")
            + f'<div class="card">{pre(snowline_txt)}</div>'
            + '</div>'),
        section(
            "Poynting-Robertson drag",
            "A dust grain re-radiates absorbed sunlight isotropically in its own "
            "frame, but aberration turns that into a faint forward headwind in the "
            "Sun's frame -- draining angular momentum so the grain spirals in. The "
            "inspiral time goes as r^2 and grain size, so micron grains at 1 AU fall "
            "in within a few thousand years; grains below the ~0.4 micron blow-out "
            "size are unbound and ejected. The zodiacal dust must be resupplied.",
            '<div class="grid">'
            + svg_card(out("poynting_robertson.svg"), "inspiral time vs grain size, blow-out and solar age marked")
            + f'<div class="card">{pre(prdrag_txt)}</div>'
            + '</div>'),
        section(
            "Toomre Q & disk stability",
            "A rotating disk balances self-gravity against pressure (small scales) "
            "and rotation via the epicyclic frequency (large scales). Toomre's "
            "Q = c_s kappa / (pi G Sigma) captures it in one number: Q > 1 is stable, "
            "Q < 1 fragments into clumps and spiral arms. The Milky Way hovers at "
            "Q ~ 1.5-2 -- marginally stable, because star formation heats a cooling "
            "disk back up, so disks self-regulate to the stability line.",
            '<div class="grid">'
            + svg_card(out("toomre.svg"), "gas Q across the galactic disk with the unstable band shaded")
            + f'<div class="card">{pre(toomre_txt)}</div>'
            + '</div>'),
        section(
            "Accretion disks: why black holes glow",
            "Gas with angular momentum settles into a disk and spirals in only as "
            "viscosity carries momentum outward, dissipating gravitational energy as "
            "heat radiated as a blackbody. The Shakura-Sunyaev profile T ~ r^(-3/4) "
            "makes the inner edge hottest: a 10-solar-mass hole peaks in soft X-rays "
            "(~keV), a billion-solar-mass one in the UV (the quasar 'big blue bump'). "
            "Both convert ~6% of rest mass to light -- ~8x fusion.",
            '<div class="grid">'
            + svg_card(out("accretion_disk.svg"), "T(r) for a stellar-mass and a supermassive disk, wavebands marked")
            + f'<div class="card">{pre(accdisk_txt)}</div>'
            + '</div>'),
        section(
            "Fermi acceleration & cosmic rays",
            "A charged particle repeatedly crossing a shock front gains energy at "
            "first order in the shock speed each time and has a fixed escape chance, "
            "producing a scale-free power-law spectrum N(E) ~ E^(-p). The index "
            "depends only on the compression ratio, p = (r+2)/(r-1), and every "
            "strong shock converges to r = 4, p = 2 -- the near-universal E^(-2) "
            "spectrum injected by supernova remnants across the Galaxy.",
            '<div class="grid">'
            + svg_card(out("fermi_acceleration.svg"), "power-law spectra steepening as the shock weakens toward p=2")
            + f'<div class="card">{pre(fermi_txt)}</div>'
            + '</div>'),
        section(
            "Stellar opacity",
            "Opacity kappa sets the photon mean free path 1/(kappa rho) and so how "
            "slowly a star leaks its light. Electron scattering is a flat floor in "
            "hot ionized gas; Kramers bound-free/free-free absorption rises with "
            "density and falls as T^(-7/2), making cool outer layers far more opaque "
            "than the core. That steep temperature dependence is what flips stellar "
            "envelopes from radiative to convective energy transport.",
            '<div class="grid">'
            + svg_card(out("opacity.svg"), "Kramers T^-3.5 fall-off flattening onto the electron-scattering floor")
            + f'<div class="card">{pre(opacity_txt)}</div>'
            + '</div>'),
        section(
            "Brunt-Vaisala frequency & convection",
            "Displace a fluid parcel upward: if it ends up denser than its new "
            "surroundings, gravity pulls it back and it oscillates at the buoyancy "
            "frequency N (internal gravity waves, ~5-10 min in the troposphere). If "
            "it ends up lighter, buoyancy runs away and the layer convects. N^2 > 0 "
            "means stable, N^2 < 0 unstable -- the sign change is exactly the "
            "Schwarzschild convection criterion, set by the adiabatic lapse rate.",
            '<div class="grid">'
            + svg_card(out("brunt_vaisala.svg"), "N^2 vs lapse rate, with the convective region beyond adiabatic shaded")
            + f'<div class="card">{pre(brunt_txt)}</div>'
            + '</div>'),
        section(
            "Ram-pressure stripping",
            "A galaxy plunging through a cluster's hot gas feels a wind of ram "
            "pressure rho v^2. The Gunn-Gott criterion strips its interstellar gas "
            "wherever that wind beats the disk's gravitational hold "
            "2 pi G Sigma_star Sigma_gas, so the galaxy keeps only the gas inside a "
            "stripping radius. In a rich cluster a Milky-Way-like spiral is stripped "
            "to a few kpc in one pass -- quenching it into a gas-poor S0.",
            '<div class="grid">'
            + svg_card(out("ram_pressure.svg"), "surviving gas radius vs infall speed for a range of ICM densities")
            + f'<div class="card">{pre(rampress_txt)}</div>'
            + '</div>'),
        section(
            "Free-fall: the universal clock of gravity",
            "Remove a body's pressure support and it collapses in the free-fall time "
            "t_ff = sqrt(3 pi / 32 G rho) -- which depends only on mean density, not "
            "size or mass. So a galaxy and a raindrop of equal density collapse in "
            "the same time. The Sun would free-fall in ~30 minutes, a molecular-cloud "
            "core in a few hundred kyr, a neutron star in under a millisecond: one "
            "1/sqrt(G rho) line spanning 37 decades of density.",
            '<div class="grid">'
            + svg_card(out("free_fall.svg"), "free-fall time vs mean density from clouds to neutron stars")
            + f'<div class="card">{pre(freefall_txt)}</div>'
            + '</div>'),
        section(
            "Shock jumps: Rankine-Hugoniot",
            "Move faster than the sound speed c_s = sqrt(gamma P/rho) and the gas "
            "cannot get out of the way -- a shock forms. Conservation of mass, "
            "momentum and energy fix the jumps from the upstream Mach number: density "
            "saturates at 4 (gamma=5/3), but pressure and temperature climb as M^2 "
            "without limit, which is why strong shocks heat gas to millions of kelvin "
            "while barely compressing it. The downstream flow is always subsonic.",
            '<div class="grid">'
            + svg_card(out("shock_jump.svg"), "density saturating at 4 while pressure and temperature diverge as M^2")
            + f'<div class="card">{pre(shock_txt)}</div>'
            + '</div>'),
        section(
            "The Stromgren sphere",
            "A hot star's ultraviolet photons ionize a bubble of hydrogen around it. "
            "In equilibrium every ionizing photon replaces one recombination, fixing "
            "the Stromgren radius R = (3Q / 4 pi n^2 alpha_B)^(1/3). Because R ~ Q^(1/3) "
            "and R ~ n^(-2/3), an O star lights up a ~25 pc nebula in diffuse gas but "
            "only a fraction of a parsec in a dense clump -- the pink emission nebulae "
            "(Orion, the Rosette) that flag recent massive-star formation.",
            '<div class="grid">'
            + svg_card(out("stromgren.svg"), "Stromgren radius shrinking as n^(-2/3) for three stellar types")
            + f'<div class="card">{pre(stromgren_txt)}</div>'
            + '</div>'),
        section(
            "Two-body relaxation & evaporation",
            "Every stellar flyby deflects a star a little; the accumulated random "
            "kicks change its velocity by order itself in the relaxation time "
            "t_relax ~ (N / 8 ln N) t_cross. Because that grows almost linearly with "
            "N, a globular cluster relaxes in ~1 Gyr and slowly evaporates, while a "
            "galaxy's relaxation time is millions of Hubble times -- effectively "
            "collisionless, which is why it keeps its spiral arms and streams.",
            '<div class="grid">'
            + svg_card(out("relaxation_time.svg"), "relaxation time vs N crossing the Hubble-time line")
            + f'<div class="card">{pre(relax_txt)}</div>'
            + '</div>'),
        section(
            "The Parker wind",
            "Parker showed a hot corona cannot stay static: an isothermal atmosphere "
            "keeps a finite pressure at infinity, far above interstellar space, so it "
            "must expand. The correct steady solution passes smoothly through Mach 1 "
            "at the sonic critical radius r_c = GM/2c_s^2 (a few solar radii), staying "
            "subsonic inside and supersonic out, and reaches a few hundred km/s by 1 "
            "AU -- the solar wind Mariner 2 confirmed.",
            '<div class="grid">'
            + svg_card(out("parker_wind.svg"), "transonic velocity profiles through the Mach-1 critical point")
            + f'<div class="card">{pre(pwind_txt)}</div>'
            + '</div>'),
        section(
            "The greenhouse effect",
            "An atmosphere transparent to sunlight but opaque in the infrared lets "
            "light in and traps the outgoing heat, so the surface runs hotter than "
            "the equilibrium temperature: T_surf = T_eq (1 + 3 tau/4)^(1/4). Earth's "
            "modest tau ~ 0.8 lifts 255 K to a life-friendly 288 K; Venus, wrapped in "
            "dense CO2 (tau ~ 150), runs away to a lead-melting 737 K; airless Mars "
            "sits at its equilibrium temperature.",
            '<div class="grid">'
            + svg_card(out("greenhouse.svg"), "surface warming vs optical depth with the terrestrial planets marked")
            + f'<div class="card">{pre(green_txt)}</div>'
            + '</div>'),
        section(
            "Rossby number & geostrophic balance",
            "On a rotating planet the Coriolis force deflects moving air at rate "
            "f = 2 Omega sin(lat). The Rossby number Ro = U/(fL) decides whether it "
            "matters: Ro << 1 (big, slow flows) means geostrophic balance -- wind "
            "blows along the isobars, so cyclones and ocean gyres are rotating "
            "vortices. Ro >> 1 (tornadoes, bathtub drains) ignores rotation entirely, "
            "which is why the draining-sink Coriolis story is a myth.",
            '<div class="grid">'
            + svg_card(out("rossby.svg"), "Rossby number vs length scale with the geostrophic band shaded")
            + f'<div class="card">{pre(rossby_txt)}</div>'
            + '</div>'),
        section(
            "Rayleigh-Benard convection",
            "Heat a fluid from below and buoyancy fights viscosity and diffusion. The "
            "Rayleigh number Ra = g alpha dT d^3 / (nu kappa) measures the contest: "
            "below Ra_c ~ 1708 the layer just conducts, above it convection switches "
            "on sharply in rolls and cells. The heat enhancement Nu ~ (Ra/Ra_c)^(1/3) "
            "climbs steeply -- and at the Ra ~ 10^30 of the solar convection zone the "
            "flow is violently turbulent, driving granulation and mantle plate motion.",
            '<div class="grid">'
            + svg_card(out("rayleigh_benard.svg"), "Nusselt number flat at 1 below onset then rising past Ra_c")
            + f'<div class="card">{pre(rb_txt)}</div>'
            + '</div>'),
        section(
            "Terminal velocity & drag",
            "A falling body speeds up until drag balances gravity. Which drag law "
            "applies is set by the Reynolds number: viscous Stokes drag (v ~ r^2) for "
            "tiny slow particles, quadratic drag (v ~ sqrt(r)) for big fast ones. So "
            "a fog droplet 200x smaller than a raindrop falls 40000x slower and "
            "effectively floats, a raindrop settles at ~9 m/s, and a belly-down "
            "skydiver tops out near 50 m/s.",
            '<div class="grid">'
            + svg_card(out("terminal_velocity.svg"), "terminal velocity vs radius bending from Stokes r^2 to quadratic sqrt(r)")
            + f'<div class="card">{pre(termv_txt)}</div>'
            + '</div>'),
        section(
            "Supernova remnant phases",
            "A supernova dumps ~10^51 erg into the ISM, and the shell ages through "
            "four phases: ballistic free expansion for a few centuries, then the long "
            "adiabatic Sedov-Taylor phase (R ~ t^2/5) for tens of thousands of years, "
            "then a radiative snowplow coasting on momentum, finally merging into the "
            "ISM near 100 pc after ~10^6 yr -- seeding the galaxy with the elements it "
            "forged along the way.",
            '<div class="grid">'
            + svg_card(out("snr_phases.svg"), "the radius-vs-age track with the phase transitions marked")
            + f'<div class="card">{pre(snr_txt)}</div>'
            + '</div>'),
        section(
            "The magnetic mirror & loss cone",
            "A charged particle spiraling along a field line conserves its magnetic "
            "moment mu = m v_perp^2 / 2B. Drifting into stronger field, v_perp must "
            "grow and v_parallel shrink until the particle reflects -- a magnetic "
            "mirror. Trapping depends only on pitch angle: sin^2(alpha) > 1/R_m holds "
            "the particle, else it falls into the loss cone and escapes. This traps "
            "the Van Allen belts and lights the aurora.",
            '<div class="grid">'
            + svg_card(out("magnetic_mirror.svg"), "loss-cone angle shrinking as the mirror ratio grows")
            + f'<div class="card">{pre(mirror_txt)}</div>'
            + '</div>'),
        section(
            "Debye shielding & the plasma frequency",
            "Ionized gas screens any test charge within the Debye length "
            "lambda_D = sqrt(eps0 kT / n e^2), and behaves as a collective plasma only "
            "when many particles sit inside a Debye sphere. Disturb the electrons and "
            "they ring at the plasma frequency omega_p = sqrt(n e^2 / eps0 m_e); waves "
            "below it are reflected -- which is why the ionosphere's ~9 MHz cutoff "
            "bounces AM radio around the Earth but lets FM escape to space.",
            '<div class="grid">'
            + svg_card(out("debye.svg"), "plasma frequency vs density with the radio bands marked")
            + f'<div class="card">{pre(debye_txt)}</div>'
            + '</div>'),
        section(
            "Spectral line broadening",
            "Atomic lines have width from three causes. Thermal Doppler motion gives a "
            "Gaussian of width ~ sqrt(T/m) -- the standard plasma thermometer. The "
            "finite excited-state lifetime gives an irreducible natural (Lorentzian) "
            "width A/4pi. Collisions add a pressure (Lorentzian) width that grows with "
            "density, so dense dwarf photospheres show broad wings absent in thin gas. "
            "The observed profile is their Voigt convolution.",
            '<div class="grid">'
            + svg_card(out("line_broadening.svg"), "a thermal Gaussian core beside a collisional Lorentzian with broad wings")
            + f'<div class="card">{pre(linebroad_txt)}</div>'
            + '</div>'),
        section(
            "The curve of growth",
            "How an absorption line's equivalent width grows with column density has "
            "three regimes: linear (W ~ N) for weak lines, a flat saturated plateau "
            "once the core goes black (W barely moves over decades of N), and a "
            "square-root damping tail (W ~ sqrt(N)) when the Lorentzian wings go thick. "
            "Matching a measured equivalent width to this curve is how stellar "
            "abundances are read from spectra.",
            '<div class="grid">'
            + svg_card(out("curve_of_growth.svg"), "the linear rise, saturated plateau and square-root damping tail")
            + f'<div class="card">{pre(cog_txt)}</div>'
            + '</div>'),
        section(
            "Sackur-Tetrode entropy",
            "Quantum state-counting fixes the absolute entropy of an ideal gas: "
            "S = N k_B [ln((V/N)(4 pi m U / 3 N h^2)^(3/2)) + 5/2]. Planck's constant "
            "enters explicitly (a phase-space cell is h^3) and the 1/N! for "
            "indistinguishable atoms resolves the Gibbs paradox. Evaluated for the "
            "noble gases at STP it reproduces the measured standard molar entropies to "
            "under a fifth of a percent -- entropy really is log of microstates.",
            '<div class="grid">'
            + svg_card(out("sackur_tetrode.svg"), "molar entropy vs temperature with measured STP values ringed")
            + f'<div class="card">{pre(sackur_txt)}</div>'
            + '</div>'),
        section(
            "Maxwell-Boltzmann speeds",
            "Molecular speeds in a gas follow f(v) ~ v^2 exp(-mv^2/2kT), whose three "
            "characteristic speeds keep a fixed ratio v_p : <v> : v_rms = "
            "1 : 1.128 : 1.225. Speeds scale as 1/sqrt(m), so hydrogen moves ~4x "
            "faster than nitrogen at the same temperature, and the thin exp(-v^2) "
            "tail -- only ~0.04% above 3 v_p -- is exactly what governs atmospheric "
            "escape and the onset of nuclear fusion.",
            '<div class="grid">'
            + svg_card(out("maxwell_boltzmann.svg"), "speed distributions for several gases with the three speeds marked")
            + f'<div class="card">{pre(mb_txt)}</div>'
            + '</div>'),
        section(
            "Three-body stability map",
            "Every pixel is a full three-body integration; colour is the time "
            "until a body escapes. The fractal boundary between long-lived and "
            "quickly-ionized initial conditions is chaos drawn in IC space. "
            "(Generated separately by stability_map_demo.py -- it's compute-heavy.)",
            '<div class="grid">'
            + svg_card(out("stability_map.svg"), "escape time over a grid of starting points")
            + '</div>'),
        section(
            "Chaos & the Lyapunov exponent",
            "Two trajectories started 1e-9 apart diverge to order unity on an "
            "exponential clock. The pythagorean 3-body has a large positive "
            "Lyapunov exponent; a regular orbit's estimate decays toward zero.",
            '<div class="grid">'
            + svg_card(out("chaos_divergence.svg"), "two runs 1e-6 apart peel apart over time")
            + f'<div class="card">{pre(chaos_txt)}</div>'
            + '</div>'),
        section(
            "Mercury's perihelion precession (general relativity)",
            "A first post-Newtonian correction makes the orbit slowly rotate "
            "instead of closing. The closed form gives Mercury's famous 43 "
            "arcsec/century; direct integration reproduces it.",
            '<div class="grid">'
            + svg_card(out("precession_rosette.svg"), "GR amplified ~8000x: a precessing rosette")
            + f'<div class="card">{pre(prec_txt)}</div>'
            + '</div>'),
        section(
            "Lagrange points & zero-velocity curves",
            "Five equilibria of the rotating-frame restricted 3-body problem. "
            "JWST parks at L2; the Trojan asteroids live at L4/L5.",
            '<div class="grid">'
            + svg_card(out("lagrange.svg"), "Earth-Moon Lagrange points + Hill curves")
            + f'<div class="card">{pre(lagr_txt)}</div>'
            + '</div>'),
        section(
            "Hohmann transfer & mission delta-v",
            "The cheapest two-burn hop between circular orbits sets every mission's "
            "delta-v budget: ~3.9 km/s from LEO to GEO, ~5.6 km/s and 259 days from "
            "Earth to Mars. The launch phase angle is why Mars windows open only "
            "every ~26 months.",
            '<div class="grid">'
            + svg_card(out("hohmann.svg"), "Earth-to-Mars transfer ellipse")
            + f'<div class="card">{pre(hohmann_txt)}</div>'
            + '</div>'),
        section(
            "Oberth effect: burn low and fast",
            "A burn's energy gain is v dv + dv^2/2, so the same dv buys far more "
            "energy deep in the gravity well where the ship moves fastest. The "
            "same burn escapes from periapsis but leaves the ship bound at "
            "apoapsis -- why probes dive in before an escape burn and powered flybys work.",
            '<div class="grid">'
            + svg_card(out("oberth.svg"), "escape speed from a fixed burn vs burn radius")
            + f'<div class="card">{pre(oberth_txt)}</div>'
            + '</div>'),
        section(
            "Escape & cosmic velocities",
            "The speed thresholds of spaceflight: orbital v1 = sqrt(GM/r), escape "
            "v2 = sqrt(2) v1 (11.2 km/s from Earth), and ~42 km/s to leave the "
            "Solar System. Push v2 to the speed of light and you recover the "
            "Schwarzschild radius -- 3 km for the Sun.",
            '<div class="grid">'
            + svg_card(out("cosmic_velocities.svg"), "escape velocity from Moon to a white dwarf")
            + f'<div class="card">{pre(cosmicv_txt)}</div>'
            + '</div>'),
        section(
            "Atmospheric escape: which worlds keep air",
            "A planet keeps a gas only if its escape speed beats ~6x the molecules' "
            "thermal speed (Jeans parameter lambda >= 36). Earth loses H2 and He but "
            "keeps N2/O2; the Moon and Mars lose the light gases; Jupiter keeps "
            "even hydrogen -- exactly the atmospheres we observe.",
            '<div class="grid">'
            + svg_card(out("atmosphere.svg"), "retained (blue) vs lost (red) across bodies and gases")
            + f'<div class="card">{pre(atm_txt)}</div>'
            + '</div>'),
    ]

    page = _TEMPLATE.replace("{{SECTIONS}}", "\n".join(sections))
    index = out("index.html")
    with open(index, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {index}")

    # Also publish a copy to docs/index.html so GitHub Pages can serve it. The
    # dashboard is fully self-contained (all SVGs inlined), so one file suffices.
    docs = os.path.join(HERE, "..", "docs")
    os.makedirs(docs, exist_ok=True)
    with open(os.path.join(docs, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {os.path.join(docs, 'index.html')} (GitHub Pages)")


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>symplectic-nbody: a computational-physics showcase</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#0d1117; color:#e6edf3;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         line-height:1.55; }
  header { padding:48px 24px 24px; max-width:960px; margin:0 auto; }
  header h1 { font-size:2rem; margin:0 0 8px; }
  header p { color:#8b949e; margin:0; }
  header a { color:#58a6ff; }
  main { max-width:960px; margin:0 auto; padding:0 24px 64px; }
  section { border-top:1px solid #21262d; padding:32px 0; }
  h2 { font-size:1.3rem; margin:0 0 6px; }
  .blurb { color:#8b949e; margin:0 0 16px; }
  pre { background:#161b22; border:1px solid #21262d; border-radius:8px;
        padding:16px; overflow-x:auto; font-size:12.5px; line-height:1.45;
        font-family:"SF Mono",Consolas,monospace; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
          gap:16px; align-items:start; }
  .card { background:#161b22; border:1px solid #21262d; border-radius:8px;
          padding:12px; }
  .card svg { width:100%; height:auto; display:block; border-radius:4px; }
  figcaption { color:#8b949e; font-size:12px; padding-top:8px; text-align:center; }
  .missing { color:#f85149; font-family:monospace; }
  footer { max-width:960px; margin:0 auto; padding:24px; color:#8b949e;
           font-size:13px; border-top:1px solid #21262d; }
</style>
</head>
<body>
<header>
  <h1>symplectic-nbody</h1>
  <p>A dependency-free computational-physics library. Every claim below is
     produced by code in this repo and checked by its test suite.
     <a href="https://github.com/awictor/symplectic-nbody">source on GitHub</a></p>
</header>
<main>
{{SECTIONS}}
</main>
<footer>
  Generated by <code>examples/build_dashboard.py</code>. Pure Python stdlib --
  no numpy, no matplotlib, no browser JavaScript. All figures are SVG.
</footer>
</body>
</html>"""


if __name__ == "__main__":
    main()
