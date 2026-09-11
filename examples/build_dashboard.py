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
    import gamow_demo
    import parallax_demo
    import standard_candle_demo
    import tully_fisher_demo
    import tolman_demo
    import olbers_demo
    import bi_elliptic_demo
    import gravity_assist_demo
    import synodic_demo
    import black_hole_shadow_demo
    import hill_sphere_demo
    import j2_precession_demo
    import solar_sail_demo
    import beaming_demo
    import relativistic_rocket_demo
    import relativistic_doppler_demo
    import de_broglie_demo
    import bohr_demo
    import photoelectric_demo
    import uncertainty_demo
    import tunneling_demo
    import particle_box_demo
    import harmonic_oscillator_demo
    import rutherford_demo
    import radioactive_decay_demo
    import mass_formula_demo
    import q_value_demo
    import quantum_stats_demo
    import debye_heat_demo
    import carnot_demo
    import adiabatic_demo
    import van_der_waals_demo
    import joule_thomson_demo
    import clausius_clapeyron_demo
    import reynolds_demo
    import bernoulli_demo
    import surface_tension_demo
    import ekman_demo
    import milankovitch_demo
    import equipartition_demo
    import osmosis_demo
    import diffusion_demo
    import peclet_demo
    import convection_demo
    import stefan_demo
    import capillary_demo
    import froude_demo
    import mach_cone_demo
    import nozzle_demo
    import blasius_demo
    import strouhal_demo
    import cluster_mass_demo
    import sersic_demo
    import grashof_demo
    import womersley_demo
    import marangoni_demo
    import kutta_joukowski_demo
    import knudsen_demo
    import richardson_demo
    import kolmogorov_demo
    import casimir_demo
    import hall_effect_demo
    import wiedemann_franz_demo
    import bragg_demo
    import diffraction_limit_demo
    import snell_demo
    import thin_film_demo
    import malus_demo
    import cherenkov_demo
    import zeeman_demo
    import rabi_demo
    import franck_hertz_demo
    import moseley_demo
    import stark_demo
    import aharonov_bohm_demo
    import josephson_demo
    import quantum_hall_demo
    import bcs_demo
    import london_demo
    import ising_mft_demo
    import percolation_demo
    import polya_demo
    import langevin_para_demo
    import buffon_demo
    import metropolis_demo
    import logistic_map_demo
    import henon_demo
    import lorenz_demo
    import double_pendulum_demo
    import mandelbrot_demo
    import van_der_pol_demo
    import duffing_demo
    import kuramoto_demo
    import sandpile_demo
    import cellular_automaton_demo
    import game_of_life_demo
    import reaction_diffusion_demo
    import boids_demo
    import dla_demo
    import benford_demo
    import coupon_collector_demo
    import secretary_demo
    import birthday_demo
    import gamblers_ruin_demo
    import parrondo_demo
    import galton_demo
    import monty_hall_demo
    import bayes_test_demo
    import shannon_demo
    import kelly_demo
    import hamming_demo
    import rsa_demo
    import diffie_hellman_demo
    import crc_demo
    import lz77_demo
    import bloom_demo
    import hyperloglog_demo
    import fenwick_demo
    import union_find_demo
    import dijkstra_demo
    import kdtree_demo
    import boyer_moore_demo
    import astar_demo
    import toposort_demo
    import levenshtein_demo
    import knapsack_demo
    import lcs_demo
    import quickselect_demo
    import aho_corasick_demo
    import floyd_warshall_demo
    import misra_gries_demo
    import reservoir_demo
    import count_min_demo
    import alias_method_demo
    import fisher_yates_demo
    import box_muller_demo
    import rejection_sampling_demo
    import welford_demo
    import kahan_demo
    import horner_demo
    import rootfind_demo
    import quadrature_demo
    import spline_demo
    import fft_demo
    import linsolve_demo
    import qr_demo
    import eigen_demo
    import conjugate_gradient_demo
    import svd_demo
    import kmeans_demo
    import regression_demo
    import decision_tree_demo
    import random_forest_demo
    import gmm_demo
    import hmm_demo
    import kalman_demo
    import pagerank_demo
    import lu_demo
    import gaussian_process_demo
    import bayes_opt_demo
    import dbscan_demo
    import hierarchical_demo
    import naive_bayes_demo
    import knn_demo
    import gradient_boosting_demo
    import spectral_clustering_demo
    import particle_filter_demo
    import simulated_annealing_demo
    import genetic_algorithm_demo
    import particle_swarm_demo
    import reed_solomon_demo
    import mutual_information_demo
    import lru_cache_demo
    import trie_demo
    import sorting_demo
    import newton_nd_demo
    import differential_evolution_demo
    import nelder_mead_demo
    import hits_demo
    import mds_demo
    import skiplist_demo
    import ant_colony_demo
    import avl_tree_demo
    import segment_tree_demo
    import convex_hull_demo

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
    gamow_txt = run("gamow_demo", gamow_demo.main, True)
    parallax_txt = run("parallax_demo", parallax_demo.main, True)
    candle_txt = run("standard_candle_demo", standard_candle_demo.main, True)
    tf_txt = run("tully_fisher_demo", tully_fisher_demo.main, True)
    tolman_txt = run("tolman_demo", tolman_demo.main, True)
    olbers_txt = run("olbers_demo", olbers_demo.main, True)
    biell_txt = run("bi_elliptic_demo", bi_elliptic_demo.main, True)
    gassist_txt = run("gravity_assist_demo", gravity_assist_demo.main, True)
    synodic_txt = run("synodic_demo", synodic_demo.main, True)
    shadow_txt = run("black_hole_shadow_demo", black_hole_shadow_demo.main, True)
    hill_txt = run("hill_sphere_demo", hill_sphere_demo.main, True)
    j2_txt = run("j2_precession_demo", j2_precession_demo.main, True)
    sail_txt = run("solar_sail_demo", solar_sail_demo.main, True)
    beaming_txt = run("beaming_demo", beaming_demo.main, True)
    rocket_txt = run("relativistic_rocket_demo", relativistic_rocket_demo.main, True)
    rdopp_txt = run("relativistic_doppler_demo", relativistic_doppler_demo.main, True)
    debroglie_txt = run("de_broglie_demo", de_broglie_demo.main, True)
    bohr_txt = run("bohr_demo", bohr_demo.main, True)
    photoel_txt = run("photoelectric_demo", photoelectric_demo.main, True)
    uncert_txt = run("uncertainty_demo", uncertainty_demo.main, True)
    tunnel_txt = run("tunneling_demo", tunneling_demo.main, True)
    pbox_txt = run("particle_box_demo", particle_box_demo.main, True)
    sho_txt = run("harmonic_oscillator_demo", harmonic_oscillator_demo.main, True)
    ruth_txt = run("rutherford_demo", rutherford_demo.main, True)
    decay_txt = run("radioactive_decay_demo", radioactive_decay_demo.main, True)
    semf_txt = run("mass_formula_demo", mass_formula_demo.main, True)
    qval_txt = run("q_value_demo", q_value_demo.main, True)
    qstats_txt = run("quantum_stats_demo", quantum_stats_demo.main, True)
    debye_txt = run("debye_heat_demo", debye_heat_demo.main, True)
    carnot_txt = run("carnot_demo", carnot_demo.main, True)
    adiab_txt = run("adiabatic_demo", adiabatic_demo.main, True)
    vdw_txt = run("van_der_waals_demo", van_der_waals_demo.main, True)
    jt_txt = run("joule_thomson_demo", joule_thomson_demo.main, True)
    cc_txt = run("clausius_clapeyron_demo", clausius_clapeyron_demo.main, True)
    reynolds_txt = run("reynolds_demo", reynolds_demo.main, True)
    bernoulli_txt = run("bernoulli_demo", bernoulli_demo.main, True)
    surface_tension_txt = run("surface_tension_demo", surface_tension_demo.main, True)
    ekman_txt = run("ekman_demo", ekman_demo.main, True)
    milankovitch_txt = run("milankovitch_demo", milankovitch_demo.main, True)
    equipartition_txt = run("equipartition_demo", equipartition_demo.main, True)
    osmosis_txt = run("osmosis_demo", osmosis_demo.main, True)
    diffusion_txt = run("diffusion_demo", diffusion_demo.main, True)
    peclet_txt = run("peclet_demo", peclet_demo.main, True)
    convection_txt = run("convection_demo", convection_demo.main, True)
    stefan_txt = run("stefan_demo", stefan_demo.main, True)
    capillary_txt = run("capillary_demo", capillary_demo.main, True)
    froude_txt = run("froude_demo", froude_demo.main, True)
    mach_cone_txt = run("mach_cone_demo", mach_cone_demo.main, True)
    nozzle_txt = run("nozzle_demo", nozzle_demo.main, True)
    blasius_txt = run("blasius_demo", blasius_demo.main, True)
    strouhal_txt = run("strouhal_demo", strouhal_demo.main, True)
    cluster_mass_txt = run("cluster_mass_demo", cluster_mass_demo.main, True)
    sersic_txt = run("sersic_demo", sersic_demo.main, True)
    grashof_txt = run("grashof_demo", grashof_demo.main, True)
    womersley_txt = run("womersley_demo", womersley_demo.main, True)
    marangoni_txt = run("marangoni_demo", marangoni_demo.main, True)
    kutta_joukowski_txt = run("kutta_joukowski_demo", kutta_joukowski_demo.main, True)
    knudsen_txt = run("knudsen_demo", knudsen_demo.main, True)
    richardson_txt = run("richardson_demo", richardson_demo.main, True)
    kolmogorov_txt = run("kolmogorov_demo", kolmogorov_demo.main, True)
    casimir_txt = run("casimir_demo", casimir_demo.main, True)
    hall_effect_txt = run("hall_effect_demo", hall_effect_demo.main, True)
    wiedemann_franz_txt = run("wiedemann_franz_demo", wiedemann_franz_demo.main, True)
    bragg_txt = run("bragg_demo", bragg_demo.main, True)
    diffraction_limit_txt = run("diffraction_limit_demo", diffraction_limit_demo.main, True)
    snell_txt = run("snell_demo", snell_demo.main, True)
    thin_film_txt = run("thin_film_demo", thin_film_demo.main, True)
    malus_txt = run("malus_demo", malus_demo.main, True)
    cherenkov_txt = run("cherenkov_demo", cherenkov_demo.main, True)
    zeeman_txt = run("zeeman_demo", zeeman_demo.main, True)
    rabi_txt = run("rabi_demo", rabi_demo.main, True)
    franck_hertz_txt = run("franck_hertz_demo", franck_hertz_demo.main, True)
    moseley_txt = run("moseley_demo", moseley_demo.main, True)
    stark_txt = run("stark_demo", stark_demo.main, True)
    aharonov_bohm_txt = run("aharonov_bohm_demo", aharonov_bohm_demo.main, True)
    josephson_txt = run("josephson_demo", josephson_demo.main, True)
    quantum_hall_txt = run("quantum_hall_demo", quantum_hall_demo.main, True)
    bcs_txt = run("bcs_demo", bcs_demo.main, True)
    london_txt = run("london_demo", london_demo.main, True)
    ising_mft_txt = run("ising_mft_demo", ising_mft_demo.main, True)
    percolation_txt = run("percolation_demo", percolation_demo.main, True)
    polya_txt = run("polya_demo", polya_demo.main, True)
    langevin_para_txt = run("langevin_para_demo", langevin_para_demo.main, True)
    buffon_txt = run("buffon_demo", buffon_demo.main, True)
    metropolis_txt = run("metropolis_demo", metropolis_demo.main, True)
    logistic_map_txt = run("logistic_map_demo", logistic_map_demo.main, True)
    henon_txt = run("henon_demo", henon_demo.main, True)
    lorenz_txt = run("lorenz_demo", lorenz_demo.main, True)
    double_pendulum_txt = run("double_pendulum_demo", double_pendulum_demo.main, True)
    mandelbrot_txt = run("mandelbrot_demo", mandelbrot_demo.main, True)
    van_der_pol_txt = run("van_der_pol_demo", van_der_pol_demo.main, True)
    duffing_txt = run("duffing_demo", duffing_demo.main, True)
    kuramoto_txt = run("kuramoto_demo", kuramoto_demo.main, True)
    sandpile_txt = run("sandpile_demo", sandpile_demo.main, True)
    cellular_automaton_txt = run("cellular_automaton_demo", cellular_automaton_demo.main, True)
    game_of_life_txt = run("game_of_life_demo", game_of_life_demo.main, True)
    reaction_diffusion_txt = run("reaction_diffusion_demo", reaction_diffusion_demo.main, True)
    boids_txt = run("boids_demo", boids_demo.main, True)
    dla_txt = run("dla_demo", dla_demo.main, True)
    benford_txt = run("benford_demo", benford_demo.main, True)
    coupon_txt = run("coupon_collector_demo", coupon_collector_demo.main, True)
    secretary_txt = run("secretary_demo", secretary_demo.main, True)
    birthday_txt = run("birthday_demo", birthday_demo.main, True)
    gamblers_ruin_txt = run("gamblers_ruin_demo", gamblers_ruin_demo.main, True)
    parrondo_txt = run("parrondo_demo", parrondo_demo.main, True)
    galton_txt = run("galton_demo", galton_demo.main, True)
    monty_hall_txt = run("monty_hall_demo", monty_hall_demo.main, True)
    bayes_test_txt = run("bayes_test_demo", bayes_test_demo.main, True)
    shannon_txt = run("shannon_demo", shannon_demo.main, True)
    kelly_txt = run("kelly_demo", kelly_demo.main, True)
    hamming_txt = run("hamming_demo", hamming_demo.main, True)
    rsa_txt = run("rsa_demo", rsa_demo.main, True)
    diffie_hellman_txt = run("diffie_hellman_demo", diffie_hellman_demo.main, True)
    crc_txt = run("crc_demo", crc_demo.main, True)
    lz77_txt = run("lz77_demo", lz77_demo.main, True)
    bloom_txt = run("bloom_demo", bloom_demo.main, True)
    hyperloglog_txt = run("hyperloglog_demo", hyperloglog_demo.main, True)
    fenwick_txt = run("fenwick_demo", fenwick_demo.main, True)
    union_find_txt = run("union_find_demo", union_find_demo.main, True)
    dijkstra_txt = run("dijkstra_demo", dijkstra_demo.main, True)
    kdtree_txt = run("kdtree_demo", kdtree_demo.main, True)
    boyer_moore_txt = run("boyer_moore_demo", boyer_moore_demo.main, True)
    astar_txt = run("astar_demo", astar_demo.main, True)
    toposort_txt = run("toposort_demo", toposort_demo.main, True)
    levenshtein_txt = run("levenshtein_demo", levenshtein_demo.main, True)
    knapsack_txt = run("knapsack_demo", knapsack_demo.main, True)
    lcs_txt = run("lcs_demo", lcs_demo.main, True)
    quickselect_txt = run("quickselect_demo", quickselect_demo.main, True)
    aho_corasick_txt = run("aho_corasick_demo", aho_corasick_demo.main, True)
    floyd_warshall_txt = run("floyd_warshall_demo", floyd_warshall_demo.main, True)
    misra_gries_txt = run("misra_gries_demo", misra_gries_demo.main, True)
    reservoir_txt = run("reservoir_demo", reservoir_demo.main, True)
    count_min_txt = run("count_min_demo", count_min_demo.main, True)
    alias_method_txt = run("alias_method_demo", alias_method_demo.main, True)
    fisher_yates_txt = run("fisher_yates_demo", fisher_yates_demo.main, True)
    box_muller_txt = run("box_muller_demo", box_muller_demo.main, True)
    rejection_sampling_txt = run("rejection_sampling_demo", rejection_sampling_demo.main, True)
    welford_txt = run("welford_demo", welford_demo.main, True)
    kahan_txt = run("kahan_demo", kahan_demo.main, True)
    horner_txt = run("horner_demo", horner_demo.main, True)
    rootfind_txt = run("rootfind_demo", rootfind_demo.main, True)
    quadrature_txt = run("quadrature_demo", quadrature_demo.main, True)
    spline_txt = run("spline_demo", spline_demo.main, True)
    fft_txt = run("fft_demo", fft_demo.main, True)
    linsolve_txt = run("linsolve_demo", linsolve_demo.main, True)
    qr_txt = run("qr_demo", qr_demo.main, True)
    eigen_txt = run("eigen_demo", eigen_demo.main, True)
    conjugate_gradient_txt = run("conjugate_gradient_demo", conjugate_gradient_demo.main, True)
    svd_txt = run("svd_demo", svd_demo.main, True)
    kmeans_txt = run("kmeans_demo", kmeans_demo.main, True)
    regression_txt = run("regression_demo", regression_demo.main, True)
    decision_tree_txt = run("decision_tree_demo", decision_tree_demo.main, True)
    random_forest_txt = run("random_forest_demo", random_forest_demo.main, True)
    gmm_txt = run("gmm_demo", gmm_demo.main, True)
    hmm_txt = run("hmm_demo", hmm_demo.main, True)
    kalman_txt = run("kalman_demo", kalman_demo.main, True)
    pagerank_txt = run("pagerank_demo", pagerank_demo.main, True)
    lu_txt = run("lu_demo", lu_demo.main, True)
    gp_txt = run("gaussian_process_demo", gaussian_process_demo.main, True)
    bayes_opt_txt = run("bayes_opt_demo", bayes_opt_demo.main, True)
    dbscan_txt = run("dbscan_demo", dbscan_demo.main, True)
    hierarchical_txt = run("hierarchical_demo", hierarchical_demo.main, True)
    naive_bayes_txt = run("naive_bayes_demo", naive_bayes_demo.main, True)
    knn_txt = run("knn_demo", knn_demo.main, True)
    gradient_boosting_txt = run("gradient_boosting_demo", gradient_boosting_demo.main, True)
    spectral_txt = run("spectral_clustering_demo", spectral_clustering_demo.main, True)
    particle_filter_txt = run("particle_filter_demo", particle_filter_demo.main, True)
    simulated_annealing_txt = run("simulated_annealing_demo", simulated_annealing_demo.main, True)
    genetic_algorithm_txt = run("genetic_algorithm_demo", genetic_algorithm_demo.main, True)
    particle_swarm_txt = run("particle_swarm_demo", particle_swarm_demo.main, True)
    reed_solomon_txt = run("reed_solomon_demo", reed_solomon_demo.main, True)
    mutual_information_txt = run("mutual_information_demo", mutual_information_demo.main, True)
    lru_cache_txt = run("lru_cache_demo", lru_cache_demo.main, True)
    trie_txt = run("trie_demo", trie_demo.main, True)
    sorting_txt = run("sorting_demo", sorting_demo.main, True)
    newton_nd_txt = run("newton_nd_demo", newton_nd_demo.main, True)
    differential_evolution_txt = run("differential_evolution_demo", differential_evolution_demo.main, True)
    nelder_mead_txt = run("nelder_mead_demo", nelder_mead_demo.main, True)
    hits_txt = run("hits_demo", hits_demo.main, True)
    mds_txt = run("mds_demo", mds_demo.main, True)
    skiplist_txt = run("skiplist_demo", skiplist_demo.main, True)
    ant_colony_txt = run("ant_colony_demo", ant_colony_demo.main, True)
    avl_tree_txt = run("avl_tree_demo", avl_tree_demo.main, True)
    segment_tree_txt = run("segment_tree_demo", segment_tree_demo.main, True)
    convex_hull_txt = run("convex_hull_demo", convex_hull_demo.main, True)

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
            "The Gamow peak",
            "Nuclei must beat an MeV Coulomb barrier to fuse, yet the Sun's core is only "
            "~1.3 keV. Two factors save it: the Maxwell-Boltzmann tail exp(-E/kT) falls "
            "with energy while quantum tunnelling exp(-sqrt(E_G/E)) rises, and their "
            "product is sharply peaked at the Gamow energy E0 = (E_G (kT)^2/4)^(1/3). "
            "Solar p-p fusion happens in a narrow window at ~6 keV; heavier nuclei need "
            "far hotter cores, the thermostat of stellar burning.",
            '<div class="grid">'
            + svg_card(out("gamow.svg"), "the Boltzmann tail and tunnelling probability multiplying to the Gamow peak")
            + f'<div class="card">{pre(gamow_txt)}</div>'
            + '</div>'),
        section(
            "Parallax & proper motion",
            "As Earth orbits the Sun a nearby star shifts by the parallax angle p, and "
            "the parsec is defined so d (pc) = 1/p (arcsec) -- the first rung of the "
            "distance ladder Gaia climbed for a billion stars. Proper motion adds the "
            "sideways drift: v_t = 4.74 mu d, combined with the Doppler radial velocity "
            "into the space velocity. Barnard's Star, the fastest, moves at 142 km/s "
            "through space.",
            '<div class="grid">'
            + svg_card(out("parallax.svg"), "the Earth-orbit baseline and the angle a nearby star subtends")
            + f'<div class="card">{pre(parallax_txt)}</div>'
            + '</div>'),
        section(
            "Standard candles & the distance ladder",
            "Know an object's true luminosity and its apparent brightness gives its "
            "distance: m - M = 5 log10(d/10 pc). Cepheids supply M through Leavitt's "
            "period-luminosity law, Type Ia supernovae (M ~ -19.3) extend it to "
            "hundreds of Mpc, and chaining parallax -> Cepheids -> supernovae is the "
            "cosmic distance ladder. Five magnitudes is exactly 100x in flux.",
            '<div class="grid">'
            + svg_card(out("standard_candle.svg"), "distance modulus vs distance with the ladder rungs marked")
            + f'<div class="card">{pre(candle_txt)}</div>'
            + '</div>'),
        section(
            "The Tully-Fisher relation",
            "Spiral galaxies obey a tight L ~ v_flat^4 scaling: because v^2 = GM/R and "
            "spirals hold roughly constant surface brightness, mass, spin and light "
            "rise together. A Milky-Way-like spiral (v ~ 220 km/s) shines ~3x10^10 "
            "L_sun. Since the rotation width is easy to measure from the 21-cm line, "
            "Tully-Fisher is a redshift-independent distance indicator reaching far "
            "beyond resolvable Cepheids -- the spiral cousin of Faber-Jackson.",
            '<div class="grid">'
            + svg_card(out("tully_fisher.svg"), "luminosity vs rotation speed with slope 4 on a log-log plot")
            + f'<div class="card">{pre(tf_txt)}</div>'
            + '</div>'),
        section(
            "The Tolman dimming test",
            "Surface brightness is distance-independent in a static universe -- flux "
            "and angular area fall together. Expansion breaks that with four factors "
            "of (1+z), so SB ~ (1+z)^-4: a z=1 galaxy is dimmed 16x per square "
            "arcsecond, a z=3 galaxy 256x. A static tired-light universe would dim "
            "only as (1+z); observations back the (1+z)^4 law -- direct evidence the "
            "redshift is real expansion, not photons losing energy en route.",
            '<div class="grid">'
            + svg_card(out("tolman.svg"), "expanding (1+z)^4 dimming diverging from the tired-light (1+z)^1 line")
            + f'<div class="card">{pre(tolman_txt)}</div>'
            + '</div>'),
        section(
            "Olbers' paradox",
            "In an infinite, eternal, static universe every line of sight would end on "
            "a star and the whole sky would blaze. It does not -- the night is dark. "
            "Stars would tile the sky only after ~10^16 light-years (the mean free "
            "path 1/n sigma), but the cosmic horizon (c x age) is a million times "
            "closer, so only ~10^-6 of the sky is covered. The finite age of the "
            "universe, not infinite space, is what makes night dark.",
            '<div class="grid">'
            + svg_card(out("olbers.svg"), "sky-covering fraction vs distance with the horizon far short of tiling")
            + f'<div class="card">{pre(olbers_txt)}</div>'
            + '</div>'),
        section(
            "Bi-elliptic transfer",
            "The Hohmann two-burn transfer is cheapest for modest orbit changes, but "
            "for large radius ratios a three-burn bi-elliptic transfer -- flinging the "
            "craft far beyond the target and dropping back -- costs less total delta-v, "
            "because the mid-course burn happens where orbital speeds are tiny. Below "
            "R = 11.94 Hohmann always wins; above R = 15.58 bi-elliptic always does. "
            "The saving is paid for with a much longer, sometimes years-long, transfer.",
            '<div class="grid">'
            + svg_card(out("bi_elliptic.svg"), "Hohmann and bi-elliptic delta-v crossing near R = 12")
            + f'<div class="card">{pre(biell_txt)}</div>'
            + '</div>'),
        section(
            "Gravity assist",
            "A spacecraft flying past a planet follows a hyperbola: it leaves at the "
            "same speed relative to the planet but bent by the turn angle "
            "sin(delta/2) = 1/e. In the Sun's frame the planet is moving, so that "
            "rotation adds up to 2 v_inf of free heliocentric speed. A slower, deeper "
            "pass bends more and steals more; Voyager chained Jupiter-Saturn-Uranus-"
            "Neptune this way to reach escape speed for almost no fuel.",
            '<div class="grid">'
            + svg_card(out("gravity_assist.svg"), "slingshot boost vs flyby periapsis for several approach speeds")
            + f'<div class="card">{pre(gassist_txt)}</div>'
            + '</div>'),
        section(
            "Synodic periods & alignments",
            "The geometry we see -- oppositions, launch windows, new Moons -- repeats "
            "on the synodic period, the beat between two orbital rates: "
            "1/S = |1/P1 - 1/P2|. Mars returns to opposition every ~780 days, exactly "
            "the ~26-month cadence of Mars missions; the synodic month is 29.5 days, "
            "longer than the 27.3-day sidereal month. Near Earth's own orbit S blows "
            "up; distant planets approach a one-year synodic period.",
            '<div class="grid">'
            + svg_card(out("synodic.svg"), "synodic period diverging at Earth's orbit and settling to one year")
            + f'<div class="card">{pre(synodic_txt)}</div>'
            + '</div>'),
        section(
            "The black-hole shadow",
            "A black hole casts a dark disk larger than its horizon: light inside the "
            "critical impact parameter b = 3 sqrt(3) GM/c^2 is captured, so lensing "
            "magnifies the shadow to 5.2 Schwarzschild radii across (vs 2 r_s for the "
            "horizon). M87* and Sgr A* each subtend only ~40-50 microarcseconds -- the "
            "angular size of an orange on the Moon -- which is why the Event Horizon "
            "Telescope had to link radio dishes across the whole Earth.",
            '<div class="grid">'
            + svg_card(out("black_hole_shadow.svg"), "horizon, photon sphere and lensed shadow edge to scale")
            + f'<div class="card">{pre(shadow_txt)}</div>'
            + '</div>'),
        section(
            "The Hill sphere",
            "A moon is held by its planet only inside the Hill sphere, "
            "r_H = a (m/3M)^(1/3), where the planet's pull beats the star's tide. "
            "Earth's is ~1.5 million km, four times the Moon's distance; Jupiter's "
            "reaches ~53 million km. Real moons survive out to ~1/2 r_H prograde, and "
            "the same balance sets the feeding zone of a forming planet and the mutual "
            "Hill spacing that keeps planetary orbits stable.",
            '<div class="grid">'
            + svg_card(out("hill_sphere.svg"), "Hill radius vs orbital distance for the planets")
            + f'<div class="card">{pre(hill_txt)}</div>'
            + '</div>'),
        section(
            "J2 orbital precession",
            "A planet's equatorial bulge (coefficient J2) makes satellite orbits "
            "precess: the node line regresses and the ellipse rotates in-plane. Tune "
            "the inclination so the nodal drift matches the Sun's 0.9856 deg/day and "
            "you get a sun-synchronous orbit (~98 deg) crossing the equator at fixed "
            "local time; at the 63.4-degree critical inclination the apsides freeze -- "
            "the Molniya orbit that parks apogee over the far north.",
            '<div class="grid">'
            + svg_card(out("j2_precession.svg"), "nodal and apsidal rates vs inclination with the special angles marked")
            + f'<div class="card">{pre(j2_txt)}</div>'
            + '</div>'),
        section(
            "Solar sails & radiation pressure",
            "Sunlight carries momentum: a mirror at 1 AU feels ~9 uPa (2F/c), a "
            "feather touch that never runs out. Since both sunlight and gravity fall "
            "as 1/r^2, the lightness number beta = radiation force / solar gravity is "
            "a fixed property of the sail; beta = 1 (a ~1.5 g/m^2 mirror) cancels the "
            "Sun's pull and beta > 1 escapes on sunlight alone. Today's sails sit at "
            "beta ~ 0.01 -- gentle but propellant-free.",
            '<div class="grid">'
            + svg_card(out("solar_sail.svg"), "lightness number vs area-to-mass with the beta=1 line and missions")
            + f'<div class="card">{pre(sail_txt)}</div>'
            + '</div>'),
        section(
            "Relativistic beaming",
            "Radiation from a source moving near light speed is swept forward into a "
            "cone of half-angle ~1/gamma and Doppler-boosted, so the observed flux "
            "scales as D^(3+alpha). An approaching jet is brightened hundreds of times "
            "while its receding twin is dimmed by the same powers -- why M87's jet "
            "looks one-sided. The same geometry makes blobs appear to move faster than "
            "light, an illusion of light-travel time.",
            '<div class="grid">'
            + svg_card(out("beaming.svg"), "Doppler factor vs viewing angle for several Lorentz factors")
            + f'<div class="card">{pre(beaming_txt)}</div>'
            + '</div>'),
        section(
            "The relativistic rocket",
            "A ship at constant 1 g follows a hyperbolic worldline: velocity "
            "c tanh(a tau/c) saturates just short of c, but proper time uses cosh/sinh, "
            "so the crew clock falls ever further behind Earth's. The galactic centre "
            "is ~10 crew-years away (27,000 pass on Earth), Andromeda ~15 -- ship time "
            "grows only logarithmically with distance. The catch is fuel: a photon "
            "drive needs exp(2 phi) times the payload mass.",
            '<div class="grid">'
            + svg_card(out("relativistic_rocket.svg"), "ship time vs Earth time diverging with distance at 1 g")
            + f'<div class="card">{pre(rocket_txt)}</div>'
            + '</div>'),
        section(
            "Relativistic Doppler shift",
            "A moving light source shifts in frequency by the classical Doppler effect "
            "times time dilation: receding sources redshift, approaching ones blueshift. "
            "The purely relativistic surprise is the transverse shift -- a source moving "
            "exactly across the line of sight still reddens by 1/gamma because its clock "
            "runs slow, the effect Ives and Stilwell measured in 1938. A measured "
            "redshift maps straight back to a speed.",
            '<div class="grid">'
            + svg_card(out("relativistic_doppler.svg"), "redshift vs speed for receding, approaching and transverse cases")
            + f'<div class="card">{pre(rdopp_txt)}</div>'
            + '</div>'),
        section(
            "de Broglie matter waves",
            "Every particle has a wavelength lambda = h/p, inversely proportional to its "
            "momentum. A baseball's is 10^-34 m -- undetectable -- but a 100 keV electron's "
            "is ~4 pm, thousands of times finer than light, which is why electron "
            "microscopes resolve atoms; a thermal neutron's ~0.1 nm matches crystal "
            "spacing for diffraction. The wave nature takes over once lambda rivals the "
            "interparticle spacing -- the onset of quantum degeneracy.",
            '<div class="grid">'
            + svg_card(out("de_broglie.svg"), "wavelength vs energy for electron/proton/neutron with reference scales")
            + f'<div class="card">{pre(debroglie_txt)}</div>'
            + '</div>'),
        section(
            "The Bohr model of hydrogen",
            "Quantizing angular momentum (L = n hbar) forces the electron onto discrete "
            "orbits: energy E_n = -13.6/n^2 eV, radius n^2 a_0 (a_0 ~ 52.9 pm). "
            "Transitions emit fixed-energy photons -- the sharp hydrogen lines: Lyman "
            "in the UV, Balmer in the visible (H-alpha at 656 nm, the red of nebulae), "
            "Paschen in the infrared. The n=1 orbital speed is alpha*c, v/c ~ 1/137.",
            '<div class="grid">'
            + svg_card(out("bohr.svg"), "the hydrogen energy levels with the Lyman/Balmer/Paschen transitions")
            + f'<div class="card">{pre(bohr_txt)}</div>'
            + '</div>'),
        section(
            "The photoelectric effect",
            "Light ejects electrons from a metal only above a threshold frequency, no "
            "matter how bright a redder beam is -- Einstein's proof that light comes in "
            "photons of energy hf. One photon gives one electron K_max = hf - phi, so "
            "the stopping voltage climbs linearly with frequency at the universal slope "
            "h/e; only the intercept (the work function) differs between metals. "
            "Millikan measured that line and pinned down Planck's constant.",
            '<div class="grid">'
            + svg_card(out("photoelectric.svg"), "stopping voltage vs frequency: parallel lines of slope h/e")
            + f'<div class="card">{pre(photoel_txt)}</div>'
            + '</div>'),
        section(
            "The uncertainty principle",
            "Position and momentum cannot both be sharp: dx dp >= hbar/2. Confining a "
            "particle to a box forces a momentum spread and thus an irreducible "
            "zero-point energy E ~ hbar^2/(m dx^2). An electron squeezed to atomic size "
            "(~0.1 nm) carries ~1 eV -- why it never falls into the nucleus -- and "
            "minimizing that against the Coulomb pull reproduces hydrogen's 13.6 eV; a "
            "nucleon in a femtometre nucleus carries MeV.",
            '<div class="grid">'
            + svg_card(out("uncertainty.svg"), "confinement energy vs box size for an electron and a nucleon")
            + f'<div class="card">{pre(uncert_txt)}</div>'
            + '</div>'),
        section(
            "Quantum tunneling",
            "A particle with too little energy to climb a barrier can still leak "
            "through: its wavefunction decays as exp(-kappa x) inside, so transmission "
            "is T ~ exp(-2 kappa L), plunging exponentially with barrier width. A "
            "nanometre barrier is essentially opaque, yet shaving an Angstrom raises "
            "the current ~8x -- the razor sensitivity that lets a scanning tunneling "
            "microscope feel individual atoms, and the physics of alpha decay and fusion.",
            '<div class="grid">'
            + svg_card(out("tunneling.svg"), "transmission falling exponentially with barrier width")
            + f'<div class="card">{pre(tunnel_txt)}</div>'
            + '</div>'),
        section(
            "The particle in a box",
            "Trap a particle in a well and only standing waves fit, quantizing the "
            "energy: E_n = n^2 h^2 / (8 m L^2). Levels rise as n^2, the ground state is "
            "nonzero (confinement zero-point energy), and every level scales as 1/L^2 -- "
            "which is why shrinking a quantum dot widens its gaps and shifts its glow "
            "bluer, giving size-tunable colour for displays and bio-markers.",
            '<div class="grid">'
            + svg_card(out("particle_box.svg"), "the first energy levels with their wavefunctions in the well")
            + f'<div class="card">{pre(pbox_txt)}</div>'
            + '</div>'),
        section(
            "The quantum harmonic oscillator",
            "Near any potential minimum a system is a spring, so the oscillator is "
            "everywhere -- molecular vibrations, phonons, cavity photons. Its levels are "
            "EVENLY spaced by hbar omega (unlike the box or atom), so a molecule absorbs "
            "one sharp infrared line per vibrational quantum (CO at 4.6 um). The ground "
            "state is nonzero: the zero-point energy (1/2) hbar omega is forced by "
            "uncertainty and keeps helium liquid at absolute zero.",
            '<div class="grid">'
            + svg_card(out("harmonic_oscillator.svg"), "equally-spaced levels inside the parabolic well")
            + f'<div class="card">{pre(sho_txt)}</div>'
            + '</div>'),
        section(
            "Rutherford scattering",
            "Firing alphas at gold foil, a few bounced almost straight back -- "
            "impossible off diffuse charge. Rutherford's Coulomb cross section "
            "dsigma/dOmega ~ 1/sin^4(theta/2) soars at small angles but stays nonzero "
            "at 180 degrees, exactly those hard bounces, revealing a tiny dense "
            "nucleus. The head-on closest approach (~45 fm for 5 MeV alphas on gold) "
            "bounded the nuclear size.",
            '<div class="grid">'
            + svg_card(out("rutherford.svg"), "the 1/sin^4 angular distribution with nonzero back-scatter")
            + f'<div class="card">{pre(ruth_txt)}</div>'
            + '</div>'),
        section(
            "Radioactive decay & decay chains",
            "Unstable nuclei decay at a constant per-nucleus rate, so a population falls "
            "exponentially N = N0 2^(-t/t_half) -- the clock behind carbon-14 dating (25% "
            "remaining is two half-lives, ~11,500 yr). In a parent-daughter chain the "
            "daughter follows the Bateman rise-and-fall, reaching secular equilibrium "
            "where its activity equals the parent's -- the principle of medical "
            "radioisotope generators and uranium-fed radon.",
            '<div class="grid">'
            + svg_card(out("radioactive_decay.svg"), "parent exponential decay and the daughter's Bateman build-up")
            + f'<div class="card">{pre(decay_txt)}</div>'
            + '</div>'),
        section(
            "Nuclear binding: the mass formula",
            "Weizsacker's liquid-drop model sums volume, surface, Coulomb, asymmetry and "
            "pairing terms into the nuclear binding energy. Their balance gives the "
            "binding-energy-per-nucleon curve, peaking near iron at ~8.8 MeV/nucleon -- "
            "which is exactly why fusion releases energy up to iron and fission beyond "
            "it. Minimizing over Z traces the valley of stability, drifting to neutron "
            "excess in heavy nuclei (U-238 sits at Z=92).",
            '<div class="grid">'
            + svg_card(out("mass_formula.svg"), "the binding-energy-per-nucleon curve peaking at iron")
            + f'<div class="card">{pre(semf_txt)}</div>'
            + '</div>'),
        section(
            "Nuclear Q-value: E = mc^2 at work",
            "A nuclear reaction releases energy equal to its mass defect times c^2 "
            "(1 amu = 931.5 MeV). D-T fusion yields 17.6 MeV, U-235 fission ~200 MeV -- "
            "only a fraction of a percent of the mass, but c^2 makes it millions of "
            "times a chemical bond: fission is ~2 million times TNT, fusion ~4x fission, "
            "and total matter-antimatter annihilation converts 100% of the mass at "
            "~9x10^16 J/kg.",
            '<div class="grid">'
            + svg_card(out("q_value.svg"), "fuel energy density from chemical to nuclear to pure mass-energy")
            + f'<div class="card">{pre(qval_txt)}</div>'
            + '</div>'),
        section(
            "Quantum statistics",
            "Identical particles come in two kinds. Fermions obey Pauli exclusion, so "
            "their Fermi-Dirac occupation never exceeds one and at T=0 fills states in a "
            "sharp step up to the Fermi level -- electron degeneracy, white-dwarf "
            "pressure. Bosons pile up without limit (Bose-Einstein), condensing into the "
            "ground state below a critical temperature. Far above the chemical potential "
            "both fade into the classical Maxwell-Boltzmann exponential.",
            '<div class="grid">'
            + svg_card(out("quantum_stats.svg"), "the fermion step, boson divergence and classical merge")
            + f'<div class="card">{pre(qstats_txt)}</div>'
            + '</div>'),
        section(
            "Debye specific heat",
            "Classically a solid stores 3R of heat per mole (Dulong-Petit), but measured "
            "heat capacities plunge toward zero in the cold. Debye treated the "
            "vibrations as quantized phonons with a maximum frequency (the Debye "
            "temperature), giving C_V ~ T^3 at low T and 3R at high T. A stiff light "
            "lattice like diamond (Theta_D ~ 2230 K) is still 'cold' at room temperature "
            "-- only ~1/6 of 3R -- while soft heavy lead has long reached the plateau.",
            '<div class="grid">'
            + svg_card(out("debye_heat.svg"), "the universal C_V/3R vs T/Theta_D curve")
            + f'<div class="card">{pre(debye_txt)}</div>'
            + '</div>'),
        section(
            "The Carnot cycle",
            "No heat engine between reservoirs at T_h and T_c can beat eta = 1 - T_c/T_h "
            "-- the second law forces some heat to be dumped to the cold side. A steam "
            "plant at 800 K exhausting to 300 K is capped at 62%; ocean-thermal at a "
            "20 K gap only 7%. Reversed, the cycle is a heat pump with COP >> 1, "
            "delivering many times the heat of the work it draws -- why heat pumps beat "
            "resistive heaters.",
            '<div class="grid">'
            + svg_card(out("carnot.svg"), "efficiency vs reservoir temperature ratio with real engines marked")
            + f'<div class="card">{pre(carnot_txt)}</div>'
            + '</div>'),
        section(
            "Adiabatic processes",
            "Compress or expand a gas with no time to shed heat and it obeys P V^gamma = "
            "const, T V^(gamma-1) = const -- and its temperature changes. A diesel "
            "engine's 22:1 squeeze reaches ~1000 K and ignites fuel without a spark; "
            "expansion cools (rising air, released spray). Sound waves compress air "
            "adiabatically, so the speed of sound carries Laplace's sqrt(gamma) factor -- "
            "the fix that corrected Newton's ~18% error.",
            '<div class="grid">'
            + svg_card(out("adiabatic.svg"), "an adiabat steeper than the isotherm through the same point")
            + f'<div class="card">{pre(adiab_txt)}</div>'
            + '</div>'),
        section(
            "The van der Waals gas",
            "Give the ideal gas molecules a finite size (b) and mutual attraction (a) "
            "and it can condense: (P + a n^2/V^2)(V - nb) = nRT. Below the critical "
            "temperature the isotherm develops an unstable loop where gas turns to "
            "liquid. The critical constants follow from a and b alone -- CO2's 304 K, "
            "7.4 MPa -- and the compressibility Pc Vc / R Tc = 3/8 is universal, the law "
            "of corresponding states.",
            '<div class="grid">'
            + svg_card(out("van_der_waals.svg"), "reduced isotherms with the sub-critical condensation loop")
            + f'<div class="card">{pre(vdw_txt)}</div>'
            + '</div>'),
        section(
            "The Joule-Thomson effect",
            "Push a real gas through a valve and it cools or warms depending on which "
            "wins: attraction (cools) or finite molecular size (warms). Below the "
            "inversion temperature T_inv = (27/4) T_c throttling cools -- so nitrogen "
            "and CO2 liquefy by repeated expansion at room temperature, but hydrogen "
            "and helium (low T_inv) warm and must be pre-cooled first. An ideal gas has "
            "no Joule-Thomson effect at all.",
            '<div class="grid">'
            + svg_card(out("joule_thomson.svg"), "the JT coefficient crossing zero at each gas's inversion temperature")
            + f'<div class="card">{pre(jt_txt)}</div>'
            + '</div>'),
        section(
            "Clausius-Clapeyron: vapor pressure",
            "Along a liquid-vapour line, pressure and temperature are locked by "
            "dP/dT = L/(T dV), integrating to P(T) = P0 exp(-(L/R)(1/T - 1/T0)) -- vapor "
            "pressure climbs exponentially, roughly doubling every ~15 K. Boiling is "
            "where it meets the ambient pressure, so water boils at 72 C atop Everest "
            "(thin air) and 121 C in a pressure cooker. The same curve sets atmospheric "
            "humidity and cloud formation.",
            '<div class="grid">'
            + svg_card(out("clausius_clapeyron.svg"), "the exponential vapor-pressure curve with altitude markers")
            + f'<div class="card">{pre(cc_txt)}</div>'
            + '</div>'),
        section(
            "The Reynolds number",
            "One dimensionless ratio Re = rho v L / mu decides whether a flow is smooth "
            "or turbulent: viscosity damps disturbances at low Re, inertia tears them "
            "into eddies at high Re, with pipe flow transitioning near Re ~ 2300. It "
            "spans 13 orders of magnitude -- a bacterium at Re ~ 1e-5 lives in pure "
            "viscosity and cannot coast, a whale at Re ~ 1e8 glides on inertia. Laminar "
            "flow follows Hagen-Poiseuille's r^4 law.",
            '<div class="grid">'
            + svg_card(out("reynolds.svg"), "systems from bacterium to whale on a log Reynolds axis")
            + f'<div class="card">{pre(reynolds_txt)}</div>'
            + '</div>'),
        section(
            "Bernoulli & the Venturi effect",
            "Along a streamline P + 1/2 rho v^2 + rho g h is constant, so speeding up a "
            "flow drops its pressure. A narrowing Venturi pipe is fastest and lowest-"
            "pressure at the throat -- drawing fuel into a carburettor, reading flow in a "
            "meter, and (with circulation) helping lift a wing. A Pitot tube runs it "
            "backwards to give airspeed, and Torricelli's sqrt(2gh) jet is the same law "
            "with the pressures cancelled.",
            '<div class="grid">'
            + svg_card(out("bernoulli.svg"), "a Venturi tube: velocity peaks and pressure dips at the throat")
            + f'<div class="card">{pre(bernoulli_txt)}</div>'
            + '</div>'),
        section(
            "Surface tension: capillary rise & Laplace pressure",
            "A liquid surface costs energy per unit area (gamma), so it behaves like a "
            "stretched skin. In a thin tube that pull lifts water against gravity by "
            "Jurin's law h = 2 gamma cos(theta) / (rho g r) -- a 1 mm bore climbs ~1.5 cm, a "
            "1 micron root pore tens of metres. A curved surface also holds a pressure jump "
            "2 gamma/r (a droplet) or 4 gamma/r (a soap bubble's two films), so smaller drops "
            "run at higher pressure and empty into larger ones.",
            '<div class="grid">'
            + svg_card(out("surface_tension.svg"), "capillary rise vs tube radius on log-log axes: narrower climbs higher")
            + f'<div class="card">{pre(surface_tension_txt)}</div>'
            + '</div>'),
        section(
            "The Ekman spiral: wind, rotation & the ocean",
            "Steady wind drags the sea surface, but Coriolis deflects the current 45 degrees "
            "to the right of the wind (northern hemisphere). Balancing friction against "
            "rotation, the current spirals clockwise and decays exponentially with depth over "
            "the Ekman depth D = pi sqrt(2 A_z/|f|). The vertically integrated transport ends "
            "up exactly 90 degrees to the right of the wind with magnitude tau/(rho |f|), "
            "independent of viscosity -- the sideways pumping behind coastal upwelling and "
            "the ocean gyres.",
            '<div class="grid">'
            + svg_card(out("ekman.svg"), "hodograph: the current vector turns clockwise and shrinks with depth")
            + f'<div class="card">{pre(ekman_txt)}</div>'
            + '</div>'),
        section(
            "Milankovitch cycles: orbits and the ice ages",
            "Slow changes in Earth's orbit -- eccentricity (~100 kyr), obliquity (~41 kyr) and "
            "precession (~23 kyr) -- redistribute sunlight between seasons and latitudes even "
            "though the annual total barely moves. Computing daily top-of-atmosphere "
            "insolation from the astronomical formula reproduces the ~478 W/m^2 65N midsummer "
            "peak; that high-latitude summer sun is the knob that decides whether winter snow "
            "survives to build ice sheets, so cool summers (low tilt, summer at aphelion) grow "
            "the glaciers.",
            '<div class="grid">'
            + svg_card(out("milankovitch.svg"), "daily insolation over latitude and season, polar day/night and the 65N target marked")
            + f'<div class="card">{pre(milankovitch_txt)}</div>'
            + '</div>'),
        section(
            "Equipartition & the heat-capacity staircase",
            "Classically every quadratic degree of freedom carries (1/2) k_B T, so an ideal "
            "gas has C_V = (f/2)R and gamma = (f+2)/f: 3R/2 and 5/3 for a monatomic gas, "
            "5R/2 and 7/5 for a diatomic at room temperature, and 3R for a solid (Dulong-"
            "Petit). But equipartition is only the hot-limit ceiling -- quantum mechanics "
            "freezes a mode out below its energy quantum, so H2 climbs a staircase from 3R/2 "
            "to 5R/2 (rotation thaws near 100 K) toward 7R/2 (vibration near 1000s K).",
            '<div class="grid">'
            + svg_card(out("equipartition.svg"), "H2 molar C_V/R vs temperature: plateaus at 3/2, 5/2, 7/2 as modes thaw")
            + f'<div class="card">{pre(equipartition_txt)}</div>'
            + '</div>'),
        section(
            "Osmotic pressure: van't Hoff across a membrane",
            "Water crosses a semipermeable membrane into a solution until the built-up head "
            "balances it. For dilute solutions the equilibrium pressure follows van't Hoff's "
            "law Pi = i c R T -- the ideal-gas law with solute particles as the gas, so a "
            "salt that splits into i ions pushes i times as hard. It reproduces seawater's "
            "~27 atm (the wall reverse-osmosis desalination must beat) and blood plasma's "
            "~7.6 atm (which sets isotonic IV fluids), and it weighs macromolecules by the "
            "tiny pressure they raise.",
            '<div class="grid">'
            + svg_card(out("osmosis.svg"), "osmotic pressure vs concentration for glucose, NaCl and CaCl2, seawater & blood marked")
            + f'<div class="card">{pre(osmosis_txt)}</div>'
            + '</div>'),
        section(
            "Fick diffusion: spreading as the root of time",
            "A random walk spreads a concentration downhill: Fick's first law J = -D dC/dx and "
            "the diffusion equation dC/dt = D d^2C/dx^2. A point release stays a Gaussian whose "
            "rms width grows as sqrt(2 D t) -- the diffusive sqrt(t), never the ballistic t -- "
            "and a step interface relaxes through an error-function profile. Because the time "
            "to cross a length scales as L^2/D, diffusion is fast across a cell (~0.1 s) but "
            "hopeless across a room (~30 years), which is why life is small and large systems "
            "need flow. Stokes-Einstein ties D to temperature and drag.",
            '<div class="grid">'
            + svg_card(out("diffusion.svg"), "a point release spreading into wider, lower Gaussians at t, 4t, 16t, 64t")
            + f'<div class="card">{pre(diffusion_txt)}</div>'
            + '</div>'),
        section(
            "The Peclet number: carried vs spreading",
            "Heat and solutes move both by being carried in the flow (advection) and by "
            "spreading down their gradient (diffusion); the Peclet number Pe = U L / D says "
            "which wins. Pe << 1 (a cell, a still cup) is diffusion-controlled; Pe >> 1 (a "
            "river, an artery) is swept along in thin plumes. It factors as Pe = Re*Pr for "
            "heat and Re*Sc for mass, so water's Pr ~ 7, air's Pr ~ 0.7 and an aqueous "
            "solute's Sc ~ 1000 set boundary-layer thicknesses, and the crossover Pe = 1 "
            "sits at the length L = D/U.",
            '<div class="grid">'
            + svg_card(out("peclet.svg"), "regime map over flow speed and length, shaded by Peclet with the Pe=1 crossover")
            + f'<div class="card">{pre(peclet_txt)}</div>'
            + '</div>'),
        section(
            "Convective heat transfer & Newton cooling",
            "A moving fluid strips heat off a surface at q = h (T_s - T_inf), and the "
            "coefficient h follows from the Nusselt number Nu = h L / k -- the ratio of "
            "convective to conductive transport. Correlations give it: Dittus-Boelter "
            "Nu = 0.023 Re^0.8 Pr^0.4 for turbulent pipe flow, 0.664 Re^0.5 Pr^(1/3) for a "
            "laminar plate. A lumped object then cools exponentially with tau = rho c_p V/(h A) "
            "provided the Biot number Bi = h L/k_solid stays below ~0.1, so an aluminium block "
            "cools in an hour in still air but seconds in forced water.",
            '<div class="grid">'
            + svg_card(out("convection.svg"), "a hot block cooling: time constant shrinks from still air to forced water")
            + f'<div class="card">{pre(convection_txt)}</div>'
            + '</div>'),
        section(
            "The Stefan problem: a freezing front as sqrt(t)",
            "A melting or freezing interface is driven by latent heat, not temperature alone: "
            "the heat released as water freezes must conduct out through the ice already "
            "formed, so the front advances as X = 2 lambda sqrt(alpha t), slowing as it "
            "deepens. The growth coefficient lambda solves the Stefan condition "
            "lambda e^(lambda^2) erf(lambda) = St/sqrt(pi), where the Stefan number "
            "St = c_p dT/L weighs sensible against latent heat. It reproduces Stefan's classic "
            "estimate -- about 10 cm of ice after a day of hard frost -- and the depth^2 time "
            "law that makes the next foot take weeks.",
            '<div class="grid">'
            + svg_card(out("stefan.svg"), "ice thickness vs time for light, hard and arctic frost -- the sqrt(t) slowdown")
            + f'<div class="card">{pre(stefan_txt)}</div>'
            + '</div>'),
        section(
            "The capillary length: surface tension vs gravity",
            "The same liquid makes a round dewdrop and a flat puddle -- the difference is size. "
            "Surface tension pulls toward a sphere, gravity flattens anything taller than the "
            "capillary length l_c = sqrt(gamma/(rho g)) (~2.7 mm for water). The Bond number "
            "Bo = (L/l_c)^2 says which wins: below 1 drops are round, above 1 they puddle out "
            "(capped at ~2 l_c deep). A moving drop adds inertia through the Weber number and "
            "shatters once We tops ~12, and a thin jet pinches into drops spaced ~9 radii "
            "apart by the Rayleigh-Plateau instability.",
            '<div class="grid">'
            + svg_card(out("capillary.svg"), "drops morphing from round spheres to flat puddles as size crosses the capillary length")
            + f'<div class="card">{pre(capillary_txt)}</div>'
            + '</div>'),
        section(
            "The Froude number: racing your own waves",
            "A surface disturbance travels at the shallow-water wave speed sqrt(g h), and the "
            "Froude number Fr = U/sqrt(g h) compares the flow to it. Fr < 1 is tranquil "
            "(subcritical) flow whose ripples run upstream; Fr > 1 is shooting (supercritical) "
            "flow that outruns its waves, so a sudden slowing throws up a hydraulic jump. For a "
            "ship the hull Froude number Fr = U/sqrt(g L) sets wave-making drag, walling a "
            "displacement hull near Fr ~ 0.4 (the 1.34 sqrt(L_ft) knots rule), and its wake "
            "wedge holds a fixed 19.47-degree half-angle at any speed.",
            '<div class="grid">'
            + svg_card(out("froude.svg"), "a hydraulic jump: thin fast supercritical water leaping to a deep slow pool")
            + f'<div class="card">{pre(froude_txt)}</div>'
            + '</div>'),
        section(
            "The Mach cone: the geometry of going supersonic",
            "When a source outruns sound, its wavelets pile into a trailing cone whose "
            "half-angle obeys sin(mu) = 1/M -- 90 degrees at Mach 1, tightening to 30 at Mach "
            "2 and 11.5 at Mach 5. That cone is the shock a ground observer hears as a sonic "
            "boom, laid down behind the overhead point and sweeping a continuous carpet along "
            "the track. Below Mach 1, thin-airfoil lift diverges by the Prandtl-Glauert factor "
            "1/sqrt(1-M^2) toward the sound barrier; above it, a flow turning a corner expands "
            "through the Prandtl-Meyer angle.",
            '<div class="grid">'
            + svg_card(out("mach_cone.svg"), "a supersonic source, its expanding wavelets, and the trailing Mach cone")
            + f'<div class="card">{pre(mach_cone_txt)}</div>'
            + '</div>'),
        section(
            "The de Laval nozzle: making exhaust supersonic",
            "Subsonic flow speeds up as a pipe narrows, but supersonic flow speeds up as it "
            "widens -- so to push exhaust past Mach 1 you squeeze the gas to a sonic throat "
            "and then expand it through a diverging bell. The isentropic relations fix "
            "everything from the local Mach number: the area-Mach relation A/A* is minimal at "
            "the throat, the flow chokes there once the pressure ratio drops below ~0.528 "
            "(air), and the exit Mach number is then set purely by the bell's area ratio -- 25 "
            "gives Mach 5. It is how every rocket and supersonic tunnel works.",
            '<div class="grid">'
            + svg_card(out("nozzle.svg"), "a converging-diverging nozzle with Mach rising through 1 and pressure falling")
            + f'<div class="card">{pre(nozzle_txt)}</div>'
            + '</div>'),
        section(
            "The Blasius boundary layer",
            "No-slip makes a thin sheared film cling to any surface in a stream. Blasius solved "
            "the laminar flat-plate case exactly: the 99%-thickness grows as "
            "delta = 5.0 x/sqrt(Re_x) -- only millimetres over the front of a wing -- with the "
            "displacement and momentum thicknesses tracking it at 1.721 and 0.664. The wall "
            "shear gives a local skin friction c_f = 0.664/sqrt(Re_x) (heaviest at the sharp "
            "leading edge) and a plate drag C_D = 1.328/sqrt(Re_L), and the layer stays laminar "
            "until Re_x ~ 5e5, where it trips to turbulence.",
            '<div class="grid">'
            + svg_card(out("blasius.svg"), "the boundary layer thickening as sqrt(x) with velocity profiles and the transition point")
            + f'<div class="card">{pre(blasius_txt)}</div>'
            + '</div>'),
        section(
            "The Strouhal number: von Karman vortex streets",
            "A blunt body in a steady flow sheds vortices alternately from each side, a "
            "staggered von Karman street whose frequency obeys St = f d / U with St ~ 0.2 "
            "nearly constant over a huge Reynolds-number range. So shedding frequency scales "
            "linearly with wind speed -- the aeolian hum of a wire (a 5 mm wire in 10 m/s wind "
            "sings at 400 Hz), the flutter of an antenna. When that frequency crosses a "
            "structure's natural frequency the flow locks in and the alternating side-force "
            "can drive destructive vortex-induced vibration, the reason chimneys wear helical "
            "strakes.",
            '<div class="grid">'
            + svg_card(out("strouhal.svg"), "vortices peeling alternately off a cylinder into the staggered von Karman wake")
            + f'<div class="card">{pre(strouhal_txt)}</div>'
            + '</div>'),
        section(
            "Weighing clusters: the virial mass & dark matter",
            "The virial theorem turns a cluster's own motion into a scale: M = alpha sigma^2 R/G "
            "from the velocity dispersion sigma and size R. Only the line-of-sight dispersion "
            "is observable, so sigma^2 = 3 sigma_los^2 for an isotropic system. This is Zwicky's "
            "1933 Coma calculation -- galaxies moving at ~1000 km/s across ~1.5 Mpc demand a "
            "dynamical mass ~10^15 solar masses, about a hundred times the visible stars. The "
            "mass-to-light ratio jumps from a few for stars to hundreds for clusters: the first "
            "evidence for dark matter.",
            '<div class="grid">'
            + svg_card(out("cluster_mass.svg"), "the mass-to-light ladder climbing from a star to a cluster")
            + f'<div class="card">{pre(cluster_mass_txt)}</div>'
            + '</div>'),
        section(
            "The Sersic profile: the shape of a galaxy's light",
            "A galaxy's surface brightness falls off as I(R) = I_e exp{-b_n[(R/R_e)^(1/n)-1]}, "
            "the Sersic law, with the index n setting the concentration. n=1 is the exponential "
            "disk of a spiral (scale length R_e/1.678); n=4 is the de Vaucouleurs law of a "
            "giant elliptical -- a bright cusped core and enormous faint wings. Half the light "
            "always sits inside the effective radius R_e (that is its definition), and "
            "integrating the profile gives the total luminosity in closed form via the gamma "
            "function, so I_e, R_e and n weigh a galaxy's stars.",
            '<div class="grid">'
            + svg_card(out("sersic.svg"), "surface brightness vs radius for n=1, 2, 4 -- all crossing at the effective radius")
            + f'<div class="card">{pre(sersic_txt)}</div>'
            + '</div>'),
        section(
            "The Grashof number: heat that stirs its own wind",
            "Natural convection has no fan -- warm fluid expands, rises, and drags cooler fluid "
            "in behind it. The Grashof number Gr = g beta dT L^3/nu^2 measures that buoyant "
            "drive against viscosity, playing the role Reynolds plays in forced flow. Heat "
            "transfer correlates against the Rayleigh number Ra = Gr Pr: Nu = 0.59 Ra^(1/4) "
            "laminar, 0.10 Ra^(1/3) turbulent past Ra ~ 1e9. The result is the gentle few "
            "W/(m^2 K) of a radiator warming a still room, an order of magnitude below forced "
            "convection; Gr/Re^2 tells you which regime rules.",
            '<div class="grid">'
            + svg_card(out("grashof.svg"), "natural-convection h vs wall height, tripping from laminar to turbulent at Ra ~ 1e9")
            + f'<div class="card">{pre(grashof_txt)}</div>'
            + '</div>'),
        section(
            "The Womersley number: why blood flow lags the pulse",
            "A pulsating pressure does not make a pulsating parabola. The Womersley number "
            "alpha = R sqrt(omega/nu) compares the heartbeat frequency to how fast viscosity "
            "diffuses momentum across a vessel. Small alpha (arterioles, capillaries) stays "
            "quasi-steady -- an in-phase Poiseuille parabola; large alpha (the aorta at "
            "alpha ~ 15) has too much core inertia to follow, so the flow lags the pressure "
            "by up to 90 degrees and flattens into a blunt plug with the shear squeezed into a "
            "thin wall layer. The pressure pulse itself races ahead at the Moens-Korteweg "
            "speed.",
            '<div class="grid">'
            + svg_card(out("womersley.svg"), "velocity profiles from quasi-steady parabola to inertial plug as alpha grows")
            + f'<div class="card">{pre(womersley_txt)}</div>'
            + '</div>'),
        section(
            "The Marangoni effect: flow along a tension gradient",
            "When surface tension varies along a surface -- from a temperature or composition "
            "gradient -- the imbalance drags the fluid from low-tension toward high-tension "
            "regions. It climbs the tears of wine up a glass, scatters pepper from a soap drop, "
            "and stirs weld pools. The Marangoni number Ma = |dgamma/dT| dT L/(mu alpha) "
            "measures the drive against diffusion, breaking a heated layer into Benard-"
            "Marangoni cells above Ma ~ 80. The dynamic Bond number Ra/Ma decides surface "
            "tension vs buoyancy: thin films and microgravity are Marangoni-driven, thick "
            "pools on the ground buoyancy-driven.",
            '<div class="grid">'
            + svg_card(out("marangoni.svg"), "regime map over layer thickness and gravity: Marangoni vs buoyancy")
            + f'<div class="card">{pre(marangoni_txt)}</div>'
            + '</div>'),
        section(
            "Kutta-Joukowski: lift is circulation",
            "A wing flies because the flow around it carries a net swirl -- circulation -- and "
            "the Kutta-Joukowski theorem makes it exact: lift per span L' = rho U Gamma. The "
            "airfoil sets its own circulation through the Kutta condition (smooth flow off the "
            "trailing edge), giving the thin-airfoil lift-slope c_l = 2 pi alpha. The same "
            "theorem is the Magnus effect -- a spinning ball drags a boundary layer around, "
            "generating circulation and a sideways curve. Lift isn't free: finite wings trail "
            "vortices and pay induced drag c_l^2/(pi AR e), so soaring birds wear long thin "
            "wings.",
            '<div class="grid">'
            + svg_card(out("kutta_joukowski.svg"), "the 2 pi lift-slope with stall, and the induced-drag penalty vs aspect ratio")
            + f'<div class="card">{pre(kutta_joukowski_txt)}</div>'
            + '</div>'),
        section(
            "The Knudsen number: when a gas stops being a fluid",
            "A gas behaves as a smooth continuum only while its mean free path lambda is tiny "
            "next to the system size L. Their ratio, the Knudsen number Kn = lambda/L, sorts "
            "flows into continuum (Kn<0.01, ordinary Navier-Stokes), slip, transitional and "
            "free-molecular (Kn>10, molecules fly wall-to-wall) regimes. Air's sea-level "
            "lambda ~ 68 nm makes everything macroscopic a perfect fluid -- but shrink L to a "
            "MEMS channel or a nanopore, or thin the air at orbital altitude, and Kn climbs "
            "past 1, so the gas slips at walls and finally must be computed molecule by "
            "molecule.",
            '<div class="grid">'
            + svg_card(out("knudsen.svg"), "regime map over system size and pressure: continuum to free-molecular")
            + f'<div class="card">{pre(knudsen_txt)}</div>'
            + '</div>'),
        section(
            "The Richardson number: shear vs stratification",
            "A stably stratified fluid resists overturning, but fast enough shear can rip the "
            "interface into billows anyway. The gradient Richardson number Ri = N^2/(du/dz)^2 "
            "weighs the buoyant stiffness against the squared shear: the Miles-Howard theorem "
            "guarantees stability where Ri > 1/4, and below it the Kelvin-Helmholtz instability "
            "curls the interface into cat's-eye billows. It governs clear-air turbulence that "
            "jolts aircraft, mixing in the ocean thermocline (Ri ~ 4, layered) and the "
            "entrainment atop a fog layer.",
            '<div class="grid">'
            + svg_card(out("richardson.svg"), "stability map over stratification and shear with the Ri=1/4 threshold and KH billows")
            + f'<div class="card">{pre(richardson_txt)}</div>'
            + '</div>'),
        section(
            "The Kolmogorov cascade: turbulence shredding into heat",
            "Turbulence hands energy down a cascade: big eddies break into smaller ones until "
            "viscosity smears the tiniest into heat. In the inertial range between, statistics "
            "depend only on the dissipation rate epsilon, giving Kolmogorov's E(k) ~ "
            "epsilon^(2/3) k^(-5/3) -- the -5/3 law seen from wind tunnels to interstellar gas. "
            "The cascade ends at the Kolmogorov scale eta = (nu^3/epsilon)^(1/4), where the "
            "eddy Reynolds number is 1, and the span L/eta ~ Re^(3/4) makes turbulence cost "
            "~Re^(9/4) grid points to simulate in 3-D.",
            '<div class="grid">'
            + svg_card(out("kolmogorov.svg"), "the E(k) spectrum with its -5/3 inertial range between injection and dissipation")
            + f'<div class="card">{pre(kolmogorov_txt)}</div>'
            + '</div>'),
        section(
            "The Casimir effect: pushed together by empty space",
            "The quantum vacuum has a zero-point energy in every field mode. Slide two "
            "conducting plates close and only the modes that fit in the gap survive between "
            "them, so the fuller vacuum outside presses them together with a pressure "
            "P = pi^2 hbar c/(240 d^4) -- a force from nothing but the structure of empty "
            "space, predicted in 1948 and measured in 1997. The steep d^-4 law makes it "
            "invisible at human gaps but crushing below 100 nm (an atmosphere by ~10 nm), "
            "where it sticks micro-machine parts together.",
            '<div class="grid">'
            + svg_card(out("casimir.svg"), "Casimir pressure vs plate gap on log-log axes, crossing one atmosphere near 10 nm")
            + f'<div class="card">{pre(casimir_txt)}</div>'
            + '</div>'),
        section(
            "The Hall effect: weighing carriers with a magnet",
            "Run a current through a conductor in a perpendicular field and the Lorentz force "
            "pushes the carriers sideways until a transverse Hall voltage V_H = I B/(n q t) "
            "balances them. Its magnitude gives the carrier density n, and its sign reveals "
            "whether the charge carriers are electrons or positive holes -- the result that "
            "classical free-electron theory could not explain and that underpins semiconductor "
            "doping. Combined with the conductivity it separates density from mobility "
            "(mu = |R_H| sigma), so a Hall bar fully characterizes a conductor.",
            '<div class="grid">'
            + svg_card(out("hall_effect.svg"), "a Hall bar: current, field, deflected carriers and the transverse Hall voltage")
            + f'<div class="card">{pre(hall_effect_txt)}</div>'
            + '</div>'),
        section(
            "Wiedemann-Franz: good conductors of charge and heat",
            "In a metal the same free electrons carry charge and heat, so their conductivities "
            "are locked together: kappa/(sigma T) = L, the Lorenz number pi^2 k_B^2/(3 e^2) = "
            "2.44e-8 W ohm/K^2. The material-specific mean free path and carrier density cancel "
            "in the ratio, leaving only fundamental constants -- so you can read a metal's "
            "thermal conductivity off an easy resistance measurement (copper's ~400 W/(m K) "
            "from its sigma). A measured Lorenz number well below L flags heat and charge "
            "decoupling, the signature of exotic 'strange metals'.",
            '<div class="grid">'
            + svg_card(out("wiedemann_franz.svg"), "predicted vs measured thermal conductivity: metals hug the Lorenz-number line")
            + f'<div class="card">{pre(wiedemann_franz_txt)}</div>'
            + '</div>'),
        section(
            "Bragg diffraction: reading a crystal with X-rays",
            "Shine X-rays on a crystal and bright reflections flash only where waves scattered "
            "from successive atomic planes add in phase: n lambda = 2 d sin(theta). Because the "
            "wavelength must match the ~0.1-0.5 nm atomic spacing, X-rays are the natural probe, "
            "and reading the diffraction spots backwards gives the structure -- the method that "
            "solved salt, DNA and countless proteins. For a cubic lattice each Miller plane "
            "(hkl) has its own spacing a/sqrt(h^2+k^2+l^2) and family of Bragg angles, and "
            "sin(theta) <= 1 caps the visible orders at 2d/lambda.",
            '<div class="grid">'
            + svg_card(out("bragg.svg"), "X-rays reflecting off two atomic planes with the 2 d sin(theta) path difference")
            + f'<div class="card">{pre(bragg_txt)}</div>'
            + '</div>'),
        section(
            "The diffraction limit: every aperture's resolution floor",
            "No lens focuses light to a point: an aperture of diameter D spreads a wave into an "
            "Airy disk of angular radius theta = 1.22 lambda/D, the Rayleigh criterion, so two "
            "sources closer than that blur into one. Bigger apertures resolve finer detail "
            "(Hubble's 2.4 m gives ~0.05 arcsec), long wavelengths need huge ones (radio "
            "dishes), and a microscope stops at the Abbe limit lambda/(2 NA) ~ 200 nm for "
            "light -- which is why electron microscopes with picometre wavelengths see atoms. "
            "A grating turns it into a spectrometer of resolving power m N.",
            '<div class="grid">'
            + svg_card(out("diffraction_limit.svg"), "resolution vs aperture with real instruments, and two sources at the Rayleigh limit")
            + f'<div class="card">{pre(diffraction_limit_txt)}</div>'
            + '</div>'),
        section(
            "Snell's law: bending, trapping, and reflecting light",
            "Light changes speed across a boundary and so must bend: n1 sin(theta1) = "
            "n2 sin(theta2). Entering a denser medium it turns toward the normal; leaving one, "
            "away from it -- until, past the critical angle arcsin(n2/n1), the refracted ray "
            "cannot exist and all the light is totally internally reflected. That perfect "
            "mirror guides light down an optical fibre and makes a diamond (24 deg critical "
            "angle) sparkle. At Brewster's angle arctan(n2/n1) the reflection is perfectly "
            "polarized, the trick behind polarizing sunglasses.",
            '<div class="grid">'
            + svg_card(out("snell.svg"), "rays leaving water bending away from the normal, then flipping to total internal reflection")
            + f'<div class="card">{pre(snell_txt)}</div>'
            + '</div>'),
        section(
            "Thin-film interference: bubble colours and lens coatings",
            "A transparent film reflects light off both surfaces, and the two waves interfere by "
            "the round-trip path 2 n t -- so a film only nanometres thick paints itself in "
            "colour, the sheen of a soap bubble or an oil slick. A half-wave phase flip on the "
            "denser-medium reflection sets which colours brighten or cancel, and is why a soap "
            "film goes black just before it bursts (2 n t -> 0, destructive everywhere). "
            "Engineered as a quarter-wave layer t = lambda/(4n) of index sqrt(n_substrate), the "
            "same interference cancels reflection -- the anti-glare coating on every lens. "
            "Newton's rings are its fringes in an air gap.",
            '<div class="grid">'
            + svg_card(out("thin_film.svg"), "soap-film bright colour vs thickness, and Newton's rings under a lens")
            + f'<div class="card">{pre(thin_film_txt)}</div>'
            + '</div>'),
        section(
            "Malus's law: dialling light down with polarizers",
            "A polarizer passes only the field component along its axis, so linearly polarized "
            "light emerges at I = I0 cos^2(theta) -- Malus's law. Unpolarized light loses "
            "exactly half through any one polarizer, and two crossed at 90 degrees pass nothing "
            "(the dark LCD pixel); yet slipping a third at 45 degrees between them rescues I0/8, "
            "light where there was none. A stack of many slightly rotated polarizers drags the "
            "polarization around while passing nearly all the light -- an optical quantum Zeno "
            "effect -- and wave plates rotate it losslessly by retarding one component.",
            '<div class="grid">'
            + svg_card(out("malus.svg"), "the cos^2 transmission law and the three-polarizer rescue vs middle angle")
            + f'<div class="card">{pre(malus_txt)}</div>'
            + '</div>'),
        section(
            "Cherenkov radiation: the blue glow of going too fast",
            "Light slows to c/n in a medium, and a charged particle can outrun it. When "
            "beta > 1/n it drags an electromagnetic shock cone behind it -- an optical sonic "
            "boom -- radiating the blue glow of a reactor pool. The cone half-angle obeys "
            "cos(theta) = 1/(n beta), just like a Mach cone, opening from threshold toward a "
            "maximum arccos(1/n) (~41 deg in water) as the particle nears beta = 1. Because the "
            "angle reads off the velocity, ring-imaging Cherenkov detectors use it to identify "
            "particles, and neutrino observatories watch for the faint cones.",
            '<div class="grid">'
            + svg_card(out("cherenkov.svg"), "cone angle vs speed for several radiators, and the cone trailing a superluminal particle")
            + f'<div class="card">{pre(cherenkov_txt)}</div>'
            + '</div>'),
        section(
            "The Zeeman effect: splitting lines with a magnetic field",
            "An atom's magnetic moment shifts its energy levels in a field, so a spectral line "
            "splits: delta_E = g_J m_J mu_B B. The normal Zeeman effect (spin cancels, g=1) "
            "gives a clean Lorentz triplet shifted by mu_B B/h = 14 GHz per tesla; the "
            "anomalous effect (g != 1, from the Lande factor 1 + [J(J+1)+S(S+1)-L(L+1)]/2J(J+1)) "
            "splits into more, unevenly spaced lines whose existence forced the discovery of "
            "electron spin. Reading the splitting backwards measures the field -- how "
            "magnetograms map sunspots.",
            '<div class="grid">'
            + svg_card(out("zeeman.svg"), "the normal triplet fanning out with field, and an anomalous sublevel ladder")
            + f'<div class="card">{pre(zeeman_txt)}</div>'
            + '</div>'),
        section(
            "Rabi oscillations: a two-level atom flopping",
            "A near-resonant field does not just excite a two-level system once -- it cycles it "
            "coherently between ground and excited at the Rabi frequency Omega = dE/hbar. On "
            "resonance P_e(t) = sin^2(Omega t/2) swings fully 0 to 1, so a pi pulse inverts the "
            "population (a qubit X gate) and a pi/2 pulse builds an equal superposition. Detuned "
            "by delta the flopping runs faster, at sqrt(Omega^2 + delta^2), but only reaches "
            "Omega^2/(Omega^2 + delta^2) -- a Lorentzian resonance of width Omega. These are the "
            "elementary operations of atomic clocks and quantum bits.",
            '<div class="grid">'
            + svg_card(out("rabi.svg"), "excited-state probability flopping in time for several detunings, and the Lorentzian resonance")
            + f'<div class="card">{pre(rabi_txt)}</div>'
            + '</div>'),
        section(
            "The Franck-Hertz experiment: energy levels in a current",
            "Fire electrons through mercury vapour and ramp the accelerating voltage: the "
            "collected current climbs, then drops sharply every 4.9 V. Electrons collide "
            "elastically until they gain the atom's excitation energy, then dump exactly that "
            "quantum in an inelastic collision and arrive too slow to be collected -- so the "
            "current falls, and the evenly spaced dips are direct proof that atomic energy is "
            "quantized (the 1914 confirmation of the Bohr atom). The excited atom relaxes by "
            "emitting a photon at that energy, mercury's 254 nm UV line.",
            '<div class="grid">'
            + svg_card(out("franck_hertz.svg"), "the current-vs-voltage sawtooth with dips at multiples of the 4.9 V excitation")
            + f'<div class="card">{pre(franck_hertz_txt)}</div>'
            + '</div>'),
        section(
            "Moseley's law: ordering the elements by X-ray colour",
            "An element struck by fast electrons fluoresces characteristic X-rays, and Moseley "
            "found the square root of the K-alpha frequency rises linearly with atomic number, "
            "sqrt(f) = a(Z-1). Equivalently the K-alpha energy is a Rydberg-like "
            "13.6 (3/4)(Z-1)^2 eV -- copper's 8 keV, molybdenum's 17 keV. This ordered the "
            "periodic table by nuclear charge rather than atomic weight, exposed gaps where "
            "undiscovered elements had to sit, and is still how an XRF gun reads which elements "
            "a sample contains from its X-ray lines.",
            '<div class="grid">'
            + svg_card(out("moseley.svg"), "the Moseley plot: sqrt(K-alpha frequency) a straight line in atomic number")
            + f'<div class="card">{pre(moseley_txt)}</div>'
            + '</div>'),
        section(
            "The Stark effect: electric fields on atoms",
            "The electric analogue of Zeeman: a field shifts atomic levels and splits lines. "
            "Hydrogen's degenerate levels give a LINEAR Stark effect -- a shift proportional to "
            "the field, since they mix into a permanent dipole -- fanning level n into 2n-1 "
            "equally spaced components. Most atoms have no permanent dipole and shift "
            "quadratically, -1/2 alpha E^2, always lowering the energy. Push hard enough and "
            "the field strips the electron: the ionization threshold scales as 1/n^4, so a "
            "Rydberg atom ionizes in a field ten billion times weaker than the ground state.",
            '<div class="grid">'
            + svg_card(out("stark.svg"), "the linear Stark fan of hydrogen n=4, and the ionizing field plummeting with n")
            + f'<div class="card">{pre(stark_txt)}</div>'
            + '</div>'),
        section(
            "The Aharonov-Bohm effect: a phase from an untouched field",
            "Classically no field means no effect, but a charged particle steered around a "
            "solenoid -- with the field entirely confined inside, zero on its path -- still has "
            "its interference fringes shift. It responds to the vector potential, picking up a "
            "phase delta_phi = q Phi/hbar set purely by the enclosed flux, proof that the "
            "potentials are physically real in quantum mechanics. The phase is periodic in the "
            "flux quantum h/q (h/2e for Cooper pairs), which quantizes flux through a "
            "superconducting ring and drives SQUID magnetometers to sense fields a billion "
            "times weaker than Earth's.",
            '<div class="grid">'
            + svg_card(out("aharonov_bohm.svg"), "interference fringes sliding with enclosed flux, and the phase winding per flux quantum")
            + f'<div class="card">{pre(aharonov_bohm_txt)}</div>'
            + '</div>'),
        section(
            "The Josephson junction: a supercurrent that defines the volt",
            "Cooper pairs tunnel through a thin barrier between superconductors with zero "
            "voltage, a supercurrent I = I_c sin(phi) set only by the quantum phase difference "
            "(DC effect). Apply a DC voltage and the phase winds, so the current oscillates at "
            "the Josephson frequency f = 2eV/h = 483.6 GHz per millivolt -- an exact voltage-"
            "to-frequency conversion through only e and h. Irradiating the junction locks it "
            "onto quantized Shapiro voltage steps n h f/2e, which is how the SI volt is now "
            "defined and how the most accurate voltmeters work.",
            '<div class="grid">'
            + svg_card(out("josephson.svg"), "the DC I = I_c sin(phi) supercurrent, and the irradiated I-V climbing in Shapiro steps")
            + f'<div class="card">{pre(josephson_txt)}</div>'
            + '</div>'),
        section(
            "The quantum Hall effect: resistance from pure constants",
            "Cool a 2D electron gas in a strong field and its Hall resistance locks onto flat "
            "plateaus R_xy = R_K/nu with R_K = h/e^2 = 25812.807 ohm -- values set by "
            "fundamental constants alone, independent of the material. The electron energies "
            "collapse into Landau levels (spacing hbar eB/m, degeneracy eB/h), and when nu of "
            "them are filled the bulk is insulating while nu chiral edge channels each carry "
            "e^2/h of conductance. Reproducible to parts per billion in any device, it now "
            "defines the ohm -- the resistance counterpart of the Josephson volt.",
            '<div class="grid">'
            + svg_card(out("quantum_hall.svg"), "the Hall resistance staircase: plateaus at R_K/nu as field sweeps a fixed density")
            + f'<div class="card">{pre(quantum_hall_txt)}</div>'
            + '</div>'),
        section(
            "BCS superconductivity: the gap that kills resistance",
            "Below T_c a phonon-mediated attraction binds electrons into Cooper pairs that "
            "condense into one coherent state carrying current without resistance. The theory's "
            "core is an energy gap Delta at the Fermi surface -- it costs 2 Delta to break a "
            "pair, so nothing scatters the condensate. BCS predicts the universal ratio "
            "2 Delta(0)/(k_B T_c) = 3.53 for every weak-coupling superconductor, a gap that "
            "closes as sqrt(1-T/Tc), and T_c = 1.13 hbar wD exp(-1/lambda) -- whose wD ~ 1/sqrt(M) "
            "gives the isotope effect that proved phonons do the pairing.",
            '<div class="grid">'
            + svg_card(out("bcs.svg"), "the gap closing as sqrt(1-T/Tc), and T_c rising with electron-phonon coupling")
            + f'<div class="card">{pre(bcs_txt)}</div>'
            + '</div>'),
        section(
            "London & Meissner: expelling the magnetic field",
            "A superconductor doesn't just conduct perfectly -- it actively pushes magnetic "
            "field out (the Meissner effect), which is why magnets levitate above one. The "
            "London equations give the field decaying into the surface as exp(-x/lambda_L) "
            "over the penetration depth lambda_L = sqrt(m/(mu0 n_s q^2)), tens of nanometres. "
            "The ratio kappa = lambda_L/xi to the coherence length splits superconductors into "
            "type I (kappa < 1/sqrt2, full expulsion) and type II (kappa > 1/sqrt2, quantized "
            "flux vortices) -- the latter surviving the huge fields of MRI and fusion magnets.",
            '<div class="grid">'
            + svg_card(out("london.svg"), "the Meissner field decaying into the surface, and materials across the type-I/II boundary")
            + f'<div class="card">{pre(london_txt)}</div>'
            + '</div>'),
        section(
            "Mean-field ferromagnetism: order from disorder",
            "Below a Curie temperature a magnet spontaneously aligns -- countless spins tip "
            "into one direction with no applied field. Weiss mean-field theory of the Ising "
            "model captures it: each spin feels the average of its neighbours, m = "
            "tanh((z J m + B)/T), with T_c = z J. Above T_c the only zero-field solution is "
            "m = 0 (paramagnet); below it a nonzero magnetization appears, vanishing near T_c "
            "as (1 - T/Tc)^(1/2) (the mean-field beta = 1/2), while the susceptibility diverges "
            "as the Curie-Weiss 1/(T - T_c) -- the hallmarks of a second-order phase transition.",
            '<div class="grid">'
            + svg_card(out("ising_mft.svg"), "magnetization collapsing to zero at the Curie point, and the diverging susceptibility")
            + f'<div class="card">{pre(ising_mft_txt)}</div>'
            + '</div>'),
        section(
            "Percolation: the sudden onset of connectivity",
            "Occupy each lattice site with probability p and ask if a connected path spans the "
            "system. Below a sharp threshold p_c (~0.59 for a 2D square lattice) the occupied "
            "sites form isolated islands; above it a single cluster abruptly spans the whole "
            "lattice -- a geometric phase transition. The largest-cluster fraction jumps from "
            "near zero to order one through p_c. The same threshold governs forest fires "
            "spreading, oil seeping through rock, disease jumping a contact network, and "
            "current finding a path through a random resistor grid.",
            '<div class="grid">'
            + svg_card(out("percolation.svg"), "the spanning probability sharpening at p_c, and lattices below/at/above threshold")
            + f'<div class="card">{pre(percolation_txt)}</div>'
            + '</div>'),
        section(
            "Polya's random walk: home, or lost forever?",
            "A random walker on an infinite lattice steps to a random neighbour forever -- does "
            "it ever return to the origin? Polya proved the answer depends only on dimension: in "
            "1D and 2D the walk is recurrent, returning with probability 1 (and visiting every "
            "site infinitely often); in 3D and above it is transient, escaping to infinity with "
            "nonzero probability (a ~0.34 chance of ever returning in 3D). The knife-edge is "
            "exactly two dimensions, because the probability of being back at the origin decays "
            "as n^(-d/2) -- summable only for d >= 3. 'A drunk man finds his way home, but a "
            "drunk bird may get lost forever.'",
            '<div class="grid">'
            + svg_card(out("polya.svg"), "return probability dropping below 1 past two dimensions -- recurrent to transient")
            + f'<div class="card">{pre(polya_txt)}</div>'
            + '</div>'),
        section(
            "Langevin paramagnetism: moments vs thermal chaos",
            "A paramagnet's independent magnetic moments each favour aligning with an applied "
            "field (energy -mu.B) while temperature randomizes them. Averaging over the "
            "Boltzmann distribution gives the Langevin function m/mu = L(x) = coth(x) - 1/x "
            "with x = mu B/(k_B T). Weak field or high temperature is the linear regime "
            "L(x) ~ x/3, so the susceptibility follows Curie's law chi ~ 1/T -- the fingerprint "
            "of a paramagnet; strong field or low temperature saturates every moment at L = 1 "
            "and the magnetization can grow no further.",
            '<div class="grid">'
            + svg_card(out("langevin_para.svg"), "the Langevin function from the Curie slope to saturation, and the 1/T susceptibility")
            + f'<div class="card">{pre(langevin_para_txt)}</div>'
            + '</div>'),
        section(
            "Buffon's needle: estimating pi by dropping sticks",
            "Rule a floor with parallel lines a distance d apart and drop a needle of length "
            "L <= d at random: it crosses a line with probability 2 L/(pi d). So counting "
            "crossings estimates pi -- pi ~ 2 L N/(d C) for N drops and C crossings -- the "
            "first problem in geometric probability (Buffon, 1777). pi emerges from a purely "
            "mechanical experiment with no measurement of pi anywhere, from the geometry of "
            "random position and angle. Convergence is the slow Monte Carlo 1/sqrt(N): 1% "
            "needs ~10000 drops, 0.1% about a million.",
            '<div class="grid">'
            + svg_card(out("buffon.svg"), "needles dropped across ruled lines (crossings red), and the pi estimate converging")
            + f'<div class="card">{pre(buffon_txt)}</div>'
            + '</div>'),
        section(
            "Metropolis Monte Carlo: sampling the Ising transition",
            "Systems too large to sum are sampled: propose a change and accept it with "
            "probability min(1, exp(-dE/T)) -- always downhill, Boltzmann-weighted uphill -- and "
            "the chain visits states with exactly the thermal probability exp(-E/T). Run on the "
            "2D Ising ferromagnet it reproduces the real phase transition that mean-field theory "
            "only approximates: spins order below the exact Onsager T_c ~ 2.269 J/k_B and "
            "disorder above it, with genuine critical fluctuations -- domains at every scale near "
            "T_c -- that mean field cannot capture.",
            '<div class="grid">'
            + svg_card(out("metropolis.svg"), "the simulated magnetization dropping to zero at the Onsager T_c, and a near-critical spin snapshot")
            + f'<div class="card">{pre(metropolis_txt)}</div>'
            + '</div>'),
        section(
            "The logistic map: period doubling into chaos",
            "The one-line map x' = r x(1-x) is the textbook birth of chaos. As the growth rate "
            "r rises, a stable population splits into a 2-cycle at r=3, then 4, 8, 16, ... in a "
            "cascade that accumulates at r ~ 3.5699 -- the onset of aperiodic, "
            "initial-condition-sensitive chaos, interrupted by periodic windows (the famous "
            "period-3 near 3.83). The bifurcation spacings shrink by the universal Feigenbaum "
            "constant 4.669, the same for any smooth single-humped map, and a positive Lyapunov "
            "exponent marks the chaotic regime.",
            '<div class="grid">'
            + svg_card(out("logistic_map.svg"), "the bifurcation diagram doubling into chaos, above the Lyapunov exponent turning positive")
            + f'<div class="card">{pre(logistic_map_txt)}</div>'
            + '</div>'),
        section(
            "The Henon map: a strange attractor in two lines",
            "Henon's 1976 map, x' = 1 - a x^2 + y, y' = b x, is the canonical low-dimensional "
            "strange attractor. At a=1.4, b=0.3 the iterates never settle and never repeat, "
            "tracing a fractal of nested arcs that -- zoomed in -- resolve into a Cantor set of "
            "ever-finer strands. It is dissipative (areas shrink by |b| each step) yet chaotic "
            "(largest Lyapunov exponent ~0.42): a blob is squeezed in area while stretched and "
            "folded, collapsing onto a fractal of dimension ~1.26. Chaos with structure at "
            "every scale.",
            '<div class="grid">'
            + svg_card(out("henon.svg"), "the Henon attractor and a zoom revealing the fractal Cantor strands")
            + f'<div class="card">{pre(henon_txt)}</div>'
            + '</div>'),
        section(
            "The Lorenz attractor: the butterfly effect",
            "Lorenz's three equations for toy convection never repeat and cannot be forecast "
            "for long. At sigma=10, beta=8/3, rho=28 the trajectory winds around two spiral "
            "lobes, jumping between them unpredictably -- the butterfly-shaped strange "
            "attractor. The flow is dissipative (phase volume shrinks at -(sigma+1+beta) so "
            "everything collapses onto the zero-volume fractal) yet chaotic: two starts a "
            "millionth apart diverge to opposite wings, the largest Lyapunov exponent ~0.9 "
            "meaning prediction error grows tenfold every ~2.5 time units. The reason weather "
            "is unforecastable beyond ~two weeks.",
            '<div class="grid">'
            + svg_card(out("lorenz.svg"), "the butterfly attractor and two nearby trajectories diverging exponentially")
            + f'<div class="card">{pre(lorenz_txt)}</div>'
            + '</div>'),
        section(
            "The double pendulum: chaos you can hang from a nail",
            "Hang one pendulum off another and you get the simplest chaotic mechanical system: "
            "fully deterministic, yet two nearly identical releases flail into totally different "
            "motions within seconds. Its coupled equations of motion have no closed form and are "
            "integrated numerically (RK4 here). Two things stay clean: the total energy is "
            "conserved (a stringent check the integrator passes over a well-resolved window), "
            "and the sensitive dependence on initial conditions is real -- a hair's difference "
            "in the start grows exponentially, the butterfly effect on a tabletop.",
            '<div class="grid">'
            + svg_card(out("double_pendulum.svg"), "the lower bob's never-repeating trace, and two near-identical pendulums drifting apart")
            + f'<div class="card">{pre(double_pendulum_txt)}</div>'
            + '</div>'),
        section(
            "The Mandelbrot set: infinite detail from z -> z^2 + c",
            "Iterate z -> z^2 + c from z=0: the Mandelbrot set is the c for which the orbit "
            "stays bounded. Since |z|>2 guarantees escape, the escape time -- how many steps to "
            "cross that radius -- colours the famous images, painting the filaments just outside "
            "the set. One quadratic rule generates a boundary of endless detail: the cardioid "
            "body, the period-2 bulb at c=-1, ever-smaller bulbs around the edge, and tiny "
            "copies of the whole set at every magnification. Its real slice is the logistic "
            "map's period-doubling route in complex dress.",
            '<div class="grid">'
            + svg_card(out("mandelbrot.svg"), "the set coloured by escape time: black interior, bright fast-escape filaments")
            + f'<div class="card">{pre(mandelbrot_txt)}</div>'
            + '</div>'),
        section(
            "The Van der Pol oscillator: a self-sustaining rhythm",
            "Unlike a pendulum that dies out, the Van der Pol oscillator x'' - mu(1-x^2)x' + x = 0 "
            "pumps itself: its damping is negative at small amplitude (energy in) and positive "
            "at large (energy out), so from almost any start it settles onto the same closed "
            "loop -- a limit cycle of amplitude ~2 that forgets its initial conditions. It is the "
            "model for self-regulated rhythms: heartbeats, firing neurons, bowed strings. Small "
            "mu gives near-sinusoidal oscillation; large mu gives relaxation oscillation -- slow "
            "charges broken by fast jumps, period ~1.6 mu.",
            '<div class="grid">'
            + svg_card(out("van_der_pol.svg"), "two starts spiralling onto the same limit cycle, and the waveform from smooth to spiky")
            + f'<div class="card">{pre(van_der_pol_txt)}</div>'
            + '</div>'),
        section(
            "The Duffing oscillator: a spring that bends the rules",
            "Add a cubic term to a spring -- x'' + delta x' + alpha x + beta x^3 = gamma cos(omega t) "
            "-- and it stops behaving linearly. Its resonance peak bends over with amplitude "
            "(the backbone sqrt(alpha + 3/4 beta A^2)), so the response is multi-valued and jumps "
            "between branches as you sweep the drive frequency (hysteresis). With alpha<0, beta>0 "
            "the potential is a double well -- a buckled beam or a bistable switch -- and a damped "
            "mass rolls into one of two stable states. Driven hard, the forced Duffing is one of "
            "the classic routes to chaos.",
            '<div class="grid">'
            + svg_card(out("duffing.svg"), "the double-well potential with a mass settling into a well, and the leaning resonance backbone")
            + f'<div class="card">{pre(duffing_txt)}</div>'
            + '</div>'),
        section(
            "The Kuramoto model: oscillators falling into sync",
            "A population of oscillators, each with its own natural frequency, pulls itself into "
            "step through coupling: dtheta_i/dt = omega_i + (K/N) sum sin(theta_j - theta_i). "
            "The synchrony is measured by the order parameter r (0 = phases scattered, 1 = all "
            "in phase), and there is a sharp phase transition -- below a critical coupling K_c "
            "the oscillators drift independently (r~0), above it a synchronized cluster "
            "spontaneously forms and r climbs toward 1. It is the canonical model of emergent "
            "collective order: fireflies flashing in unison, pacemaker cells, applause locking "
            "into rhythm, generators on a grid.",
            '<div class="grid">'
            + svg_card(out("kuramoto.svg"), "the synchronization transition r(K), and phase circles scattered vs clustered")
            + f'<div class="card">{pre(kuramoto_txt)}</div>'
            + '</div>'),
        section(
            "The Abelian sandpile: self-organized criticality",
            "Drop grains one at a time; wherever a pile reaches 4 it topples, one grain to each "
            "neighbour, and a single grain can set off an avalanche of any size. With no "
            "parameter tuning the pile drives itself to a critical state where avalanche sizes "
            "follow a power law -- mostly tiny, rarely system-spanning. The toppling is Abelian "
            "(the final state is independent of relaxation order). It is the founding model of "
            "self-organized criticality, a candidate for the scale-free statistics of "
            "earthquakes, forest fires, and neuronal avalanches.",
            '<div class="grid">'
            + svg_card(out("sandpile.svg"), "a relaxed self-similar sandpile pattern, and the heavy-tailed avalanche-size distribution")
            + f'<div class="card">{pre(sandpile_txt)}</div>'
            + '</div>'),
        section(
            "Elementary cellular automata: complexity from 8 bits",
            "A row of 0/1 cells, each updated from itself and its two neighbours -- 2^8 = 256 "
            "possible rules, and from that trivial definition comes Wolfram's whole zoo: rule 0 "
            "dies to uniform, rule 90 draws the Sierpinski fractal by XOR, rule 30 is chaotic "
            "enough to have served as Mathematica's random-number generator, and rule 110 is "
            "Turing-complete -- a universal computer from an eight-bit lookup table. That "
            "computation needs almost no ingredients is one of the most surprising results in "
            "the field.",
            '<div class="grid">'
            + svg_card(out("cellular_automaton.svg"), "space-time diagrams of rules 90 (fractal), 30 (chaos), and 110 (complex)")
            + f'<div class="card">{pre(cellular_automaton_txt)}</div>'
            + '</div>'),
        section(
            "Conway's Game of Life: a universe from four rules",
            "On a 2D grid, a live cell survives with 2-3 live neighbours and a dead cell is born "
            "with exactly 3 (B3/S23) -- and that is the whole rule. From it come still lifes "
            "(the block, unchanging), oscillators (the blinker, period 2), and spaceships (the "
            "glider, translating one cell diagonally every four generations). Because gliders "
            "can be fired from guns and collided to build logic gates, Life is Turing-complete: "
            "a computer can be built inside it. The canonical proof that simple local rules "
            "generate open-ended complexity.",
            '<div class="grid">'
            + svg_card(out("game_of_life.svg"), "the glider's four phases and a mixed board of still life, oscillator, and spaceship")
            + f'<div class="card">{pre(game_of_life_txt)}</div>'
            + '</div>'),
        section(
            "Reaction-diffusion: Turing's spots and stripes",
            "Turing showed in 1952 that patterns can form from chemistry alone: a slowly "
            "diffusing self-promoting activator and a fast-diffusing inhibitor make a uniform "
            "mixture unstable, and it settles into standing spots or stripes with no template. "
            "The Gray-Scott model du/dt = Du lap(u) - u v^2 + F(1-u), dv/dt = Dv lap(v) + u v^2 "
            "- (F+k)v produces, depending on the feed F and kill k rates, spots, stripes, mazes, "
            "self-replicating blobs, or waves -- a working model of morphogenesis behind leopard "
            "spots and seashell ridges.",
            '<div class="grid">'
            + svg_card(out("reaction_diffusion.svg"), "the autocatalyst field growing from a seed into standing Turing spots")
            + f'<div class="card">{pre(reaction_diffusion_txt)}</div>'
            + '</div>'),
        section(
            "Boids: flocking from three local rules",
            "Reynolds showed that a flock needs no leader -- each boid just follows its "
            "neighbours by three rules: separation (avoid crowding), alignment (match heading), "
            "and cohesion (stay together). Sum those urges into an acceleration and a swarm of "
            "identical agents produces lifelike murmurations from a random scatter: the "
            "alignment (polarization) climbs toward 1 while separation keeps them from "
            "colliding, all bottom-up with no flock-level rule. The model behind starling "
            "murmurations, sardine bait balls, and the crowds in films and games.",
            '<div class="grid">'
            + svg_card(out("boids.svg"), "a random scatter organizing into aligned flocking, with polarization rising over time")
            + f'<div class="card">{pre(boids_txt)}</div>'
            + '</div>'),
        section(
            "Diffusion-limited aggregation: a fractal from random walkers",
            "Release a particle far from a seed and let it random-walk until it touches the "
            "cluster, where it sticks; repeat. What grows is not a blob but a feathery, "
            "self-similar fractal, because a wanderer almost always brushes an outer tip long "
            "before it can diffuse into an interior fjord -- the tips screen the inside and "
            "grow faster still. The cluster's mass scales as N(r) ~ r^D with D ~ 1.71 in the "
            "plane, not 2: the branches leave most of the plane empty. The same instability "
            "draws mineral dendrites, electrodeposits, viscous fingers in a Hele-Shaw cell, "
            "lightning, and soot.",
            '<div class="grid">'
            + svg_card(out("dla.svg"), "the branching cluster and the log-log mass-radius scaling whose slope is the fractal dimension")
            + f'<div class="card">{pre(dla_txt)}</div>'
            + '</div>'),
        section(
            "Benford's law: why leading digits are not uniform",
            "Count the leading digit of river areas, physical constants, stock prices, or file "
            "sizes and you do not get each of 1-9 a ninth of the time: 1 leads about 30% and 9 "
            "barely 4.6%, following P(d) = log10(1 + 1/d). The reason is scale invariance -- a "
            "quantity spanning many orders of magnitude is uniform in its logarithm, and a "
            "uniform log-mantissa maps to this logarithmic digit law, the only distribution "
            "invariant under a change of units. Multiplicative data (Fibonacci numbers, powers, "
            "factorials, populations) obey it, and departures flag fabricated accounting and "
            "election returns, which is why forensic auditors test for it.",
            '<div class="grid">'
            + svg_card(out("benford.svg"), "the Benford curve with the Fibonacci leading digits hugging it while a uniform control does not")
            + f'<div class="card">{pre(benford_txt)}</div>'
            + '</div>'),
        section(
            "The coupon collector: how long to collect the whole set",
            "Each cereal box holds one of n equally likely coupons; how many boxes to collect "
            "all n? Once you hold k of them a fresh box is new with probability (n-k)/n, so the "
            "wait for the next new one is geometric with mean n/(n-k), and summing gives "
            "E[T] = n H_n ~ n ln n. The last few coupons dominate: collecting the final one "
            "alone averages n boxes. The number needed is sharply concentrated, with a tail "
            "bound P(T > n ln n + c n) <= e^{-c}. The same law sets cache warmup, random "
            "test-coverage of n branches, and how many samples it takes to see every category "
            "at least once -- here the analytic E[T], variance, and completion CDF are checked "
            "against a seeded Monte-Carlo run.",
            '<div class="grid">'
            + svg_card(out("coupon_collector.svg"), "the collection-progress curve (the last coupons cost the most) and the completion-probability CDF")
            + f'<div class="card">{pre(coupon_txt)}</div>'
            + '</div>'),
        section(
            "The secretary problem: optimal stopping and the 1/e rule",
            "Interview n candidates one at a time in random order, accept or reject on the spot, "
            "and you only care about landing the single best. The optimal policy is a cutoff: "
            "reject the first r-1 (a look phase), then take the first later candidate who beats "
            "all seen so far. The win probability P(r) = (r-1)/n sum_{i=r}^{n} 1/(i-1) is "
            "maximized near r ~ n/e, and as n grows both the optimal look-fraction and the win "
            "probability tend to 1/e ~ 0.368: look at 37% of the field, then leap at the next "
            "record, and you land the very best about 37% of the time no matter how large n is. "
            "The same optimal-stopping law governs flat-hunting, parking, and online auctions -- "
            "here the exact probabilities are checked against a seeded Monte-Carlo run.",
            '<div class="grid">'
            + svg_card(out("secretary.svg"), "the win probability peaking near the 1/e look-fraction, and the optimum converging to 1/e as n grows")
            + f'<div class="card">{pre(secretary_txt)}</div>'
            + '</div>'),
        section(
            "The birthday problem: coincidences are more common than they feel",
            "How many people before two share a birthday with better-than-even odds? Only 23, "
            "not hundreds, because k people make k(k-1)/2 pairs and it is the pair count, "
            "growing like k^2, that drives collisions. The probability all k are distinct is "
            "prod (365-i)/365, so a collision passes 1/2 at 23 and 99.9% by 70. In general a "
            "collision becomes likely once k ~ 1.177 sqrt(d), a square-root law that sizes hash "
            "tables and UUID spaces and sets the birthday attack: a b-bit hash collides after "
            "~2^(b/2) tries, not 2^b, which is why collision resistance needs twice the bits of "
            "preimage resistance. Exact and Poisson-approximate probabilities are checked "
            "against a seeded Monte-Carlo run.",
            '<div class="grid">'
            + svg_card(out("birthday.svg"), "the collision-probability curve crossing 50% at 23 people, and the crossover growing like sqrt(days)")
            + f'<div class="card">{pre(birthday_txt)}</div>'
            + '</div>'),
        section(
            "Gambler's ruin: the walk that ends at a wall",
            "Start with i dollars, bet $1 a round with win probability p, and stop at broke (0) "
            "or a target N -- a random walk with two absorbing walls. In a fair game the ruin "
            "chance is exactly 1 - i/N (your stake as a fraction of the table) and the game "
            "lasts i(N-i) rounds. But shift the odds a hair to p=0.49 and, starting at the "
            "halfway mark, the ruin chance leaps from 50% to 88%; against an infinitely rich "
            "house any p <= 1/2 is ruin with certainty. That asymmetry is why the house always "
            "wins, and the same absorbing-walk math models allele fixation in a finite "
            "population and sequential hypothesis tests. Exact ruin probabilities and durations "
            "are checked against a seeded Monte-Carlo run.",
            '<div class="grid">'
            + svg_card(out("gamblers_ruin.svg"), "ruin probability vs starting stake for several win rates, and sample walks absorbed at a wall")
            + f'<div class="card">{pre(gamblers_ruin_txt)}</div>'
            + '</div>'),
        section(
            "Parrondo's paradox: two losing games that together win",
            "Two gambling games, each a sure loser played alone, can be alternated -- or chosen "
            "at random each round -- to make your capital drift UP. Game A is a slightly-losing "
            "flat coin; game B flips a terrible coin whenever your capital is a multiple of 3 "
            "and a good one otherwise, and loses because the walk gets stuck in the bad state "
            "too often. Mixing in game A reshuffles that state occupancy so the good coin comes "
            "up more, and the combined drift -- the stationary average of (2p-1) over the "
            "capital-mod-3 Markov chain -- turns positive. The same flashing-ratchet mechanism "
            "drives molecular motors, pumping directed motion from noise. Here the exact "
            "stationary drift is checked against a seeded Monte-Carlo trajectory.",
            '<div class="grid">'
            + svg_card(out("parrondo.svg"), "capital rising for the mixture while both games fall, and the winning window in the mixing fraction")
            + f'<div class="card">{pre(parrondo_txt)}</div>'
            + '</div>'),
        section(
            "The Galton board: coin flips converge to a bell curve",
            "Galton's bean machine is a board of n staggered peg rows; a bead bounces left or "
            "right with probability 1/2 at each row and lands in one of n+1 slots. Its slot is "
            "just the number of right-bounces in n coin flips, so the slot occupancy is the "
            "binomial C(n,k) p^k (1-p)^(n-k) -- and because it is a sum of n independent steps, "
            "the central limit theorem makes the histogram converge to a Gaussian of mean np "
            "and variance np(1-p) as n grows. It is the CLT made physical: no bead is steered, "
            "yet thousands pile into a smooth bell curve, and biasing the pegs slides the peak "
            "to np. The binomial-to-Gaussian distance shrinks like 1/sqrt(n), verified here "
            "against exact values and a seeded Monte-Carlo bead drop.",
            '<div class="grid">'
            + svg_card(out("galton.svg"), "the simulated slot histogram matching the CLT Gaussian, and the binomial-to-Gaussian distance falling like 1/sqrt(rows)")
            + f'<div class="card">{pre(galton_txt)}</div>'
            + '</div>'),
        section(
            "The Monty Hall problem: why switching doors wins",
            "A car hides behind one of three doors; you pick one, and the host -- who knows "
            "where the car is -- opens a different door revealing a goat and offers a switch. "
            "Switching wins 2/3 of the time, staying only 1/3, because your first pick is right "
            "just 1/3 of the time and the host's forced reveal concentrates the whole remaining "
            "2/3 onto the other closed door. The paradox lives in the host's knowledge: if he "
            "opened a door blindly and it happened to show a goat, switching would only be 1/2. "
            "Generalized to N doors with the host opening all but one other, switching wins "
            "(N-1)/N -- 99% at 100 doors. The exact stay and switch probabilities are checked "
            "against a seeded Monte-Carlo play.",
            '<div class="grid">'
            + svg_card(out("monty_hall.svg"), "the classic 2/3-vs-1/3 win rates (and the 1/2 blind-host variant), and switching approaching certainty as doors grow")
            + f'<div class="card">{pre(monty_hall_txt)}</div>'
            + '</div>'),
        section(
            "Bayes and the base-rate fallacy: a positive test can still mean healthy",
            "A disease affects 1 in 1000; a test is 99% sensitive and 99% specific; you test "
            "positive. The chance you are actually sick is not 99% but about 9%. Bayes' theorem "
            "combines the prior with the test's likelihoods, and the rare base rate makes false "
            "positives swamp the true ones: among 100,000 people, 99 true positives are buried "
            "under 999 false ones. A positive becomes more-likely-than-not only once the "
            "prevalence passes (1-spec)/(sens+1-spec) = 1%, and two independent positives push "
            "the posterior above 90%. This is the base-rate fallacy behind medical screening, "
            "spam filters, and security profiling -- here the posterior, likelihood ratios, and "
            "retest odds are checked against a seeded Monte-Carlo cohort.",
            '<div class="grid">'
            + svg_card(out("bayes_test.svg"), "the positive predictive value rising with prevalence (50-50 only at 1%), and a cohort where 91% of positives are false")
            + f'<div class="card">{pre(bayes_test_txt)}</div>'
            + '</div>'),
        section(
            "Shannon entropy and Huffman coding: the limit of lossless compression",
            "How few bits, on average, to record one symbol from a source? Shannon's entropy "
            "H = -sum p log2 p is the answer, and no lossless code can beat it: the average "
            "codeword length L obeys L >= H, with a code always existing at L < H+1. Entropy is "
            "maximal (log2 n) for a uniform source and zero when one symbol is certain -- it "
            "measures surprise. Huffman's algorithm merges the two least-likely symbols "
            "repeatedly to build the optimal prefix code, provably landing in the [H, H+1) band "
            "and obeying the Kraft inequality sum 2^-len <= 1. This module computes entropy, "
            "builds the Huffman code, verifies the Shannon bound and a lossless encode/decode "
            "round-trip -- the Huffman stage inside ZIP, JPEG, and MP3.",
            '<div class="grid">'
            + svg_card(out("shannon.svg"), "the binary-entropy curve peaking at a fair coin, and Huffman codeword lengths hugging the entropy limit")
            + f'<div class="card">{pre(shannon_txt)}</div>'
            + '</div>'),
        section(
            "The Kelly criterion: how much to bet to grow fastest",
            "Given an edge -- a bet paying b-to-1 that wins with probability p above break-even "
            "-- how much of your bankroll should you stake? Kelly's answer maximizes long-run "
            "compound growth and is a fixed fraction f* = p - (1-p)/b, the edge over the odds. "
            "The growth rate g(f) = p ln(1+bf) + (1-p) ln(1-f) is a concave curve peaking at "
            "f*: betting less is safe but slow, betting past 2f* drives the growth rate "
            "negative and you go broke despite a winning edge. Half-Kelly keeps about 3/4 of "
            "the growth at far less volatility, which is why traders bet fractional Kelly. The "
            "same log-optimal rule (which Kelly derived from Shannon's channel capacity) sizes "
            "positions in quantitative finance -- here it is checked against a seeded "
            "Monte-Carlo of the compounding bankroll.",
            '<div class="grid">'
            + svg_card(out("kelly.svg"), "the growth-rate curve peaking at f* and going negative past break-even, with sample bankrolls under-, Kelly-, and over-betting")
            + f'<div class="card">{pre(kelly_txt)}</div>'
            + '</div>'),
        section(
            "Hamming codes: correcting a bit error from the syndrome",
            "Bits flip on a noisy channel. A parity bit can detect a single error but not fix "
            "it; Hamming's 1950 codes correct it. Placing parity bits at the power-of-two "
            "positions so each data bit is covered by a unique combination of checks, the "
            "pattern of failed checks -- the syndrome -- reads out, in binary, the exact "
            "position of the flipped bit. The classic Hamming(7,4) carries 4 data bits in 7 and "
            "corrects any single error; Hamming(2^m-1, 2^m-1-m) needs only m parity bits, so the "
            "overhead shrinks as blocks grow. Every Hamming code has minimum distance 3, and "
            "one extra overall parity bit gives SECDED (single-correct, double-detect), the "
            "scheme in ECC memory. Verified by exhaustively correcting every single-bit error in "
            "every codeword.",
            '<div class="grid">'
            + svg_card(out("hamming.svg"), "each parity bit's coverage that makes the syndrome name the error position, and the code rate rising toward 1")
            + f'<div class="card">{pre(hamming_txt)}</div>'
            + '</div>'),
        section(
            "RSA: public-key cryptography from the hardness of factoring",
            "RSA lets two strangers communicate secretly without ever sharing a key, resting on "
            "one asymmetry: multiplying two large primes into n = pq is easy, but factoring n "
            "back apart is (as far as anyone knows) astronomically hard. Pick primes p, q, take "
            "phi = (p-1)(q-1), a public exponent e coprime to phi, and the private d = e^-1 mod "
            "phi. Encryption is c = m^e mod n, decryption m = c^d mod n, and they undo each "
            "other because ed = 1 mod phi (Euler). The same keys sign: encrypt with the private "
            "key, verify with the public. This pure-stdlib reference implements Miller-Rabin "
            "primality (catching Carmichael numbers that fool Fermat), the extended Euclidean "
            "inverse, fast square-and-multiply exponentiation, and full encrypt/decrypt/sign/"
            "verify round-trips -- the number theory behind TLS and SSH.",
            '<div class="grid">'
            + svg_card(out("rsa.svg"), "the one-way public-key flow an eavesdropper cannot invert, and the log-time cost of modular exponentiation")
            + f'<div class="card">{pre(rsa_txt)}</div>'
            + '</div>'),
        section(
            "Diffie-Hellman: agreeing on a secret in the open",
            "Two people who have never met, on a wiretapped line, can agree on a shared secret "
            "the eavesdropper cannot learn. Fix a prime p and generator g; Alice sends g^a mod "
            "p, Bob sends g^b mod p, and each raises what they received to their own secret, "
            "both landing on g^(ab) mod p while the wire carried only g^a and g^b. Stealing the "
            "secret means recovering a from g^a mod p -- the discrete-logarithm problem, easy "
            "to state and (for large p) astronomically hard. This module generates safe-prime "
            "parameters, finds a generator, runs the exchange, and includes a baby-step/"
            "giant-step discrete-log solver whose sqrt(p) cost dwarfs the parties' log(p) work "
            "-- the gap that keeps the secret safe. Without authentication a man-in-the-middle "
            "can still intercept, which is why real protocols sign the exchange.",
            '<div class="grid">'
            + svg_card(out("diffie_hellman.svg"), "the exchange both sides complete to the same secret, and the attacker's sqrt(p) cost against the honest log(p)")
            + f'<div class="card">{pre(diffie_hellman_txt)}</div>'
            + '</div>'),
        section(
            "CRC: catching transmission errors with polynomial division",
            "A cyclic redundancy check is the checksum on almost every digital frame -- "
            "Ethernet packets, ZIP files, PNG chunks, disk sectors. Treat the message as a "
            "polynomial over GF(2) (arithmetic mod 2, addition = XOR), divide by a fixed "
            "generator, and append the remainder; the receiver divides again and a nonzero "
            "remainder flags corruption. A degree-r generator guarantees detection of every "
            "single-bit error, every burst shorter than r+1 bits, and misses a random error "
            "only with probability ~2^-r -- 1 in 4 billion for CRC-32, computed with nothing "
            "but shifts and XORs. This module does bit-at-a-time polynomial division for "
            "CRC-8/16/32, reproducing the published '123456789' check values and matching "
            "zlib.crc32 exactly. The detection companion to the Hamming code (which corrects).",
            '<div class="grid">'
            + svg_card(out("crc.svg"), "the data-plus-CRC frame layout and the miss probability 2^-r shrinking with the number of check bits")
            + f'<div class="card">{pre(crc_txt)}</div>'
            + '</div>'),
        section(
            "LZ77: compression by pointing back at what you have seen",
            "Lempel and Ziv's 1977 algorithm is the engine inside ZIP, gzip, and PNG. Scanning "
            "the data, whenever the next bytes have appeared recently it emits a back-reference "
            "-- a (distance, length) pair meaning 'copy length bytes from distance back' -- "
            "instead of repeating them; only genuinely new bytes are stored literally. A "
            "sliding window holds the recent history, and the decompressor replays the tokens "
            "from its own growing output, so an overlapping copy expands a whole run from one "
            "token. Repetitive data (text, code, logs) compresses enormously while random data "
            "cannot shrink at all -- Shannon's entropy limit showing through. This module "
            "implements the encoder and decoder with a guaranteed lossless round-trip; LZ77 "
            "plus Huffman together are DEFLATE.",
            '<div class="grid">'
            + svg_card(out("lz77.svg"), "the token stream of literals and back-references, and the compression ratio climbing with repetition")
            + f'<div class="card">{pre(lz77_txt)}</div>'
            + '</div>'),
        section(
            "Bloom filters: membership in a fraction of the space",
            "A Bloom filter answers 'have I seen this?' with a bit array and a few hash "
            "functions, in a tiny fraction of the memory the items would take. The trade is "
            "one-sided: it may say 'possibly present' for something never added (a false "
            "positive) but NEVER says 'absent' for something you did add -- no false negatives. "
            "Add an item by setting its k bits; test by checking all k are set. After n items "
            "in m bits the false-positive rate is (1 - e^{-kn/m})^k, minimized at k = (m/n) ln "
            "2, needing only ~1.44 log2(1/p) bits per item regardless of item size -- a million "
            "URLs at 1% error in about 1.2 MB. Web caches, spell checkers, and databases use "
            "one as a fast pre-filter. Verified here: zero false negatives and an observed "
            "false-positive rate matching theory.",
            '<div class="grid">'
            + svg_card(out("bloom.svg"), "the false-positive rate rising as the filter fills (observed tracking theory) and the optimal number of hash functions")
            + f'<div class="card">{pre(bloom_txt)}</div>'
            + '</div>'),
        section(
            "HyperLogLog: counting distinct items in kilobytes",
            "How many DISTINCT items in a stream, when exact counting means storing every one? "
            "HyperLogLog estimates the cardinality to a percent or two in fixed tiny memory -- a "
            "billion distinct items in ~1.5 KB. Hash each item; the longest run of leading zeros "
            "seen hints at the count (k zeros suggests ~2^k items). To tame the noise, the first "
            "p bits pick one of m = 2^p registers each holding its max leading-zero rank, and "
            "the harmonic mean across registers gives E = alpha_m m^2 / sum 2^-M[j] with "
            "relative error ~1.04/sqrt(m). Small counts get a linear-counting correction, and "
            "two sketches merge by register-wise max, so counts are trivially distributed -- "
            "which is why Redis, Presto, and BigQuery all ship it. Verified against true "
            "cardinalities across four orders of magnitude.",
            '<div class="grid">'
            + svg_card(out("hyperloglog.svg"), "the estimate hugging the exact diagonal over four orders of magnitude, and the relative error staying within the standard-error band")
            + f'<div class="card">{pre(hyperloglog_txt)}</div>'
            + '</div>'),
        section(
            "Fenwick trees: running sums that update in log time",
            "Keeping an array while asking for prefix sums, a plain array gives instant updates "
            "but O(n) sums, and a prefix-sum array the reverse. Fenwick's binary indexed tree "
            "does both in O(log n) using the binary structure of the indices: node i stores the "
            "partial sum of the range ending at i whose length is its lowest set bit i & -i. A "
            "prefix sum strips the low bit each step (i -= i & -i); an update adds it "
            "(i += i & -i) -- each walk touches only one node per set bit. Range sums come by "
            "subtraction, and because cumulative sums are monotone you can binary-search the "
            "tree for the smallest index whose prefix reaches a target, an O(log n) 'select' "
            "for weighted sampling and rank queries. Every operation is cross-checked here "
            "against a brute-force array over thousands of mixed updates and queries.",
            '<div class="grid">'
            + svg_card(out("fenwick.svg"), "each node's low-bit-sized coverage range over the array, and the log n vs naive n cost per query")
            + f'<div class="card">{pre(fenwick_txt)}</div>'
            + '</div>'),
        section(
            "Union-Find: connectivity in near-constant time, and Kruskal's MST",
            "Given a stream of 'these two are connected' facts, answer whether any two items are "
            "in the same group -- the disjoint-set problem. Union-Find solves a sequence of m "
            "operations in O(m alpha(n)) time, where the inverse Ackermann alpha(n) is at most 4 "
            "for any conceivable n: effectively constant. Each set is a tree with a "
            "representative root; union by rank hangs the shorter tree under the taller, and "
            "path compression repoints every node visited during a find straight at the root, so "
            "trees stay flat. A cycle in a graph is exactly two endpoints already in the same "
            "set, which makes Union-Find the whole of Kruskal's minimum-spanning-tree algorithm. "
            "It also drives image segmentation, percolation, and account-merging. Verified "
            "against a brute-force flood fill and known minimum spanning trees.",
            '<div class="grid">'
            + svg_card(out("union_find.svg"), "the MST edges chosen on a weighted graph, and the component count falling as edges are merged")
            + f'<div class="card">{pre(union_find_txt)}</div>'
            + '</div>'),
        section(
            "Dijkstra's algorithm: shortest paths, greedily",
            "Given a graph with nonnegative edge weights, Dijkstra finds the cheapest route from "
            "a source to every node by a greedy rule: keep a tentative distance to each node, "
            "repeatedly settle the unsettled node with the smallest distance, and relax its "
            "edges. Once settled, a node's distance is final -- which holds precisely because "
            "no nonnegative detour can improve a shorter path. With a binary-heap priority "
            "queue it runs in O((V+E) log V). This module builds a weighted graph, runs "
            "Dijkstra (returning distances and a predecessor tree for path reconstruction), and "
            "includes a from-scratch binary min-heap and a Bellman-Ford implementation used to "
            "verify every distance. Add a goal heuristic to the priority and it becomes A*, the "
            "workhorse of map and game routing -- shown here solving a grid maze.",
            '<div class="grid">'
            + svg_card(out("dijkstra.svg"), "the distance flood from the source colouring the grid, with the shortest S-to-G route picked out")
            + f'<div class="card">{pre(dijkstra_txt)}</div>'
            + '</div>'),
        section(
            "k-d trees: fast nearest-neighbour search in space",
            "\"Which point is closest to this query?\" is asked constantly in graphics, "
            "robotics, machine learning, and geographic search, and scanning every point is O(n). "
            "A k-d tree organizes the points by recursive median splitting -- the root splits on "
            "x, the next level on y, then z, cycling axes -- so a query descends to its leaf and "
            "then unwinds, only crossing a splitting plane into the far subtree when that "
            "hyper-rectangle could hold something nearer. Whole branches are pruned, giving "
            "O(log n) typical queries. The same descent-and-prune serves k-nearest-neighbours "
            "and radius queries. This module builds a balanced tree and does exact nearest, "
            "k-nearest, and radius search, each cross-checked against a brute-force scan in 2D "
            "and 3D. It powers k-NN classification, particle neighbour lists, and map search.",
            '<div class="grid">'
            + svg_card(out("kdtree.svg"), "a point cloud with the query, its nearest neighbour, its five nearest, and a radius query circle")
            + f'<div class="card">{pre(kdtree_txt)}</div>'
            + '</div>'),
        section(
            "Boyer-Moore: the string search that skips ahead",
            "Finding a pattern in text is one of computing's most-run operations. The naive scan "
            "checks every position in O(n*m); Boyer-Moore -- the algorithm your editor's find "
            "actually uses -- matches the pattern right-to-left and, on a mismatch, jumps forward "
            "by more than one position, often by nearly the whole pattern. Two precomputed rules "
            "set the skip: the bad-character rule shifts so the last occurrence of the mismatched "
            "text character lines up (or past it entirely if absent), and the good-suffix rule "
            "realigns an already-matched suffix without undoing confirmed matches. Taking the "
            "larger shift keeps it safe and makes it sublinear on real text -- most characters "
            "are never examined. This module builds both tables and finds first/all/overlapping "
            "occurrences, proven correct exhaustively against a naive search over 5000 random "
            "cases.",
            '<div class="grid">'
            + svg_card(out("boyer_moore.svg"), "character comparisons vs pattern length: the naive scan rises while Boyer-Moore falls")
            + f'<div class="card">{pre(boyer_moore_txt)}</div>'
            + '</div>'),
        section(
            "A* search: Dijkstra with a sense of direction",
            "Dijkstra explores outward in every direction equally; A* keeps the same guarantee of "
            "an optimal path but adds a heuristic h(n) estimating the remaining distance to the "
            "goal, ordering its frontier by f(n) = g(n) + h(n) -- known cost so far plus the "
            "guess ahead -- so it pushes toward the goal instead of flooding. If the heuristic "
            "never overestimates the true remaining cost (it is admissible), the path is still "
            "guaranteed optimal; with h = 0 it degenerates exactly to Dijkstra. On a grid the "
            "Manhattan distance is admissible for 4-directional movement and the octile distance "
            "for 8-directional. This module runs A* on a weighted grid with obstacles, returns "
            "the path and the expanded-node set, and verifies A* finds the same optimal cost as "
            "Dijkstra while expanding no more nodes -- the standard for game and robot "
            "navigation.",
            '<div class="grid">'
            + svg_card(out("astar.svg"), "A* and Dijkstra side by side on the same maze: identical path, but A* explores far fewer cells")
            + f'<div class="card">{pre(astar_txt)}</div>'
            + '</div>'),
        section(
            "Topological sort: ordering tasks so prerequisites come first",
            "A directed acyclic graph encodes dependencies -- an edge u -> v means u must come "
            "before v -- and a topological order is a linear arrangement in which every edge "
            "points forward: the order you can build the modules, install the packages, or "
            "recompute the spreadsheet cells. It exists exactly when the graph has no cycle. "
            "Kahn's algorithm repeatedly emits a node with no remaining incoming edges (BFS on "
            "in-degrees); the DFS method reverses finish-times and spots a cycle as a back-edge "
            "to a node still on the stack. Both run in O(V+E). With a duration on each task, the "
            "longest path through the DAG is the critical path -- the minimum time to finish "
            "everything. This module implements both sorts, cycle detection, and the critical "
            "path, and verifies every order it returns respects all edges across hundreds of "
            "random DAGs.",
            '<div class="grid">'
            + svg_card(out("toposort.svg"), "a dependency DAG laid out in topological layers with the critical (longest-duration) path highlighted")
            + f'<div class="card">{pre(toposort_txt)}</div>'
            + '</div>'),
        section(
            "Levenshtein edit distance: how far apart are two strings",
            "The edit distance is the fewest single-character edits -- insert, delete, "
            "substitute -- that turn one string into another, the measure behind spell-checkers, "
            "fuzzy search, diff, and DNA alignment. Dynamic programming fills a table where "
            "d[i][j] is the distance between prefixes, each cell the cheapest of a match, "
            "substitution, insertion, or deletion, in O(mn); backtracing the choices recovers "
            "the actual alignment, not just the count. The distance is a true metric (symmetric, "
            "zero only for equal strings, triangle-inequality-respecting). This module computes "
            "the distance, a memory-lean two-row variant, the alignment operations, a similarity "
            "ratio, and the Damerau variant that treats an adjacent-character swap as one edit "
            "(the commonest typo), all verified against known values and the metric axioms.",
            '<div class="grid">'
            + svg_card(out("levenshtein.svg"), "the dynamic-programming cost table with the backtrace path that spells out the minimal edits")
            + f'<div class="card">{pre(levenshtein_txt)}</div>'
            + '</div>'),
        section(
            "The 0/1 knapsack: packing the most value under a weight limit",
            "Items each have a weight and a value; which subset maximizes value without exceeding "
            "a capacity W, taking each item whole or not at all? Brute force checks 2^n subsets, "
            "but dynamic programming solves it in pseudo-polynomial O(nW): best[i][w] is the most "
            "value from the first i items within capacity w, each cell the better of skipping "
            "item i or taking it (freeing w - weight_i of room). Backtracing recovers which items "
            "to take; a rolling array (iterating capacity downward so each item is used once) "
            "cuts memory to O(W). The related subset-sum question and the unbounded knapsack "
            "(unlimited copies, iterate capacity upward) are the same table. This module solves "
            "all four, reconstructs the chosen items, and is verified exhaustively against a "
            "brute-force subset search -- the model for budget allocation, cargo loading, and "
            "portfolio selection under a hard cap.",
            '<div class="grid">'
            + svg_card(out("knapsack.svg"), "the DP value table filling row by row to the optimum, and the optimal value climbing with capacity")
            + f'<div class="card">{pre(knapsack_txt)}</div>'
            + '</div>'),
        section(
            "Longest common subsequence: the engine behind diff",
            "A subsequence keeps some elements in order but may skip others; the longest common "
            "subsequence of two sequences is the longest ordering appearing in both, and it is "
            "what diff, git, patch, and bioinformatics comparison are built on -- the complement "
            "of the LCS is exactly the lines to add or delete, so a bigger LCS means a smaller "
            "diff. The dynamic program fills L[i][j] = L[i-1][j-1]+1 on a match, else "
            "max(L[i-1][j], L[i][j-1]), in O(mn); backtracing recovers an actual longest "
            "subsequence, and turning the walk into keep/delete/insert steps yields the diff. "
            "The LCS length also gives the insert/delete edit distance, m + n - 2*LCS. This "
            "module computes the length, one subsequence, the diff edit-script, and that "
            "distance, all verified exhaustively against a brute-force subsequence search.",
            '<div class="grid">'
            + svg_card(out("lcs.svg"), "the DP length table with the match-cell diagonal that spells out the longest common subsequence")
            + f'<div class="card">{pre(lcs_txt)}</div>'
            + '</div>'),
        section(
            "Quickselect: the k-th smallest without sorting",
            "Finding the median, a percentile, or the top-k threshold looks like it needs a full "
            "O(n log n) sort -- it does not. Quickselect partitions the array around a pivot and "
            "recurses only into the side containing rank k, discarding half the data each step "
            "for O(n) expected time; you never touch the elements you do not need. A bad pivot "
            "would degrade it to O(n^2), so the median-of-medians algorithm (medians of groups "
            "of five, recursively) picks a pivot provably better than 30% and worse than 30% of "
            "the data, guaranteeing worst-case linear time -- the classic proof that selection "
            "beats sorting. This module implements quickselect with a randomized pivot and with "
            "median-of-medians, plus median, k-th smallest/largest, and percentile wrappers, "
            "each verified against a full sort over 1000 random arrays.",
            '<div class="grid">'
            + svg_card(out("quickselect.svg"), "comparisons to find the median: quickselect grows linearly while a full sort grows as n log n")
            + f'<div class="card">{pre(quickselect_txt)}</div>'
            + '</div>'),
        section(
            "Aho-Corasick: finding many patterns in one pass",
            "A spam filter or virus scanner must match hundreds of patterns at once; running a "
            "single-pattern search per pattern costs O(n * patterns). The Aho-Corasick automaton "
            "finds every occurrence of every pattern in a single left-to-right scan, in "
            "O(n + total pattern length + matches) -- independent of the pattern count. It builds "
            "a trie of the patterns, then adds failure links (fall back to the longest proper "
            "suffix that is also a pattern prefix, so no character is re-examined -- KMP "
            "generalized to many patterns) and output links (report every pattern ending at the "
            "current state, catching overlapping and nested matches like 'he', 'she', 'hers' in "
            "'ushers'). Built once by breadth-first traversal, then one text pass reports all "
            "matches. This module builds the automaton and finds all matches, verified "
            "exhaustively against a brute-force per-pattern search.",
            '<div class="grid">'
            + svg_card(out("aho_corasick.svg"), "the pattern trie with its failure links (dashed) and the nodes where a pattern ends")
            + f'<div class="card">{pre(aho_corasick_txt)}</div>'
            + '</div>'),
        section(
            "Floyd-Warshall: shortest paths between every pair",
            "Dijkstra gives shortest paths from one source; for a routing table or a road-network "
            "distance matrix you need them between ALL pairs. Floyd-Warshall does it in one "
            "elegant O(V^3) dynamic program that also handles negative edge weights (which "
            "Dijkstra cannot) and detects negative cycles. It allows ever-larger sets of "
            "intermediate nodes: dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]) as k runs "
            "over every node -- three nested loops, no priority queue. Recording the next hop per "
            "pair reconstructs the routes; a negative value on the diagonal means a negative "
            "cycle; and swapping min for boolean OR gives the transitive closure (who can reach "
            "whom). This module computes the distance matrix, paths, cycle detection, and "
            "closure, verified against running Dijkstra from every source on 300 random graphs.",
            '<div class="grid">'
            + svg_card(out("floyd_warshall.svg"), "the all-pairs distance matrix as a heatmap, nearer pairs darker and unreachable pairs gray")
            + f'<div class="card">{pre(floyd_warshall_txt)}</div>'
            + '</div>'),
        section(
            "Misra-Gries: frequent items of a stream in tiny memory",
            "Which items appear more than n/k times in a stream, when a counter per distinct "
            "value is impossible for billions of distinct items? The Misra-Gries summary finds "
            "every such heavy hitter with only k-1 counters in a single pass. Its rule is a "
            "generalized vote: increment a tracked item, start tracking a new one if a slot is "
            "free, else decrement every counter (the incoming item cancels one of each). Any "
            "item over n/k is guaranteed to survive (no false negatives), and a cheap second "
            "pass counts the survivors exactly to drop the false positives; the carried counts "
            "underestimate by at most n/k. The special case k=2 is the Boyer-Moore majority "
            "vote -- the strict-majority element with one counter. This module builds the "
            "summary, verifies candidates, and does majority vote, all checked exhaustively "
            "against exact counting.",
            '<div class="grid">'
            + svg_card(out("misra_gries.svg"), "the summary's underestimated counts against the true counts, and its fixed memory versus exact counting")
            + f'<div class="card">{pre(misra_gries_txt)}</div>'
            + '</div>'),
        section(
            "Reservoir sampling: a uniform sample from an endless stream",
            "Keep a uniform random sample of k items from a stream whose length you do not know "
            "and cannot store -- log lines, sensor readings, a file too big for memory. "
            "Vitter's Algorithm R does it in one pass with O(k) memory: fill the reservoir with "
            "the first k items, then keep the i-th item with probability k/i, evicting a random "
            "existing one. A short induction shows every item ever seen is in the final sample "
            "with probability exactly k/n, whatever n turns out to be (k=1 is the classic "
            "'random line from a file'). The weighted Efraimidis-Spirakis variant gives each "
            "item a key u^(1/w) and keeps the largest, sampling in proportion to weight. This "
            "module implements both plus a streaming reservoir object, and verifies the "
            "uniformity with a chi-square test over tens of thousands of runs.",
            '<div class="grid">'
            + svg_card(out("reservoir.svg"), "the flat selection-frequency histogram proving uniformity, and weighted sampling tracking the weights")
            + f'<div class="card">{pre(reservoir_txt)}</div>'
            + '</div>'),
        section(
            "Count-Min sketch: frequency estimates in sublinear memory",
            "How often has each item appeared, when a counter per distinct key is impossible? "
            "The Count-Min sketch estimates every item's count from a fixed d x w grid of "
            "counters with d hash functions. Add an item by incrementing one counter per row; "
            "query it by taking the MINIMUM of its d counters -- since collisions only inflate a "
            "counter, the smallest is the tightest overestimate and the true count is never "
            "above it. With width e/epsilon and depth ln(1/delta), the estimate exceeds the "
            "truth by more than epsilon * total with probability at most delta, so a few "
            "kilobytes track a stream of any size. Counts combine additively, so two sketches "
            "merge by element-wise addition -- counting is distributed. This module builds the "
            "sketch, queries and merges, and verifies it never underestimates and stays within "
            "the error bound across many skewed streams.",
            '<div class="grid">'
            + svg_card(out("count_min.svg"), "estimate-vs-true points all on or above the diagonal (never under), and the max error shrinking as the table widens")
            + f'<div class="card">{pre(count_min_txt)}</div>'
            + '</div>'),
        section(
            "The alias method: O(1) sampling from a weighted die",
            "To draw outcome i with probability p_i, the obvious way builds a cumulative "
            "distribution and binary-searches a random number at O(log n) per draw. Walker's "
            "alias method does it in O(1) per draw after O(n) setup. It chops the distribution "
            "into n equal-area columns, each holding at most two outcomes -- a main and an alias "
            "-- by repeatedly pairing an under-full outcome with an over-full one until every "
            "column has area 1. A draw is one integer roll (pick a column) plus one float flip "
            "(main outcome or its alias): two operations, no search, however many outcomes "
            "there are. The result is exact -- long-run frequencies equal the weights. This "
            "module builds the table by Vose's algorithm and samples from it, verified against "
            "the target weights with chi-square tests. The standard for loot tables, particle "
            "spawning, and any hot loop sampling one categorical distribution.",
            '<div class="grid">'
            + svg_card(out("alias_method.svg"), "the sampled frequencies matching the target weights, and the equal-area alias columns each split main/alias")
            + f'<div class="card">{pre(alias_method_txt)}</div>'
            + '</div>'),
        section(
            "Fisher-Yates: the only correct way to shuffle",
            "Shuffling looks trivial and almost everyone gets it wrong: the naive 'swap each "
            "position with a random position anywhere' makes n^n equally likely swap sequences "
            "but only n! permutations, and since n^n is not divisible by n! some orderings come "
            "up more often. Fisher-Yates fixes it by shrinking the range -- to place position i, "
            "swap it with a random position in [i, n), only the unshuffled tail -- so each of "
            "the n! permutations results from exactly one choice sequence and every ordering is "
            "equally likely. The same sweep gives a partial-shuffle k-sample (uniform without "
            "replacement), and restricting swaps to strictly earlier positions (Sattolo) yields "
            "a uniform random single cycle. This module implements all of these and demonstrates "
            "the bias by enumerating every permutation: Fisher-Yates is flat, the naive shuffle "
            "measurably lumpy.",
            '<div class="grid">'
            + svg_card(out("fisher_yates.svg"), "permutation frequencies: Fisher-Yates hugs the uniform line while the naive shuffle is visibly biased")
            + f'<div class="card">{pre(fisher_yates_txt)}</div>'
            + '</div>'),
        section(
            "Box-Muller: turning uniform randomness into a bell curve",
            "Random generators give uniform values, but almost every simulation -- Brownian "
            "motion, Monte-Carlo finance, noise models -- needs Gaussians. The Box-Muller "
            "transform converts a pair of uniforms into a pair of independent standard normals: "
            "z0 = sqrt(-2 ln u1) cos(2 pi u2), z1 = sqrt(-2 ln u1) sin(2 pi u2). It is exact -- "
            "the polar change of variables onto the 2D Gaussian, whose radius has r^2 "
            "exponentially distributed and whose angle is uniform. Marsaglia's polar method "
            "rejects to the unit disc and reuses its coordinates, skipping the trig. Scaling by "
            "sigma and shifting by mu gives any N(mu, sigma^2). This module implements both, and "
            "the tests verify the output's mean, variance, skewness (~0), kurtosis (~3), and the "
            "68-95-99.7 rule over large samples.",
            '<div class="grid">'
            + svg_card(out("box_muller.svg"), "the transformed-uniform histogram landing exactly on the analytic Gaussian density with its 1/2/3-sigma bands")
            + f'<div class="card">{pre(box_muller_txt)}</div>'
            + '</div>'),
        section(
            "Rejection sampling: drawing from any density you can evaluate",
            "You can compute a density f(x) but cannot sample it directly -- an unnormalized "
            "posterior, a physics distribution, a hand-drawn shape. Rejection sampling turns "
            "'I can evaluate f' into 'I can sample f': draw a candidate from a simpler proposal "
            "g you can sample, and accept it with probability f(x)/(M g(x)) where M bounds "
            "f <= M g. Geometrically you throw darts uniformly under the envelope M g and keep "
            "those below f -- the kept points are distributed exactly as f, even when f is only "
            "known up to a constant (which is why it underlies Bayesian computation). The price "
            "is efficiency: the acceptance rate is the area ratio 1/M, so a loose envelope or a "
            "high dimension wastes darts. This module does box and general rejection sampling, "
            "and the tests verify the sampled moments, the acceptance-rate theory, and a "
            "histogram chi-square against the target.",
            '<div class="grid">'
            + svg_card(out("rejection_sampling.svg"), "accepted (green) and rejected (red) darts under a bimodal density, and the kept-sample histogram matching it")
            + f'<div class="card">{pre(rejection_sampling_txt)}</div>'
            + '</div>'),
        section(
            "Welford's algorithm: mean and variance in one stable pass",
            "The textbook variance E[x^2] - E[x]^2 is a numerical disaster: it subtracts two "
            "large, nearly-equal numbers, so on offset data (temperatures near 1e6, timestamps, "
            "prices) catastrophic cancellation can even return a NEGATIVE variance. Welford's "
            "algorithm updates a running mean and the sum of squared deviations as each datum "
            "arrives -- delta = x - mean; mean += delta/n; M2 += delta*(x - new_mean) -- never "
            "forming those giant intermediates, so it is both online (no need to store the data) "
            "and numerically stable. Terriberry's extension carries M3 and M4 for skewness and "
            "kurtosis, and two accumulators merge by combining counts, means, and M2 with a "
            "correction -- so statistics over shards combine in parallel, exactly. This module "
            "provides the accumulator and merge, verified against a two-pass computation and "
            "shown staying exact where the naive formula collapses.",
            '<div class="grid">'
            + svg_card(out("welford.svg"), "the running mean and standard deviation converging onto their true values as the stream flows")
            + f'<div class="card">{pre(welford_txt)}</div>'
            + '</div>'),
        section(
            "Kahan summation: adding floats without losing the small ones",
            "Add a million small numbers naively and the answer drifts: once the total is large, "
            "each tiny addend has fewer mantissa bits to land in and its low-order part is "
            "rounded away, so the error grows with n. Kahan's compensated summation carries a "
            "correction term c for the bits lost on the previous addition -- y = x - c; t = sum "
            "+ y; c = (t - sum) - y -- so the error stays bounded by a small constant, as if the "
            "sum were computed in twice the precision. Neumaier's variant also handles the case "
            "where the next addend exceeds the running total (catastrophic cancellation), and "
            "pairwise summation gives O(log n) error growth with no correction term. This module "
            "implements all of these plus a compensated dot product and running-mean "
            "accumulator, verified against Python's exact math.fsum on ill-conditioned inputs.",
            '<div class="grid">'
            + svg_card(out("kahan.svg"), "the relative error versus the number of terms: naive climbs with n while Kahan stays flat at machine precision")
            + f'<div class="card">{pre(kahan_txt)}</div>'
            + '</div>'),
        section(
            "Horner's method: evaluating a polynomial the fast, stable way",
            "Evaluating a polynomial by computing each power separately costs ~2n "
            "multiplications and sums terms of wildly different sizes. Horner rewrites it as "
            "nested multiplication -- p(x) = (...(a_n x + a_{n-1})x + ...)x + a_0 -- evaluating "
            "in exactly n mults and n adds with far better rounding. The same sweep IS synthetic "
            "division: the intermediate values are the quotient of dividing by (x - r) and the "
            "final value is the remainder p(r) (the Remainder Theorem); a second sweep gives "
            "p'(r) for free, which makes Horner the engine of Newton's method for polynomial "
            "roots. This module evaluates by Horner, does synthetic division and derivatives, "
            "and finds real roots by Newton refinement plus deflation -- verified against direct "
            "power-sum evaluation and by substituting the roots back.",
            '<div class="grid">'
            + svg_card(out("horner.svg"), "the multiplication count (Horner n vs direct 2n) and Newton's quadratic convergence to a root")
            + f'<div class="card">{pre(horner_txt)}</div>'
            + '</div>'),
        section(
            "Bracketing root-finders: bisection, secant, false position, Brent",
            "Newton is fast but can diverge; when you have a bracket [a, b] where f changes sign, "
            "bracketing methods guarantee convergence. Bisection halves the interval each step "
            "(foolproof, linear -- one bit per iteration); the secant method fits a line through "
            "the last two points (superlinear, order ~1.618, but not guaranteed); false position "
            "keeps the secant inside the bracket (safe and faster than bisection); and Brent's "
            "method combines bisection's safety with inverse quadratic interpolation's speed, "
            "falling back to bisection when the fast step misbehaves -- which is why it is the "
            "default root-finder in most numerical libraries. This module implements all four "
            "with a shared bracket interface plus a sign-change scanner, verified against roots "
            "of polynomials and transcendentals and checked to agree.",
            '<div class="grid">'
            + svg_card(out("rootfind.svg"), "the error per iteration on log scale: bisection's steady linear slope against the secant method's steepening superlinear one")
            + f'<div class="card">{pre(rootfind_txt)}</div>'
            + '</div>'),
        section(
            "Numerical quadrature: integrating what algebra cannot",
            "Most integrals have no closed form, so you approximate the area under f by sampling "
            "it. The trapezoid rule joins samples with lines (error O(h^2)); Simpson fits "
            "parabolas (O(h^4), exact for cubics); Romberg applies Richardson extrapolation to a "
            "ladder of halved-step trapezoid estimates, cancelling error terms two orders at a "
            "time to reach machine precision in a handful of levels; adaptive Simpson subdivides "
            "only where the function is hard; and Gauss-Legendre places n nodes optimally to "
            "integrate polynomials of degree 2n-1 exactly. This module implements all five and "
            "verifies them against integrals with known values (polynomials, exp, trig, the "
            "Gaussian bell), confirming the convergence orders -- trapezoid error quarters and "
            "Simpson's sixteenths each time the step is halved.",
            '<div class="grid">'
            + svg_card(out("quadrature.svg"), "error vs samples on log-log axes, where each method's slope is its convergence order")
            + f'<div class="card">{pre(quadrature_txt)}</div>'
            + '</div>'),
        section(
            "Cubic spline interpolation: a smooth curve through every point",
            "A single high-degree polynomial through many points oscillates wildly between them "
            "(Runge's phenomenon). A cubic spline instead fits a separate cubic to each interval "
            "and stitches them C^2 -- value, slope, AND curvature match at every join -- giving "
            "the smoothest interpolant, the shape a flexible draftsman's ruler naturally takes. "
            "The construction reduces to solving for the knot second derivatives, a tridiagonal "
            "system solved in O(n) by the Thomas algorithm; the natural spline sets zero end "
            "curvature, the clamped spline fixes end slopes. This module builds and evaluates "
            "the spline and its derivatives, verified to pass through every knot, be C^2, "
            "reproduce cubics exactly, and stay near the true Runge curve where a single "
            "polynomial explodes to ~1.9. The interpolation behind fonts, animation, and CAD.",
            '<div class="grid">'
            + svg_card(out("spline.svg"), "spline (green) and single polynomial (red) through the same knots on Runge's function -- the polynomial oscillates, the spline stays smooth")
            + f'<div class="card">{pre(spline_txt)}</div>'
            + '</div>'),
        section(
            "The Fast Fourier Transform: O(n log n) instead of O(n^2)",
            "The discrete Fourier transform turns n samples into their n frequency components -- "
            "the recipe behind audio and image compression, spectrum analysis, and fast "
            "polynomial multiplication. Computed directly it costs O(n^2); the Cooley-Tukey FFT "
            "does the same transform in O(n log n) by splitting the samples into even- and "
            "odd-indexed halves, transforming each recursively, and combining them with "
            "twiddle-factor butterflies -- collapsing a million-sample transform from 10^12 "
            "operations to ~2x10^7, ~50,000x faster. The same butterfly runs backwards for the "
            "inverse, and the convolution theorem turns an O(n^2) convolution into three FFTs. "
            "This module implements the radix-2 FFT, inverse, naive DFT check, and FFT "
            "convolution using only built-in complex numbers, verified to match the DFT, "
            "round-trip exactly, recover known frequencies, and satisfy Parseval's identity.",
            '<div class="grid">'
            + svg_card(out("fft.svg"), "a two-tone signal in time decomposing into two sharp peaks in its frequency spectrum")
            + f'<div class="card">{pre(fft_txt)}</div>'
            + '</div>'),
        section(
            "Gaussian elimination and LU decomposition: solving A x = b",
            "A x = b -- n equations in n unknowns -- is the most-solved problem in computation: "
            "circuit analysis, structural mechanics, least squares, the linearized step of every "
            "nonlinear solver. Gaussian elimination row-reduces to triangular form and "
            "back-substitutes in O(n^3), and done once it factors A = L U into lower- and "
            "upper-triangular matrices, after which each new right-hand side is solved in O(n^2) "
            "by two triangular sweeps. Partial pivoting swaps in the largest pivot at each step "
            "to keep it numerically stable, recorded as a permutation P so P A = L U. The "
            "determinant is the product of U's diagonal times the permutation sign, and the "
            "inverse comes from solving against each unit column. This module builds the "
            "factorization, solves systems, and computes determinants and inverses, verified by "
            "residuals and P A = L U over hundreds of random systems.",
            '<div class="grid">'
            + svg_card(out("linsolve.svg"), "the L and U triangular factors that a pivoted matrix splits into")
            + f'<div class="card">{pre(linsolve_txt)}</div>'
            + '</div>'),
        section(
            "QR decomposition and least squares: fitting more data than parameters",
            "Any matrix A (m >= n) factors as A = Q R with Q orthonormal (Q^T Q = I) and R "
            "upper-triangular. It is the workhorse of overdetermined systems: to fit a model to "
            "more data points than parameters, solving the least-squares problem min ||A x - b|| "
            "by R x = Q^T b is far more stable than the normal equations A^T A x = A^T b (which "
            "square the condition number). The construction is the Gram-Schmidt process -- "
            "subtract from each column the components along the earlier orthonormal directions, "
            "then normalize -- in its modified form, which subtracts each projection immediately "
            "to stay orthogonal under rounding. QR also drives eigenvalue iteration and "
            "orthogonal regression. This module builds the thin QR by modified Gram-Schmidt, "
            "solves least-squares and square systems, and verifies Q^T Q = I, Q R = A, and that "
            "the residual is orthogonal to the column space.",
            '<div class="grid">'
            + svg_card(out("qr.svg"), "the least-squares best-fit line through scattered points, with the residuals it minimizes")
            + f'<div class="card">{pre(qr_txt)}</div>'
            + '</div>'),
        section(
            "Power iteration: eigenvalues without the characteristic polynomial",
            "An eigenvector is a direction a matrix only stretches: A v = lambda v. They govern "
            "vibration modes, Markov stationary distributions, PCA axes, and PageRank -- and for "
            "anything beyond 2x2 the characteristic polynomial is a poor way to find them. Power "
            "iteration is the simplest alternative: start from a random vector and repeatedly "
            "multiply by A and normalize; each multiply amplifies the largest-|eigenvalue| "
            "direction most, so the vector converges to the dominant eigenvector and the "
            "Rayleigh quotient v^T A v / v^T v gives its eigenvalue. Shifted inverse iteration "
            "power-iterates (A - mu I)^-1 to target the eigenvalue nearest mu, and deflation "
            "peels off found eigenpairs to recover a symmetric matrix's whole spectrum. This "
            "module implements all three, verified by A v = lambda v, the trace/determinant "
            "identities, and analytic cases.",
            '<div class="grid">'
            + svg_card(out("eigen.svg"), "the Rayleigh quotient converging to the dominant eigenvalue, and the full spectrum recovered by deflation")
            + f'<div class="card">{pre(eigen_txt)}</div>'
            + '</div>'),
        section(
            "Conjugate gradient: huge sparse SPD systems without a factorization",
            "For a symmetric positive-definite A, solving A x = b by LU costs O(n^3) and stores "
            "the whole factorization -- impossible at millions of rows (finite-element meshes, "
            "image operators, graph Laplacians). Conjugate gradient solves it with nothing but "
            "matrix-vector products, so a sparse A costs O(nnz) per step and O(n) memory. It "
            "minimizes the energy (1/2)x^T A x - b^T x, choosing each search direction "
            "A-conjugate to all previous ones so it never undoes earlier progress -- converging "
            "in at most n steps exactly, far fewer in practice at a rate set by sqrt(kappa), "
            "which is why preconditioning (here the Jacobi diagonal) is the whole game. This "
            "module implements CG and preconditioned CG, verified against a dense LU solve, the "
            "<= n step guarantee, and the monotone residual decay, and shown beating steepest "
            "descent's zig-zag.",
            '<div class="grid">'
            + svg_card(out("conjugate_gradient.svg"), "the residual plunging to machine precision under CG while steepest descent crawls (log scale)")
            + f'<div class="card">{pre(conjugate_gradient_txt)}</div>'
            + '</div>'),
        section(
            "SVD and PCA: the axes a matrix acts along",
            "Every matrix A factors as A = U S V^T: orthonormal input directions V, orthonormal "
            "outputs U, and non-negative singular values S saying how much A stretches each -- "
            "geometrically, A sends the unit sphere to an ellipsoid whose semi-axes are the "
            "singular values. It is the most informative factorization: the rank, the 2-norm and "
            "condition number, the best low-rank approximation (Eckart-Young, the basis of image "
            "compression), and the pseudo-inverse all read off it. Here it is built via the "
            "symmetric eigendecomposition of A^T A (reusing the power-iteration eigensolver). "
            "Principal component analysis is SVD of mean-centred data: the top singular vectors "
            "are the directions of greatest variance. This module computes the thin SVD, "
            "low-rank reconstruction, and PCA with explained variance, verified by A = U S V^T, "
            "orthonormality, and the variance ordering.",
            '<div class="grid">'
            + svg_card(out("svd.svg"), "a tilted data cloud with its PCA principal axes, and the singular-value spectrum")
            + f'<div class="card">{pre(svd_txt)}</div>'
            + '</div>'),
        section(
            "k-means clustering: finding groups in unlabelled data",
            "Given points and a number k, k-means partitions them so each belongs to the nearest "
            "centroid, minimizing the total within-cluster squared distance (the inertia). "
            "Lloyd's algorithm alternates two steps -- assign each point to its nearest centroid, "
            "then move each centroid to its members' mean -- each of which can only lower the "
            "inertia, so it converges (to a local minimum). k-means++ seeding spreads the "
            "initial centres by picking each with probability proportional to its squared "
            "distance from the nearest chosen one, avoiding bad starts, and a few restarts keep "
            "the best. The inertia elbow and the silhouette score both flag the natural number "
            "of clusters. This module implements Lloyd's with random and k-means++ init, "
            "multi-restart selection, and the silhouette, verified to recover well-separated "
            "blobs with monotone inertia.",
            '<div class="grid">'
            + svg_card(out("kmeans.svg"), "points coloured by recovered cluster with their centroids, and the inertia elbow marking the true k")
            + f'<div class="card">{pre(kmeans_txt)}</div>'
            + '</div>'),
        section(
            "Linear and logistic regression: fitting a line and a decision boundary",
            "The two workhorses of supervised learning. Linear regression fits y = w.x + b by "
            "minimizing squared error -- solved in closed form by QR least squares (avoiding the "
            "normal equations' condition-number squaring) or by gradient descent for large data. "
            "Logistic regression predicts a probability sigmoid(w.x + b) for binary labels, fit "
            "by gradient descent on the convex log-loss, and its decision boundary w.x + b = 0 "
            "is a separating hyperplane. Both are linear models; logistic just squashes through "
            "the sigmoid to stay a probability, and an L2 penalty shrinks the weights for "
            "generalization. This module fits linear regression by both QR and gradient descent, "
            "logistic by gradient descent with optional L2, and reports R^2 for regression and "
            "accuracy / log-loss for classification, verified against exact fits and separable "
            "data.",
            '<div class="grid">'
            + svg_card(out("regression.svg"), "the least-squares line through noisy points, and the logistic sigmoid crossing 0.5 at the decision boundary")
            + f'<div class="card">{pre(regression_txt)}</div>'
            + '</div>'),
        section(
            "Decision trees: classification by the best yes/no questions",
            "The most interpretable model, and the base learner of random forests and gradient "
            "boosting -- the workhorses of tabular ML. CART grows a tree greedily: at each node it "
            "tries every feature and every threshold and keeps the split that most reduces the "
            "IMPURITY of the children, where Gini impurity is 1 - sum p^2 (the chance two random "
            "draws differ) and entropy is -sum p log2 p (bits of surprise), both zero for a pure "
            "node. Recursing carves the plane into axis-aligned rectangles, each a leaf that votes "
            "the majority class; a max-depth or minimum-node-size cap fights the overfitting a "
            "fully grown tree invites. The path from root to leaf reads as a plain if/else rule, "
            "and no feature scaling is needed. This module builds the classifier with Gini or "
            "entropy, exposes the learned rules and feature importances, and is checked on "
            "separable blobs, a train/test split, and a known single split.",
            '<div class="grid">'
            + svg_card(out("decision_tree.svg"), "the axis-aligned decision regions the tree carves out, with training points and the learned rules")
            + f'<div class="card">{pre(decision_tree_txt)}</div>'
            + '</div>'),
        section(
            "Random forests: a committee of decorrelated trees",
            "A single decision tree overfits -- it memorizes noise by growing pure leaves. A random "
            "forest averages many trees deliberately weakened to disagree, so their errors cancel "
            "while their signal adds. Two randomizations decorrelate them: BAGGING trains each tree "
            "on a bootstrap sample (n rows drawn with replacement, ~63% distinct), and FEATURE "
            "SUBSAMPLING lets each split consider only a random sqrt(d) subset of features so no one "
            "strong feature dominates every tree. Prediction is a majority vote. Because ~37% of "
            "rows are out-of-bag for each tree -- never seen by it -- voting each row over only its "
            "out-of-bag trees gives a free, honest validation estimate needing no held-out set. "
            "This module builds a bagged forest of the CART learner with per-node feature sampling, "
            "majority-vote prediction, out-of-bag scoring, and averaged feature importances, "
            "verified to beat an overfit single tree on a noisy problem with its OOB estimate "
            "tracking true test error.",
            '<div class="grid">'
            + svg_card(out("random_forest.svg"), "the jagged single-tree boundary beside the smoother forest boundary on the same noisy data")
            + f'<div class="card">{pre(random_forest_txt)}</div>'
            + '</div>'),
        section(
            "Gaussian mixtures & EM: soft, probabilistic clustering",
            "k-means assigns each point hard to its nearest centroid; a Gaussian mixture instead "
            "models the data as drawn from k Gaussians and asks, for each point, the PROBABILITY it "
            "came from each -- a soft assignment that lets clusters differ in size, weight, and "
            "spread, and that comes with a likelihood. The fit is Expectation-Maximization: the "
            "E-step computes each point's responsibility (posterior over components) with the "
            "parameters fixed, and the M-step re-estimates each component as the "
            "responsibility-weighted mean, variance, and weight of the data. Each round provably "
            "cannot decrease the log-likelihood -- that monotone climb is the standard correctness "
            "check -- and EM converges to a local optimum, so it is run from several inits and the "
            "best kept. This module fits a diagonal-covariance mixture in any dimension with "
            "log-sum-exp numerics, gives soft responsibilities and hard labels, and reports AIC/BIC "
            "for choosing k -- verified to recover known mixture parameters, climb the "
            "log-likelihood every iteration, and let BIC select the true number of components.",
            '<div class="grid">'
            + svg_card(out("gmm.svg"), "points tinted by their soft responsibilities with the fitted 2-sigma component ellipses, and the BIC curve dipping at the true k")
            + f'<div class="card">{pre(gmm_txt)}</div>'
            + '</div>'),
        section(
            "Hidden Markov models: decoding sequences with hidden state",
            "A hidden Markov model describes a sequence you can see (emissions) generated by a chain "
            "of states you cannot (the hidden path) -- the weather driving whether you carry an "
            "umbrella, or a dealer switching between a fair and a loaded die. Three questions have "
            "three exact dynamic-programming answers over the trellis: the FORWARD algorithm sums "
            "the probability of every consistent path in O(T k^2) to evaluate the sequence (a naive "
            "sum is O(k^T)); VITERBI is the same recursion with max for sum plus backpointers, "
            "giving the single most-likely hidden path; and BAUM-WELCH is EM -- forward-backward "
            "gives the posterior of each state and transition at each step, and those soft counts "
            "re-estimate the matrices, the likelihood provably climbing each round. It all runs in "
            "log space with log-sum-exp so long sequences never underflow. This module implements "
            "forward, Viterbi, forward-backward posteriors, and Baum-Welch training, verified that "
            "Viterbi recovers a planted path, forward and backward agree on the likelihood, and "
            "Baum-Welch relearns a known loaded-die model with monotone log-likelihood.",
            '<div class="grid">'
            + svg_card(out("hmm.svg"), "the true hidden fair/loaded path, the Viterbi decode, the posterior P(loaded) ribbon, and the relearned emission distributions")
            + f'<div class="card">{pre(hmm_txt)}</div>'
            + '</div>'),
        section(
            "The Kalman filter: optimal tracking of a hidden state",
            "Where a hidden Markov model tracks a DISCRETE hidden state, the Kalman filter tracks a "
            "CONTINUOUS one -- a position and velocity, a trajectory -- from noisy measurements in "
            "real time, with no growing history. It is the optimal estimator for a linear system "
            "with Gaussian noise, and the math behind GPS, guidance, and sensor fusion. The world "
            "is a linear-Gaussian state space: the true state evolves as x <- F x + process noise, "
            "the sensor reports z = H x + measurement noise. The filter carries a Gaussian belief "
            "(mean, covariance) and alternates PREDICT (push through the dynamics, uncertainty "
            "grows) and UPDATE (fold in a measurement weighted by the Kalman gain "
            "K = P H' (H P H' + R)^-1, uncertainty shrinks). Because both model and sensor are "
            "noisy, the fused estimate beats either alone -- its variance provably below the "
            "sensor's -- and a backward RTS smoother, using future data, beats the causal filter. "
            "This module implements the multivariate filter and smoother with self-contained "
            "matrix helpers, verified on constant-velocity tracking: error and variance fall below "
            "the raw measurements', a steady-state gain is reached, a perfect sensor is trusted "
            "exactly and a useless one ignored, and the smoother improves on the filter.",
            '<div class="grid">'
            + svg_card(out("kalman.svg"), "the true track, noisy measurements, filtered and smoothed estimates, and the estimate variance collapsing to a steady state")
            + f'<div class="card">{pre(kalman_txt)}</div>'
            + '</div>'),
        section(
            "PageRank: ranking a graph by its random walk",
            "The algorithm that launched Google, and a clean application of Markov chains and the "
            "dominant eigenvector. PageRank scores every node of a directed graph by one recursive "
            "idea -- a node is important if important nodes link to it -- formalized as a random "
            "surfer who with probability d follows a random out-link and with probability 1-d "
            "teleports to a uniformly random page. The score is the fraction of time the surfer "
            "spends on each page: the STATIONARY DISTRIBUTION of that chain, equivalently the "
            "dominant eigenvector of the Google matrix G = d M + (1-d)/N 11'. The teleport makes G "
            "strictly positive, so Perron-Frobenius guarantees a unique positive stationary vector "
            "and POWER ITERATION converges to it geometrically at rate d. Dangling nodes (no "
            "out-links) would leak probability, so their mass is redistributed by teleport, and it "
            "is all done sparsely without forming the dense NxN matrix. This module computes "
            "PageRank by sparse power iteration with damping and correct dangling handling, plus "
            "the personalized variant, verified against the analytic stationary distribution of "
            "small chains, ring symmetry, and the fixed-point property.",
            '<div class="grid">'
            + svg_card(out("pagerank.svg"), "a small web graph with each node sized by its PageRank, and the power-iteration convergence curve on a log scale")
            + f'<div class="card">{pre(pagerank_txt)}</div>'
            + '</div>'),
        section(
            "LU & Cholesky: factoring a matrix to solve, invert, and take determinants",
            "Solving A x = b once is easy; solving it for many right-hand sides, or getting det(A) "
            "or A^-1, is where FACTORIZATION pays. LU decomposition writes any square matrix as "
            "P A = L U -- a permutation of row swaps (partial pivoting, for stability), a "
            "unit-lower-triangular L, and an upper-triangular U -- by Gaussian elimination in "
            "O(n^3) once; afterwards each solve is two O(n^2) triangular sweeps, the determinant is "
            "the signed product of U's diagonal, and the inverse is n solves. For a symmetric "
            "positive-definite matrix, CHOLESKY A = L L' is the smaller, stabler special case: half "
            "the work, no pivoting, and it succeeds if and only if the matrix is positive definite "
            "-- so attempting it IS the standard SPD test, the backbone of least squares and Kalman "
            "filters. This module implements LU with partial pivoting, Cholesky, triangular and "
            "general solves, determinant, and inverse, verified by reconstructing P A = L U and "
            "A = L L', cross-checking determinants against cofactors, confirming Cholesky rejects "
            "non-positive-definite matrices, and round-tripping A A^-1 = I.",
            '<div class="grid">'
            + svg_card(out("lu.svg"), "the L and U factors of A and the L, L' factors of an SPD matrix, shaded by magnitude so the triangular zero-structure shows")
            + f'<div class="card">{pre(lu_txt)}</div>'
            + '</div>'),
        section(
            "Gaussian process regression: prediction with honest error bars",
            "Where linear regression fits fixed coefficients, a Gaussian process fits a "
            "distribution over FUNCTIONS and returns a calibrated error bar that widens where there "
            "is no data. It assumes any finite set of function values is jointly Gaussian with "
            "covariance set by a KERNEL -- the RBF kernel k(x,x') = sigma^2 exp(-||x-x'||^2/2l^2) "
            "encoding 'smooth, with length scale l'. Conditioning that joint Gaussian on the "
            "observations gives the posterior in closed form: mean = k*'(K+sigma_n^2 I)^-1 y and "
            "variance = k(x*,x*) - k*'(K+sigma_n^2 I)^-1 k*, the single linear solve done by a "
            "Cholesky factorization of the SPD matrix (K + noise) and reused for the log MARGINAL "
            "LIKELIHOOD that scores hyperparameters. Noise-free, the posterior interpolates the "
            "data exactly with zero variance there; far from data the variance rises back to the "
            "prior. This module builds GP regression with the RBF kernel, posterior mean and "
            "variance, and marginal-likelihood length-scale selection -- verified to interpolate "
            "noise-free data exactly, grow uncertainty away from data, recover a known smooth "
            "function, and peak the marginal likelihood near the true length scale.",
            '<div class="grid">'
            + svg_card(out("gaussian_process.svg"), "the posterior mean tracking the true function with a 2-sigma band that pinches shut at observations and flares wide in the gap and beyond")
            + f'<div class="card">{pre(gp_txt)}</div>'
            + '</div>'),
        section(
            "Bayesian optimization: minimizing an expensive black box",
            "Some objectives are costly to evaluate -- a hyperparameter sweep that trains a model "
            "each time, a physical experiment, a slow simulation -- and grid or random search "
            "wastes most of the budget on uninteresting regions. Bayesian optimization builds a "
            "cheap probabilistic surrogate (a Gaussian process) of the objective and spends each "
            "expensive evaluation where an ACQUISITION FUNCTION says the expected payoff is "
            "highest, balancing EXPLOITATION (sample where the surrogate predicts a low value) "
            "against EXPLORATION (sample where it is uncertain). The classic acquisition is "
            "EXPECTED IMPROVEMENT: with current best f_best and posterior (mu, sigma), "
            "EI = (f_best - mu) Phi(z) + sigma phi(z) where z = (f_best - mu)/sigma -- zero at "
            "observed points, large where the surrogate is both promising and unsure, so "
            "maximizing it trades the two off automatically. This module implements EI and the full "
            "loop over a bounded domain, verified to locate the minima of a 1-D multimodal function "
            "(to within 0.001 of optimum in 20 evaluations) and the 2-D Branin function, and to "
            "beat random search at equal budget.",
            '<div class="grid">'
            + svg_card(out("bayes_opt.svg"), "the GP surrogate and 2-sigma band over the true function with sampled points and the EI curve, beside the best-so-far convergence outpacing random search")
            + f'<div class="card">{pre(bayes_opt_txt)}</div>'
            + '</div>'),
        section(
            "DBSCAN: density clustering of arbitrary shapes",
            "k-means and Gaussian mixtures need you to pick k and assume blobby clusters. DBSCAN "
            "assumes neither: it finds clusters as connected regions of high point density, "
            "discovers their number automatically, handles arbitrary shapes (interlocking moons, "
            "concentric rings), and explicitly labels outliers as NOISE instead of forcing every "
            "point into a cluster. Two parameters set 'dense enough' -- a radius EPS and a count "
            "MIN_PTS: a CORE point has at least min_pts neighbours within eps, a BORDER point is "
            "within eps of a core but not itself core, and everything else is NOISE. A cluster "
            "grows by starting at a core point and flood-filling through core-to-core "
            "neighbourhoods, so an S-curve is recovered whole where k-means would slice it. This "
            "module implements DBSCAN with the core/border/noise classification and a k-distance "
            "helper for choosing eps, verified to separate two interlocking half-moons that k-means "
            "cannot, flag sparse outliers as noise, discover the cluster count on its own, and "
            "degenerate sensibly at extreme parameters.",
            '<div class="grid">'
            + svg_card(out("dbscan.svg"), "two moons coloured by discovered cluster with noise points marked as grey crosses, and the k-distance graph whose elbow suggests eps")
            + f'<div class="card">{pre(dbscan_txt)}</div>'
            + '</div>'),
        section(
            "Hierarchical clustering: a tree of nested groupings",
            "k-means and DBSCAN give one flat partition; hierarchical clustering gives the whole "
            "family at once, as a tree. Agglomerative clustering starts with every point its own "
            "cluster and repeatedly merges the two closest, recording each merge and the distance "
            "at which it happened -- the DENDROGRAM. Cut it at any height to read off a flat "
            "clustering, so the number of clusters comes from where you cut, often the biggest gap "
            "in merge heights, rather than being fixed in advance. What 'closest' means is the "
            "LINKAGE: SINGLE (nearest points, tends to chain along filaments), COMPLETE (farthest "
            "points, compact clusters), AVERAGE (mean pairwise, UPGMA), and WARD (least increase in "
            "within-cluster variance, tight and spherical). Merge heights are monotone for these, "
            "so the tree has no crossings. This module builds the full dendrogram by the "
            "Lance-Williams update and cuts it into k clusters, verified to recover well-separated "
            "blobs with all linkages, climb merge heights monotonically, and show single-linkage "
            "chaining where complete linkage stays compact.",
            '<div class="grid">'
            + svg_card(out("hierarchical.svg"), "points coloured by the three-cluster cut beside the dendrogram whose branch heights are merge distances, with the cut line marked")
            + f'<div class="card">{pre(hierarchical_txt)}</div>'
            + '</div>'),
        section(
            "Naive Bayes: fast probabilistic classification",
            "Naive Bayes applies Bayes' theorem with one bold simplification -- it assumes the "
            "features are conditionally independent given the class. That is 'naive' (words and "
            "pixels are not independent), but it collapses a joint distribution into a product of "
            "per-feature terms, so training is a single counting pass and prediction a sum of logs. "
            "The posterior is P(c | x) proportional to P(c) prod_i P(x_i | c); in log space we pick "
            "the class maximizing log P(c) + sum_i log P(x_i | c). Two flavours differ only in "
            "P(x_i | c): GAUSSIAN models each continuous feature as a per-class Normal (estimate "
            "mean and variance), while MULTINOMIAL models word counts with Laplace add-alpha "
            "smoothing so an unseen word never zeroes the product -- the workhorse of spam filters. "
            "This module implements both entirely in log space, verified that the Gaussian model "
            "separates blobs and matches a hand-computed posterior exactly, that the multinomial "
            "model classifies documents and its smoothing prevents zero probabilities, and that the "
            "predicted class-probabilities are normalized. Despite the crude assumption it is the "
            "strong baseline every fancier classifier must beat.",
            '<div class="grid">'
            + svg_card(out("naive_bayes.svg"), "the Gaussian decision regions with per-class means, and the multinomial spam filter's per-word log-odds as a diverging bar chart")
            + f'<div class="card">{pre(naive_bayes_txt)}</div>'
            + '</div>'),
        section(
            "k-nearest-neighbours: lazy, instance-based learning",
            "k-NN is the ultimate lazy learner: it builds no model. To predict a point it finds the "
            "k nearest training points and lets them VOTE (classification) or AVERAGES their values "
            "(regression) -- all the work at query time, a non-parametric decision boundary that "
            "traces the data. The number of neighbours k trades variance for bias: k=1 fits every "
            "point exactly (a jagged, noise-fitting boundary) while large k averages over a wide "
            "region (smooth but blurred), and the vote can be uniform or DISTANCE-WEIGHTED "
            "(weight = 1/distance, so closer neighbours count more). Because it compares raw "
            "coordinates it is sensitive to feature scaling, so standardization is included. A "
            "brute-force search is O(n) per query; a k-d tree cuts that to O(log n) in low "
            "dimensions. This module implements k-NN classification and regression with both "
            "weightings and leave-one-out cross-validation, verified that 1-NN memorizes the "
            "training labels, that it recovers separable classes and a smooth regression target, "
            "that distance-weighting follows the closest neighbour, that LOO selects k>1 under "
            "label noise, and that its neighbours agree with the k-d tree.",
            '<div class="grid">'
            + svg_card(out("knn.svg"), "the jagged k=1 decision boundary beside the smoother cross-validation-selected k on the same noisy two-class data")
            + f'<div class="card">{pre(knn_txt)}</div>'
            + '</div>'),
        section(
            "Gradient boosting: shallow trees that correct each other",
            "A random forest averages independent deep trees; gradient boosting grows trees in "
            "SEQUENCE, each correcting the errors of those before it. It is gradient descent in "
            "FUNCTION space: start with a constant, then repeatedly fit a small tree to the "
            "negative gradient of the loss and add a shrunken step of it to the running model. For "
            "squared-error regression that gradient is exactly the RESIDUAL y - F(x), so each tree "
            "learns what the ensemble still gets wrong; for logistic classification it is "
            "y - sigmoid(F(x)). Three knobs trade bias for variance: the number of trees adds "
            "capacity, the LEARNING RATE shrinks each tree's contribution (small rates need more "
            "trees but generalize better -- shrinkage is regularization), and tree DEPTH caps the "
            "feature interactions each weak learner captures. This module implements boosting for "
            "squared-error regression and log-loss binary classification over self-contained "
            "regression trees, with staged predictions, verified that training loss decreases "
            "monotonically as trees are added, that the ensemble beats a single tree by ~250x on a "
            "noisy target, that a smaller learning rate needs more trees, and that it separates a "
            "circular class boundary. This is the method that wins most tabular-data competitions.",
            '<div class="grid">'
            + svg_card(out("gradient_boosting.svg"), "the regression fit sharpening from 1 to 120 trees over the noisy data, beside the training loss falling monotonically on a log scale")
            + f'<div class="card">{pre(gradient_boosting_txt)}</div>'
            + '</div>'),
        section(
            "Spectral clustering: cutting a graph by its Laplacian",
            "k-means splits space by distance to a centroid, so it fails on non-convex shapes -- "
            "concentric rings, interlocking moons. Spectral clustering escapes that by working on a "
            "GRAPH: connect nearby points with weighted edges, then cut the graph into pieces dense "
            "inside and sparse between. Remarkably, that combinatorial cut is solved (relaxed) by "
            "linear algebra -- the eigenvectors of the graph LAPLACIAN. Build a Gaussian affinity "
            "matrix, form the normalized Laplacian L = I - D^-1/2 W D^-1/2, take the eigenvectors of "
            "its k smallest eigenvalues (the number near zero equals the number of connected "
            "components), embed each point by its coordinates there, and run k-means in that space "
            "-- where the tangled shapes become tight, linearly separable blobs. This module builds "
            "the affinity graph and Laplacians, extracts the low eigenvectors by reusing a symmetric "
            "eigensolver on cI - L (turning smallest into largest), and clusters the embedding, "
            "verified to separate concentric rings and two moons that k-means cannot, and that the "
            "Laplacian's zero-eigenvalue multiplicity counts the graph's connected components. Built "
            "on the eigen and k-means modules.",
            '<div class="grid">'
            + svg_card(out("spectral_clustering.svg"), "two concentric rings correctly split by spectral clustering, beside the two-eigenvector embedding in which the tangled rings become separable point clouds")
            + f'<div class="card">{pre(spectral_txt)}</div>'
            + '</div>'),
        section(
            "Particle filters: nonlinear, non-Gaussian tracking",
            "The Kalman filter is optimal but only for LINEAR dynamics with GAUSSIAN noise. When "
            "the motion or sensor is nonlinear, or the belief is multimodal, its single Gaussian "
            "breaks. A particle filter drops that assumption: it represents the belief as a CLOUD "
            "of weighted samples and propagates them through the true, arbitrary dynamics -- "
            "sequential Monte Carlo. Each step PREDICTS (push every particle through the motion "
            "model plus noise), WEIGHTS (reweight by the likelihood of the actual measurement), and "
            "RESAMPLES (draw a new equal-weight set in proportion to the weights, killing unlikely "
            "particles and duplicating likely ones). Resampling is the crux: without it a few "
            "particles hoard all the weight (DEGENERACY) and the cloud stops representing the "
            "posterior. The EFFECTIVE SAMPLE SIZE 1/sum(w^2) measures that, and we resample only "
            "when it drops below N/2, using low-variance systematic resampling. This module "
            "implements a generic bootstrap filter with adaptive systematic resampling, verified on "
            "a nonlinear tracking problem: its estimate beats the raw sensor by ~60%, resampling "
            "keeps the effective sample size high where a weight-only filter collapses to a single "
            "particle, and more particles reduce the error. Built with its own Gaussian sampler.",
            '<div class="grid">'
            + svg_card(out("particle_filter.svg"), "the particle-filter estimate tracking the nonlinear truth below the noisy sensor, beside the effective sample size staying healthy with resampling and collapsing without it")
            + f'<div class="card">{pre(particle_filter_txt)}</div>'
            + '</div>'),
        section(
            "Simulated annealing: escaping local minima by cooling",
            "Greedy search only moves downhill and gets trapped in the first local minimum. "
            "Simulated annealing escapes by sometimes moving UPHILL, with a probability that shrinks "
            "over time -- the metallurgical analogy: heat a metal and cool it slowly so its atoms "
            "settle into a low-energy crystal, not a brittle freeze. At each step it proposes a "
            "random neighbour and applies the METROPOLIS criterion: a move lowering the cost is "
            "always taken; one raising it by delta is taken with probability exp(-delta/T). High T "
            "(early) accepts almost anything and roams freely out of local basins; as T cools only "
            "improving moves survive. The COOLING SCHEDULE is the key knob -- cool too fast and you "
            "quench into a poor minimum; cool slowly (geometric T <- alpha T) and you approach the "
            "global optimum. This module implements generic annealing over any state plus a "
            "travelling-salesman solver with 2-opt segment-reversal moves, verified to find the "
            "global minimum of a multimodal function that greedy descent misses, converge a square "
            "TSP tour to its exact optimal perimeter, beat nearest-neighbour greedy on random "
            "tours, and shrink its acceptance rate as it cools.",
            '<div class="grid">'
            + svg_card(out("simulated_annealing.svg"), "the greedy nearest-neighbour tour beside the shorter annealed tour, and the tour length falling as the temperature cools with early uphill excursions")
            + f'<div class="card">{pre(simulated_annealing_txt)}</div>'
            + '</div>'),
        section(
            "Genetic algorithms: optimization by simulated evolution",
            "Where simulated annealing perturbs a single state, a genetic algorithm evolves a whole "
            "POPULATION, letting good solutions breed -- a direct metaphor for natural selection. "
            "Each generation SELECTS parents biased toward high fitness (tournament: the best of k "
            "random individuals), applies CROSSOVER to splice two parents' genes into offspring, "
            "MUTATES to inject new variation, and keeps the best few via ELITISM so the best-so-far "
            "never regresses. Being population-based it explores many basins at once and needs no "
            "gradients, making it strong on rugged, discrete, or black-box landscapes. This module "
            "implements a generic GA (tournament selection, one-point crossover, elitism) over both "
            "binary and real-valued genomes plus a 0/1 knapsack solver, verified to solve the "
            "OneMax bit problem to all-ones, maximize a multimodal real function, match the "
            "brute-force optimum of a small knapsack, keep the best fitness monotone under elitism, "
            "and beat random search at equal budget.",
            '<div class="grid">'
            + svg_card(out("genetic_algorithm.svg"), "the best and mean population fitness climbing each generation on the OneMax bit problem and the bumpy real-function optimization")
            + f'<div class="card">{pre(genetic_algorithm_txt)}</div>'
            + '</div>'),
        section(
            "Particle swarm optimization: a flock homing on the optimum",
            "A third metaheuristic beside simulated annealing (one state) and genetic algorithms "
            "(breeding a population): a SWARM of candidate solutions flies through the search space, "
            "each remembering its own best spot and pulled toward the best any member has found -- "
            "inspired by flocking birds. Each particle carries a position and a VELOCITY updated by "
            "three pulls: INERTIA (coast on the old velocity, exploring), COGNITIVE (toward this "
            "particle's own best, individual memory), and SOCIAL (toward the swarm's global best, "
            "shared knowledge), with fresh randoms keeping it stochastic. High inertia explores, low "
            "inertia exploits, so it is often decayed over the run. No gradients, just local rules, "
            "and the swarm balances exploration against convergence. This module implements PSO over "
            "a bounded box with velocity clamping and linearly-decaying inertia, verified to find "
            "the global minimum of the Sphere, Rastrigin, and Rosenbrock benchmarks, drive the "
            "global best down monotonically, beat random search at equal budget, and converge faster "
            "when inertia decays.",
            '<div class="grid">'
            + svg_card(out("particle_swarm.svg"), "the final swarm clustered at the Rastrigin optimum, beside the global-best convergence with decaying vs fixed inertia on a log scale")
            + f'<div class="card">{pre(particle_swarm_txt)}</div>'
            + '</div>'),
        section(
            "Reed-Solomon codes: recovering data from errors",
            "The error-correcting code behind QR codes, CDs, DVDs, and deep-space probes. "
            "Reed-Solomon treats a message as the coefficients of a polynomial over the finite "
            "field GF(256) and appends 2t parity symbols so the whole codeword is divisible by a "
            "fixed generator polynomial. Any corruption breaks that divisibility in a way that "
            "pinpoints both WHERE the errors are and WHAT they should have been -- correcting up to "
            "t byte-errors per block no matter how they are distributed, which is why it survives a "
            "scratch on a disc or a fading radio link. The field is GF(2^8): bytes with XOR as "
            "addition and multiplication modulo 0x11d, so every nonzero byte is a power of the "
            "generator and multiplication is log-table addition. Decoding is the classic pipeline: "
            "SYNDROMES (evaluate at the code roots), BERLEKAMP-MASSEY (the error-locator "
            "polynomial), a CHIEN search (its roots = error positions), and FORNEY's formula (the "
            "error magnitudes). This module implements GF(256) arithmetic, encoding, and full "
            "syndrome decoding, verified that a clean codeword is unchanged, that up to t corrupted "
            "bytes anywhere (including in the parity) are corrected exactly across dozens of random "
            "trials, and that one error past the limit is flagged rather than mis-corrected.",
            '<div class="grid">'
            + svg_card(out("reed_solomon.svg"), "the received codeword with injected errors in red, and the decoded codeword with the same positions repaired in green, split at the message/parity boundary")
            + f'<div class="card">{pre(reed_solomon_txt)}</div>'
            + '</div>'),
        section(
            "Mutual information: measuring dependence between variables",
            "Entropy measures the uncertainty in one variable; MUTUAL INFORMATION I(X;Y) measures "
            "how much knowing one reduces the uncertainty in the other -- the shared information. "
            "Unlike correlation it captures ANY relationship, linear or not: I = 0 exactly when X "
            "and Y are independent, and it equals H(X) for a deterministic copy. Everything is "
            "built from Shannon entropy: I(X;Y) = H(X) + H(Y) - H(X,Y) = H(X) - H(X|Y). The "
            "Kullback-Leibler divergence D(p||q) = sum p log2(p/q) -- the extra bits to code "
            "p-samples with a q-code -- is the asymmetric distance from which MI is the divergence "
            "of the joint from the product of marginals, and INFORMATION GAIN (the decision-tree "
            "split criterion) is exactly the mutual information between a feature and the label. "
            "This module estimates entropies and mutual information from samples or a joint "
            "distribution with KL divergence and normalized MI, verified that independent variables "
            "have zero MI, a copy has I = H (maximal), the H(X)+H(Y)-H(X,Y) and conditional-entropy "
            "identities hold, KL is nonnegative and zero only for equal distributions, and against "
            "hand-computed values -- and it catches a nonlinear dependence that correlation reports "
            "as ~0.",
            '<div class="grid">'
            + svg_card(out("mutual_information.svg"), "mutual information falling from 1 bit to 0 as a binary channel's flip probability rises, matching the theoretical 1 - H(f) curve")
            + f'<div class="card">{pre(mutual_information_txt)}</div>'
            + '</div>'),
        section(
            "LRU & LFU caches: O(1) eviction policies",
            "A cache holds a fixed number of items and must EVICT one when full; which one decides "
            "the hit rate. LRU (Least Recently Used) evicts the item untouched longest, betting on "
            "temporal locality -- the default in CPU caches, page tables, and web caches. LFU "
            "(Least Frequently Used) evicts the least-accessed, betting popularity persists. The "
            "craft is doing it in O(1): a naive LRU scans for the oldest item on every eviction, "
            "but a HASH MAP (key -> node) for lookup plus a DOUBLY-LINKED LIST ordered by recency "
            "makes touch-and-promote and tail-eviction O(1); LFU groups keys into frequency buckets "
            "so increments and min-frequency eviction are O(1) amortized. This module implements "
            "both with hit/miss statistics, verified that LRU evicts in true least-recently-used "
            "order (checked against a brute-force reference over 60 random workloads), that "
            "touching an item spares it, that LFU evicts the least-frequent breaking ties by "
            "recency, that capacity is never exceeded, and that a skewed hot-key workload gives LFU "
            "a higher hit rate than LRU.",
            '<div class="grid">'
            + svg_card(out("lru_cache.svg"), "LRU and LFU hit rates side by side across uniform, skewed, and looping workloads, showing no single policy wins everywhere")
            + f'<div class="card">{pre(lru_cache_txt)}</div>'
            + '</div>'),
        section(
            "Tries: the prefix tree behind autocomplete",
            "A trie stores strings as a tree where each edge is a character and each root-to-node "
            "path spells a prefix. Words sharing a prefix share its path, so the structure is "
            "naturally compressed by common beginnings and every operation runs in O(length of the "
            "key) -- INDEPENDENT of how many keys are stored, unlike a hash set's collisions or a "
            "balanced tree's O(log n) whole-string comparisons. That per-character walk is exactly "
            "what powers autocomplete (every word under a prefix), longest-prefix matching (IP "
            "routers), and dictionary spell-check. Deletion unmarks a word and prunes now-childless "
            "non-terminal nodes on the way back up, and building a trie over all SUFFIXES of a text "
            "turns it into a substring index -- any substring is a prefix of some suffix. This "
            "module implements insert, search, prefix membership, autocomplete (alphabetical or "
            "frequency-ranked), deletion with pruning, longest-prefix matching, and a suffix-trie "
            "substring index, verified that it distinguishes a stored word from a mere prefix, "
            "autocompletes exactly the words under a prefix, deletes without disturbing siblings or "
            "shared prefixes, and detects substrings and their positions.",
            '<div class="grid">'
            + svg_card(out("trie.svg"), "the trie drawn as a character-labelled tree with word-ending nodes filled green and interior prefix nodes outlined")
            + f'<div class="card">{pre(trie_txt)}</div>'
            + '</div>'),
        section(
            "Comparison sorts: the classic four and their trade-offs",
            "Sorting is computing's most-studied problem, and the classic comparison algorithms each "
            "make a different trade among speed, memory, stability, and worst case -- all bounded "
            "below by the O(n log n) decision-tree limit. INSERTION sort is O(n^2) but fast on "
            "nearly-sorted data and the base case big sorts fall back to; MERGE sort is O(n log n) "
            "ALWAYS and stable but needs O(n) scratch; QUICK sort is usually fastest and in-place "
            "but O(n^2) on adversarial input unless the pivot is chosen well (here median-of-three "
            "plus an insertion cutoff); HEAP sort is O(n log n) worst-case AND in-place, built on a "
            "binary heap that doubles as a priority queue. This module implements all four plus the "
            "heap with a comparison counter and a key function, verified that every sort matches "
            "Python's built-in on random, sorted, reverse, and duplicate-heavy inputs, that merge "
            "and insertion are stable while quick and heap are not, that comparison counts scale as "
            "O(n log n) for the good sorts and O(n^2) for insertion, that median-of-three keeps "
            "quick sort fast on its classic sorted/reverse adversaries, and that the heap is a "
            "correct priority queue.",
            '<div class="grid">'
            + svg_card(out("sorting.svg"), "comparison counts vs input size on a log-log plot, insertion's O(n^2) slope pulling away from the parallel O(n log n) lines of merge, quick, and heap")
            + f'<div class="card">{pre(sorting_txt)}</div>'
            + '</div>'),
        section(
            "Newton's method in n dimensions: solving nonlinear systems",
            "One-dimensional Newton iterates x <- x - f/f'; in n dimensions the derivative becomes "
            "the JACOBIAN matrix and the division becomes solving a linear system J(x) delta = "
            "-F(x), then x <- x + delta. Each step linearizes the system at the current point, jumps "
            "to that linear model's root, and repeats -- and near a solution it converges "
            "QUADRATICALLY, the number of correct digits roughly doubling each step. It is the "
            "engine inside circuit simulation, inverse kinematics, chemical equilibrium, and "
            "optimization. When the analytic Jacobian is unavailable it is approximated by finite "
            "differences; because plain Newton can overshoot and diverge far from a root, a damped "
            "line search backtracks the step until the residual actually decreases for global "
            "robustness; and a Broyden quasi-Newton mode updates a Jacobian approximation instead "
            "of recomputing it. This module implements all three, each solving the linear step via "
            "LU with partial pivoting, verified on a circle-line intersection and the Rosenbrock "
            "stationary point, that convergence is quadratic near the root, that the "
            "finite-difference Jacobian matches an analytic one, that damping rescues a start where "
            "plain Newton diverges (arctan from far out), and that Broyden converges too.",
            '<div class="grid">'
            + svg_card(out("newton_nd.svg"), "the residual norm plunging near-vertically for Newton (quadratic convergence) and a few steps slower for Broyden, on a log scale")
            + f'<div class="card">{pre(newton_nd_txt)}</div>'
            + '</div>'),
        section(
            "Differential evolution: optimization by vector differences",
            "A population optimizer for continuous spaces that, unlike genetic algorithms (mutating "
            "bits) or particle swarms (tracking velocities), mutates by ADDING SCALED DIFFERENCES "
            "between population members. That self-referential step is its signature: the spread of "
            "the population itself sets the mutation scale, so the search takes large steps while "
            "dispersed (early, exploring) and small ones as it converges (late, refining) -- no "
            "cooling schedule or velocity tuning. The classic DE/rand/1/bin per target: pick three "
            "other members and form a donor v = a + F*(b - c); build a trial by taking each "
            "coordinate from the donor with probability CR (else from the target); keep whichever "
            "of trial and target has the lower cost (greedy selection, so the best never worsens). "
            "This module implements DE/rand/1/bin over a bounded box with bound reflection and a "
            "random-search baseline, verified that it finds the global minimum of the Sphere, "
            "Rastrigin, and Rosenbrock benchmarks, that the best cost is monotone, that it beats "
            "random search at equal budget, that the solution stays in bounds, and that it scales "
            "to 10-20 dimensions. One of the most robust black-box optimizers for continuous "
            "problems.",
            '<div class="grid">'
            + svg_card(out("differential_evolution.svg"), "the best cost of all three benchmarks falling monotonically to their global minima on a log scale")
            + f'<div class="card">{pre(differential_evolution_txt)}</div>'
            + '</div>'),
        section(
            "Nelder-Mead: derivative-free optimization by a crawling simplex",
            "Newton needs a Jacobian and gradient descent a gradient; Nelder-Mead needs NEITHER. It "
            "minimizes using only function VALUES, maintaining a simplex of n+1 points (a triangle "
            "in 2-D, a tetrahedron in 3-D) that tumbles and shrinks downhill -- the workhorse behind "
            "'just minimize this black box' (SciPy's gradient-free default, MATLAB's fminsearch). "
            "Each iteration reflects the worst vertex through the centroid of the others and, by how "
            "good that reflection is, EXPANDS further in a promising direction, CONTRACTS back "
            "toward the centroid, or SHRINKS the whole simplex toward the best vertex. The simplex "
            "crawls like an amoeba, stretching down valleys and squeezing through the curved "
            "Rosenbrock banana. This module implements the standard algorithm with the classic "
            "coefficients, value- and size-based convergence, and restarts, verified that it finds "
            "the minimum of the Sphere, Rosenbrock, and Beale benchmarks from several starts with "
            "no gradient, that the best vertex improves monotonically, that it even minimizes a "
            "non-smooth objective, and that restarting refines the result.",
            '<div class="grid">'
            + svg_card(out("nelder_mead.svg"), "the best-vertex value plunging as the simplex crawls the Rosenbrock valley, on a log scale, using function values alone")
            + f'<div class="card">{pre(nelder_mead_txt)}</div>'
            + '</div>'),
        section(
            "HITS: hubs and authorities",
            "PageRank's contemporary rival, and it splits importance into TWO complementary scores. "
            "An AUTHORITY is a page many good hubs point to (a definitive source); a HUB is a page "
            "that points to many good authorities (a good list of links). The definitions are "
            "mutually recursive -- a good authority is linked by good hubs, a good hub links to "
            "good authorities -- resolved by iterating to a fixed point: authority(p) = sum of hub "
            "scores linking to p, hub(p) = sum of authority scores p links to, normalized each "
            "round. Written with the adjacency matrix A the update is a = A'h, h = Aa, so "
            "authorities are the dominant eigenvector of A'A and hubs of AA' -- HITS is power "
            "iteration on those matrices. Unlike PageRank's single query-independent score, HITS "
            "yields the two roles separately, so a curated link list and the source everyone cites "
            "rank differently. This module computes HITS by power iteration with normalization "
            "plus the eigenvector check, verified that on a hub-and-spoke graph the hub score "
            "flags the linker and the authority score the linked-to targets, that scores converge "
            "and are unit-normalized, that a pure authority has zero hub score and vice versa, and "
            "that the results match the dominant eigenvectors of A'A and AA'.",
            '<div class="grid">'
            + svg_card(out("hits.svg"), "the same web graph drawn twice, node size by hub score (link lists) then by authority score (cited sources), showing the two roles fall on different nodes")
            + f'<div class="card">{pre(hits_txt)}</div>'
            + '</div>'),
        section(
            "Classical MDS: a map from a table of distances",
            "Given only the pairwise DISTANCES between objects -- cities on a map, dissimilar survey "
            "responses, aligned sequences -- multidimensional scaling reconstructs COORDINATES whose "
            "distances match, answering 'where do these sit relative to each other?' from distances "
            "alone. Classical (Torgerson) MDS solves it in closed form by DOUBLE CENTERING: from the "
            "squared-distance matrix, B = -1/2 J D2 J turns distances into the centered "
            "inner-product (Gram) matrix B = X X', and eigendecomposing B = V L V' gives the "
            "coordinates X = V L^{1/2} -- the top k eigenvectors scaled by the square roots of "
            "their eigenvalues, with the eigenvalues themselves reporting how much shape each "
            "dimension carries. The map is unique only up to rotation, reflection, and translation. "
            "This module builds the squared-distance and double-centered matrices, extracts the "
            "embedding by eigendecomposition, reports the eigenvalue spectrum, and Procrustes-aligns "
            "a reconstruction to a known map, verified that it recovers a square, a line, and random "
            "point sets so their reconstructed distances match, that a flat configuration has "
            "exactly two positive eigenvalues, and that the stress is essentially zero for Euclidean "
            "inputs. Built on the eigen module.",
            '<div class="grid">'
            + svg_card(out("mds.svg"), "true city positions and the MDS reconstruction Procrustes-aligned on top of them, beside the eigenvalue scree showing two positive dimensions")
            + f'<div class="card">{pre(mds_txt)}</div>'
            + '</div>'),
        section(
            "Skip lists: a probabilistic ordered dictionary",
            "A balanced binary search tree gives O(log n) search, insert, and delete but needs "
            "intricate rotations; a SKIP LIST reaches the same expected bounds with almost no "
            "bookkeeping, using RANDOMNESS instead. It is an ordered linked list with EXPRESS "
            "LANES: each node is promoted to the next level up with probability p (~1/2), so level "
            "0 holds every element, level 1 about half, level 2 a quarter -- a tower of sparser and "
            "sparser shortcut lists. A search drops down from the top lane, skipping far along each "
            "level before descending, covering the list in O(log n) expected hops. Because "
            "promotion is a coin flip there are no rotations: insert picks a random height and "
            "splices in, delete unlinks, and the structure stays probabilistically balanced on its "
            "own (Redis sorted sets use it). This module implements a skip-list ordered map with "
            "insert, search, delete, ordered iteration, range queries, and min/max, verified "
            "against a brute-force sorted dictionary over 4000 random operations (every search, "
            "deletion, and traversal agrees), that keys iterate sorted, that duplicates update "
            "rather than duplicate, that range queries return exactly the in-range keys, and that "
            "the level distribution is geometric as designed.",
            '<div class="grid">'
            + svg_card(out("skiplist.svg"), "the skip list drawn as stacked express lanes, taller towers skipping more keys, over the fully-populated level-0 sorted list")
            + f'<div class="card">{pre(skiplist_txt)}</div>'
            + '</div>'),
        section(
            "Ant colony optimization: pheromone trails for the travelling salesman",
            "Real ants find short paths without a map: each lays a PHEROMONE trail, shorter paths "
            "get traversed sooner so their trails are reinforced first, and other ants prefer "
            "strongly-scented edges -- positive feedback that converges the colony onto good routes. "
            "Ant colony optimization turns that into a TSP solver. Each iteration a swarm of "
            "artificial ants each builds a tour, at every step choosing the next city with "
            "probability proportional to tau^alpha * eta^beta, where tau is the learned pheromone on "
            "an edge and eta = 1/distance is the greedy heuristic (alpha weights experience, beta "
            "greed). Then pheromone EVAPORATES (forgetting stale trails) and each ant DEPOSITS an "
            "amount inversely proportional to its tour length, so shorter tours leave stronger "
            "trails and the pheromone concentrates on good edges. This module implements ant system "
            "for the symmetric TSP with the standard transition rule, evaporation, length-weighted "
            "deposit, and elitist reinforcement, verified that it recovers the optimal perimeter of "
            "a square and a circle's polygon, beats the nearest-neighbour greedy tour on random "
            "cities, drives the best length down monotonically, and concentrates pheromone on short "
            "edges.",
            '<div class="grid">'
            + svg_card(out("ant_colony.svg"), "the converged tour drawn bold over the pheromone field (strong edges glowing yellow), beside the best-tour-length convergence curve")
            + f'<div class="card">{pre(ant_colony_txt)}</div>'
            + '</div>'),
        section(
            "AVL trees: a self-balancing binary search tree",
            "A plain binary search tree degrades to a linked list -- O(n) operations -- if keys "
            "arrive sorted. An AVL tree (the first self-balancing BST) prevents that by keeping "
            "every node HEIGHT-BALANCED: its two subtrees' heights differ by at most 1. After each "
            "insert or delete it checks the balance factor up the path to the root and, wherever it "
            "exceeds the bound, restores it with a local ROTATION -- a constant-time pointer "
            "rewiring that shortens the tall side. Four cases cover every imbalance: left-left and "
            "right-right take a single rotation, left-right and right-left a double. The strict "
            "invariant makes AVL the most rigidly balanced classic BST (shorter than a red-black "
            "tree), so lookups are fast at the cost of a little more rotation on updates -- where a "
            "skip list stays balanced probabilistically, an AVL tree does so deterministically. "
            "This module implements an AVL ordered map with insert, delete, search, ordered "
            "traversal, range queries, and min/max, verified against a brute-force sorted "
            "dictionary over 5000 random operations, that the height-balance invariant holds "
            "throughout, that the height stays O(log n) even for sorted insertions (where a naive "
            "BST would be linear), and that all four rotation cases trigger.",
            '<div class="grid">'
            + svg_card(out("avl_tree.svg"), "a balanced AVL tree drawn in-order left-to-right and by depth top-to-bottom, staying log-deep regardless of insertion order")
            + f'<div class="card">{pre(avl_tree_txt)}</div>'
            + '</div>'),
        section(
            "Segment trees with lazy propagation: range query and range update",
            "A Fenwick tree answers prefix sums with point updates; a SEGMENT TREE handles arbitrary "
            "RANGE queries (sum, min, max, ...) AND RANGE updates -- add a value to every element in "
            "[l, r] -- all in O(log n). Each node stores the aggregate of a contiguous segment; the "
            "root covers everything and each node splits its range in half between two children, so "
            "a query descends only into the O(log n) nodes whose segments tile the range. The trick "
            "for range UPDATES is LAZY PROPAGATION: rather than touch every leaf (O(n)), a node "
            "records a pending update as a lazy tag, applies it to itself immediately, and pushes it "
            "down to its children only when a later operation actually visits them -- so a full-array "
            "update touches ~2 log n nodes, not n. This module implements a segment tree "
            "parameterized by the aggregate (sum, min, or max) with lazy range-add updates, point "
            "updates, and range queries, verified against a brute-force array over 3000 random mixed "
            "operations for all three aggregates, that overlapping range-adds accumulate correctly, "
            "that point updates match a plain list, and that it handles single-element and "
            "full-array edge ranges.",
            '<div class="grid">'
            + svg_card(out("segment_tree.svg"), "the segment tree drawn as a binary tree of segment sums, the root covering the whole array and each level halving the range")
            + f'<div class="card">{pre(segment_tree_txt)}</div>'
            + '</div>'),
        section(
            "Convex hull: the tightest polygon enclosing points",
            "The convex hull is the smallest convex polygon containing a point set -- the shape a "
            "rubber band snaps to around a scatter of pins, and the foundation of computational "
            "geometry (collision detection, shape analysis, path-planning boundaries). This module "
            "builds it by ANDREW'S MONOTONE CHAIN in O(n log n): sort the points, then sweep "
            "left-to-right building the lower hull and right-to-left the upper, keeping only left "
            "turns. The engine is the CROSS PRODUCT, whose sign gives orientation -- (b-a) x (c-a) > "
            "0 is a counter-clockwise turn -- so the chain pops any vertex that would make a "
            "non-left turn, leaving only the outer boundary. From the hull come the enclosed AREA "
            "(shoelace formula), perimeter, whether an arbitrary point lies inside (orientation "
            "tests against each edge), and the DIAMETER (farthest pair, which always lies on the "
            "hull). This module implements the hull, area, perimeter, point-in-hull, and diameter, "
            "verified that a square's hull is its four corners (interior points dropped), that "
            "collinear and duplicate points are handled, that the hull is convex and "
            "counter-clockwise, that its area matches an independent shoelace value, that every "
            "input point lies inside it, and that the diameter is the true farthest pair.",
            '<div class="grid">'
            + svg_card(out("convex_hull.svg"), "a scatter of points with the convex-hull boundary outlined in blue, interior points grey, and the diameter (farthest pair) marked with a dashed line")
            + f'<div class="card">{pre(convex_hull_txt)}</div>'
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
