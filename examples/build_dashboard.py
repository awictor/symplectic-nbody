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
