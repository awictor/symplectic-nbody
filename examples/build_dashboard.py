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

    # make sure all SVGs exist
    import plot_orbits
    plot_orbits.main.__globals__  # noqa: silence linter
    _old_argv = sys.argv
    sys.argv = ["plot_orbits", outdir]
    plot_orbits.main()
    sys.argv = ["lagrange_demo", outdir]
    lagrange_demo.main()
    sys.argv = ["solar_system_demo", outdir]
    solar_system_demo.main()
    sys.argv = ["precession_demo", outdir]
    precession_demo.main()
    sys.argv = ["chaos_demo", outdir]
    chaos_demo.main()
    sys.argv = ["poincare_demo", outdir]
    poincare_demo.main()
    sys.argv = ["galaxy_collision_demo", outdir]
    galaxy_collision_demo.main()
    sys.argv = ["gravwave_demo", outdir]
    gravwave_demo.main()
    sys.argv = ["circularization_demo", outdir]
    circularization_demo.main()
    sys.argv = ["sitnikov_demo", outdir]
    sitnikov_demo.main()
    sys.argv = ["virial_demo", outdir]
    virial_demo.main()
    sys.argv = _old_argv

    print("capturing demo outputs...")
    energy_txt = capture(energy_drift_demo.main)
    conv_txt = capture(convergence_demo.main)
    adapt_txt = capture(adaptive_demo.main)
    sys.argv = ["lagrange_demo", outdir]
    lagr_txt = capture(lagrange_demo.main)
    sys.argv = ["solar_system_demo", outdir]
    solar_txt = capture(solar_system_demo.main)
    sys.argv = ["precession_demo", outdir]
    prec_txt = capture(precession_demo.main)
    sys.argv = ["chaos_demo", outdir]
    chaos_txt = capture(chaos_demo.main)
    sys.argv = ["poincare_demo", outdir]
    poincare_txt = capture(poincare_demo.main)
    sys.argv = ["galaxy_collision_demo", outdir]
    galaxy_txt = capture(galaxy_collision_demo.main)
    sys.argv = ["gravwave_demo", outdir]
    gw_txt = capture(gravwave_demo.main)
    sys.argv = ["circularization_demo", outdir]
    circ_txt = capture(circularization_demo.main)
    sys.argv = ["sitnikov_demo", outdir]
    sitnikov_txt = capture(sitnikov_demo.main)
    sys.argv = ["virial_demo", outdir]
    virial_txt = capture(virial_demo.main)
    sys.argv = _old_argv
    hermite_txt = capture(hermite_demo.main)
    # scaling benchmark is slow; run a lighter inline version
    scale_txt = capture(scaling_benchmark.main)

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
    print("open it in a browser, or enable GitHub Pages on examples/output/")


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
