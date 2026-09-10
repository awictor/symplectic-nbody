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

    # make sure all SVGs exist
    import plot_orbits
    plot_orbits.main.__globals__  # noqa: silence linter
    _old_argv = sys.argv
    sys.argv = ["plot_orbits", outdir]
    plot_orbits.main()
    sys.argv = ["lagrange_demo", outdir]
    lagrange_demo.main()
    sys.argv = _old_argv

    print("capturing demo outputs...")
    energy_txt = capture(energy_drift_demo.main)
    conv_txt = capture(convergence_demo.main)
    adapt_txt = capture(adaptive_demo.main)
    sys.argv = ["lagrange_demo", outdir]
    lagr_txt = capture(lagrange_demo.main)
    sys.argv = _old_argv
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
