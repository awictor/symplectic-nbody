"""Demo: Adams-Bashforth-Moulton multistep ODE integration.

Integrates the harmonic oscillator with the 4th-order predictor-corrector, shows the measured
convergence order matches theory, and compares its accuracy and cost against plain Adams-Bashforth and
RK4. Draws the error-vs-step-size curves.

    python examples/adams_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adams import (  # noqa: E402
    rk4,
    adams_bashforth,
    predictor_corrector,
    max_error,
    convergence_order,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Adams-Bashforth-Moulton: multistep ODE integration, one new f-eval per step\n")

    # exponential decay for the clean convergence table
    f = lambda t, y: [-y[0]]
    exact = lambda t: [math.exp(-t)]

    print("  Solving y' = -y on [0, 3] (exact e^-t). Error vs number of steps:\n")
    print(f"  {'steps':>6}  {'AB4 error':>12}  {'PECE error':>12}  {'RK4 error':>12}")
    curve = []
    for n in [10, 20, 40, 80, 160]:
        tsa, ya = adams_bashforth(f, [1.0], 0, 3, n, order=4)
        tsp, yp = predictor_corrector(f, [1.0], 0, 3, n, order=4)
        tsr, yr = rk4(f, [1.0], 0, 3, n)
        ea = max_error(ya, tsa, exact)
        ep = max_error(yp, tsp, exact)
        er = max_error(yr, tsr, exact)
        curve.append((n, ea, ep, er))
        print(f"  {n:>6}  {ea:>12.2e}  {ep:>12.2e}  {er:>12.2e}")

    o = convergence_order(f, [1.0], 0, 3, exact, method=predictor_corrector, n_coarse=20, order=4)
    print(f"\n  Measured PECE convergence order: {o:.2f} (theory: 4).")
    print("  Each PECE step uses only 2 f-evaluations vs RK4's 4, at comparable accuracy -- the")
    print("  multistep payoff when the right-hand side f is expensive to evaluate.\n")

    # harmonic oscillator trajectory
    fho = lambda t, y: [y[1], -y[0]]
    ts, ys = predictor_corrector(fho, [1.0, 0.0], 0, 20, 400, order=4)
    energies = [0.5 * (y[0] ** 2 + y[1] ** 2) for y in ys]
    print(f"  Harmonic oscillator over 20 units: energy stays {min(energies):.5f}..{max(energies):.5f}")
    print("  (analytic 0.5), so the phase-space orbit barely drifts.")

    _svg(os.path.join(outdir, "adams.svg"), curve)
    print(f"\n  wrote {os.path.join(outdir, 'adams.svg')}")


def _svg(path, curve, width=760, height=410):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Error vs steps (log-log): all 4th-order methods fall as h^4 (slope -4)</text>',
    ]
    ox, oy, ow, oh = 60, 55, width - 110, height - 110
    ns = [c[0] for c in curve]
    lx = [math.log10(n) for n in ns]
    series = {
        "AB4": ("#ff922b", [math.log10(max(c[1], 1e-16)) for c in curve]),
        "PECE": ("#06d6a0", [math.log10(max(c[2], 1e-16)) for c in curve]),
        "RK4": ("#4dabf7", [math.log10(max(c[3], 1e-16)) for c in curve]),
    }
    allv = [v for _, ys in series.values() for v in ys]
    ymin, ymax = min(allv), max(allv)

    def px(v):
        return ox + ow * (v - lx[0]) / (lx[-1] - lx[0])

    def py(v):
        return oy + oh * (1 - (v - ymin) / (ymax - ymin or 1))

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    ly = oy + 14
    for name, (col, ys) in series.items():
        pts = " ".join(f"{px(lx[i]):.1f},{py(ys[i]):.1f}" for i in range(len(ns)))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        for i in range(len(ns)):
            parts.append(f'<circle cx="{px(lx[i]):.1f}" cy="{py(ys[i]):.1f}" r="2.5" fill="{col}"/>')
        parts.append(f'<text x="{ox+ow-70}" y="{ly}" fill="{col}" font-size="10">{name}</text>')
        ly += 15
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+22:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">log10 steps</text>')
    parts.append(f'<text x="{ox-6}" y="{oy-4}" fill="#8b949e" font-size="10" text-anchor="end">'
                 f'log10 error</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
