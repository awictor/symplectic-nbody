"""Demo: Bulirsch-Stoer integrating a Kepler orbit to machine precision, order rising with each column.

Integrates a two-body Kepler orbit and the harmonic oscillator, showing near-perfect energy
conservation, and displays how a single Bulirsch-Stoer step's error plummets as more modified-midpoint
sweeps are extrapolated. Draws the Kepler orbit and the per-level error decay.

    python examples/bulirsch_stoer_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bulirsch_stoer import (  # noqa: E402
    bulirsch_stoer_step, solve_fixed, kepler_2d, kepler_energy, harmonic_oscillator, harmonic_energy,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Bulirsch-Stoer: modified midpoint + Richardson extrapolation for extreme ODE accuracy\n")

    # error of one BS step on y'=y vs number of extrapolation levels
    f_exp = lambda t, y: [y[0]]
    print(f"  one Bulirsch-Stoer step of y'=y over H=1 (true answer e = {math.e:.15f}):")
    print(f"    {'levels':>8}{'estimate':>20}{'error':>12}")
    for k in range(1, 9):
        est, _ = bulirsch_stoer_step(f_exp, 0, [1.0], 1.0, k_max=k)
        print(f"    {k:>8}{est[0]:>20.15f}{abs(est[0] - math.e):>12.1e}")
    print(f"  each added midpoint sweep jumps TWO error orders -- machine precision in one step.\n")

    # Kepler orbit: mildly eccentric
    kf = kepler_2d(1.0)
    # elliptical orbit: start at perihelion-ish with a slightly slow tangential speed
    y0 = [1.0, 0.0, 0.0, 0.9]
    period = 2 * math.pi  # approximate
    ts, ys = solve_fixed(kf, 0, y0, 4 * math.pi, steps=400)
    e0 = kepler_energy(y0)
    drift = max(abs(kepler_energy(y) - e0) for y in ys)
    print(f"  Kepler orbit (eccentric), two revolutions, 400 steps:")
    print(f"    energy drift: {drift:.2e}  (near machine precision -- no secular growth)")

    # harmonic energy over many periods
    hf = harmonic_oscillator(1.0)
    tsh, ysh = solve_fixed(hf, 0, [1.0, 0.0], 50 * math.pi, steps=500)
    hdrift = max(abs(harmonic_energy(y) - harmonic_energy([1.0, 0.0])) for y in ysh)
    print(f"    harmonic oscillator, 25 periods: energy drift {hdrift:.2e}\n")

    print(f"  Bulirsch-Stoer is the method of choice for smooth, high-precision trajectory problems")
    print(f"  -- ephemerides, celestial mechanics -- where a few big extrapolated steps beat many")
    print(f"  small Runge-Kutta ones.")

    _svg(os.path.join(outdir, "bulirsch_stoer.svg"), ys, f_exp)
    print(f"\n  wrote {os.path.join(outdir, 'bulirsch_stoer.svg')}")


def _svg(path, orbit, f_exp, width=760, height=420):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="24" fill="#e6edf3" font-size="14">'
        f'Bulirsch-Stoer: a Kepler orbit (left) and one-step error vs extrapolation levels (right)</text>',
    ]
    # left: orbit
    xs = [s[0] for s in orbit]
    ys = [s[1] for s in orbit]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    span = max(maxx - minx, maxy - miny) * 1.1
    cx0 = (minx + maxx) / 2
    cy0 = (miny + maxy) / 2
    lox, loy, lw = 40, 50, 320

    def sx(x):
        return lox + lw / 2 + lw * (x - cx0) / span

    def sy(y):
        return loy + lw / 2 - lw * (y - cy0) / span

    parts.append(f'<rect x="{lox}" y="{loy}" width="{lw}" height="{lw}" fill="none" stroke="#30363d"/>')
    # central mass
    parts.append(f'<circle cx="{sx(0):.1f}" cy="{sy(0):.1f}" r="5" fill="#ffd43b"/>')
    pts = " ".join(f"{sx(orbit[i][0]):.1f},{sy(orbit[i][1]):.1f}" for i in range(len(orbit)))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="1.5"/>')
    parts.append(f'<text x="{lox+lw/2:.0f}" y="{loy+lw+16:.0f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="middle">two revolutions, energy conserved</text>')

    # right: error vs levels (log)
    box, boy, bw, bh = 420, 60, width - 460, 280
    errs = []
    for k in range(1, 9):
        est, _ = bulirsch_stoer_step(f_exp, 0, [1.0], 1.0, k_max=k)
        errs.append(max(abs(est[0] - math.e), 1e-17))
    lo = math.log10(min(errs))
    hi = math.log10(max(errs))

    def px(i):
        return box + bw * i / (len(errs) - 1)

    def py(e):
        return boy + bh * (1 - (math.log10(e) - lo) / (hi - lo + 1e-12))

    parts.append(f'<rect x="{box}" y="{boy}" width="{bw}" height="{bh}" fill="none" stroke="#30363d"/>')
    d = math.floor(lo)
    while d <= hi:
        y = py(10 ** d)
        parts.append(f'<line x1="{box}" y1="{y:.1f}" x2="{box+bw}" y2="{y:.1f}" stroke="#161b22"/>')
        parts.append(f'<text x="{box-6}" y="{y+3:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="end">1e{int(d)}</text>')
        d += 3
    pts = " ".join(f"{px(i):.1f},{py(errs[i]):.1f}" for i in range(len(errs)))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    for i in range(len(errs)):
        parts.append(f'<circle cx="{px(i):.1f}" cy="{py(errs[i]):.1f}" r="3" fill="#ffd43b"/>')
    parts.append(f'<text x="{box+bw/2:.0f}" y="{boy+bh+18:.0f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="middle">extrapolation levels (one step of the exponential ODE)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
