"""Demo: Wang-Landau sampling -- one run gives the density of states, hence every temperature.

Estimates the density of states g(E) of a small Ising lattice by Wang-Landau flat-histogram sampling,
compares the recovered ln g to the exact brute-force enumeration, and then -- from that SINGLE run --
computes the specific-heat curve across all temperatures, showing the peak near the critical point.

    python examples/wang_landau_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wang_landau import (  # noqa: E402
    wang_landau,
    exact_density_of_states,
    thermodynamics,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Wang-Landau: measure the density of states once, get all temperatures\n")

    n = 4
    print(f"  {n}x{n} Ising ferromagnet ({n * n} spins, 2^{n * n} = {1 << (n * n)} configurations)\n")

    ex_e, ex_lng, ex_counts = exact_density_of_states(n)
    wl_e, wl_lng = wang_landau(n, seed=3, flat=0.8, f_final=1e-5, max_sweeps=250000)

    print(f"  Recovered ln g(E) vs exact enumeration:")
    print(f"    {'E':>6}{'exact count':>14}{'exact ln g':>13}{'WL ln g':>11}")
    for i in range(len(ex_e)):
        if ex_lng[i] != float("-inf"):
            print(f"    {ex_e[i]:>6}{ex_counts[i]:>14}{ex_lng[i]:>13.3f}{wl_lng[i]:>11.3f}")

    # specific heat curve from the single WL run and from exact g, for comparison
    temps = [0.5 + 0.25 * k for k in range(0, 22)]
    curve_wl = []
    curve_ex = []
    for T in temps:
        _, c_wl = thermodynamics(wl_e, wl_lng, T)
        _, c_ex = thermodynamics(ex_e, ex_lng, T)
        curve_wl.append((T, c_wl / (n * n)))    # per spin
        curve_ex.append((T, c_ex / (n * n)))

    peak_T = max(curve_ex, key=lambda tc: tc[1])[0]
    print(f"\n  Specific heat per spin peaks near T = {peak_T:.2f}")
    print(f"  (2D Ising T_c -> 2.269 in the thermodynamic limit; small lattices peak a bit high).")
    print(f"  All of this thermodynamics came from ONE Wang-Landau run -- no re-simulation per T.")

    _svg(os.path.join(outdir, "wang_landau.svg"), curve_wl, curve_ex, peak_T)
    print(f"\n  wrote {os.path.join(outdir, 'wang_landau.svg')}")


def _svg(path, curve_wl, curve_ex, peak_T, width=760, height=400):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Specific heat per spin from one Wang-Landau run (dots) vs exact g(E) (line)</text>',
    ]
    ox, oy, ow, oh = 60, 55, width - 110, height - 110
    tmin = min(t for t, _ in curve_ex)
    tmax = max(t for t, _ in curve_ex)
    cmax = max(max(c for _, c in curve_ex), max(c for _, c in curve_wl)) * 1.1

    def px(t):
        return ox + ow * (t - tmin) / (tmax - tmin)

    def py(c):
        return oy + oh * (1 - c / cmax)

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    # peak marker
    parts.append(f'<line x1="{px(peak_T):.1f}" y1="{oy}" x2="{px(peak_T):.1f}" y2="{oy+oh}" '
                 f'stroke="#ff6b6b" stroke-width="1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{px(peak_T)+4:.0f}" y="{oy+14}" fill="#ff6b6b" font-size="10">'
                 f'peak T~{peak_T:.2f}</text>')
    # exact line
    exline = " ".join(f"{px(t):.1f},{py(c):.1f}" for t, c in curve_ex)
    parts.append(f'<polyline points="{exline}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    # WL dots
    for t, c in curve_wl:
        parts.append(f'<circle cx="{px(t):.1f}" cy="{py(c):.1f}" r="3" fill="#4dabf7"/>')

    parts.append(f'<rect x="{ox+ow-150}" y="{oy+6}" width="12" height="4" fill="#06d6a0"/>')
    parts.append(f'<text x="{ox+ow-134}" y="{oy+11}" fill="#e6edf3" font-size="10">exact g(E)</text>')
    parts.append(f'<circle cx="{ox+ow-144}" cy="{oy+24}" r="3" fill="#4dabf7"/>')
    parts.append(f'<text x="{ox+ow-134}" y="{oy+27}" fill="#e6edf3" font-size="10">Wang-Landau</text>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+24:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">temperature T (units of J/k_B)</text>')
    parts.append(f'<text x="{ox-40}" y="{oy+oh/2:.0f}" fill="#8b949e" font-size="10" '
                 f'transform="rotate(-90 {ox-40} {oy+oh/2:.0f})" text-anchor="middle">'
                 f'specific heat per spin</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
