"""Demo: the Sunyaev-Zeldovich effect -- clusters shadowing the CMB.

Computes the Compton y-parameter and Rayleigh-Jeans temperature decrement for
clusters of increasing mass, and renders the decrement vs y to SVG. Highlights
that the signal is redshift-independent.

    python examples/sz_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sz import (y_from_kev, rj_temperature_decrement, rj_decrement_kelvin,  # noqa: E402
                is_redshift_independent, MPC)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("The Sunyaev-Zeldovich effect: hot clusters distorting the CMB\n")
    print(f"  {'cluster':<16}{'n_e (/m^3)':>12}{'kT (keV)':>10}{'y':>12}{'dT (uK)':>12}")
    print("  " + "-" * 62)
    # (name, n_e, kT keV, path Mpc)
    cases = [("group", 3e2, 2.0, 0.5), ("Coma-like", 1e3, 8.0, 1.0),
             ("massive", 3e3, 12.0, 2.0)]
    curve = []
    for name, n_e, kT, Lmpc in cases:
        y = y_from_kev(n_e, kT, Lmpc * MPC)
        dT = rj_decrement_kelvin(y) * 1e6
        curve.append((y, dT))
        print(f"  {name:<16}{n_e:>12.0e}{kT:>10.1f}{y:>12.2e}{dT:>12.1f}")
    print(f"\n  In the Rayleigh-Jeans band a cluster is a COLD spot: dT/T = -2y.")
    print(f"  redshift-independent? {is_redshift_independent()} -- the SZ signal does not dim")
    print(f"  with distance, so SZ surveys find clusters clear across the universe.")

    _svg(os.path.join(outdir, "sz.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'sz.svg')}")


def _svg(path, size=720, pad=64):
    # decrement magnitude vs y (linear -> straight line, illustrating dT = -2 y T)
    ys = [1e-6 + (2e-4 - 1e-6) * i / 100 for i in range(101)]
    dTs = [abs(rj_decrement_kelvin(y)) * 1e6 for y in ys]  # uK
    ymax = ys[-1]
    dmax = max(dTs) * 1.1

    def sx(y):
        return pad + y / ymax * (size - 2 * pad)

    def sy(d):
        return size - pad - d / dmax * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(ys[i]):.1f},{sy(dTs[i]):.1f}" for i in range(len(ys)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#8338ec" stroke-width="2.2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'SZ Rayleigh-Jeans decrement vs Compton y</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'dT = -2 y T_CMB: a cluster is a cold spot, redshift-independent</text>')
    parts.append(f'<text x="{pad}" y="{size-pad+24}" fill="#8b949e" font-size="11">'
                 f'Compton y-parameter -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'|dT| (microkelvin)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
