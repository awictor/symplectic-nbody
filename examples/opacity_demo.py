"""Demo: stellar opacity across the temperature range of a star.

Prints opacity contributions at representative interior points, then draws opacity
vs temperature at fixed density, showing the T^(-7/2) Kramers fall-off giving way to
the flat electron-scattering floor at high temperature.

    python examples/opacity_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from opacity import (electron_scattering, kramers_bound_free,  # noqa: E402
                     kramers_free_free, total_opacity, mean_free_path)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Stellar opacity: how slowly light escapes matter (kappa, m^2/kg)\n")
    print(f"  electron scattering floor: {electron_scattering():.4f} m^2/kg "
          f"(T- and rho-independent)\n")
    print(f"  {'region':>16}{'rho':>10}{'T (K)':>10}{'kappa_es':>10}"
          f"{'Kramers':>10}{'total':>9}")
    print("  " + "-" * 65)
    regions = [
        ("solar centre", 1.5e5, 1.5e7),
        ("radiative zone", 2e4, 5e6),
        ("near surface", 1e-3, 1e5),
        ("photosphere", 1e-4, 6e3),
    ]
    for name, rho, T in regions:
        kes = electron_scattering()
        kkr = kramers_bound_free(rho, T) + kramers_free_free(rho, T)
        kt = total_opacity(rho, T)
        print(f"  {name:>16}{rho:>10.1e}{T:>10.1e}{kes:>10.3f}{kkr:>10.3f}{kt:>9.3f}")

    print("\n  Kramers opacity ~ rho T^(-7/2): the cool outer layers are far more")
    print("  opaque than the blazing core, which is why energy switches from")
    print("  radiative diffusion to convection in stellar envelopes. In the hot,")
    print("  dilute deep interior everything drops to the electron-scattering floor")
    print("  -- the same opacity the Eddington luminosity is built on.")
    mfp = mean_free_path(1.5e5, total_opacity(1.5e5, 1.5e7))
    print(f"\n  photon mean free path at the solar centre: {mfp*1e6:.1f} microns")
    print("  -- a photon random-walks out over ~100,000 years.")

    _svg(os.path.join(outdir, "opacity.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'opacity.svg')}")


def _svg(path, size=720, pad=68, rho=1e2):
    T = [10 ** (4.0 + 0.05 * i) for i in range(0, 101)]   # 1e4 .. 1e9 K
    kes = electron_scattering()
    ktot = [total_opacity(rho, t) for t in T]
    kkr = [kramers_bound_free(rho, t) + kramers_free_free(rho, t) for t in T]

    lx = [math.log10(t) for t in T]
    all_y = ktot + [kes]
    ly_all = [math.log10(y) for y in all_y if y > 0]
    xmin, xmax = lx[0], lx[-1]
    ymin, ymax = min(ly_all), max(ly_all)

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    # electron-scattering floor
    yf = sy(math.log10(kes))
    parts.append(f'<line x1="{pad}" y1="{yf:.1f}" x2="{size-pad}" y2="{yf:.1f}" '
                 f'stroke="#8b949e" stroke-width="1.3" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{pad+8}" y="{yf-6:.1f}" fill="#8b949e" '
                 f'font-size="11">electron-scattering floor</text>')

    # Kramers-only curve (dimmer)
    kkr_c = [math.log10(y) if y > 0 else ymin for y in kkr]
    polyk = " ".join(f"{sx(lx[i]):.1f},{sy(min(max(kkr_c[i], ymin), ymax)):.1f}"
                     for i in range(len(T)))
    parts.append(f'<polyline points="{polyk}" fill="none" stroke="#f06595" '
                 f'stroke-width="1.6" stroke-dasharray="3 3"/>')

    # total opacity curve
    poly = " ".join(f"{sx(lx[i]):.1f},{sy(math.log10(ktot[i])):.1f}" for i in range(len(T)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#ffd43b" stroke-width="2.6"/>')

    parts.append(f'<text x="{pad+10}" y="{pad+22}" fill="#ffd43b" font-size="12">total kappa</text>')
    parts.append(f'<text x="{pad+10}" y="{pad+38}" fill="#f06595" font-size="12">Kramers (~rho T^-3.5)</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Stellar opacity vs temperature</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'Kramers T^-3.5 fall-off flattening onto the electron-scattering floor</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 temperature (K) -&gt;</text>')
    parts.append(f'<text x="{pad-14}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 opacity (m^2/kg)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
