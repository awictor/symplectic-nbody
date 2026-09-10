"""Demo: photon-photon pair production and the gamma-ray horizon.

Shows the 511 keV head-on threshold, the partner-energy threshold vs background
photon energy, and which gamma rays are absorbed by the CMB / infrared / optical
backgrounds. Renders the threshold gamma-ray energy vs background photon energy
to a log-log SVG.

    python examples/pair_production_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pair_production import (electron_rest_energy_kev, head_on_threshold,  # noqa: E402
                             partner_threshold_energy, gamma_ray_horizon_case,
                             KEV, EV, GEV, TEV)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Photon-photon pair production: gamma + gamma -> e+ e-\n")
    print(f"  electron rest energy: {electron_rest_energy_kev():.0f} keV")
    print(f"  head-on threshold (each photon): {head_on_threshold()/KEV:.0f} keV\n")
    print(f"  {'background photon':>20}{'E (eV)':>10}{'threshold gamma':>18}")
    print("  " + "-" * 48)
    for name, eV in (("CMB", 6e-4), ("infrared (EBL)", 0.1),
                     ("optical (EBL)", 2.0), ("X-ray", 1e3)):
        Eg = partner_threshold_energy(eV * EV)
        if Eg > TEV:
            estr = f"{Eg/TEV:.0f} TeV"
        elif Eg > GEV:
            estr = f"{Eg/GEV:.0f} GeV"
        else:
            estr = f"{Eg/1e6/EV:.0f} MeV"
        print(f"  {name:>20}{eV:>10.1e}{estr:>18}")
    print("\n  A gamma ray is absorbed once its energy exceeds (m_e c^2)^2 / E_bg,")
    print("  so TeV photons from distant blazars are eaten by starlight/IR and")
    print("  PeV photons by the CMB -- the universe has a gamma-ray horizon that")
    print("  shrinks as the photon energy rises.")

    _svg(os.path.join(outdir, "pair_production.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'pair_production.svg')}")


def _svg(path, size=720, pad=64):
    # threshold gamma energy (GeV) vs background photon energy (eV), log-log
    bgs = [10 ** (-4 + 0.08 * i) for i in range(0, 101)]  # 1e-4 .. 1e4 eV
    Egs = [partner_threshold_energy(b * EV) / GEV for b in bgs]
    lb = [math.log10(b) for b in bgs]
    le = [math.log10(E) for E in Egs]
    bmin, bmax = lb[0], lb[-1]
    emin, emax = min(le), max(le)

    def sx(x):
        return pad + (x - bmin) / (bmax - bmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - emin) / (emax - emin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    poly = " ".join(f"{sx(lb[i]):.1f},{sy(le[i]):.1f}" for i in range(len(bgs)))
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#8338ec" stroke-width="2.2"/>')
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Gamma-ray horizon: pair-production threshold</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'E_gamma = (m_e c^2)^2 / E_background: high-E gammas absorbed by low-E fields</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 background photon energy (eV) -&gt;</text>')
    parts.append(f'<text x="{pad-10}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 threshold gamma energy (GeV)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
