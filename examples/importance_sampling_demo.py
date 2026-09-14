"""Importance sampling demo: rare-tail estimation and the proposal-shift that makes it work (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import importance_sampling as IS


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
RED = "#ff6b6b"
YELLOW = "#ffd43b"


def main(outdir=None):
    lines = []
    lines.append("Importance sampling: estimating rare-event probabilities")
    lines.append("=" * 58)
    lines.append("target P(Z > t) for a standard normal, n = 20000 samples")
    lines.append("")
    lines.append(f"{'t':>5}{'exact':>13}{'naive MC':>13}{'IS':>13}{'IS ESS':>9}")
    for t in (2.0, 3.0, 4.0, 5.0, 6.0):
        exact = IS.normal_tail_exact(t)
        naive = IS.naive_tail_probability(t, n=20000, seed=1)
        ise, istd, ess = IS.is_tail_probability(t, n=20000, seed=1)
        lines.append(f"{t:>5.1f}{exact:>13.3e}{naive:>13.3e}{ise:>13.3e}{ess:>9.0f}")
    lines.append("")
    lines.append("Naive Monte Carlo returns exactly 0 once the event is rarer than ~1/n:")
    lines.append("no sample ever lands in the tail. Importance sampling shifts the proposal")
    lines.append("to N(t, 1) so half the samples DO land past t, then reweights by p/q -- and")
    lines.append("stays accurate to many significant figures even at P ~ 1e-9.")
    lines.append("")
    # variance-reduction comparison at a moderate tail
    t = 3.0
    exact = IS.normal_tail_exact(t)
    is_e = [IS.is_tail_probability(t, n=2000, seed=s)[0] for s in range(1, 51)]
    nv_e = [IS.naive_tail_probability(t, n=2000, seed=s) for s in range(1, 51)]

    def stats(a):
        m = sum(a) / len(a)
        v = sum((x - m) ** 2 for x in a) / len(a)
        return m, math.sqrt(v)

    im, isd = stats(is_e)
    nm, nsd = stats(nv_e)
    lines.append(f"50 repeats at t=3, n=2000 each (exact {exact:.3e}):")
    lines.append(f"  naive MC:  mean {nm:.3e}  std {nsd:.3e}")
    lines.append(f"  IS:        mean {im:.3e}  std {isd:.3e}")
    lines.append(f"  variance reduction factor: {(nsd/isd)**2:.0f}x")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        W, H = 720, 380
        ml, mt, w, h = 55, 55, 610, 250
        lo, hi = -4.0, 8.0
        t = 4.0
        shift = 4.0

        def pdf(x, mu, sig):
            return IS._normal_pdf(x, mu, sig)

        pmax = pdf(0, 0, 1)

        def sx(x):
            return ml + (x - lo) / (hi - lo) * w

        def sy(d):
            return mt + h - d / pmax * h

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="30" fill="{TEXT}" font-size="15">'
                 f'Importance sampling: shift the proposal into the rare tail</text>')
        s.append(f'<line x1="{ml}" y1="{mt+h}" x2="{ml+w}" y2="{mt+h}" stroke="{GRAY}"/>')
        # target p = N(0,1)
        cp = " ".join(f"{sx(lo+i/300*(hi-lo)):.1f},{sy(pdf(lo+i/300*(hi-lo),0,1)):.1f}" for i in range(301))
        s.append(f'<polyline points="{cp}" fill="none" stroke="{BLUE}" stroke-width="2"/>')
        # proposal q = N(shift,1)
        cq = " ".join(f"{sx(lo+i/300*(hi-lo)):.1f},{sy(pdf(lo+i/300*(hi-lo),shift,1)):.1f}" for i in range(301))
        s.append(f'<polyline points="{cq}" fill="none" stroke="{GREEN}" stroke-width="2"/>')
        # target tail region x>t shaded
        s.append(f'<line x1="{sx(t):.1f}" y1="{mt}" x2="{sx(t):.1f}" y2="{mt+h}" '
                 f'stroke="{RED}" stroke-width="1.5" stroke-dasharray="4,3"/>')
        s.append(f'<text x="{sx(t)+4:.1f}" y="{mt+16}" fill="{RED}" font-size="11">t = {t:g} (rare tail)</text>')
        s.append(f'<text x="{ml+w-210}" y="{mt+20}" fill="{BLUE}" font-size="11">target p = N(0,1)</text>')
        s.append(f'<text x="{ml+w-210}" y="{mt+36}" fill="{GREEN}" font-size="11">proposal q = N(t,1)</text>')
        s.append(f'<text x="{ml}" y="{H-14}" fill="{GRAY}" font-size="10">'
                 f'Under p (blue) almost no samples pass t; under q (green) about half do. '
                 f'Reweighting by p/q recovers the true tail probability with tiny variance.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "importance_sampling.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
