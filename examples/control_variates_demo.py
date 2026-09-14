"""Control variates demo: variance reduction vs correlation, and the estimator-spread comparison (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import control_variates as CV


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
RED = "#ff6b6b"
YELLOW = "#ffd43b"


def main(outdir=None):
    true = math.e - 1

    lines = []
    lines.append("Control variates: variance reduction for free")
    lines.append("=" * 52)
    lines.append("estimate E[e^U], U ~ Uniform(0,1);  true value = e - 1 = "
                 f"{true:.6f}")
    lines.append("control g = U (known mean 0.5), correlated with f = e^U")
    lines.append("")
    rng = CV._Rng(1)
    n = 40000
    U = [rng.uniform() for _ in range(n)]
    f = [math.exp(u) for u in U]
    est, se, red = CV.estimate(f, U, 0.5)
    rho = CV.correlation(f, U)
    lines.append(f"correlation rho(f, g):      {rho:.4f}")
    lines.append(f"optimal coefficient c*:     {CV.optimal_coefficient(f, U):.4f}")
    lines.append(f"plain MC estimate:          {CV._mean(f):.6f}")
    lines.append(f"control-variate estimate:   {est:.6f}")
    lines.append(f"variance reduction (emp):   {red:.5f}")
    lines.append(f"theoretical (1 - rho^2):    {CV.theoretical_reduction(f, U):.5f}")
    lines.append(f"-> effective sample multiplier: {1/red:.0f}x")
    lines.append("")

    # reduction vs correlation, using g = U^k to dial correlation
    lines.append("variance reduction as the control's correlation varies:")
    lines.append(f"{'control':>14}{'rho':>9}{'reduction':>12}{'1-rho^2':>10}")
    controls = [("U (linear)", U, 0.5),
                ("U^2", [u * u for u in U], 1.0 / 3.0),
                ("sqrt(U)", [math.sqrt(u) for u in U], 2.0 / 3.0),
                ("U^5", [u ** 5 for u in U], 1.0 / 6.0)]
    for name, gv, gm in controls:
        _, _, r = CV.estimate(f, gv, gm)
        rr = CV.correlation(f, gv)
        lines.append(f"{name:>14}{rr:>9.4f}{r:>12.5f}{1-rr*rr:>10.5f}")
    lines.append("")
    lines.append("Any control with a known mean works and stays unbiased; the closer its")
    lines.append("correlation to f, the more variance vanishes -- the reduction is exactly 1 - rho^2.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # SVG: distribution of the estimator over many small-n repeats, plain vs CV
        reps = 400
        small_n = 200
        plain_ests = []
        cv_ests = []
        for s in range(reps):
            r = CV._Rng(1000 + s)
            u = [r.uniform() for _ in range(small_n)]
            ff = [math.exp(x) for x in u]
            plain_ests.append(CV._mean(ff))
            e, _, _ = CV.estimate(ff, u, 0.5)
            cv_ests.append(e)

        W, H = 700, 360
        ml, mt, w, h = 55, 55, 600, 250
        lo, hi = true - 0.25, true + 0.25
        nb = 40

        def hist(vals):
            c = [0] * nb
            for v in vals:
                if lo <= v < hi:
                    c[int((v - lo) / (hi - lo) * nb)] += 1
            return c

        hp = hist(plain_ests)
        hc = hist(cv_ests)
        ymax = max(max(hp), max(hc))

        def sx(v):
            return ml + (v - lo) / (hi - lo) * w

        def sy(c):
            return mt + h - c / ymax * h

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="30" fill="{TEXT}" font-size="15">'
                 f'Estimator spread over {reps} runs (n={small_n}): plain MC vs control variate</text>')
        s.append(f'<line x1="{ml}" y1="{mt+h}" x2="{ml+w}" y2="{mt+h}" stroke="{GRAY}"/>')
        binw = (hi - lo) / nb
        for k in range(nb):
            x0 = lo + k * binw
            bx = sx(x0)
            bw = sx(x0 + binw) - bx
            s.append(f'<rect x="{bx:.1f}" y="{sy(hp[k]):.1f}" width="{max(bw-0.6,0.6):.1f}" '
                     f'height="{mt+h-sy(hp[k]):.1f}" fill="{RED}" fill-opacity="0.45"/>')
            s.append(f'<rect x="{bx:.1f}" y="{sy(hc[k]):.1f}" width="{max(bw-0.6,0.6):.1f}" '
                     f'height="{mt+h-sy(hc[k]):.1f}" fill="{GREEN}" fill-opacity="0.55"/>')
        s.append(f'<line x1="{sx(true):.1f}" y1="{mt}" x2="{sx(true):.1f}" y2="{mt+h}" '
                 f'stroke="{YELLOW}" stroke-width="1.5" stroke-dasharray="4,3"/>')
        s.append(f'<text x="{sx(true)+4:.1f}" y="{mt+14}" fill="{YELLOW}" font-size="11">true e-1</text>')
        s.append(f'<text x="{ml+w-180}" y="{mt+20}" fill="{RED}" font-size="11">plain MC (wide)</text>')
        s.append(f'<text x="{ml+w-180}" y="{mt+36}" fill="{GREEN}" font-size="11">control variate (tight)</text>')
        s.append(f'<text x="{ml}" y="{H-14}" fill="{GRAY}" font-size="10">'
                 f'Both estimators center on the true value, but the control-variate estimator '
                 f'(green) is far more concentrated -- same samples, a fraction of the variance.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "control_variates.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
