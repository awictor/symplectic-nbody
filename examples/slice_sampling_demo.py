"""Slice sampling demo: draw from a bimodal density and overlay the histogram on the true curve (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import slice_sampling as SS


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
YELLOW = "#ffd43b"


def main(outdir=None):
    # bimodal target: mixture of two gaussians
    comps = [(0.6, -2.0, 0.6), (0.4, 2.5, 1.0)]
    logf = SS.log_mixture(comps)

    def pdf(x):
        return sum(w / (s * math.sqrt(2 * math.pi)) * math.exp(-0.5 * ((x - m) / s) ** 2)
                   for w, m, s in comps)

    xs = SS.sample(logf, x0=0.0, n=20000, w=3.0, burn=2000, seed=1)

    lines = []
    lines.append("Slice sampling: self-tuning MCMC for a bimodal density")
    lines.append("=" * 56)
    lines.append("target: 0.6*N(-2, 0.6) + 0.4*N(2.5, 1.0)")
    lines.append(f"drew {len(xs)} samples, no proposal-width tuning (stepping-out + shrinkage)")
    lines.append("")
    # analytic vs empirical moments
    true_mean = sum(w * m for w, m, s in comps)
    true_second = sum(w * (s * s + m * m) for w, m, s in comps)
    true_var = true_second - true_mean ** 2
    lines.append(f"{'quantity':>12}{'empirical':>12}{'analytic':>12}")
    lines.append(f"{'mean':>12}{SS.mean(xs):>12.3f}{true_mean:>12.3f}")
    lines.append(f"{'variance':>12}{SS.variance(xs):>12.3f}{true_var:>12.3f}")
    left = sum(1 for x in xs if x < 0) / len(xs)
    lines.append(f"{'P(x<0)':>12}{left:>12.3f}{0.6:>12.3f}")
    lines.append("")
    lines.append("Both modes are explored with the right relative mass, with no accept/reject tuning:")
    lines.append("the slice width adapts to the local shape of the distribution.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        W, H = 700, 380
        ml, mt, w, h = 50, 50, 600, 280
        lo, hi = -5.0, 6.0
        nb = 60
        counts = [0] * nb
        for x in xs:
            if lo <= x < hi:
                counts[int((x - lo) / (hi - lo) * nb)] += 1
        # normalize histogram to a density
        binw = (hi - lo) / nb
        dens = [c / (len(xs) * binw) for c in counts]
        pmax = max(max(dens), max(pdf(lo + (k + 0.5) * binw) for k in range(nb)))

        def sx(x):
            return ml + (x - lo) / (hi - lo) * w

        def sy(d):
            return mt + h - d / pmax * h

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="30" fill="{TEXT}" font-size="15">'
                 f'Slice-sampled histogram vs the true bimodal density</text>')
        s.append(f'<line x1="{ml}" y1="{mt+h}" x2="{ml+w}" y2="{mt+h}" stroke="{GRAY}"/>')
        # histogram bars
        for k in range(nb):
            x0 = lo + k * binw
            bx = sx(x0)
            bw = sx(x0 + binw) - bx
            by = sy(dens[k])
            s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{max(bw-0.6,0.6):.1f}" '
                     f'height="{mt+h-by:.1f}" fill="{BLUE}" fill-opacity="0.5"/>')
        # true density curve
        curve = " ".join(f"{sx(lo + i/300*(hi-lo)):.1f},{sy(pdf(lo + i/300*(hi-lo))):.1f}"
                         for i in range(301))
        s.append(f'<polyline points="{curve}" fill="none" stroke="{YELLOW}" stroke-width="2.5"/>')
        s.append(f'<text x="{ml+w-160}" y="{mt+20}" fill="{YELLOW}" font-size="11">true density</text>')
        s.append(f'<text x="{ml+w-160}" y="{mt+36}" fill="{BLUE}" font-size="11">sampled histogram</text>')
        s.append(f'<text x="{ml}" y="{H-14}" fill="{GRAY}" font-size="10">'
                 f'20000 slice samples, no proposal-width tuning; the histogram tracks both modes '
                 f'and their relative weights.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "slice_sampling.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
