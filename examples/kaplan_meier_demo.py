"""Demo: Kaplan-Meier survival curves for two groups, with censoring and the log-rank test.

Simulates a clinical-trial-style dataset -- a treatment and a control arm, each with right-censoring --
estimates the Kaplan-Meier survival curves with confidence bands, reports median survival, and runs the
log-rank test for a difference. Draws the two step-function survival curves with censoring ticks.

    python examples/kaplan_meier_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kaplan_meier import (  # noqa: E402
    kaplan_meier, median_survival, logrank_test, confidence_band,
)


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def _simulate(seed, scale, censor_time, n):
    """Exponential survival times with administrative censoring at censor_time."""
    rng = _lcg(seed)
    times, events = [], []
    for _ in range(n):
        t = -scale * math.log(1 - rng())  # exponential lifetime
        if t > censor_time:
            times.append(censor_time)
            events.append(0)  # censored
        else:
            times.append(round(t, 2))
            events.append(1)  # event observed
    return times, events


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Kaplan-Meier: censored survival curves and the log-rank test\n")

    # treatment survives longer (larger scale) than control
    t_ctrl, e_ctrl = _simulate(seed=1, scale=8.0, censor_time=24, n=60)
    t_trt, e_trt = _simulate(seed=2, scale=16.0, censor_time=24, n=60)

    et_c, S_c, v_c = kaplan_meier(t_ctrl, e_ctrl)
    et_t, S_t, v_t = kaplan_meier(t_trt, e_trt)

    print(f"  control:   {len(t_ctrl)} subjects, {sum(e_ctrl)} events, "
          f"{sum(1 for x in e_ctrl if x == 0)} censored")
    print(f"  treatment: {len(t_trt)} subjects, {sum(e_trt)} events, "
          f"{sum(1 for x in e_trt if x == 0)} censored\n")

    mc = median_survival(et_c, S_c)
    mt = median_survival(et_t, S_t)
    print(f"  median survival:  control {mc},  treatment {mt}")

    chi2, O1, E1 = logrank_test(t_ctrl, e_ctrl, t_trt, e_trt)
    # p-value from chi-squared with 1 dof: p = erfc(sqrt(chi2/2))
    p = math.erfc(math.sqrt(chi2 / 2)) if chi2 >= 0 else 1.0
    print(f"\n  log-rank test:  chi2 = {chi2:.2f}  (1 dof),  p ~ {p:.4f}")
    print(f"  observed control events {O1:.0f} vs {E1:.1f} expected under the null")
    verdict = "significant difference" if p < 0.05 else "no significant difference"
    print(f"  -> {verdict} between the survival curves")
    print(f"\n  Censoring is handled exactly: censored subjects stay in the at-risk set until")
    print(f"  their censoring time, then leave without causing a drop in the curve.")

    _svg(os.path.join(outdir, "kaplan_meier.svg"), et_c, S_c, et_t, S_t, t_ctrl, e_ctrl, t_trt, e_trt)
    print(f"\n  wrote {os.path.join(outdir, 'kaplan_meier.svg')}")


def _svg(path, etc, Sc, ett, St, tc, ec, tt, et, width=760, height=400):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Kaplan-Meier survival: treatment (blue) outlives control (orange); ticks = censoring</text>',
    ]
    ox, oy, ow, oh = 55, 50, width - 100, height - 100
    tmax = max(max(etc) if etc else 1, max(ett) if ett else 1)

    def sx(t):
        return ox + ow * t / tmax

    def sy(s):
        return oy + oh * (1 - s)

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = sy(frac)
        parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+ow}" y2="{y:.1f}" stroke="#161b22"/>')
        parts.append(f'<text x="{ox-6}" y="{y+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{frac:.2f}</text>')
    # median line
    parts.append(f'<line x1="{ox}" y1="{sy(0.5):.1f}" x2="{ox+ow}" y2="{sy(0.5):.1f}" '
                 f'stroke="#484f58" stroke-dasharray="4 3"/>')

    def step(et, S, color, times, events):
        # build the step path starting at S=1
        pts = [f"{sx(0):.1f},{sy(1.0):.1f}"]
        prev_s = 1.0
        for t, s in zip(et, S):
            pts.append(f"{sx(t):.1f},{sy(prev_s):.1f}")
            pts.append(f"{sx(t):.1f},{sy(s):.1f}")
            prev_s = s
        pts.append(f"{sx(tmax):.1f},{sy(prev_s):.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
                     f'stroke-width="2"/>')
        # censoring ticks
        # approximate survival at each censor time
        for t, e in zip(times, events):
            if e == 0:
                s_here = 1.0
                for et_i, s_i in zip(et, S):
                    if et_i <= t:
                        s_here = s_i
                x = sx(t)
                y = sy(s_here)
                parts.append(f'<line x1="{x:.1f}" y1="{y-4:.1f}" x2="{x:.1f}" y2="{y+4:.1f}" '
                             f'stroke="{color}" stroke-width="1"/>')

    step(etc, Sc, "#ff922b", tc, ec)
    step(ett, St, "#4dabf7", tt, et)

    parts.append(f'<rect x="{ox+ow-140}" y="{oy+8}" width="12" height="3" fill="#4dabf7"/>')
    parts.append(f'<text x="{ox+ow-124}" y="{oy+12}" fill="#e6edf3" font-size="10">treatment</text>')
    parts.append(f'<rect x="{ox+ow-140}" y="{oy+24}" width="12" height="3" fill="#ff922b"/>')
    parts.append(f'<text x="{ox+ow-124}" y="{oy+28}" fill="#e6edf3" font-size="10">control</text>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+22:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">time</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
