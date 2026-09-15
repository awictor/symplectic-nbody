"""Ornstein-Uhlenbeck demo: mean-reverting paths, exponential relaxation of the moments, and exact vs Euler."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import ornstein_uhlenbeck as ou


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    p = ou.OrnsteinUhlenbeck(theta=1.0, mu=1.0, sigma=0.6)

    lines = []
    lines.append("Ornstein-Uhlenbeck process -- the mean-reverting random walk")
    lines.append("=" * 74)
    lines.append("")
    lines.append("  dX = theta (mu - X) dt + sigma dW      (spring pull to mu + white noise)")
    lines.append(f"  theta={p.theta} (rate)   mu={p.mu} (mean)   sigma={p.sigma} (noise)")
    lines.append("")
    lines.append(f"Exact solvability -- every moment in closed form:")
    lines.append(f"  correlation time 1/theta = {p.correlation_time():.3f}")
    lines.append(f"  stationary law: Normal(mu={p.mu}, var=sigma^2/2theta={p.stationary_variance():.4f})")
    lines.append("")

    lines.append("Relaxation from x0=4 toward mu=1 -- Monte Carlo (6000 paths) tracks the analytic curves:")
    lines.append("    t      E[X_t] analytic   MC mean     Var analytic   MC var")
    lines.append("   " + "-" * 60)
    for t in (0.2, 0.5, 1.0, 2.0, 4.0):
        m, v = p.empirical_moments(4.0, t=t, dt=0.02, paths=6000, seed=5, exact=True)
        lines.append(f"   {t:4.1f}      {p.mean(4.0,t):7.4f}       {m:7.4f}     "
                     f"{p.variance(t):7.4f}      {v:7.4f}")
    lines.append("")
    lines.append("Mean decays exponentially to mu; variance climbs from 0 to the stationary value.")
    lines.append("")

    lines.append("Exact Gaussian stepper has NO time-step error; Euler-Maruyama is biased at large dt:")
    lines.append("    dt      analytic Var(dt)   exact stepper   Euler-Maruyama")
    lines.append("   " + "-" * 58)
    for dt in (0.1, 0.5, 1.0, 2.0):
        _, ve = p.empirical_moments(1.0, t=dt, dt=dt, paths=8000, seed=9, exact=True)
        _, vu = p.empirical_moments(1.0, t=dt, dt=dt, paths=8000, seed=9, exact=False)
        lines.append(f"   {dt:4.1f}       {p.variance(dt):8.4f}        {ve:8.4f}       {vu:8.4f}")
    lines.append("")
    lines.append("Euler injects sigma^2*dt of variance per step regardless of relaxation -> overshoots badly.")
    lines.append("The exact stepper matches truth at any dt because the OU transition is Gaussian.")

    text = "\n".join(lines)
    print(text)

    svg = _svg(p)
    return text, svg


def _svg(p):
    W, H = 640, 460
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Mean-reverting paths relaxing to mu, with the analytic mean +/- std band</text>')

    # ---- TOP: sample paths X(t) ----
    x0v = 4.0
    T = 5.0
    dt = 0.02
    steps = int(T / dt)
    x0, x1 = 60, W - 30
    y0, y1 = 55, 260
    vmin, vmax = -0.5, 4.5
    def tx(i): return x0 + i / steps * (x1 - x0)
    def ty(v): return y1 - (v - vmin) / (vmax - vmin) * (y1 - y0)

    # mu line
    parts.append(f'<line x1="{x0}" y1="{ty(p.mu):.1f}" x2="{x1}" y2="{ty(p.mu):.1f}" '
                 f'stroke="{P["gray"]}" stroke-width="1" stroke-dasharray="4,3"/>')
    parts.append(f'<text x="{x1}" y="{ty(p.mu)-4:.1f}" fill="{P["gray"]}" font-size="10" text-anchor="end">mu={p.mu}</text>')

    # analytic mean +/- 1 std band
    band_top, band_bot = [], []
    for i in range(steps + 1):
        t = i * dt
        m = p.mean(x0v, t)
        sd = math.sqrt(p.variance(t))
        band_top.append(f"{tx(i):.1f},{ty(m + sd):.1f}")
        band_bot.append(f"{tx(i):.1f},{ty(m - sd):.1f}")
    poly = " ".join(band_top) + " " + " ".join(reversed(band_bot))
    parts.append(f'<polygon points="{poly}" fill="{P["blue"]}" opacity="0.14"/>')
    # analytic mean curve
    mean_pts = " ".join(f"{tx(i):.1f},{ty(p.mean(x0v, i*dt)):.1f}" for i in range(steps + 1))
    parts.append(f'<polyline points="{mean_pts}" fill="none" stroke="{P["yellow"]}" stroke-width="2"/>')

    # a few sample paths
    cols = [P["green"], P["red"], P["purple"], P["blue"]]
    for k in range(4):
        rng = ou._Rng((k + 1) * 7919 + 1)
        path = p.simulate(x0v, dt, steps, rng, exact=True)
        pts = " ".join(f"{tx(i):.1f},{ty(path[i]):.1f}" for i in range(0, len(path), 3))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{cols[k]}" stroke-width="0.8" opacity="0.75"/>')
    parts.append(f'<text x="{x0}" y="{y1+16:.1f}" fill="{P["yellow"]}" font-size="10">yellow: analytic mean;  band: +/-1 std;  thin: sample paths</text>')

    # ---- BOTTOM: exact vs Euler variance vs dt ----
    gx0, gx1 = 70, W - 40
    gy0, gy1 = 320, 420
    dts = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    an = [p.variance(d) for d in dts]
    eu = [p.sigma ** 2 * d for d in dts]  # Euler one-step variance = sigma^2 dt
    vmax2 = max(eu) * 1.05
    def gx(d): return gx0 + d / 3.0 * (gx1 - gx0)
    def gy(v): return gy1 - v / vmax2 * (gy1 - gy0)
    parts.append(f'<line x1="{gx0}" y1="{gy1}" x2="{gx1}" y2="{gy1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<line x1="{gx0}" y1="{gy0}" x2="{gx0}" y2="{gy1}" stroke="{P["gray"]}" stroke-width="1"/>')
    # stationary variance asymptote
    sv = p.stationary_variance()
    parts.append(f'<line x1="{gx0}" y1="{gy(sv):.1f}" x2="{gx1}" y2="{gy(sv):.1f}" '
                 f'stroke="{P["gray"]}" stroke-width="0.5" stroke-dasharray="3,3"/>')
    parts.append(f'<text x="{gx1}" y="{gy(sv)-3:.1f}" fill="{P["gray"]}" font-size="9" text-anchor="end">stationary var</text>')
    an_pts = " ".join(f"{gx(d):.1f},{gy(v):.1f}" for d, v in zip(dts, an))
    eu_pts = " ".join(f"{gx(d):.1f},{gy(v):.1f}" for d, v in zip(dts, eu))
    parts.append(f'<polyline points="{an_pts}" fill="none" stroke="{P["green"]}" stroke-width="2"/>')
    parts.append(f'<polyline points="{eu_pts}" fill="none" stroke="{P["red"]}" stroke-width="2"/>')
    parts.append(f'<text x="{gx0+6}" y="{gy0+12:.1f}" fill="{P["green"]}" font-size="10">exact / analytic variance (bounded)</text>')
    parts.append(f'<text x="{gx0+6}" y="{gy0+26:.1f}" fill="{P["red"]}" font-size="10">Euler one-step variance = sigma^2 dt (unbounded)</text>')
    parts.append(f'<text x="{gx0}" y="{gy1+16:.1f}" fill="{P["gray"]}" font-size="10">time step dt</text>')
    parts.append(f'<text x="{gx1}" y="{gy1+16:.1f}" fill="{P["gray"]}" font-size="10" text-anchor="end">3.0</text>')

    parts.append(f'<text x="20" y="{H-8}" fill="{P["gray"]}" font-size="11">'
                 f'The exact stepper (green) saturates at the true stationary variance; Euler (red) grows without bound in dt.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
