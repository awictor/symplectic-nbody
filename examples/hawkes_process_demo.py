"""Hawkes process demo: self-exciting events clustering, the intensity path, and rate amplified by 1/(1-n)."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import hawkes_process as hp


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    lines = []
    lines.append("Hawkes process -- self-exciting events (earthquakes, spikes, orders)")
    lines.append("=" * 68)
    lines.append("")

    mu = 1.0
    beta = 2.0
    lines.append(f"Baseline mu={mu}, decay beta={beta}. Self-excitation amplifies the rate by 1/(1-n).")
    lines.append("")
    lines.append("Branching ratio n = alpha/beta controls clustering and the stationary rate mu/(1-n):")
    lines.append("   alpha   n=a/b    theoretical rate   empirical rate")
    lines.append("   " + "-" * 50)
    T = 2000.0
    for alpha in (0.0, 0.5, 1.0, 1.5, 1.8):
        n = hp.branching_ratio(alpha, beta)
        theo = hp.stationary_rate(mu, alpha, beta)
        rates = [hp.empirical_rate(hp.simulate_fast(mu, alpha, beta, T, seed=r + 1), T) for r in range(10)]
        emp = sum(rates) / len(rates)
        lines.append(f"   {alpha:5.1f}   {n:.3f}    {theo:8.3f}          {emp:.3f}")
    lines.append("")
    lines.append("n=0 is plain Poisson; n->1 the rate blows up as excitation feeds on itself.")
    lines.append("")

    # clustering vs Poisson
    ev_h = hp.simulate_fast(1.0, 0.8, 2.0, 1000.0, seed=5)
    ev_p = hp.simulate_fast(hp.empirical_rate(ev_h, 1000.0), 0.0, 2.0, 1000.0, seed=5)
    def cv2(ev):
        g = [ev[i + 1] - ev[i] for i in range(len(ev) - 1)]
        m = sum(g) / len(g)
        return sum((x - m) ** 2 for x in g) / len(g) / (m * m)
    lines.append("Clustering: coefficient of variation^2 of inter-event gaps (Poisson = 1):")
    lines.append(f"  Hawkes  (n=0.4): CV^2 = {cv2(ev_h):.2f}  (bursty -- gaps very uneven)")
    lines.append(f"  Poisson (same rate): CV^2 = {cv2(ev_p):.2f}  (memoryless)")

    text = "\n".join(lines)
    print(text)

    svg = _svg(mu, 1.2, beta)
    return text, svg


def _svg(mu, alpha, beta):
    W, H = 640, 430
    P = PALETTE
    T = 30.0
    events = hp.simulate_fast(mu, alpha, beta, T, seed=11)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Conditional intensity lambda(t) spiking at each self-excited event</text>')

    x0, x1, y0, y1 = 50, 615, 55, 320

    def px(t):
        return x0 + t / T * (x1 - x0)

    # sample the intensity on a fine grid
    N = 1200
    lam_vals = []
    for i in range(N + 1):
        t = T * i / N
        lam_vals.append(hp.intensity(t, events, mu, alpha, beta))
    lam_max = max(lam_vals) * 1.05

    def py(l):
        return y1 - l / lam_max * (y1 - y0)

    # baseline mu line
    parts.append(f'<line x1="{x0}" y1="{py(mu):.1f}" x2="{x1}" y2="{py(mu):.1f}" '
                 f'stroke="{P["gray"]}" stroke-width="1" stroke-dasharray="4,3"/>')
    parts.append(f'<text x="{x1 - 60}" y="{py(mu) - 4:.1f}" fill="{P["gray"]}" font-size="10">baseline mu</text>')

    # intensity curve
    pts = " ".join(f"{px(T * i / N):.1f},{py(lam_vals[i]):.1f}" for i in range(N + 1))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{P["yellow"]}" stroke-width="1.5"/>')

    # event ticks
    for e in events:
        parts.append(f'<line x1="{px(e):.1f}" y1="{y1}" x2="{px(e):.1f}" y2="{y1 + 12}" '
                     f'stroke="{P["red"]}" stroke-width="1.2"/>')
    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')

    parts.append(f'<text x="{x0}" y="{y1 + 30:.1f}" fill="{P["red"]}" font-size="11">'
                 f'red ticks = events ({len(events)} in t=[0,{T:.0f}])</text>')
    parts.append(f'<text x="{x0 + 260}" y="{y1 + 30:.1f}" fill="{P["yellow"]}" font-size="11">'
                 f'yellow = intensity lambda(t)</text>')
    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'each event jumps lambda by alpha, then it decays -- events bunch into bursts, '
                 f'unlike a flat Poisson rate.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
