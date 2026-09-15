"""FitzHugh-Nagumo demo: threshold excitability, the phase-plane, and the Hopf onset of repetitive spiking."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import fitzhugh_nagumo as fn


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    lines = []
    lines.append("FitzHugh-Nagumo -- the two-variable cartoon of a spiking neuron")
    lines.append("=" * 74)
    lines.append("")
    lines.append("  dv/dt = v - v^3/3 - w + I     (fast voltage-like variable)")
    lines.append("  dw/dt = eps (v + a - b w)     (slow recovery, eps << 1)")
    lines.append("")

    rest = fn.FitzHughNagumo(I=0.0)
    v_rest, w_rest = rest.fixed_point()
    lines.append(f"At rest (I=0): fixed point (v*,w*) = ({v_rest:.3f}, {w_rest:.3f}), "
                 f"stable = {rest.fixed_point_stable()}.")
    lines.append("")

    lines.append("Threshold excitability -- a kick must clear the middle branch of the cubic to spike:")
    lines.append("   kick to v      peak v reached     spike?")
    lines.append("   " + "-" * 44)
    for dv in (0.1, 0.3, 0.5, 0.8, 1.5):
        vk = v_rest + dv
        _, vs, _ = rest.simulate(vk, w_rest, 0.05, 6000)
        peak = max(vs)
        spk = rest.spikes(vk)
        lines.append(f"   {vk:+6.3f}         {peak:+6.3f}          {'SPIKE' if spk else 'decays'}")
    lines.append("")
    lines.append("Small kicks decay straight back; past threshold the cubic amplifies the kick into a")
    lines.append("full ~2-unit spike before the slow recovery drags v home. No switch is hard-coded.")
    lines.append("")

    lines.append("Inject steady current I -> Hopf bifurcation -> repetitive firing (a relaxation oscillator):")
    on, off = fn.hopf_current_window()
    lines.append(f"   fixed point is unstable for I in ({on:.3f}, {off:.3f}) -- two Hopf points.")
    lines.append("   I        fixed pt      behaviour")
    lines.append("   " + "-" * 40)
    for I in (0.0, 0.2, 0.35, 0.6, 1.0, 1.4, 1.8):
        neuron = fn.FitzHughNagumo(I=I)
        stab = "stable  " if neuron.fixed_point_stable() else "UNSTABLE"
        osc = "repetitive spiking" if neuron.is_oscillating() else "rest"
        lines.append(f"   {I:4.2f}     {stab}      {osc}")
    lines.append("")
    lines.append("Below and above the window the neuron rests; inside it fires forever -- the excitable")
    lines.append("cell has become an oscillator. Same equations, current is the only knob.")

    text = "\n".join(lines)
    print(text)

    svg = _svg()
    return text, svg


def _svg():
    W, H = 640, 460
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Left: phase-plane spike vs decay. Right: sub- and supra-threshold voltage traces</text>')

    # ---- LEFT: phase plane (v horizontal, w vertical) ----
    lx0, lx1 = 55, 300
    ly0, ly1 = 60, 320
    vmin, vmax = -2.5, 2.5
    wmin, wmax = -0.8, 1.6
    def px(v): return lx0 + (v - vmin) / (vmax - vmin) * (lx1 - lx0)
    def py(w): return ly1 - (w - wmin) / (wmax - wmin) * (ly1 - ly0)

    # axes
    parts.append(f'<line x1="{lx0}" y1="{py(0):.1f}" x2="{lx1}" y2="{py(0):.1f}" stroke="{P["gray"]}" stroke-width="0.5" opacity="0.5"/>')
    parts.append(f'<line x1="{px(0):.1f}" y1="{ly0}" x2="{px(0):.1f}" y2="{ly1}" stroke="{P["gray"]}" stroke-width="0.5" opacity="0.5"/>')
    parts.append(f'<text x="{lx1}" y="{py(0)+14:.1f}" fill="{P["gray"]}" font-size="10" text-anchor="end">v</text>')
    parts.append(f'<text x="{px(0)+4:.1f}" y="{ly0+10:.1f}" fill="{P["gray"]}" font-size="10">w</text>')

    # v-nullcline: w = v - v^3/3 + I  (I=0), the cubic
    cub = []
    v = vmin
    while v <= vmax:
        cub.append(f"{px(v):.1f},{py(v - v**3/3.0):.1f}")
        v += 0.05
    parts.append(f'<polyline points="{" ".join(cub)}" fill="none" stroke="{P["yellow"]}" stroke-width="1.5"/>')
    parts.append(f'<text x="{px(2.1):.1f}" y="{py(-0.3):.1f}" fill="{P["yellow"]}" font-size="10">v-nullcline</text>')

    # w-nullcline: v + a - b w = 0 -> w = (v + a)/b   a=0.7 b=0.8
    a, b = 0.7, 0.8
    wl = [f"{px(v):.1f},{py((v + a)/b):.1f}" for v in (vmin, vmax)]
    parts.append(f'<polyline points="{" ".join(wl)}" fill="none" stroke="{P["green"]}" stroke-width="1.5"/>')
    parts.append(f'<text x="{px(-2.3):.1f}" y="{py(-0.5):.1f}" fill="{P["green"]}" font-size="10">w-nullcline</text>')

    rest = fn.FitzHughNagumo(I=0.0)
    vr, wr = rest.fixed_point()
    parts.append(f'<circle cx="{px(vr):.1f}" cy="{py(wr):.1f}" r="4" fill="{P["text"]}"/>')

    # spike trajectory (supra-threshold) and decay trajectory (sub-threshold)
    for dv, col in ((1.5, P["red"]), (0.3, P["blue"])):
        _, vs, ws = rest.simulate(vr + dv, wr, 0.04, 3000)
        pts = " ".join(f"{px(vs[i]):.1f},{py(ws[i]):.1f}" for i in range(0, len(vs), 4))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.3" opacity="0.9"/>')

    parts.append(f'<text x="{lx0}" y="{ly1+22:.1f}" fill="{P["red"]}" font-size="10">red: supra-threshold -> spike loop</text>')
    parts.append(f'<text x="{lx0}" y="{ly1+36:.1f}" fill="{P["blue"]}" font-size="10">blue: sub-threshold -> straight back to rest</text>')

    # ---- RIGHT: voltage traces v(t) ----
    rx0, rx1 = 355, W - 30
    ry0, ry1 = 60, 320
    T_steps = 3000
    dt = 0.04
    def tx(i): return rx0 + i / T_steps * (rx1 - rx0)
    def ty(v): return ry1 - (v - (-2.5)) / (2.5 - (-2.5)) * (ry1 - ry0)
    parts.append(f'<line x1="{rx0}" y1="{ty(0):.1f}" x2="{rx1}" y2="{ty(0):.1f}" stroke="{P["gray"]}" stroke-width="0.5" opacity="0.5"/>')
    parts.append(f'<text x="{rx0}" y="{ry0-2:.1f}" fill="{P["gray"]}" font-size="10">v(t)</text>')

    for dv, col in ((1.5, P["red"]), (0.3, P["blue"])):
        _, vs, _ = rest.simulate(vr + dv, wr, dt, T_steps)
        pts = " ".join(f"{tx(i):.1f},{ty(vs[i]):.1f}" for i in range(0, len(vs), 4))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.3"/>')

    # a repetitive-firing trace for I in the Hopf window
    firing = fn.FitzHughNagumo(I=0.6)
    fv, fw = firing.fixed_point()
    _, vs, _ = firing.simulate(fv + 0.01, fw, dt, T_steps)
    pts = " ".join(f"{tx(i):.1f},{ty(vs[i]):.1f}" for i in range(0, len(vs), 4))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{P["purple"]}" stroke-width="1.3" opacity="0.9"/>')
    parts.append(f'<text x="{rx0}" y="{ry1+22:.1f}" fill="{P["purple"]}" font-size="10">purple: I=0.6 in the Hopf window -> repetitive spiking</text>')

    parts.append(f'<text x="20" y="{H-26}" fill="{P["gray"]}" font-size="11">'
                 f'The resting fixed point sits where the cubic (yellow) and line (green) nullclines cross.</text>')
    parts.append(f'<text x="20" y="{H-10}" fill="{P["gray"]}" font-size="11">'
                 f'A kick past the middle branch loops out into a spike; steady current makes it fire forever.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
