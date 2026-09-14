"""Wiener filter demo: denoise a noisy signal and show clean/noisy/recovered + the frequency gain (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import wiener_filter as W
from bluestein import dft


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
RED = "#ff6b6b"
YELLOW = "#ffd43b"


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def main(outdir=None):
    rnd = _lcg(20260913)

    def gauss():
        return math.sqrt(-2 * math.log(rnd() + 1e-12)) * math.cos(2 * math.pi * rnd())

    N = 128
    clean = [math.sin(2 * math.pi * 5 * n / N) + 0.6 * math.sin(2 * math.pi * 12 * n / N)
             for n in range(N)]
    noise_amp = 0.7
    noisy = [clean[n] + noise_amp * gauss() for n in range(N)]
    Spow = [abs(v) ** 2 for v in dft(clean)]
    Npow = [noise_amp ** 2 * N] * N
    recovered = W.denoise(noisy, Spow, Npow)
    H = W.wiener_gain(Spow, Npow)

    lines = []
    lines.append("Wiener filtering: minimum-MSE denoising")
    lines.append("=" * 50)
    lines.append("optimal linear filter: H(f) = S(f) / (S(f) + N(f)), a per-bin trust weight")
    lines.append("")
    lines.append(f"signal: two tones (5 & 12 cyc), additive noise amplitude {noise_amp}")
    lines.append(f"  SNR noisy:     {W.snr(clean, noisy):6.2f} dB")
    lines.append(f"  SNR recovered: {W.snr(clean, recovered):6.2f} dB  "
                 f"(+{W.snr(clean, recovered) - W.snr(clean, noisy):.1f} dB)")
    lines.append(f"  MSE noisy:     {W.mse(clean, noisy):.4f}")
    lines.append(f"  MSE recovered: {W.mse(clean, recovered):.4f}")
    lines.append("")
    lines.append("the gain passes the signal bins (near 1) and crushes the noise-only bins (near 0):")
    # show gain at the two signal bins and a noise bin
    lines.append(f"  gain at bin 5 (tone):  {H[5]:.3f}")
    lines.append(f"  gain at bin 12 (tone): {H[12]:.3f}")
    lines.append(f"  gain at bin 40 (noise):{H[40]:.3f}")
    lines.append("")
    lines.append("Wiener's filter is optimal among ALL linear time-invariant filters for stationary")
    lines.append("signals -- the ancestor of spectral subtraction, deconvolution, and image restoration.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        W_px, H_px = 720, 420
        ml, w = 50, 620

        def panel_wave(s, series, oy, h, col, title):
            lo = min(min(sr) for sr in series)
            hi = max(max(sr) for sr in series)
            rng = hi - lo or 1

            def sx(i):
                return ml + i / (N - 1) * w

            def sy(v):
                return oy + h - (v - lo) / rng * h
            s.append(f'<text x="{ml}" y="{oy-4}" fill="{GRAY}" font-size="11">{title}</text>')
            for sr, c in zip(series, col):
                pts = " ".join(f"{sx(i):.1f},{sy(sr[i]):.1f}" for i in range(N))
                s.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="1.5"/>')

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_px}" height="{H_px}" '
             f'viewBox="0 0 {W_px} {H_px}" font-family="monospace">']
        s.append(f'<rect width="{W_px}" height="{H_px}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="24" fill="{TEXT}" font-size="15">'
                 f'Wiener denoising: noisy vs recovered vs clean, and the frequency gain</text>')
        panel_wave(s, [noisy], 50, 90, [RED], "noisy input")
        panel_wave(s, [recovered, clean], 165, 90, [GREEN, YELLOW],
                   "recovered (green) over clean (yellow)")
        # gain panel
        oy = 290
        h = 90
        s.append(f'<text x="{ml}" y="{oy-4}" fill="{GRAY}" font-size="11">'
                 f'Wiener gain H(f) over the first {N//2} bins (1 = keep, 0 = kill)</text>')
        s.append(f'<line x1="{ml}" y1="{oy+h}" x2="{ml+w}" y2="{oy+h}" stroke="{GRAY}"/>')
        half = N // 2
        for k in range(half):
            x = ml + k / (half - 1) * w
            s.append(f'<line x1="{x:.1f}" y1="{oy+h:.1f}" x2="{x:.1f}" y2="{oy+h-H[k]*h:.1f}" '
                     f'stroke="{BLUE}" stroke-width="2"/>')
        s.append(f'<text x="{ml}" y="{H_px-10}" fill="{GRAY}" font-size="10">'
                 f'The gain spikes to ~1 exactly at the two signal bins (5, 12) and stays near 0 '
                 f'everywhere else, so only signal energy survives.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "wiener_filter.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
