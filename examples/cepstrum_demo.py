"""Cepstrum demo: recover a voiced-speech pitch from its cepstral peak, and detect an echo (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cepstrum as C


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
RED = "#ff6b6b"
YELLOW = "#ffd43b"


def main(outdir=None):
    N = 256
    fs = 8000
    f0 = 125
    # voiced-speech-like: harmonic stack of f0 shaped by a formant-ish envelope
    voiced = [sum(math.sin(2 * math.pi * f0 * h * n / fs) / h for h in range(1, 16))
              for n in range(N)]

    lines = []
    lines.append("Cepstrum: the spectrum of a log-spectrum")
    lines.append("=" * 50)
    lines.append("cepstrum(x) = IDFT( log|DFT(x)| ); its axis is 'quefrency' (time-like)")
    lines.append("a convolution (envelope * excitation) becomes a SUM after the log, so the")
    lines.append("two separate cleanly in the cepstral domain.")
    lines.append("")
    lines.append(f"pitch detection: voiced signal, f0 = {f0} Hz, fs = {fs}")
    T = fs / f0
    q = C.pitch_quefrency(voiced, min_q=int(fs / 300), max_q=int(fs / 80))
    lines.append(f"  true pitch period: {T:.1f} samples ({f0} Hz)")
    lines.append(f"  cepstral peak at quefrency {q} samples -> {fs / q:.1f} Hz")
    lines.append("")
    # echo
    # broadband base (deterministic pseudo-noise) so the echo's periodicity dominates the cepstrum
    st = 12345

    def rnd():
        nonlocal_state[0] = (1664525 * nonlocal_state[0] + 1013904223) & 0xFFFFFFFF
        return (nonlocal_state[0] >> 8) / (1 << 24) * 2 - 1
    nonlocal_state = [st]
    base = [rnd() for _ in range(N)]
    d = 35
    echo = [base[n] + 0.7 * base[n - d] if n >= d else base[n] for n in range(N)]
    qe = C.echo_delay(echo, min_q=10, max_q=90)
    lines.append(f"echo detection: signal + a copy delayed by {d} samples")
    lines.append(f"  cepstral peak at quefrency {qe} samples (true delay {d})")
    lines.append("")
    lines.append("Cepstral pitch tracking finds the fundamental even when it is weak or missing,")
    lines.append("because it keys on the harmonic SPACING, not the fundamental's own energy.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        W, H = 720, 380
        ml, w = 55, 620
        # top: cepstrum of the voiced signal (pitch peak); bottom: cepstrum of the echo signal
        cv = C.real_cepstrum(voiced)
        ce = C.real_cepstrum(echo)
        qmax = 110

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="26" fill="{TEXT}" font-size="15">'
                 f'Cepstra: pitch peak (top) and echo peak (bottom)</text>')

        def panel(c, oy, h, mark, mark_col, title):
            band = [abs(c[q]) for q in range(2, qmax)]
            cmax = max(band) or 1
            s.append(f'<text x="{ml}" y="{oy-6}" fill="{GRAY}" font-size="11">{title}</text>')
            s.append(f'<line x1="{ml}" y1="{oy+h}" x2="{ml+w}" y2="{oy+h}" stroke="{GRAY}"/>')
            for q in range(2, qmax):
                x = ml + (q - 2) / (qmax - 2) * w
                v = abs(c[q]) / cmax
                s.append(f'<line x1="{x:.1f}" y1="{oy+h:.1f}" x2="{x:.1f}" '
                         f'y2="{oy+h-v*h:.1f}" stroke="{BLUE}" stroke-width="2"/>')
            mx = ml + (mark - 2) / (qmax - 2) * w
            s.append(f'<line x1="{mx:.1f}" y1="{oy}" x2="{mx:.1f}" y2="{oy+h}" '
                     f'stroke="{mark_col}" stroke-width="1.5" stroke-dasharray="4,3"/>')
            s.append(f'<text x="{mx+3:.1f}" y="{oy+12:.1f}" fill="{mark_col}" font-size="10">'
                     f'q={mark}</text>')

        panel(cv, 55, 120, q, GREEN, f"voiced pitch: peak at quefrency {q} ({fs/q:.0f} Hz)")
        panel(ce, 230, 120, qe, RED, f"echo: peak at quefrency {qe} (delay {d})")
        s.append(f'<text x="{ml}" y="{H-10}" fill="{GRAY}" font-size="10">'
                 f'quefrency (samples) ->  a peak marks the periodicity: pitch period or echo delay</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "cepstrum.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
