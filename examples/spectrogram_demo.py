"""Spectrogram demo: render the time-frequency image of a chirp + tones as a heatmap (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import spectrogram as SP


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"


def _heat(t):
    """Dark -> blue -> green -> yellow -> white by magnitude t in [0,1]."""
    t = max(0.0, min(1.0, t))
    stops = [(0.0, (13, 17, 30)), (0.3, (30, 60, 130)), (0.55, (6, 160, 120)),
             (0.78, (255, 212, 59)), (1.0, (255, 255, 255))]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            w = (t - t0) / (t1 - t0) if t1 > t0 else 0
            return f"#{int(c0[0]+w*(c1[0]-c0[0])):02x}{int(c0[1]+w*(c1[1]-c0[1])):02x}{int(c0[2]+w*(c1[2]-c0[2])):02x}"
    return "#ffffff"


def main(outdir=None):
    fs = 256
    N = 2048
    # a signal: a rising chirp plus a steady tone at 40 Hz plus a late burst at 90 Hz
    sig = []
    for n in range(N):
        t = n / fs
        x = math.cos(2 * math.pi * (10 + 60 * (n / N)) * n / fs)     # chirp 10->70 Hz
        x += 0.7 * math.cos(2 * math.pi * 40 * n / fs)              # steady 40 Hz
        if n > N * 0.6:
            x += 0.8 * math.cos(2 * math.pi * 95 * n / fs)         # late burst 95 Hz
        sig.append(x)

    frame_len = 128
    hop = 32
    spec = SP.spectrogram(sig, frame_len=frame_len, hop=hop, window="hann")
    n_frames = len(spec)
    n_bins = len(spec[0])

    lines = []
    lines.append("Spectrogram: short-time Fourier transform")
    lines.append("=" * 50)
    lines.append(f"signal {N} samples at {fs} Hz: chirp 10->70 Hz + steady 40 Hz + late 95 Hz burst")
    lines.append(f"STFT: frame {frame_len}, hop {hop}, Hann window -> {n_frames} frames x {n_bins} bins")
    lines.append(f"bin resolution: {fs/frame_len:.2f} Hz, time resolution: {hop/fs*1000:.1f} ms")
    lines.append("")
    # dominant track
    track = SP.dominant_frequency_track(sig, frame_len, hop, "hann", sample_rate=fs)
    lines.append("dominant-frequency ridge over time (first / mid / last frames):")
    lines.append(f"  start: {track[0]:.1f} Hz, middle: {track[len(track)//2]:.1f} Hz, "
                 f"end: {track[-1]:.1f} Hz")
    lines.append("")
    lines.append("The uncertainty principle: a shorter frame sharpens timing but blurs frequency;")
    lines.append("a longer frame does the reverse. The Hann window suppresses spectral leakage.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # render the spectrogram: x = time (frames), y = frequency (bins), color = log magnitude
        W, H = 720, 400
        ml, mt = 55, 50
        pw, ph = 620, 300
        cw = pw / n_frames
        ch = ph / n_bins
        # log-scale magnitudes, normalized
        flat = [math.log(1 + m) for row in spec for m in row]
        mx = max(flat) or 1
        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="30" fill="{TEXT}" font-size="15">'
                 f'Spectrogram: chirp (diagonal), steady 40 Hz (horizontal), late 95 Hz burst</text>')
        for fi in range(n_frames):
            for bi in range(n_bins):
                t = math.log(1 + spec[fi][bi]) / mx
                if t < 0.04:
                    continue
                x = ml + fi * cw
                y = mt + ph - (bi + 1) * ch       # low freq at bottom
                s.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cw+0.6:.2f}" '
                         f'height="{ch+0.6:.2f}" fill="{_heat(t)}"/>')
        # axes labels
        s.append(f'<text x="{ml}" y="{mt+ph+18}" fill="{GRAY}" font-size="10">time -></text>')
        s.append(f'<text x="{ml-45}" y="{mt+ph/2:.0f}" fill="{GRAY}" font-size="10">freq (Hz)</text>')
        for hz in (0, 40, 80, 120):
            bi = int(hz * frame_len / fs)
            if bi < n_bins:
                y = mt + ph - (bi + 0.5) * ch
                s.append(f'<text x="{ml-8}" y="{y+3:.0f}" fill="{GRAY}" font-size="9" '
                         f'text-anchor="end">{hz}</text>')
        s.append(f'<text x="{ml}" y="{H-12}" fill="{GRAY}" font-size="10">'
                 f'The diagonal ridge is the chirp; the flat line is the steady tone; the bright block '
                 f'at upper-right is the burst that only appears late.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "spectrogram.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
