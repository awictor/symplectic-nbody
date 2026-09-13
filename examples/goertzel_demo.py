"""Demo: Goertzel algorithm -- one DFT bin cheaply, and decoding a phone keypad.

Detects a tone buried in noise with a Goertzel detector bank (far cheaper than a full FFT), then
synthesizes and decodes a dialled DTMF string. Draws the detector-bank response and the DTMF
row/column power grid for one digit.

    python examples/goertzel_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from goertzel import (  # noqa: E402
    goertzel_power_hz,
    detector_bank,
    dtmf_tone,
    dtmf_decode,
    DTMF_ROWS,
    DTMF_COLS,
)


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Goertzel algorithm: single-frequency detection without the full FFT\n")

    fs = 8000
    n = 800
    rng = _lcg(2024)
    # a 1000 Hz tone buried in noise
    signal = [math.sin(2 * math.pi * 1000 * i / fs) + 1.5 * (rng() * 2 - 1) for i in range(n)]

    print("  Tone detection: a 1000 Hz sinusoid buried in strong noise.")
    probe = [500, 800, 1000, 1200, 1500, 2000]
    powers = detector_bank(signal, probe, fs)
    pmax = max(powers.values())
    print(f"  {'freq (Hz)':>10}  {'Goertzel power':>15}")
    for f in probe:
        bar = "#" * int(powers[f] / pmax * 30)
        print(f"  {f:>10}  {powers[f]:>15.1f}  {bar}")
    detected = max(powers, key=lambda f: powers[f])
    print(f"    -> strongest at {detected} Hz (the hidden tone), one O(n) recurrence per probe.\n")

    # DTMF
    print("  DTMF touch-tone decoding: dial a number, decode from the audio.")
    dial = "1-800-2468"
    decoded = ""
    for ch in dial:
        if ch == "-":
            decoded += "-"
            continue
        tone = dtmf_tone(ch, 0.05, fs)
        decoded += dtmf_decode(tone, fs) or "?"
    print(f"    dialled:  {dial}")
    print(f"    decoded:  {decoded}   (match: {decoded == dial})")

    print("\n  Each key is a sum of a row and a column tone; two Goertzel detectors per band find")
    print("  the strongest, and the row x column intersection names the key -- how real phones do it.")

    _svg(os.path.join(outdir, "goertzel.svg"), signal, fs, probe, powers)
    print(f"\n  wrote {os.path.join(outdir, 'goertzel.svg')}")


def _svg(path, signal, fs, probe, powers, width=760, height=410):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Left: Goertzel power at probe frequencies. Right: DTMF row x column grid for key 5</text>',
    ]

    # ---- left: detector bar chart -------------------------------------------------------
    ox, oy, ow, oh = 50, 60, 300, 300
    pmax = max(powers.values())
    bw = ow / len(probe)
    parts.append(f'<line x1="{ox}" y1="{oy+oh}" x2="{ox+ow}" y2="{oy+oh}" stroke="#8b949e"/>')
    for i, f in enumerate(probe):
        h = oh * powers[f] / pmax
        x = ox + i * bw
        col = "#06d6a0" if powers[f] == pmax else "#4dabf7"
        parts.append(f'<rect x="{x+4:.0f}" y="{oy+oh-h:.0f}" width="{bw-10:.0f}" height="{h:.0f}" '
                     f'fill="{col}"/>')
        parts.append(f'<text x="{x+bw/2:.0f}" y="{oy+oh+16:.0f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{f}</text>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+30:.0f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="middle">probe frequency (Hz)</text>')

    # ---- right: DTMF power grid for key '5' --------------------------------------------
    tone5 = dtmf_tone("5", 0.05, fs)
    rp = detector_bank(tone5, DTMF_ROWS, fs)
    cp = detector_bank(tone5, DTMF_COLS, fs)
    gx, gy, cell = 440, 70, 62
    allmax = max(max(rp.values()), max(cp.values()))
    keys = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"], ["*", "0", "#"]]
    # column headers (col tone power)
    for ci in range(3):
        t = cp[DTMF_COLS[ci]] / allmax
        parts.append(f'<text x="{gx + ci*cell + cell/2:.0f}" y="{gy-6}" fill="#ffd43b" '
                     f'font-size="8" text-anchor="middle">{DTMF_COLS[ci]}</text>')
    for ri in range(4):
        parts.append(f'<text x="{gx-8}" y="{gy + ri*cell + cell/2:.0f}" fill="#ffd43b" '
                     f'font-size="8" text-anchor="end">{DTMF_ROWS[ri]}</text>')
        for ci in range(3):
            # cell brightness = product of row & col detection (only key 5 lights up)
            lit = (keys[ri][ci] == "5")
            fill = "#06d6a0" if lit else "#161b22"
            parts.append(f'<rect x="{gx + ci*cell}" y="{gy + ri*cell}" width="{cell-4}" '
                         f'height="{cell-4}" rx="4" fill="{fill}" stroke="#30363d"/>')
            tc = "#0d1117" if lit else "#8b949e"
            parts.append(f'<text x="{gx + ci*cell + cell/2:.0f}" y="{gy + ri*cell + cell/2+4:.0f}" '
                         f'fill="{tc}" font-size="16" text-anchor="middle">{keys[ri][ci]}</text>')
    parts.append(f'<text x="{gx}" y="{gy + 4*cell + 16}" fill="#8b949e" font-size="9">'
                 f'697+1336 Hz light up key 5</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
