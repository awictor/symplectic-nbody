"""Demo: page-replacement algorithms -- counting faults and Belady's paradox.

Runs FIFO, LRU, Clock, LFU, and Belady's optimal over a reference string, compares their page-fault
counts against the optimal minimum, and demonstrates Belady's anomaly (more frames -> more FIFO faults).
Draws the fault counts and the anomaly curve.

    python examples/page_replacement_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from page_replacement import (fifo, lru, clock, lfu, optimal, compare,  # noqa: E402
                              belady_anomaly_example)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Page replacement: which page to evict, and the optimal we can only approximate\n")

    refs = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
    frames = 3
    print(f"  reference string ({len(refs)} accesses), {frames} frames:")
    print(f"    {refs}\n")
    c = compare(refs, frames)
    opt = c["OPTIMAL"]
    print("    policy     faults   excess over optimal")
    for name in ("OPTIMAL", "LFU", "LRU", "CLOCK", "FIFO"):
        print(f"    {name:<8}   {c[name]:>4}      {'+' + str(c[name] - opt) if c[name] > opt else '(optimal)'}")
    print("    (Belady's optimal evicts the page whose next use is farthest away -- unbeatable but")
    print("     needs the future; real policies approximate it from the past)\n")

    # Belady's anomaly
    anom_refs, f3, f4 = belady_anomaly_example()
    print("  Belady's anomaly -- MORE frames, MORE faults (only FIFO):")
    print(f"    string {anom_refs}")
    for fr in (3, 4):
        print(f"    FIFO with {fr} frames: {fifo(anom_refs, fr)[0]} faults"
              + ("   <- more frames, more faults!" if fr == 4 else ""))
    print(f"    LRU is immune: {lru(anom_refs, 3)[0]} faults at 3, {lru(anom_refs, 4)[0]} at 4\n")

    print("  LRU and Belady's optimal are STACK algorithms -- adding frames can never increase faults --")
    print("  so they are anomaly-free; FIFO ignores usage and can misbehave. Clock is the cheap LRU")
    print("  approximation real kernels run, trading a little accuracy for O(1) eviction.")

    _svg(os.path.join(outdir, "page_replacement.svg"), refs)
    print(f"\n  wrote {os.path.join(outdir, 'page_replacement.svg')}")


def _svg(path, refs, width=760, height=420):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="16">'
        f'Page faults by policy (top); Belady\'s anomaly under FIFO (bottom)</text>',
    ]

    # top: bar chart of faults at 3 frames
    c = compare(refs, 3)
    names = ["OPTIMAL", "LFU", "LRU", "CLOCK", "FIFO"]
    cols = {"OPTIMAL": "#06d6a0", "LFU": "#4dabf7", "LRU": "#b197fc",
            "CLOCK": "#ffd43b", "FIFO": "#ff6b6b"}
    ox, oy = 60, 210
    pw, ph = width - 120, 150
    fmax = max(c.values())
    bw = pw / len(names)
    for i, nm in enumerate(names):
        h = ph * c[nm] / fmax
        x = ox + i * bw
        parts.append(f'<rect x="{x:.1f}" y="{oy-h:.1f}" width="{bw-14:.1f}" height="{h:.1f}" '
                     f'fill="{cols[nm]}"/>')
        parts.append(f'<text x="{x+(bw-14)/2:.1f}" y="{oy-h-6:.1f}" fill="{cols[nm]}" '
                     f'font-size="12" text-anchor="middle">{c[nm]}</text>')
        parts.append(f'<text x="{x+(bw-14)/2:.1f}" y="{oy+16:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{nm}</text>')
    parts.append(f'<text x="{ox}" y="{oy-ph-4:.0f}" fill="#8b949e" font-size="11">'
                 f'fewer faults is better; OPTIMAL is the unbeatable floor</text>')

    # bottom: FIFO faults vs frame count (anomaly)
    anom, _, _ = belady_anomaly_example()
    bx, by = 60, 400
    bw2, bh = width - 120, 130
    frames_range = list(range(1, 7))
    ff = [fifo(anom, fr)[0] for fr in frames_range]
    fl = [lru(anom, fr)[0] for fr in frames_range]
    fmax2 = max(max(ff), max(fl))
    fmin2 = min(min(ff), min(fl))
    span = (fmax2 - fmin2) or 1

    def px(i):
        return bx + i / (len(frames_range) - 1) * bw2

    def py(v):
        return by - (v - fmin2) / span * bh

    for series, col, lab in ((ff, "#ff6b6b", "FIFO"), (fl, "#b197fc", "LRU")):
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(series))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        for i, v in enumerate(series):
            parts.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="3" fill="{col}"/>')
    parts.append(f'<text x="{bx}" y="{by-bh-6:.0f}" fill="#8b949e" font-size="11">'
                 f'FIFO (red) faults rise from 3 to 4 frames -- Belady\'s anomaly; LRU (purple) only falls</text>')
    for i, fr in enumerate(frames_range):
        parts.append(f'<text x="{px(i):.0f}" y="{by+16:.0f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{fr}f</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
