"""Demo: the Burrows-Wheeler transform and the bzip2 compression pipeline.

Shows BWT rearranging text so like-context characters cluster into runs (the mean run length jumps),
then chains move-to-front and run-length encoding to compress a repetitive string, confirming the
whole pipeline is lossless. Draws the runniness gain and the compression ratio.

    python examples/bwt_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bwt import (bwt_transform, bwt_inverse, move_to_front_encode, rle_encode,  # noqa: E402
                 compress, decompress, bwt_runniness_gain, _mean_run_length)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Burrows-Wheeler transform + the bzip2-style pipeline\n")

    # BWT clusters like-context characters
    print("  BWT rearranges text so characters sharing a context cluster into runs:")
    for text in ["banana", "mississippi", "abracadabra"]:
        t = bwt_transform(text)
        print(f"    {text:>12} -> BWT {t.replace(chr(0), '$')!r}  "
              f"(reversible: {bwt_inverse(t) == text})")

    print("\n  On structured text the mean run length jumps (more compressible):")
    for label, text in [("repeated phrase", "the cat sat on the mat " * 5),
                        ("DNA-like", "ACGTACGTACGT" * 6),
                        ("English-ish", "she sells sea shells by the sea shore " * 3)]:
        before, after = bwt_runniness_gain(text)
        print(f"    {label:>16}: mean run {before:.2f} -> {after:.2f}  "
              f"({after / before:.1f}x runnier)")

    # the full pipeline on a repetitive string
    print("\n  Full pipeline (BWT -> move-to-front -> run-length) on repetitive input:")
    for text in ["ababababab" * 10, "aaaaabbbbbccccc" * 8, "the theme of theses " * 6]:
        pairs, alpha = compress(text)
        restored = decompress(pairs, alpha)
        print(f"    input {len(text):>4} chars -> {len(pairs):>4} RLE pairs "
              f"({len(text) / len(pairs):.1f}x fewer tokens), lossless: {restored == text}")

    # a worked mini-example of all three stages
    sample = "banana"
    bw = bwt_transform(sample)
    codes, alpha = move_to_front_encode(bw)
    rle = rle_encode(codes)
    print(f"\n  Worked example on {sample!r}:")
    print(f"    BWT:            {bw.replace(chr(0), '$')!r}")
    print(f"    move-to-front:  {codes}")
    print(f"    run-length:     {rle}")
    print(f"    decode:         {decompress(rle, alpha)!r}")

    print("\n  BWT alone compresses nothing -- it is a reversible permutation -- but it makes the")
    print("  data 'runnier' by clustering same-context bytes. Move-to-front then turns runs into")
    print("  small numbers (mostly zeros), and run-length collapses those into (value, count) pairs.")
    print("  Every stage inverts exactly, so the pipeline is lossless: this is bzip2's core.")

    _svg(os.path.join(outdir, "bwt.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'bwt.svg')}")


def _svg(path, width=760, height=400):
    texts = [("repeated\nphrase", "the cat sat on the mat " * 5),
             ("DNA-like", "ACGTACGTACGT" * 6),
             ("English", "she sells sea shells by the sea shore " * 3),
             ("random", "qwertyuiopasdfgh" * 4)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Burrows-Wheeler: mean run length before (grey) vs after (green) the transform</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'a runnier string compresses better; structured text gains most, random text little</text>',
    ]
    y0, y1 = height - 55, 75
    plot_h = y0 - y1
    maxrun = max(max(bwt_runniness_gain(t)) for _, t in texts)
    gw = (width - 90) / len(texts)
    bw = gw * 0.3
    for i, (label, text) in enumerate(texts):
        before, after = bwt_runniness_gain(text)
        cx = 70 + i * gw
        for j, (val, col) in enumerate([(before, "#8b949e"), (after, "#06d6a0")]):
            h = val / maxrun * plot_h
            x = cx + j * (bw + 4)
            parts.append(f'<rect x="{x:.1f}" y="{y0 - h:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                         f'fill="{col}"/>')
            parts.append(f'<text x="{x + bw / 2:.1f}" y="{y0 - h - 4:.1f}" fill="{col}" '
                         f'font-size="9" text-anchor="middle">{val:.1f}</text>')
        parts.append(f'<text x="{cx + bw:.1f}" y="{y0 + 16:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{label.replace(chr(10), " ")}</text>')
    parts.append(f'<line x1="55" y1="{y0}" x2="{width-30}" y2="{y0}" stroke="#8b949e" stroke-width="1.1"/>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
