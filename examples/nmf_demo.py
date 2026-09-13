"""Demo: non-negative matrix factorization -- parts-based decomposition by multiplicative updates.

Runs NMF as a tiny topic model: a term-document count matrix built from two hidden topics is
factored back into topic-word and document-topic matrices. Shows the recovered topics and the
monotone error curve. Draws the factorization V ~ W H and the error decay.

    python examples/nmf_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nmf import nmf, frobenius_error, reconstruct  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Non-negative matrix factorization: parts-based decomposition (Lee-Seung)\n")

    # a tiny term-document matrix, 6 words x 8 documents, built from two latent topics:
    # topic SPACE: {orbit, star, galaxy}; topic COOKING: {recipe, oven, flour}
    words = ["orbit", "star", "galaxy", "recipe", "oven", "flour"]
    # documents: first 4 are space-ish, last 4 cooking-ish, with a little overlap
    V = [
        [5, 4, 3, 4, 0, 0, 1, 0],   # orbit
        [4, 5, 4, 3, 0, 1, 0, 0],   # star
        [3, 3, 5, 4, 1, 0, 0, 0],   # galaxy
        [0, 0, 1, 0, 5, 4, 3, 4],   # recipe
        [0, 1, 0, 0, 4, 5, 4, 3],   # oven
        [1, 0, 0, 0, 3, 3, 5, 4],   # flour
    ]

    print(f"  Term-document count matrix: {len(words)} words x {len(V[0])} documents.")
    W, H, curve = nmf(V, k=2, iterations=300, seed=7, track_error=True)

    print(f"\n  Factored V ~ W H with k=2 topics. Final error {curve[-1]:.3f} "
          f"(from {curve[0]:.3f}).\n")

    # normalize W columns to read topics as word distributions
    print("  Recovered topics (top words by weight in each column of W):")
    for t in range(2):
        col = [(words[i], W[i][t]) for i in range(len(words))]
        col.sort(key=lambda x: -x[1])
        top = ", ".join(f"{w}({v:.1f})" for w, v in col[:3])
        print(f"    topic {t}: {top}")

    print("\n  Document-topic weights (columns of H, which topic each doc leans on):")
    for d in range(len(V[0])):
        t0, t1 = H[0][d], H[1][d]
        lean = 0 if t0 > t1 else 1
        print(f"    doc {d}: topic0 {t0:5.2f}  topic1 {t1:5.2f}  -> topic {lean}")

    print("\n  Because everything is non-negative, topics ADD rather than cancel -- the columns of W")
    print("  are interpretable word groups, exactly what PCA's signed components cannot give.")

    _svg(os.path.join(outdir, "nmf.svg"), V, W, H, curve, words)
    print(f"\n  wrote {os.path.join(outdir, 'nmf.svg')}")


def _svg(path, V, W, H, curve, words, width=760, height=440):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Left: V ~ W H heatmaps. Right: reconstruction error decays monotonically</text>',
    ]

    def heatmap(M, x0, y0, cw, ch, vmax, title, colhi="#4dabf7"):
        rows, cols = len(M), len(M[0])
        parts.append(f'<text x="{x0}" y="{y0-6}" fill="#8b949e" font-size="10">{title}</text>')
        for i in range(rows):
            for j in range(cols):
                t = min(1.0, M[i][j] / vmax) if vmax > 0 else 0
                # blend dark -> colhi by intensity
                r = int(0x16 + t * (int(colhi[1:3], 16) - 0x16))
                g = int(0x1b + t * (int(colhi[3:5], 16) - 0x1b))
                b = int(0x22 + t * (int(colhi[5:7], 16) - 0x22))
                parts.append(f'<rect x="{x0 + j*cw:.0f}" y="{y0 + i*ch:.0f}" width="{cw-1:.0f}" '
                             f'height="{ch-1:.0f}" fill="rgb({r},{g},{b})"/>')

    vmax = max(max(row) for row in V)
    heatmap(V, 40, 70, 16, 18, vmax, "V (6 words x 8 docs)")
    wmax = max(max(row) for row in W)
    heatmap(W, 200, 70, 18, 18, wmax, "W (words x 2 topics)", "#06d6a0")
    hmax = max(max(row) for row in H)
    heatmap(H, 40, 240, 16, 22, hmax, "H (2 topics x 8 docs)", "#ffd43b")

    # error curve (right)
    ox, oy, ow, oh = 330, 90, 380, 300
    cmax = max(curve)
    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" '
                 f'stroke="#30363d"/>')
    pts = " ".join(f"{ox + ow*i/(len(curve)-1):.1f},{oy + oh*(1 - curve[i]/cmax):.1f}"
                   for i in range(len(curve)))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#ff6b6b" stroke-width="2"/>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+22:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">iteration</text>')
    parts.append(f'<text x="{ox-6}" y="{oy-4}" fill="#8b949e" font-size="10" text-anchor="end">'
                 f'||V - WH||</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
