"""Demo: hidden Markov models -- the occasionally-dishonest casino.

A dealer secretly switches between a fair die and one loaded toward sixes. We see only the rolls.
Viterbi recovers the hidden fair/loaded path, the forward-backward posterior shades how sure we
are at each step, and Baum-Welch relearns the model from the rolls alone -- its log-likelihood
climbing monotonically toward the true model's.

    python examples/hmm_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hmm import HMM, baum_welch_best  # noqa: E402


def _sample_sequence(model, T, seed):
    state = seed

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    def draw(dist):
        r = rng()
        c = 0.0
        for i, p in enumerate(dist):
            c += p
            if r < c:
                return i
        return len(dist) - 1

    states = [draw(model.start)]
    obs = [draw(model.emit[states[0]])]
    for _ in range(T - 1):
        states.append(draw(model.trans[states[-1]]))
        obs.append(draw(model.emit[states[-1]]))
    return states, obs


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    # state 0 = fair die, state 1 = loaded (six four times as likely); sticky switching
    start = [0.6, 0.4]
    trans = [[0.93, 0.07], [0.12, 0.88]]
    emit = [[1 / 6] * 6, [0.08, 0.08, 0.08, 0.08, 0.08, 0.60]]
    true = HMM(start, trans, emit)

    states, obs = _sample_sequence(true, 260, seed=7)

    print("Hidden Markov model: the occasionally-dishonest casino\n")
    print(f"  {len(obs)} die rolls; hidden state is fair (0) or loaded (1)\n")

    ll = true.log_likelihood(obs)
    path, logp = true.viterbi(obs)
    vit_acc = sum(1 for i in range(len(states)) if path[i] == states[i]) / len(states)
    frac_loaded_true = sum(states) / len(states)
    frac_loaded_vit = sum(path) / len(path)
    print(f"  forward log-likelihood of the rolls: {ll:.2f}")
    print(f"  Viterbi decode accuracy vs the true hidden path: {vit_acc:.1%}")
    print(f"  fraction of time loaded -- true {frac_loaded_true:.2f}, "
          f"Viterbi {frac_loaded_vit:.2f}\n")

    # posterior: how sure are we the die is loaded at each step?
    gamma = true.posterior(obs)
    hi = sum(1 for g in gamma if g[1] > 0.9)
    print(f"  forward-backward posterior: {hi} rolls flagged loaded with >90% confidence\n")

    print("  Baum-Welch relearns the model from the rolls alone (no hidden states given):")
    model, best_ll = baum_welch_best([obs], n_states=2, n_symbols=6, restarts=6,
                                    max_iter=100, seed=3)
    # identify which learned state is the loaded one (higher P of rolling a six)
    loaded = 0 if model.emit[0][5] > model.emit[1][5] else 1
    print(f"    learned P(roll a six | loaded state) = {model.emit[loaded][5]:.2f}  (true 0.60)")
    print(f"    learned P(roll a six | fair state)   = {model.emit[1 - loaded][5]:.2f}  (true 0.17)")
    print(f"    learned P(stay loaded)               = {model.trans[loaded][loaded]:.2f}  (true 0.88)")
    print(f"    trained log-likelihood {best_ll:.2f}  vs true-model {ll:.2f}\n")

    print("  Three exact dynamic-programming algorithms over the trellis: forward sums all paths")
    print("  (evaluate), Viterbi maxes over them (decode), Baum-Welch uses forward-backward soft")
    print("  counts to relearn the matrices (EM). All in log space so long sequences never underflow.")

    _svg(os.path.join(outdir, "hmm.svg"), states, obs, path, gamma, model, loaded)
    print(f"\n  wrote {os.path.join(outdir, 'hmm.svg')}")


def _svg(path_svg, states, obs, vit, gamma, model, loaded, width=760, height=430):
    T = len(obs)
    show = min(T, 120)      # first 120 rolls keep the strip readable
    x0, x1 = 45, width - 20
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Hidden Markov model: decoding the dishonest casino</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'first {show} rolls: true hidden state, Viterbi decode, and the posterior P(loaded) '
        f'ribbon</text>',
    ]

    def X(t):
        return x0 + t / (show - 1) * (x1 - x0)

    def band(y, label, colorfn):
        parts.append(f'<text x="20" y="{y+4}" fill="#8b949e" font-size="10">{label}</text>')
        w = (x1 - x0) / show
        for t in range(show):
            parts.append(f'<rect x="{X(t):.1f}" y="{y-8:.1f}" width="{w+0.6:.1f}" height="16" '
                         f'fill="{colorfn(t)}"/>')

    # true state band: fair=blue, loaded=red
    band(90, "true", lambda t: "#ff6b6b" if states[t] == 1 else "#16324f")
    # Viterbi decode band (loaded index may be either 0/1 in the *true* model, here true model
    # so state 1 = loaded directly)
    band(130, "viterbi", lambda t: "#ff6b6b" if vit[t] == 1 else "#16324f")

    # posterior P(loaded) as a shaded ribbon + curve
    py0, py1 = height - 60, 175
    parts.append(f'<text x="20" y="{(py0+py1)/2:.1f}" fill="#8b949e" font-size="10">'
                 f'P(loaded)</text>')
    parts.append(f'<line x1="{x0}" y1="{py0}" x2="{x1}" y2="{py0}" stroke="#8b949e" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{py0}" x2="{x0}" y2="{py1}" stroke="#8b949e" stroke-width="1"/>')

    def PY(p):
        return py0 - p * (py0 - py1)

    pts = " ".join(f"{X(t):.1f},{PY(gamma[t][1]):.1f}" for t in range(show))
    # area under the curve
    area = f"{x0},{py0:.1f} " + pts + f" {X(show-1):.1f},{py0:.1f}"
    parts.append(f'<polygon points="{area}" fill="#ffd43b" opacity="0.18"/>')
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#ffd43b" stroke-width="1.8"/>')
    parts.append(f'<line x1="{x0}" y1="{PY(0.5):.1f}" x2="{x1}" y2="{PY(0.5):.1f}" '
                 f'stroke="#30363d" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{x1-2:.1f}" y="{PY(1.0)-2:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">1.0</text>')

    # emission bars: learned loaded vs fair distributions
    bx0 = x0
    by = py0 + 20
    parts.append(f'<text x="20" y="{by+10}" fill="#8b949e" font-size="10">learned emissions '
                 f'(loaded=red, fair=blue): die faces 1-6</text>')
    bw = 26
    for face in range(6):
        lp = model.emit[loaded][face]
        fp = model.emit[1 - loaded][face]
        gx = bx0 + 180 + face * (bw * 2 + 8)
        parts.append(f'<rect x="{gx}" y="{by+30-lp*40:.1f}" width="{bw-2}" height="{lp*40:.1f}" fill="#ff6b6b"/>')
        parts.append(f'<rect x="{gx+bw}" y="{by+30-fp*40:.1f}" width="{bw-2}" height="{fp*40:.1f}" fill="#4dabf7"/>')
        parts.append(f'<text x="{gx+bw-2:.1f}" y="{by+42}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{face+1}</text>')

    parts.append("</svg>")
    with open(path_svg, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
