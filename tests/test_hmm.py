"""Tests for hmm: forward/backward agreement, Viterbi decoding, Baum-Welch EM."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hmm import HMM, baum_welch, baum_welch_best, _logsumexp, _log

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- helpers ---------------------------------------------------------------
check("logsumexp basic", approx(_logsumexp([0.0, 0.0]), math.log(2.0), 1e-12))
check("log of zero is -inf", _log(0.0) == float("-inf"))

# --- a well-separated, sticky 2-state model --------------------------------
start = [0.5, 0.5]
trans = [[0.9, 0.1], [0.1, 0.9]]
emit = [[0.45, 0.45, 0.05, 0.05], [0.05, 0.05, 0.45, 0.45]]
h = HMM(start, trans, emit)

# --- sample a sequence with a known hidden path (LCG, high bits) -----------
state = 11


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def sample(dist):
    r = rng()
    c = 0.0
    for i, p in enumerate(dist):
        c += p
        if r < c:
            return i
    return len(dist) - 1


S = [sample(start)]
O = [sample(emit[S[0]])]
for _ in range(300):
    S.append(sample(trans[S[-1]]))
    O.append(sample(emit[S[-1]]))

# --- forward and backward agree on the likelihood --------------------------
alpha = h.forward(O)
beta = h.backward(O)
ll_forward = h.log_likelihood(O)
ll_fb = _logsumexp([alpha[0][i] + beta[0][i] for i in range(h.n)])
check("forward == backward likelihood", approx(ll_forward, ll_fb, 1e-6))
# and it equals logsumexp over any time slice
mid = 150
ll_mid = _logsumexp([alpha[mid][i] + beta[mid][i] for i in range(h.n)])
check("likelihood constant across time slices", approx(ll_forward, ll_mid, 1e-6))

# --- Viterbi recovers the planted state path (up to label swap) ------------
path, logp = h.viterbi(O)
acc = sum(1 for i in range(len(S)) if path[i] == S[i]) / len(S)
acc = max(acc, 1 - acc)
check("Viterbi recovers planted path", acc >= 0.90)
check("Viterbi path length matches", len(path) == len(O))
# Viterbi path is one specific path, so its logprob <= total forward logprob
check("Viterbi logprob <= forward logprob", logp <= ll_forward + 1e-9)

# --- posterior marginals are valid probabilities ---------------------------
gamma = h.posterior(O)
check("posterior rows sum to 1", all(approx(sum(row), 1.0, 1e-9) for row in gamma))
check("posterior in [0,1]", all(0.0 <= v <= 1.0 for row in gamma for v in row))
# posterior argmax should also mostly match the true path
pacc = sum(1 for i in range(len(S)) if (0 if gamma[i][0] > gamma[i][1] else 1) == S[i]) / len(S)
pacc = max(pacc, 1 - pacc)
check("posterior decode recovers path", pacc >= 0.90)

# --- a tiny hand-checkable forward computation -----------------------------
# 1 state that always emits symbol 0: P(obs) = 1 for any all-zero sequence
solo = HMM([1.0], [[1.0]], [[1.0]])
check("degenerate model gives logL 0", approx(solo.log_likelihood([0, 0, 0]), 0.0, 1e-12))
check("degenerate Viterbi path", solo.viterbi([0, 0])[0] == [0, 0])

# --- Baum-Welch increases the log-likelihood every iteration ---------------
model, hist = baum_welch([O], n_states=2, n_symbols=4, max_iter=60, seed=1)
mono = all(hist[i + 1] >= hist[i] - 1e-6 for i in range(len(hist) - 1))
check("Baum-Welch log-likelihood monotone", mono)
check("Baum-Welch improves over init", hist[-1] > hist[0])
# trained model should reach near the true-model likelihood on this data
check("Baum-Welch reaches good likelihood", hist[-1] >= h.log_likelihood(O) - 15.0)

# --- learned model separates the two emission regimes ----------------------
# one learned state should favour symbols {0,1}, the other {2,3}
s0_low = model.emit[0][0] + model.emit[0][1]
s1_low = model.emit[1][0] + model.emit[1][1]
check("learned states specialize on symbol groups", abs(s0_low - s1_low) > 0.5)

# --- multi-restart returns the best of several fits ------------------------
best_model, best_ll = baum_welch_best([O], 2, 4, restarts=4, max_iter=50, seed=2)
check("best-of-restarts >= single seed", best_ll >= hist[-1] - 20.0)
check("best model is an HMM", isinstance(best_model, HMM))

# --- rows of every matrix are valid distributions --------------------------
check("start sums to 1", approx(sum(model.start), 1.0, 1e-6))
check("trans rows sum to 1", all(approx(sum(r), 1.0, 1e-6) for r in model.trans))
check("emit rows sum to 1", all(approx(sum(r), 1.0, 1e-6) for r in model.emit))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all hmm tests passed")
