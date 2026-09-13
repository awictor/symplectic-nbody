"""Tests for QFT: matches classical DFT, inverse round-trip, basis-state ramp, phase estimation."""

import cmath
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qft import (  # noqa: E402
    qft,
    inverse_qft,
    classical_dft,
    state_from_amplitudes,
    phase_estimation,
)
from quantum_circuit import QuantumState  # noqa: E402


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def _close(a, b, tol=1e-9):
    return all(abs(a[i] - b[i]) < tol for i in range(len(a)))


def main():
    rng = _lcg(1)

    # ---- 1. QFT matches classical DFT for random states, n = 1..4 -----------------------
    ok = True
    which = None
    for n in [1, 2, 3, 4]:
        N = 1 << n
        # random normalized complex amplitudes
        amps = [complex(rng() - 0.5, rng() - 0.5) for _ in range(N)]
        norm = math.sqrt(sum(abs(a) ** 2 for a in amps))
        amps = [a / norm for a in amps]
        st = state_from_amplitudes(amps)
        qft(st)
        ref = classical_dft(amps, inverse=False)
        if not _close(st.amp, ref, 1e-9):
            ok = False
            which = n
            # show max diff
            md = max(abs(st.amp[i] - ref[i]) for i in range(N))
            check("QFT == classical DFT", False, f"n={n} max diff {md:.2e}")
            break
    if ok:
        check("QFT == classical DFT (n=1..4)", True)

    # ---- 2. inverse QFT undoes QFT (round-trip to identity) -----------------------------
    n = 4
    N = 1 << n
    amps = [complex(rng() - 0.5, rng() - 0.5) for _ in range(N)]
    norm = math.sqrt(sum(abs(a) ** 2 for a in amps))
    amps = [a / norm for a in amps]
    st = state_from_amplitudes(amps)
    qft(st)
    inverse_qft(st)
    check("inverse_qft(qft(x)) == x", _close(st.amp, amps, 1e-9),
          f"max diff {max(abs(st.amp[i]-amps[i]) for i in range(N)):.2e}")

    # ---- 3. QFT of |0> is the uniform superposition -------------------------------------
    n = 3
    st = QuantumState(n)  # |000>
    qft(st)
    N = 1 << n
    check("QFT|0> = uniform superposition",
          all(abs(abs(st.amp[k]) - 1 / math.sqrt(N)) < 1e-9 for k in range(N)))

    # ---- 4. QFT of uniform superposition is |0> -----------------------------------------
    N = 1 << n
    amps = [1 / math.sqrt(N)] * N
    st = state_from_amplitudes(amps)
    qft(st)
    check("QFT(uniform) = |0>", abs(st.amp[0] - 1) < 1e-9 and all(abs(st.amp[k]) < 1e-9 for k in range(1, N)))

    # ---- 5. QFT of a basis state |k> is a uniform-magnitude phase ramp ------------------
    n = 3
    N = 1 << n
    k0 = 3
    st = QuantumState(n)
    st.amp = [0j] * N
    st.amp[k0] = 1 + 0j
    qft(st)
    # every amplitude has magnitude 1/sqrt(N)
    check("QFT|k> uniform magnitude", all(abs(abs(st.amp[j]) - 1 / math.sqrt(N)) < 1e-9 for j in range(N)))
    # phase of amp[j] should be 2 pi k0 j / N (up to global)
    ok = True
    for j in range(N):
        expected = cmath.exp(2j * math.pi * k0 * j / N) / math.sqrt(N)
        if abs(st.amp[j] - expected) > 1e-9:
            ok = False
            break
    check("QFT|k> phase ramp correct", ok)

    # ---- 6. QFT is unitary (norm preserved) ---------------------------------------------
    amps = [complex(rng() - 0.5, rng() - 0.5) for _ in range(16)]
    norm = math.sqrt(sum(abs(a) ** 2 for a in amps))
    amps = [a / norm for a in amps]
    st = state_from_amplitudes(amps)
    qft(st)
    check("QFT preserves norm", abs(st.norm() - 1.0) < 1e-9, f"norm {st.norm()}")

    # ---- 7. inverse QFT matches inverse classical DFT -----------------------------------
    n = 3
    N = 1 << n
    amps = [complex(rng() - 0.5, rng() - 0.5) for _ in range(N)]
    norm = math.sqrt(sum(abs(a) ** 2 for a in amps))
    amps = [a / norm for a in amps]
    st = state_from_amplitudes(amps)
    inverse_qft(st)
    ref = classical_dft(amps, inverse=True)
    check("inverse QFT == inverse DFT", _close(st.amp, ref, 1e-9),
          f"max diff {max(abs(st.amp[i]-ref[i]) for i in range(N)):.2e}")

    # ---- 8. phase estimation recovers exact dyadic phases -------------------------------
    for n in [3, 4, 5]:
        N = 1 << n
        # choose a phase exactly representable in n bits
        m_true = 3
        phi = m_true / N
        est, m = phase_estimation(phi, n)
        check(f"phase estimation n={n}: exact phi={phi}", m == m_true, f"got m={m}")

    # ---- 9. phase estimation approximates non-dyadic phases -----------------------------
    n = 6
    N = 1 << n
    phi = 0.31
    est, m = phase_estimation(phi, n)
    check("phase estimation approximates 0.31", abs(est - phi) <= 1.0 / N + 1e-9,
          f"est {est:.4f} vs {phi} (res 1/{N})")

    # ---- 10. QFT on a subset of qubits leaves others coherent (norm still 1) ------------
    st = QuantumState(4)
    st.apply_gate(__import__("quantum_circuit").X, 0)  # |1000>
    qft(st, qubits=[1, 2, 3])
    check("partial QFT preserves norm", abs(st.norm() - 1.0) < 1e-9)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
