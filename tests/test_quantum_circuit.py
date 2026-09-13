"""Tests for the quantum circuit simulator: unitarity, Bell entanglement, Deutsch-Jozsa, Grover."""

import cmath
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantum_circuit import (  # noqa: E402
    QuantumState,
    H, X, Y, Z, S, T, I2,
    bell_pair,
    deutsch_jozsa,
    grover_search,
    grover_optimal_iterations,
    is_unitary_on_state,
    rx, ry, phase_gate,
)


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


def main():
    # ---- 1. all standard gates are unitary (norm-preserving) ----------------------------
    for name, g in [("H", H), ("X", X), ("Y", Y), ("Z", Z), ("S", S), ("T", T),
                    ("rx", rx(0.7)), ("ry", ry(1.1)), ("phase", phase_gate(0.5))]:
        check(f"{name} is unitary", is_unitary_on_state(g))

    # ---- 2. H|0> is an equal superposition ----------------------------------------------
    st = QuantumState(1)
    st.apply_gate(H, 0)
    p = st.probabilities()
    check("H|0> = equal superposition", abs(p[0] - 0.5) < 1e-12 and abs(p[1] - 0.5) < 1e-12)

    # ---- 3. HH = I ----------------------------------------------------------------------
    st = QuantumState(1)
    st.apply_gate(H, 0)
    st.apply_gate(H, 0)
    check("HH|0> = |0>", abs(st.amp[0] - 1) < 1e-12 and abs(st.amp[1]) < 1e-12)

    # ---- 4. X flips |0> to |1> ----------------------------------------------------------
    st = QuantumState(1)
    st.apply_gate(X, 0)
    check("X|0> = |1>", abs(st.amp[1] - 1) < 1e-12)

    # ---- 5. every state stays normalized after a mixed circuit --------------------------
    st = QuantumState(3)
    st.apply_gate(H, 0)
    st.apply_gate(ry(0.9), 1)
    st.cnot(0, 2)
    st.toffoli(0, 1, 2)
    st.apply_gate(T, 1)
    check("norm preserved through circuit", abs(st.norm() - 1.0) < 1e-12, f"norm {st.norm()}")

    # ---- 6. Bell pair: only 00 and 11, each 1/2 -----------------------------------------
    bell = bell_pair()
    corr = bell.measure_correlations()
    check("Bell: P(00)=P(11)=1/2", abs(corr["00"] - 0.5) < 1e-12 and abs(corr["11"] - 0.5) < 1e-12)
    check("Bell: P(01)=P(10)=0", abs(corr["01"]) < 1e-12 and abs(corr["10"]) < 1e-12)

    # ---- 7. CNOT truth table on basis states --------------------------------------------
    for c_in, t_in, t_out in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        st = QuantumState(2)
        if c_in:
            st.apply_gate(X, 0)
        if t_in:
            st.apply_gate(X, 1)
        st.cnot(0, 1)
        idx = (c_in << 1) | t_out
        check(f"CNOT({c_in},{t_in})->({c_in},{t_out})", abs(st.amp[idx] - 1) < 1e-12)

    # ---- 8. Toffoli flips target only when both controls are 1 --------------------------
    st = QuantumState(3)
    st.apply_gate(X, 0)
    st.apply_gate(X, 1)  # both controls 1
    st.toffoli(0, 1, 2)
    check("Toffoli(1,1,0)->(1,1,1)", abs(st.amp[0b111] - 1) < 1e-12)
    st = QuantumState(3)
    st.apply_gate(X, 0)  # only one control
    st.toffoli(0, 1, 2)
    check("Toffoli(1,0,0) unchanged", abs(st.amp[0b100] - 1) < 1e-12)

    # ---- 9. Deutsch-Jozsa: constant oracles -> 'constant' -------------------------------
    n = 3
    check("DJ constant-0", deutsch_jozsa(lambda x: 0, n) == "constant")
    check("DJ constant-1", deutsch_jozsa(lambda x: 1, n) == "constant")

    # ---- 10. Deutsch-Jozsa: balanced oracles -> 'balanced' ------------------------------
    check("DJ balanced (parity)", deutsch_jozsa(lambda x: bin(x).count("1") & 1, n) == "balanced")
    check("DJ balanced (first bit)", deutsch_jozsa(lambda x: (x >> (n - 1)) & 1, n) == "balanced")

    # ---- 11. Grover amplifies the marked state ------------------------------------------
    for n in [3, 4, 5]:
        marked = (1 << n) - 2  # some non-trivial index
        st, iters = grover_search(marked, n)
        pm = st.probability(marked)
        check(f"Grover n={n} finds marked (P>0.9)", pm > 0.9, f"P={pm:.3f} in {iters} iters")
        check(f"Grover n={n} most_likely == marked", st.most_likely() == marked)

    # ---- 12. Grover iteration count matches the (pi/4)sqrt(N) formula -------------------
    check("Grover iters n=4", grover_optimal_iterations(4) == max(1, round((math.pi / 4) * 4)))

    # ---- 13. Grover single iteration on n=2 hits the marked state exactly ---------------
    # for N=4, one Grover iteration gives probability 1 on the marked state
    st, iters = grover_search(2, 2, iterations=1)
    check("Grover n=2 exact after 1 iter", abs(st.probability(2) - 1.0) < 1e-12,
          f"P={st.probability(2):.6f}")

    # ---- 14. brute-force amplitude check for one Grover step ----------------------------
    # after H^n, amp = 1/sqrt(N) each; oracle flips marked; diffusion 2*mean - amp.
    n = 3
    N = 1 << n
    marked = 5
    amp = [1 / math.sqrt(N)] * N
    amp[marked] = -amp[marked]
    mean = sum(amp) / N
    amp = [2 * mean - a for a in amp]
    st, _ = grover_search(marked, n, iterations=1)
    match = max(abs(st.amp[i] - amp[i]) for i in range(N))
    check("Grover step == brute amplitude calc", match < 1e-12, f"max diff {match:.2e}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
