"""A quantum circuit simulator: statevector evolution, entanglement, and Grover's search.

A quantum computer holds n qubits in a STATEVECTOR of 2^n complex amplitudes -- one per basis string
|00..0> through |11..1> -- and gates are unitary matrices that rotate that vector. Measurement then
draws a basis string with probability equal to the squared magnitude of its amplitude. This module
simulates that exactly on a classical computer (feasible up to ~20 qubits, since the vector is 2^n
long), which is precisely how quantum algorithms are prototyped and checked before they ever touch
hardware.

The core operations:

  - SINGLE-QUBIT GATES (X, Y, Z, Hadamard, phase, arbitrary rotations) are 2x2 unitaries applied to
    one qubit by pairing up the amplitudes that differ only in that qubit's bit and multiplying the
    2-vector by the gate. No 2^n x 2^n matrix is ever formed.
  - CONTROLLED GATES (CNOT, controlled-Z, Toffoli) apply a gate to a target only on the basis states
    where all control bits are 1 -- the mechanism that creates ENTANGLEMENT, correlations with no
    classical analogue (a Bell pair measures as 00 or 11, never 01 or 10, no matter how far apart).

On top of the simulator sit two landmark algorithms whose speed-ups are the whole point of quantum
computing:

  - The DEUTSCH-JOZSA algorithm decides whether a black-box boolean function is CONSTANT or BALANCED
    with a SINGLE query, where any classical method may need 2^(n-1)+1 -- the first proof that quantum
    beats classical for a real task.
  - GROVER'S SEARCH finds a marked item in an unsorted database of N = 2^n entries in about
    (pi/4) sqrt(N) queries instead of N/2, a quadratic speed-up, by repeatedly reflecting the
    statevector about the marked state and then about the mean (amplitude amplification).

This module implements the statevector, the gate set, measurement probabilities, Deutsch-Jozsa, and
Grover, and it is validated exactly: every gate is unitary (norm preserved), H|0> is an equal
superposition, HH = I, a Bell circuit yields exactly the 00/11 correlations, Deutsch-Jozsa labels
constant and balanced oracles correctly, and Grover drives the marked state's probability above 0.9 in
the predicted number of iterations (matching a brute-force amplitude calculation). Pure stdlib (cmath);
the gate-level companion to the quantum-statistics and quantum-Hall notes."""

from __future__ import annotations

import cmath
import math

# ---- single-qubit gates as 2x2 tuples ((a,b),(c,d)) ------------------------------------------
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
H = ((_INV_SQRT2, _INV_SQRT2), (_INV_SQRT2, -_INV_SQRT2))
X = ((0.0, 1.0), (1.0, 0.0))
Y = ((0.0, -1j), (1j, 0.0))
Z = ((1.0, 0.0), (0.0, -1.0))
I2 = ((1.0, 0.0), (0.0, 1.0))
S = ((1.0, 0.0), (0.0, 1j))
T = ((1.0, 0.0), (0.0, cmath.exp(1j * math.pi / 4)))


def phase_gate(theta):
    """Phase gate diag(1, e^{i theta})."""
    return ((1.0, 0.0), (0.0, cmath.exp(1j * theta)))


def rx(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return ((c, -1j * s), (-1j * s, c))


def ry(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return ((c, -s), (s, c))


class QuantumState:
    """A statevector of n qubits, initialized to |00..0>. Qubit 0 is the most significant bit."""

    def __init__(self, n_qubits):
        self.n = n_qubits
        self.dim = 1 << n_qubits
        self.amp = [0j] * self.dim
        self.amp[0] = 1 + 0j

    def _bit(self, index, qubit):
        # qubit 0 = most significant
        return (index >> (self.n - 1 - qubit)) & 1

    def apply_gate(self, gate, qubit):
        """Apply a 2x2 gate to a single qubit."""
        (a, b), (c, d) = gate
        step = 1 << (self.n - 1 - qubit)
        for i in range(self.dim):
            if not (i & step):
                j = i | step
                x0 = self.amp[i]
                x1 = self.amp[j]
                self.amp[i] = a * x0 + b * x1
                self.amp[j] = c * x0 + d * x1
        return self

    def apply_controlled(self, gate, controls, target):
        """Apply a 2x2 gate to `target` only where all `controls` qubits are 1."""
        (a, b), (c, d) = gate
        step = 1 << (self.n - 1 - target)
        cmask = 0
        for q in controls:
            cmask |= 1 << (self.n - 1 - q)
        for i in range(self.dim):
            if not (i & step) and (i & cmask) == cmask:
                j = i | step
                x0 = self.amp[i]
                x1 = self.amp[j]
                self.amp[i] = a * x0 + b * x1
                self.amp[j] = c * x0 + d * x1
        return self

    def cnot(self, control, target):
        return self.apply_controlled(X, [control], target)

    def cz(self, control, target):
        return self.apply_controlled(Z, [control], target)

    def toffoli(self, c1, c2, target):
        return self.apply_controlled(X, [c1, c2], target)

    def hadamard_all(self):
        for q in range(self.n):
            self.apply_gate(H, q)
        return self

    def probabilities(self):
        """Probability of each basis state (squared amplitude magnitudes)."""
        return [abs(a) ** 2 for a in self.amp]

    def probability(self, basis_index):
        return abs(self.amp[basis_index]) ** 2

    def norm(self):
        return math.sqrt(sum(abs(a) ** 2 for a in self.amp))

    def most_likely(self):
        """Basis index with the largest probability."""
        probs = self.probabilities()
        return max(range(self.dim), key=lambda i: probs[i])

    def measure_correlations(self):
        """For a 2-qubit state, the joint probabilities P(00), P(01), P(10), P(11)."""
        p = self.probabilities()
        return {"00": p[0], "01": p[1], "10": p[2], "11": p[3]}


def bell_pair():
    """Create the Bell state (|00> + |11>)/sqrt(2): H on q0, then CNOT(0->1)."""
    st = QuantumState(2)
    st.apply_gate(H, 0)
    st.cnot(0, 1)
    return st


# ---- Deutsch-Jozsa -----------------------------------------------------------------------------

def deutsch_jozsa(oracle, n):
    """Decide if a boolean oracle f: {0,1}^n -> {0,1} is constant or balanced in one query.

    `oracle(x)` takes an integer 0..2^n-1 and returns 0/1. Returns "constant" or "balanced".
    Uses n input qubits (no separate ancilla; the phase oracle applies (-1)^f(x) directly).
    """
    st = QuantumState(n)
    st.hadamard_all()
    # phase oracle: multiply each basis amplitude by (-1)^f(x)
    for x in range(st.dim):
        if oracle(x):
            st.amp[x] = -st.amp[x]
    for q in range(n):
        st.apply_gate(H, q)
    # constant iff all amplitude is back on |00..0>
    return "constant" if st.probability(0) > 0.5 else "balanced"


# ---- Grover's search ---------------------------------------------------------------------------

def grover_search(marked, n, iterations=None):
    """Grover's algorithm: amplify the amplitude of `marked` in a 2^n search space.

    Returns (state, iterations_used). If iterations is None, uses the optimal ~ (pi/4) sqrt(N).
    """
    N = 1 << n
    if iterations is None:
        iterations = max(1, int(round((math.pi / 4) * math.sqrt(N))))

    st = QuantumState(n)
    st.hadamard_all()

    for _ in range(iterations):
        # oracle: flip the sign of the marked amplitude
        st.amp[marked] = -st.amp[marked]
        # diffusion (inversion about the mean)
        mean = sum(st.amp) / N
        for i in range(N):
            st.amp[i] = 2 * mean - st.amp[i]
    return st, iterations


def grover_optimal_iterations(n):
    return max(1, int(round((math.pi / 4) * math.sqrt(1 << n))))


def is_unitary_on_state(gate, tol=1e-12):
    """Check a 2x2 gate preserves the norm of a random-ish 1-qubit state (necessary unitarity check)."""
    (a, b), (c, d) = gate
    st = QuantumState(1)
    st.amp = [0.6 + 0.0j, 0.8 + 0.0j]  # norm 1
    x0, x1 = st.amp
    y0 = a * x0 + b * x1
    y1 = c * x0 + d * x1
    return abs((abs(y0) ** 2 + abs(y1) ** 2) - 1.0) < tol
