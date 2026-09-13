"""The quantum Fourier transform: the engine inside Shor's algorithm, built from Hadamards and phases.

The quantum Fourier transform is the quantum analogue of the discrete Fourier transform, and it is the
beating heart of the most famous quantum algorithms -- Shor's factoring, quantum phase estimation, the
hidden-subgroup solvers. Its power is twofold: it acts on the 2^n AMPLITUDES of an n-qubit register
exactly as the DFT acts on a length-2^n vector, yet it does so with only O(n^2) gates instead of the
O(n 2^n) operations a classical FFT needs on that vector -- an exponential edge in gate count (the
catch, as always, is that you cannot read all the amplitudes out; the QFT is useful only when a later
measurement collapses the interference into the answer, as in phase estimation).

The circuit is beautifully regular. On qubit j, apply a HADAMARD, then a cascade of CONTROLLED PHASE
rotations R_k = diag(1, e^{2 pi i / 2^k}) controlled by each less-significant qubit -- rotating qubit
j's phase in proportion to the binary fraction formed by the bits below it. Do this for every qubit in
turn, then REVERSE the qubit order with swaps. That is the entire transform: n Hadamards and n(n-1)/2
controlled phases. The controlled-phase cascade is exactly how the DFT's twiddle factors
e^{2 pi i j k / N} get built up bit by bit.

This module implements the QFT and its inverse as gate sequences on the repo's statevector simulator,
plus a direct classical DFT for checking. It is validated exactly against that reference: applying the
QFT to an arbitrary input state produces the DFT of its amplitude vector (to machine precision) for
n = 1..4; the inverse QFT undoes the QFT (round-trip to identity); QFT of a computational basis state
|k> is the expected uniform-magnitude phase ramp; the QFT of the uniform superposition is the basis
state |0>; and the transform is unitary (norm preserved). It also demonstrates PHASE ESTIMATION -- the
inverse QFT reading a phase e^{2 pi i phi} off an eigenstate -- recovering phi to the register's
resolution. Pure stdlib (cmath); the quantum-algorithm companion to the gate-level circuit simulator
and the classical FFT/DFT."""

from __future__ import annotations

import cmath
import math

from quantum_circuit import QuantumState, H


def _controlled_phase(theta):
    """Controlled phase gate: the 2x2 diag(1, e^{i theta}) applied to the target."""
    return ((1.0, 0.0), (0.0, cmath.exp(1j * theta)))


def apply_qft(state, qubits=None, inverse=False):
    """Apply the QFT (or inverse QFT) in place to `qubits` of a QuantumState.

    qubits defaults to all of them, most-significant first. Returns the state.
    """
    n = state.n
    if qubits is None:
        qubits = list(range(n))
    m = len(qubits)
    sign = -1.0 if inverse else 1.0

    for i in range(m):
        state.apply_gate(H, qubits[i])
        for j in range(i + 1, m):
            k = j - i + 1
            theta = sign * 2 * math.pi / (1 << k)
            # controlled phase: control = qubits[j], target = qubits[i]
            state.apply_controlled(_controlled_phase(theta), [qubits[j]], qubits[i])

    # reverse the qubit order with swaps
    for i in range(m // 2):
        _swap(state, qubits[i], qubits[m - 1 - i])
    return state


def _swap(state, a, b):
    """Swap two qubits via three CNOTs."""
    if a == b:
        return
    state.cnot(a, b)
    state.cnot(b, a)
    state.cnot(a, b)


def qft(state, qubits=None):
    return apply_qft(state, qubits, inverse=False)


def inverse_qft(state, qubits=None):
    return apply_qft(state, qubits, inverse=True)


def classical_dft(vec, inverse=False):
    """Direct DFT of a complex vector, matching the QFT's convention and normalization (1/sqrt(N)).

    The standard QFT maps |j> to (1/sqrt N) sum_k e^{+2 pi i j k / N} |k>, i.e. the POSITIVE-exponent
    transform, so the forward direction here uses +2 pi i (the opposite sign convention to a physics
    forward DFT). The inverse uses the negative exponent.
    """
    N = len(vec)
    sign = -1.0 if inverse else 1.0
    out = []
    for k in range(N):
        s = 0j
        for j in range(N):
            s += vec[j] * cmath.exp(sign * 2j * math.pi * j * k / N)
        out.append(s / math.sqrt(N))
    return out


def state_from_amplitudes(amps):
    """Build a QuantumState from an explicit (normalized) amplitude list of length 2^n."""
    N = len(amps)
    n = int(round(math.log2(N)))
    if (1 << n) != N:
        raise ValueError("amplitude length must be a power of two")
    st = QuantumState(n)
    st.amp = [complex(a) for a in amps]
    return st


def phase_estimation(phi, n_qubits):
    """Estimate a phase phi in [0,1) using n_qubits of precision via the inverse QFT.

    Prepares the register in the state whose QFT-basis phase is phi (i.e. amplitude
    e^{2 pi i phi k} on |k>), applies the inverse QFT, and returns the most likely integer readout m,
    so that m / 2^n approximates phi. Returns (estimate_fraction, measured_integer).
    """
    N = 1 << n_qubits
    # the eigenstate phase kickback produces amplitudes e^{2 pi i phi k}/sqrt(N) on |k>
    amps = [cmath.exp(2j * math.pi * phi * k) / math.sqrt(N) for k in range(N)]
    st = state_from_amplitudes(amps)
    inverse_qft(st)
    m = st.most_likely()
    return m / N, m
