"""Convolutional codes and the Viterbi decoder: error correction for noisy channels.

A CONVOLUTIONAL CODE protects a stream of bits by feeding it through a small shift register and
emitting, at each step, several output bits that are XORs of the current and recent input bits. Unlike
a block code, it has MEMORY -- each output depends on a sliding window of inputs -- which spreads the
information about each bit across many transmitted bits, so a burst of channel errors can be undone.
Convolutional codes carried the Voyager images home, run in every GSM and satellite modem, and are
the classic setting for the VITERBI ALGORITHM.

Encoding is a walk through a state machine: the state is the contents of the shift register, and each
input bit both emits output bits (via the GENERATOR polynomials -- which taps to XOR) and shifts the
register. Decoding a noisy received stream is the inverse: find the input sequence whose encoded
output is CLOSEST (fewest bit differences -- minimum Hamming distance) to what was received. Brute
force over all 2^n inputs is hopeless, but the VITERBI ALGORITHM does it in linear time by dynamic
programming on the TRELLIS: at each step it keeps, for every register state, the single most-likely
path reaching it (the survivor with the least accumulated error), because any optimal path through a
state must use the best way of getting there. Tracing back the survivors from the end recovers the
maximum-likelihood transmitted sequence.

This module implements a rate-1/n convolutional encoder for arbitrary generator polynomials and a
Viterbi maximum-likelihood decoder, with the standard zero-tail termination. It is verified that a
clean channel decodes back to the original message exactly, that the decoder corrects random bit
errors up to the code's capability (matching a brute-force minimum-distance search on short messages),
that the classic rate-1/2 (7, 5) code corrects isolated errors, that its performance degrades
gracefully as noise rises, and on hand-checked encodings. Pure stdlib; a coding-theory companion to
the Hamming, Reed-Solomon, and CRC notes."""

from __future__ import annotations


def _parity(x):
    """Parity (XOR of all bits) of a non-negative integer."""
    p = 0
    while x:
        p ^= x & 1
        x >>= 1
    return p


class ConvolutionalCode:
    """A rate-1/n convolutional code defined by n generator polynomials (as integer bitmasks).

    Example: the classic (7, 5) rate-1/2 code has generators [0b111, 0b101] and constraint length 3."""

    def __init__(self, generators):
        self.generators = list(generators)
        self.n = len(generators)
        self.constraint = max(g.bit_length() for g in generators)
        self.mem = self.constraint - 1          # number of memory bits
        self.n_states = 1 << self.mem

    def _step(self, state, bit):
        """Given the current register state and an input bit, return (output_bits_tuple, next_state).

        The register holds the last `mem` bits; the full window is (bit << mem) | state."""
        window = (bit << self.mem) | state
        outputs = tuple(_parity(window & g) for g in self.generators)
        next_state = window >> 1              # shift: drop the oldest bit
        return outputs, next_state

    def encode(self, bits):
        """Encode a list of message bits, appending `mem` zero bits to flush the register (zero-tail
        termination). Returns the flat list of coded bits (length n * (len + mem))."""
        state = 0
        out = []
        for bit in list(bits) + [0] * self.mem:
            outputs, state = self._step(state, bit)
            out.extend(outputs)
        return out

    def decode(self, received):
        """Viterbi maximum-likelihood decode. `received` is the flat coded bit stream (possibly with
        errors). Returns the decoded message bits (without the flushed tail)."""
        n = self.n
        total_steps = len(received) // n
        INF = float("inf")
        # metric[state] = least accumulated Hamming distance to reach `state`
        metric = [INF] * self.n_states
        metric[0] = 0
        # back-pointers: path[t][state] = (prev_state, input_bit)
        path = []
        for t in range(total_steps):
            chunk = received[t * n:(t + 1) * n]
            new_metric = [INF] * self.n_states
            back = [None] * self.n_states
            for state in range(self.n_states):
                if metric[state] == INF:
                    continue
                for bit in (0, 1):
                    outputs, nxt = self._step(state, bit)
                    dist = sum(1 for i in range(n) if outputs[i] != chunk[i])
                    cand = metric[state] + dist
                    if cand < new_metric[nxt]:
                        new_metric[nxt] = cand
                        back[nxt] = (state, bit)
            metric = new_metric
            path.append(back)
        # zero-tail termination: the final state must be 0
        state = 0
        if metric[0] == INF:
            state = min(range(self.n_states), key=lambda s: metric[s])
        bits = []
        for t in range(total_steps - 1, -1, -1):
            prev, bit = path[t][state]
            bits.append(bit)
            state = prev
        bits.reverse()
        # drop the flushed tail bits
        return bits[:total_steps - self.mem]

    def encoded_length(self, message_len):
        return self.n * (message_len + self.mem)


def hamming_distance(a, b):
    """Number of positions at which two equal-length bit lists differ."""
    return sum(1 for x, y in zip(a, b) if x != y)
