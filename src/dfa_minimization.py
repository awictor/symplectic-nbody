"""DFA minimization by Hopcroft's algorithm: the unique smallest automaton for a regular language.

A deterministic finite automaton recognises a language -- the set of strings it accepts. Many DFAs
recognise the SAME language, differing only in redundant states. The Myhill-Nerode theorem says each
regular language has a UNIQUE minimal DFA (up to renaming): the number of states equals the number of
equivalence classes of strings under "leads to the same future". Minimizing is not just tidying --
it is a canonical form, so two DFAs recognise the same language exactly when their minimal DFAs are
isomorphic. It powers regex engines, lexer generators, and model checking.

Two states are EQUIVALENT if no input string distinguishes them: from either, every string is
accepted or rejected identically. Minimization merges each equivalence class into one state. The
naive test refines a partition pairwise in O(n^2 * |alphabet|). HOPCROFT'S ALGORITHM (1971) does it
in O(n log n * |alphabet|) by refining smartly: start with the partition {accepting, non-accepting},
and repeatedly pick a "splitter" set A and a symbol c, then split every block by whether its states
transition on c INTO A. The trick that gives the log factor is, when a block splits, to add only the
SMALLER half to the worklist -- each state's block halves at most log n times.

This module builds a DFA (states, alphabet, transitions, start, accepting), removes unreachable
states, minimizes by Hopcroft, tests string acceptance, and checks language equivalence of two DFAs
by minimizing both and comparing canonically. A brute-force partition-refinement minimizer serves as
the independent reference.

Validated by language preservation and minimality: the minimized DFA accepts exactly the same
strings as the original over an exhaustive set of short strings; it has no more states than the
brute-force minimizer (and the same count); minimizing an already-minimal DFA is a no-op; equivalent
DFAs built differently minimize to isomorphic machines; and unreachable and dead states are handled.
Pure stdlib; the automata-theory companion to the Thompson NFA regex engine and the suffix
automaton."""

from __future__ import annotations

from collections import deque


class DFA:
    """A deterministic finite automaton.

    states: iterable of hashable state ids.
    alphabet: iterable of symbols.
    transitions: dict {(state, symbol): next_state}. Must be total after completion, but this class
        completes partial DFAs with an implicit dead state when asked.
    start: the start state. accepting: set of accepting states."""

    def __init__(self, states, alphabet, transitions, start, accepting):
        self.states = set(states)
        self.alphabet = list(alphabet)
        self.transitions = dict(transitions)
        self.start = start
        self.accepting = set(accepting)
        if start not in self.states:
            raise ValueError("start state not in states")
        if not self.accepting <= self.states:
            raise ValueError("accepting states must be a subset of states")

    def step(self, state, symbol):
        return self.transitions.get((state, symbol))

    def accepts(self, string):
        """True if the DFA accepts the string (a sequence of symbols)."""
        s = self.start
        for c in string:
            s = self.transitions.get((s, c))
            if s is None:
                return False  # missing transition = implicit dead state
        return s in self.accepting

    def reachable_states(self):
        seen = {self.start}
        q = deque([self.start])
        while q:
            s = q.popleft()
            for c in self.alphabet:
                t = self.step(s, c)
                if t is not None and t not in seen:
                    seen.add(t)
                    q.append(t)
        return seen

    def remove_unreachable(self):
        """Return a copy with only reachable states."""
        reach = self.reachable_states()
        trans = {(s, c): t for (s, c), t in self.transitions.items() if s in reach and t in reach}
        return DFA(reach, self.alphabet, trans, self.start, self.accepting & reach)

    def _complete(self):
        """Return (dfa-with-total-transitions, dead_state) adding a dead state for missing edges."""
        dead = ("__dead__",)
        states = set(self.states)
        trans = dict(self.transitions)
        need_dead = False
        for s in self.states:
            for c in self.alphabet:
                if (s, c) not in trans:
                    trans[(s, c)] = dead
                    need_dead = True
        if need_dead:
            states.add(dead)
            for c in self.alphabet:
                trans[(dead, c)] = dead
        return DFA(states, self.alphabet, trans, self.start, self.accepting), (dead if need_dead else None)


def minimize(dfa):
    """Minimize by Hopcroft's algorithm. Returns a new DFA with merged equivalence classes,
    unreachable states removed first. State ids are frozensets of original states."""
    dfa = dfa.remove_unreachable()
    dfa, dead = dfa._complete()
    states = dfa.states
    alphabet = dfa.alphabet

    accepting = dfa.accepting & states
    nonaccepting = states - accepting

    # initial partition
    partition = []
    if accepting:
        partition.append(set(accepting))
    if nonaccepting:
        partition.append(set(nonaccepting))

    # worklist of (block, symbol) splitters; use the smaller of the two initial blocks
    worklist = deque()
    for c in alphabet:
        # add the smaller initial block for each symbol
        if accepting and nonaccepting:
            smaller = accepting if len(accepting) <= len(nonaccepting) else nonaccepting
            worklist.append((frozenset(smaller), c))
        elif accepting:
            worklist.append((frozenset(accepting), c))
        elif nonaccepting:
            worklist.append((frozenset(nonaccepting), c))

    # precompute inverse transitions: pred[(target, c)] = set of states s with step(s,c)=target
    pred = {}
    for (s, c), t in dfa.transitions.items():
        pred.setdefault((t, c), set()).add(s)

    while worklist:
        A, c = worklist.popleft()
        # X = states that transition on c into A
        X = set()
        for target in A:
            X |= pred.get((target, c), set())
        if not X:
            continue
        new_partition = []
        for Y in partition:
            inter = Y & X
            diff = Y - X
            if inter and diff:
                new_partition.append(inter)
                new_partition.append(diff)
                # update worklist
                for sym in alphabet:
                    inY = (frozenset(Y), sym)
                    # replace Y by the two halves; add the smaller (or both if Y was queued)
                    smaller = inter if len(inter) <= len(diff) else diff
                    # standard Hopcroft: if Y in worklist for sym replace by both, else add smaller
                    replaced = False
                    for idx, (blk, s2) in enumerate(worklist):
                        if blk == frozenset(Y) and s2 == sym:
                            worklist[idx] = (frozenset(inter), sym)
                            worklist.append((frozenset(diff), sym))
                            replaced = True
                            break
                    if not replaced:
                        worklist.append((frozenset(smaller), sym))
            else:
                new_partition.append(Y)
        partition = new_partition

    # build the minimized DFA: one state per block
    block_of = {}
    for blk in partition:
        key = frozenset(blk)
        for s in blk:
            block_of[s] = key

    new_states = set(block_of.values())
    new_start = block_of[dfa.start]
    new_accept = {block_of[s] for s in accepting}
    new_trans = {}
    for blk in new_states:
        rep = next(iter(blk))
        for c in alphabet:
            t = dfa.step(rep, c)
            if t is not None:
                new_trans[(blk, c)] = block_of[t]

    # canonical complete minimal DFA (a dead/trap state, if the language needs one, is kept --
    # this is the unique minimal complete DFA of the Myhill-Nerode theorem)
    return DFA(new_states, alphabet, new_trans, new_start, new_accept)


def equivalent(dfa1, dfa2):
    """True if two DFAs recognise the same language, by minimizing both and comparing canonically."""
    m1 = minimize(dfa1)
    m2 = minimize(dfa2)
    if len(m1.states) != len(m2.states):
        return False
    if len(m1.accepting) != len(m2.accepting):
        return False
    # canonical BFS isomorphism check from the start states
    return _iso(m1, m2)


def _iso(m1, m2):
    """BFS from both start states in lockstep, matching states by transition structure."""
    alphabet = m1.alphabet
    if set(alphabet) != set(m2.alphabet):
        return False
    mapping = {m1.start: m2.start}
    q = deque([m1.start])
    while q:
        s1 = q.popleft()
        s2 = mapping[s1]
        if (s1 in m1.accepting) != (s2 in m2.accepting):
            return False
        for c in alphabet:
            t1 = m1.step(s1, c)
            t2 = m2.step(s2, c)
            if (t1 is None) != (t2 is None):
                return False
            if t1 is None:
                continue
            if t1 in mapping:
                if mapping[t1] != t2:
                    return False
            else:
                mapping[t1] = t2
                q.append(t1)
    return True


# --- brute-force reference ---------------------------------------------------
def brute_minimize_state_count(dfa):
    """Independent minimal-state count by iterative pairwise partition refinement (Moore's method)."""
    dfa = dfa.remove_unreachable()
    dfa, _ = dfa._complete()
    states = list(dfa.states)
    # equivalence classes start as accepting / non-accepting
    def cls0(s):
        return s in dfa.accepting
    label = {s: cls0(s) for s in states}
    while True:
        # refine: two states split if a symbol sends them to different current classes
        signature = {}
        for s in states:
            sig = (label[s],) + tuple(label[dfa.step(s, c)] for c in dfa.alphabet)
            signature[s] = sig
        # relabel by distinct signatures
        sigs = {}
        new_label = {}
        for s in states:
            if signature[s] not in sigs:
                sigs[signature[s]] = len(sigs)
            new_label[s] = sigs[signature[s]]
        if len(set(new_label.values())) == len(set(label.values())):
            break
        label = new_label
    return len(set(label.values()))
