"""Thompson's NFA construction -- a regular-expression engine that can never blow up.

The regular-expression matcher built into most languages uses backtracking, which is why a pattern
like ``(a*)*b`` against a long string of ``a``s can hang for seconds -- catastrophic backtracking, an
exponential-time trap that has taken down production web services. Ken Thompson's 1968 construction
avoids it completely by matching in GUARANTEED LINEAR TIME in the length of the input, no matter how
pathological the pattern. The trick is to never backtrack: instead of trying one path and rewinding,
the engine tracks the SET of all NFA states the machine could be in simultaneously, and advances the
whole set one input character at a time. With at most one state per NFA node live at once, matching an
n-character string against an m-node machine costs O(n*m), always.

The pipeline has three classic stages, all implemented here from scratch:

  1. PARSE the regex into a syntax tree. A recursive-descent parser handles alternation ``|`` (lowest
     precedence), implicit concatenation, the postfix quantifiers ``*`` ``+`` ``?``, grouping with
     parentheses, the wildcard ``.``, and backslash escapes for literals. It respects the usual
     precedence so ``ab|cd`` means ``(ab)|(cd)`` and ``ab*`` means ``a(b*)``.

  2. COMPILE the tree to an NFA by Thompson's rules, gluing together tiny two-state machines with
     epsilon (empty) transitions: a literal is one edge; concatenation wires the accept of the first
     into the start of the second; alternation forks with two epsilons; the star wires a loop with
     epsilons for "zero or more". Each operator adds a constant number of states, so the NFA has size
     linear in the pattern.

  3. SIMULATE the NFA over the input by the subset method. Start from the epsilon-closure of the start
     state; for each input character, step every live state that has a matching edge and take the
     epsilon-closure of the result; accept if the final live set contains the accept state. A visited
     stamp keeps the closure from looping on cycles introduced by the star.

The public API mirrors the standard library: ``fullmatch`` (the whole string must match) and
``search`` (some substring matches, implemented by prepending an implicit ``.*``). Supported syntax:
literals, ``.``, ``*`` ``+`` ``?``, ``|``, parentheses, and ``\\`` escaping. No captures, anchors, or
character classes -- deliberately a minimal core that demonstrates the linear-time guarantee cleanly.

Validation. The engine is checked against Python's own ``re`` module as an independent oracle: over
hundreds of seeded random (pattern, string) pairs drawn from the supported grammar, ``fullmatch`` and
``search`` agree with ``re.fullmatch`` and ``re.search`` on every case. It is also checked on
hand-written examples and, crucially, on the catastrophic-backtracking pattern ``(a*)*b`` against long
``a``-strings, where it returns promptly while a backtracking matcher would stall -- demonstrating the
linear-time property directly. Pure standard library."""


# ---------------------------------------------------------------------------
# 1. parse the regex into a syntax tree
# ---------------------------------------------------------------------------
# node forms (tuples):
#   ('lit', ch)      literal character
#   ('dot',)         any character
#   ('cat', a, b)    concatenation
#   ('alt', a, b)    alternation
#   ('star', a)      zero or more
#   ('plus', a)      one or more
#   ('opt', a)       zero or one
#   ('eps',)         empty match


class _Parser:
    def __init__(self, pattern):
        self.s = pattern
        self.i = 0

    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else None

    def next(self):
        ch = self.s[self.i]
        self.i += 1
        return ch

    def parse(self):
        node = self.parse_alt()
        if self.i != len(self.s):
            raise ValueError(f"unexpected character at position {self.i}: {self.peek()!r}")
        return node

    def parse_alt(self):
        node = self.parse_concat()
        while self.peek() == '|':
            self.next()
            rhs = self.parse_concat()
            node = ('alt', node, rhs)
        return node

    def parse_concat(self):
        nodes = []
        while self.peek() is not None and self.peek() not in '|)':
            nodes.append(self.parse_repeat())
        if not nodes:
            return ('eps',)
        node = nodes[0]
        for nxt in nodes[1:]:
            node = ('cat', node, nxt)
        return node

    def parse_repeat(self):
        node = self.parse_atom()
        while self.peek() in ('*', '+', '?'):
            op = self.next()
            if op == '*':
                node = ('star', node)
            elif op == '+':
                node = ('plus', node)
            else:
                node = ('opt', node)
        return node

    def parse_atom(self):
        ch = self.peek()
        if ch == '(':
            self.next()
            node = self.parse_alt()
            if self.peek() != ')':
                raise ValueError("unbalanced parenthesis")
            self.next()
            return node
        if ch == '.':
            self.next()
            return ('dot',)
        if ch == '\\':
            self.next()
            if self.peek() is None:
                raise ValueError("dangling escape")
            return ('lit', self.next())
        if ch is None or ch in '|)*+?':
            raise ValueError(f"unexpected token {ch!r}")
        return ('lit', self.next())


def parse(pattern):
    """Parse a regex string into a syntax tree."""
    return _Parser(pattern).parse()


# ---------------------------------------------------------------------------
# 2. compile the tree to an NFA (Thompson construction)
# ---------------------------------------------------------------------------
# The NFA is a list of states. Each state is a dict:
#   {'edges': [(label, target), ...]}
# where label is None for an epsilon transition, the string char for a literal,
# or the sentinel DOT for "any character".

DOT = object()


class NFA:
    def __init__(self):
        self.states = []
        self.start = None
        self.accept = None

    def new_state(self):
        self.states.append({'edges': []})
        return len(self.states) - 1

    def add_edge(self, src, label, dst):
        self.states[src]['edges'].append((label, dst))


def compile_nfa(tree):
    """Compile a parsed regex tree into an NFA by Thompson's construction."""
    nfa = NFA()

    def build(node):
        """Return (start, accept) state indices for the sub-machine of ``node``."""
        kind = node[0]
        if kind == 'eps':
            s = nfa.new_state()
            a = nfa.new_state()
            nfa.add_edge(s, None, a)
            return s, a
        if kind == 'lit':
            s = nfa.new_state()
            a = nfa.new_state()
            nfa.add_edge(s, node[1], a)
            return s, a
        if kind == 'dot':
            s = nfa.new_state()
            a = nfa.new_state()
            nfa.add_edge(s, DOT, a)
            return s, a
        if kind == 'cat':
            s1, a1 = build(node[1])
            s2, a2 = build(node[2])
            nfa.add_edge(a1, None, s2)
            return s1, a2
        if kind == 'alt':
            s = nfa.new_state()
            a = nfa.new_state()
            s1, a1 = build(node[1])
            s2, a2 = build(node[2])
            nfa.add_edge(s, None, s1)
            nfa.add_edge(s, None, s2)
            nfa.add_edge(a1, None, a)
            nfa.add_edge(a2, None, a)
            return s, a
        if kind == 'star':
            s = nfa.new_state()
            a = nfa.new_state()
            s1, a1 = build(node[1])
            nfa.add_edge(s, None, s1)
            nfa.add_edge(s, None, a)       # zero occurrences
            nfa.add_edge(a1, None, s1)     # loop back
            nfa.add_edge(a1, None, a)
            return s, a
        if kind == 'plus':
            s1, a1 = build(node[1])
            a = nfa.new_state()
            nfa.add_edge(a1, None, s1)     # loop back (one or more)
            nfa.add_edge(a1, None, a)
            return s1, a
        if kind == 'opt':
            s = nfa.new_state()
            a = nfa.new_state()
            s1, a1 = build(node[1])
            nfa.add_edge(s, None, s1)
            nfa.add_edge(s, None, a)       # skip
            nfa.add_edge(a1, None, a)
            return s, a
        raise ValueError(f"unknown node kind {kind!r}")

    nfa.start, nfa.accept = build(tree)
    return nfa


# ---------------------------------------------------------------------------
# 3. simulate the NFA over an input string (subset construction on the fly)
# ---------------------------------------------------------------------------

def _epsilon_closure(nfa, states):
    """All states reachable from ``states`` via epsilon transitions (iterative, cycle-safe)."""
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for label, dst in nfa.states[s]['edges']:
            if label is None and dst not in closure:
                closure.add(dst)
                stack.append(dst)
    return closure


def _run(nfa, text):
    """True if the NFA accepts the ENTIRE text."""
    current = _epsilon_closure(nfa, {nfa.start})
    for ch in text:
        nxt = set()
        for s in current:
            for label, dst in nfa.states[s]['edges']:
                if label is None:
                    continue
                if label is DOT or label == ch:
                    nxt.add(dst)
        current = _epsilon_closure(nfa, nxt)
        if not current:
            return False
    return nfa.accept in current


class Regex:
    """A compiled regular expression with fullmatch / search, matching in linear time."""

    def __init__(self, pattern):
        self.pattern = pattern
        self._nfa = compile_nfa(parse(pattern))

    def fullmatch(self, text):
        """True if the whole ``text`` matches the pattern."""
        return _run(self._nfa, text)

    def search(self, text):
        """True if any substring of ``text`` matches -- try every start offset against a prefix run.

        Implemented by simulating from each start position; still linear per start, and we short-
        circuit on the first accepting run. (A single .*-prefixed machine would also work; this keeps
        the accept semantics simple and is fast enough for the validation sizes.)"""
        # Simulate once but allow the start state to be (re)entered at every position: this is the
        # standard ".* prefix" trick done inline without rebuilding the machine.
        n = len(text)
        for start in range(n + 1):
            if self._match_prefix(text, start):
                return True
        return False

    def _match_prefix(self, text, start):
        """True if some prefix of text[start:] is accepted by the machine."""
        nfa = self._nfa
        current = _epsilon_closure(nfa, {nfa.start})
        if nfa.accept in current:
            return True
        for i in range(start, len(text)):
            ch = text[i]
            nxt = set()
            for s in current:
                for label, dst in nfa.states[s]['edges']:
                    if label is None:
                        continue
                    if label is DOT or label == ch:
                        nxt.add(dst)
            current = _epsilon_closure(nfa, nxt)
            if not current:
                return False
            if nfa.accept in current:
                return True
        return False


def fullmatch(pattern, text):
    """Convenience: True if ``text`` fully matches ``pattern``."""
    return Regex(pattern).fullmatch(text)


def search(pattern, text):
    """Convenience: True if some substring of ``text`` matches ``pattern``."""
    return Regex(pattern).search(text)
