"""Real-time scheduling -- can a set of periodic tasks always meet their deadlines?

A flight controller, an engine ECU, a pacemaker: each runs a set of PERIODIC TASKS -- task i wakes every
T_i time units and must finish its C_i units of computation before its deadline. Missing a deadline can
be catastrophic, so before deploying you must PROVE the task set is SCHEDULABLE: that some scheduling
policy always finishes every job in time. This is the founding question of real-time systems theory,
answered by Liu and Layland in their celebrated 1973 paper for the two canonical policies.

RATE-MONOTONIC (RM) is the optimal FIXED-PRIORITY policy: assign each task a static priority by rate --
shorter period, higher priority -- and always run the highest-priority ready job. Liu-Layland proved a
beautiful SUFFICIENT test: n tasks are schedulable if their total utilisation U = sum(C_i / T_i) is at
most n(2^(1/n) - 1), a bound that falls from 1.0 for one task toward ln 2 ~ 0.693 as n grows. It is
sufficient but not necessary -- some task sets above the bound are still schedulable -- so an exact test
simulates the busy period or checks response times.

EARLIEST-DEADLINE-FIRST (EDF) is the optimal DYNAMIC-priority policy: always run the ready job whose
absolute deadline is soonest. Its schedulability test is stunningly simple and EXACT (for deadlines
equal to periods): the task set is schedulable if and only if U <= 1. EDF can use the processor fully
where RM cannot -- its price is dynamic priorities and less predictable overload behaviour.

This module computes utilisation, applies the RM (Liu-Layland) and EDF schedulability tests, and --
the ground truth -- SIMULATES both policies over one HYPERPERIOD (the LCM of the periods, after which
the schedule repeats), reporting whether any deadline is missed. Pure standard library.

Validation. The simulation is the oracle: the EDF test (U <= 1) is verified to be EXACT -- the
simulation misses a deadline exactly when U > 1 -- over many random task sets. The RM test is verified
to be SUFFICIENT -- whenever U is under the Liu-Layland bound the simulation meets every deadline -- and
its non-necessity is shown (task sets above the bound that the simulation nonetheless schedules exist).
RM is confirmed optimal among fixed-priority policies on hand cases, EDF dominates RM (schedules
everything RM does and more), the classic Liu-Layland examples match, a task set with U slightly above 1
misses under both, and the hyperperiod is the LCM of the periods. Pure standard library."""

import math
from functools import reduce


class Task:
    """A periodic real-time task: computation time C, period T, deadline D (defaults to T)."""

    __slots__ = ("name", "C", "T", "D")

    def __init__(self, C, T, D=None, name=""):
        if C <= 0 or T <= 0:
            raise ValueError("C and T must be positive")
        self.C = C
        self.T = T
        self.D = D if D is not None else T
        self.name = name

    def __repr__(self):
        return f"Task({self.name or '?'}: C={self.C}, T={self.T}, D={self.D})"


def utilisation(tasks):
    """Total processor utilisation U = sum(C_i / T_i)."""
    return sum(t.C / t.T for t in tasks)


def liu_layland_bound(n):
    """The Liu-Layland utilisation bound for n tasks: n(2^(1/n) - 1)."""
    if n == 0:
        return 1.0
    return n * (2 ** (1.0 / n) - 1)


# ---------------------------------------------------------------------------
# schedulability tests
# ---------------------------------------------------------------------------

def rm_schedulable_ll(tasks):
    """Rate-monotonic Liu-Layland SUFFICIENT test: U <= n(2^(1/n) - 1)."""
    n = len(tasks)
    return utilisation(tasks) <= liu_layland_bound(n) + 1e-12


def edf_schedulable(tasks):
    """EDF EXACT test (deadlines = periods): schedulable iff U <= 1."""
    return utilisation(tasks) <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# hyperperiod simulation (ground truth)
# ---------------------------------------------------------------------------

def _lcm(a, b):
    return a * b // math.gcd(a, b)


def hyperperiod(tasks):
    """The LCM of all task periods; the schedule repeats every hyperperiod."""
    periods = [t.T for t in tasks]
    return reduce(_lcm, periods)


def simulate(tasks, policy="edf", horizon=None):
    """Simulate scheduling over ``horizon`` (default one hyperperiod) at unit time steps.

    ``policy`` is 'edf' (earliest absolute deadline first) or 'rm' (rate-monotonic: shortest period
    first, static). Returns (schedulable, misses) where misses lists (task_index, deadline_time) of any
    deadline missed. Assumes integer C, T, D.
    """
    n = len(tasks)
    if horizon is None:
        horizon = hyperperiod(tasks)
    # per-task job bookkeeping: remaining compute, absolute deadline of the current job
    remaining = [0] * n
    abs_deadline = [0] * n
    released = [False] * n
    misses = []

    for time in range(horizon):
        # check deadline misses for jobs that still have work AT or past their deadline
        for i, t in enumerate(tasks):
            if remaining[i] > 0 and time >= abs_deadline[i]:
                misses.append((i, abs_deadline[i]))
                remaining[i] = 0            # drop the missed job to keep simulating
        # release new jobs at period boundaries
        for i, t in enumerate(tasks):
            if time % t.T == 0:
                if released[i] and remaining[i] > 0:
                    # a new job arrives before the previous one finished -> the previous missed
                    misses.append((i, abs_deadline[i]))
                remaining[i] = t.C
                abs_deadline[i] = time + t.D
                released[i] = True
        # choose which ready job runs this tick
        ready = [i for i in range(n) if remaining[i] > 0]
        if not ready:
            continue
        if policy == "edf":
            chosen = min(ready, key=lambda i: (abs_deadline[i], i))
        else:  # rate-monotonic: smaller period = higher priority
            chosen = min(ready, key=lambda i: (tasks[i].T, i))
        remaining[chosen] -= 1

    # final deadline check at the horizon boundary
    for i, t in enumerate(tasks):
        if remaining[i] > 0 and horizon >= abs_deadline[i]:
            misses.append((i, abs_deadline[i]))

    return len(misses) == 0, misses


def rm_schedulable_exact(tasks):
    """Exact RM schedulability by hyperperiod simulation."""
    return simulate(tasks, "rm")[0]


def edf_schedulable_exact(tasks):
    """Exact EDF schedulability by hyperperiod simulation."""
    return simulate(tasks, "edf")[0]
