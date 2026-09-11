"""Boids: a flock with no leader, from three local rules.

Craig Reynolds showed in 1987 that the coordinated swirl of a bird flock or fish school needs
no choreographer and no global plan -- just each individual ("boid") following three simple
rules based only on its nearby neighbours:

    SEPARATION: steer away from neighbours that are too close (avoid collisions),
    ALIGNMENT:  steer toward the average heading of neighbours (match velocity),
    COHESION:   steer toward the average position of neighbours (stay together).

Sum those three urges into an acceleration each step and a swarm of identical agents produces
lifelike, ever-changing flocking -- an emergent order that appears in starling murmurations,
sardine bait balls, and locust swarms, and that drives the crowd and creature animation in
films and games. There is no flock-level rule anywhere; the pattern is entirely bottom-up.

This module holds a population of boids (position + velocity in 2D), computes the three
steering vectors from each boid's neighbours within a radius, updates the flock one step
(with a speed limit), and measures the flock's polarization (how aligned the headings are) and
mean neighbour spacing. It reproduces the rise of alignment from a random start and the
separation rule keeping boids apart. Pure stdlib (seeded LCG for the initial scatter); the
emergent-collective-motion companion to the Kuramoto and cellular-automaton notes.
"""

from __future__ import annotations

import math


class _Rng:
    """Seeded LCG; high bits."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def uniform(self, a: float, b: float) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return a + (self.state >> 8) / (1 << 24) * (b - a)


def random_flock(n: int, size: float = 100.0, speed: float = 2.0, seed: int = 1):
    """Create n boids as (x, y, vx, vy) with random positions in a size x size box and random
    headings at the given speed."""
    rng = _Rng(seed)
    flock = []
    for _ in range(n):
        ang = rng.uniform(0, 2 * math.pi)
        flock.append((rng.uniform(0, size), rng.uniform(0, size),
                      speed * math.cos(ang), speed * math.sin(ang)))
    return flock


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def steering(flock, i, radius=20.0, sep_radius=8.0):
    """Return the (ax, ay) steering acceleration on boid i from separation, alignment, and
    cohesion with neighbours inside `radius` (separation only within sep_radius)."""
    bi = flock[i]
    sep_x = sep_y = 0.0
    ali_x = ali_y = 0.0
    coh_x = coh_y = 0.0
    n_ali = n_coh = 0
    for j, bj in enumerate(flock):
        if j == i:
            continue
        d = _dist(bi, bj)
        if d < sep_radius and d > 0:
            sep_x += (bi[0] - bj[0]) / d
            sep_y += (bi[1] - bj[1]) / d
        if d < radius:
            ali_x += bj[2]; ali_y += bj[3]; n_ali += 1
            coh_x += bj[0]; coh_y += bj[1]; n_coh += 1
    ax = ay = 0.0
    # separation
    ax += 1.5 * sep_x; ay += 1.5 * sep_y
    # alignment: toward average neighbour velocity
    if n_ali:
        ax += 1.0 * (ali_x / n_ali - bi[2]); ay += 1.0 * (ali_y / n_ali - bi[3])
    # cohesion: toward average neighbour position
    if n_coh:
        ax += 0.02 * (coh_x / n_coh - bi[0]); ay += 0.02 * (coh_y / n_coh - bi[1])
    return ax, ay


def step(flock, dt=1.0, max_speed=3.0, size=100.0, **kw):
    """Advance the flock one step: apply the three-rule steering, limit speed, integrate, and
    wrap positions in the toroidal box. Returns the new flock."""
    new = []
    for i, b in enumerate(flock):
        ax, ay = steering(flock, i, **kw)
        vx, vy = b[2] + ax * dt, b[3] + ay * dt
        sp = math.hypot(vx, vy)
        if sp > max_speed:
            vx, vy = vx / sp * max_speed, vy / sp * max_speed
        x = (b[0] + vx * dt) % size
        y = (b[1] + vy * dt) % size
        new.append((x, y, vx, vy))
    return new


def evolve(flock, steps, **kw):
    """Advance the flock `steps` steps; return the final flock."""
    for _ in range(steps):
        flock = step(flock, **kw)
    return flock


def polarization(flock) -> float:
    """Order parameter: magnitude of the mean unit heading, 0 (headings scattered) to 1 (all
    boids moving the same way) -- the flock's alignment."""
    n = len(flock)
    if n == 0:
        return 0.0
    cx = cy = 0.0
    for b in flock:
        sp = math.hypot(b[2], b[3])
        if sp > 0:
            cx += b[2] / sp; cy += b[3] / sp
    return math.hypot(cx, cy) / n


def mean_nearest_distance(flock) -> float:
    """Average distance from each boid to its nearest neighbour -- a measure of how tightly
    packed the flock is (separation keeps this above ~the separation radius)."""
    n = len(flock)
    if n < 2:
        return 0.0
    total = 0.0
    for i, bi in enumerate(flock):
        nn = min(_dist(bi, flock[j]) for j in range(n) if j != i)
        total += nn
    return total / n
