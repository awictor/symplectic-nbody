"""Rapidly-exploring random trees -- finding a path through a cluttered space by growing toward chaos.

A robot arm threading between obstacles, a drone crossing a room of furniture, a character navigating a
game level: all face MOTION PLANNING -- find a collision-free path from start to goal through a space
peppered with obstacles. When the space is low-dimensional you can grid it and run a shortest-path
search, but that explodes in high dimensions (a 7-joint arm lives in a 7D configuration space) and wastes
effort on regions the path never visits. SAMPLING-BASED planning sidesteps the curse of dimensionality:
instead of discretising everything, randomly probe the space and connect the probes that are reachable,
building a tree that reaches into the free space until it touches the goal.

The RAPIDLY-EXPLORING RANDOM TREE (LaValle, 1998) is the classic. Start a tree at the initial
configuration. Repeatedly: draw a random sample from the space, find the nearest node already in the tree,
and STEER from that node a small step toward the sample, adding the new point if the connecting segment
is collision-free. The magic is in the name -- because new samples are most likely to fall in the large
unexplored regions, the nearest existing node is usually on the frontier, so the tree is pulled OUTWARD
and rapidly fills the reachable free space (a Voronoi-bias argument makes this precise). Bias a small
fraction of samples toward the goal and the tree homes in; when a new node can see the goal, you are done.

Plain RRT finds *a* path, not a good one -- it can be needlessly jagged and long. RRT* (Karaman &
Frazzoli, 2011) adds two rewiring steps that make it ASYMPTOTICALLY OPTIMAL: when adding a new node,
connect it to the neighbour that gives the lowest cost-to-come (not merely the nearest), and then check
whether routing each nearby node THROUGH the new node would shorten its own path, rewiring if so. As
samples accumulate the tree relaxes toward the shortest path. This module implements both, on a 2D plane
with circular obstacles and a straight-line steering/collision model, driven by a seeded generator so
runs reproduce.

Validation. (1) Every returned path is genuinely COLLISION-FREE: each segment is checked against every
obstacle by point sampling, and it starts at the start and ends within tolerance of the goal. (2) Tree
edges never pass through an obstacle. (3) On an obstacle-free space the found path length is close to the
straight-line optimum. (4) RRT* returns a path no longer -- and on cluttered maps meaningfully shorter --
than plain RRT from the same samples, the whole point of the rewiring. (5) When the goal is walled off,
the planner reports failure rather than returning an invalid path. (6) Reproducibility under a fixed
seed. Pure standard library -- ``math`` only."""

import math


class _LCG:
    """Seeded linear-congruential generator for reproducible sampling."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def random(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def uniform(self, lo, hi):
        return lo + (hi - lo) * self.random()


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class World:
    """A rectangular 2D world [0,width] x [0,height] with circular obstacles (x, y, radius)."""

    def __init__(self, width, height, obstacles=None):
        self.width = width
        self.height = height
        self.obstacles = list(obstacles or [])

    def in_bounds(self, p):
        return 0 <= p[0] <= self.width and 0 <= p[1] <= self.height

    def point_free(self, p):
        if not self.in_bounds(p):
            return False
        for ox, oy, r in self.obstacles:
            if (p[0] - ox) ** 2 + (p[1] - oy) ** 2 <= r * r:
                return False
        return True

    def segment_free(self, a, b, step=0.5):
        """True if the straight segment a-b avoids every obstacle (sampled at ``step`` resolution)."""
        d = _dist(a, b)
        n = max(1, int(d / step))
        for i in range(n + 1):
            t = i / n
            p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if not self.point_free(p):
                return False
        return True


def _steer(from_p, to_p, max_step):
    """Return a point at most ``max_step`` from from_p in the direction of to_p."""
    d = _dist(from_p, to_p)
    if d <= max_step:
        return to_p
    t = max_step / d
    return (from_p[0] + t * (to_p[0] - from_p[0]), from_p[1] + t * (to_p[1] - from_p[1]))


def _reconstruct(parents, goal_idx, nodes):
    path = []
    i = goal_idx
    while i is not None:
        path.append(nodes[i])
        i = parents[i]
    return path[::-1]


def rrt(world, start, goal, max_step=5.0, goal_bias=0.05, max_iters=5000,
        goal_tol=2.0, seed=12345):
    """Plain RRT. Returns (path, nodes, parents) where path is a list of points or None if it fails."""
    rng = _LCG(seed)
    nodes = [start]
    parents = [None]

    for _ in range(max_iters):
        if rng.random() < goal_bias:
            sample = goal
        else:
            sample = (rng.uniform(0, world.width), rng.uniform(0, world.height))
        # nearest existing node
        nearest = min(range(len(nodes)), key=lambda i: _dist(nodes[i], sample))
        new_p = _steer(nodes[nearest], sample, max_step)
        if not world.point_free(new_p) or not world.segment_free(nodes[nearest], new_p):
            continue
        nodes.append(new_p)
        parents.append(nearest)
        if _dist(new_p, goal) <= goal_tol and world.segment_free(new_p, goal):
            nodes.append(goal)
            parents.append(len(nodes) - 2)
            return _reconstruct(parents, len(nodes) - 1, nodes), nodes, parents
    return None, nodes, parents


def rrt_star(world, start, goal, max_step=5.0, goal_bias=0.05, max_iters=5000,
             goal_tol=2.0, radius=8.0, seed=12345):
    """RRT* with rewiring for asymptotic optimality. Returns (path, nodes, parents)."""
    rng = _LCG(seed)
    nodes = [start]
    parents = [None]
    cost = [0.0]                              # cost-to-come for each node
    goal_idx = None

    for _ in range(max_iters):
        if rng.random() < goal_bias:
            sample = goal
        else:
            sample = (rng.uniform(0, world.width), rng.uniform(0, world.height))
        nearest = min(range(len(nodes)), key=lambda i: _dist(nodes[i], sample))
        new_p = _steer(nodes[nearest], sample, max_step)
        if not world.point_free(new_p) or not world.segment_free(nodes[nearest], new_p):
            continue

        # choose the parent among nearby nodes that minimises cost-to-come
        near = [i for i in range(len(nodes)) if _dist(nodes[i], new_p) <= radius
                and world.segment_free(nodes[i], new_p)]
        if not near:
            near = [nearest]
        best_parent = min(near, key=lambda i: cost[i] + _dist(nodes[i], new_p))
        new_idx = len(nodes)
        nodes.append(new_p)
        parents.append(best_parent)
        cost.append(cost[best_parent] + _dist(nodes[best_parent], new_p))

        # rewire: can any near node reach a lower cost THROUGH the new node?
        for i in near:
            new_cost = cost[new_idx] + _dist(new_p, nodes[i])
            if new_cost < cost[i] and world.segment_free(new_p, nodes[i]):
                parents[i] = new_idx
                cost[i] = new_cost

        # track the best goal connection seen so far, keep improving
        if _dist(new_p, goal) <= goal_tol and world.segment_free(new_p, goal):
            gcost = cost[new_idx] + _dist(new_p, goal)
            if goal_idx is None or gcost < cost[goal_idx]:
                if goal_idx is None:
                    goal_idx = len(nodes)
                    nodes.append(goal)
                    parents.append(new_idx)
                    cost.append(gcost)
                else:
                    parents[goal_idx] = new_idx
                    cost[goal_idx] = gcost

    if goal_idx is None:
        return None, nodes, parents
    return _reconstruct(parents, goal_idx, nodes), nodes, parents


def path_length(path):
    """Total Euclidean length of a path (list of points)."""
    if not path or len(path) < 2:
        return 0.0
    return sum(_dist(path[i], path[i + 1]) for i in range(len(path) - 1))


def path_is_valid(world, path, start, goal, goal_tol=2.0, step=0.5):
    """True if the path starts at start, ends at goal, and every segment is collision-free."""
    if not path or len(path) < 2:
        return False
    if _dist(path[0], start) > 1e-9:
        return False
    if _dist(path[-1], goal) > goal_tol + 1e-9:
        return False
    for i in range(len(path) - 1):
        if not world.segment_free(path[i], path[i + 1], step):
            return False
    return True
