"""Tests for rrt: collision-free paths, valid endpoints, RRT* <= RRT length, failure on walled goal."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rrt import (World, rrt, rrt_star, path_length, path_is_valid, _dist, _steer)  # noqa: E402


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
    # ---- 1. steering respects max step -----------------------------------------------
    p = _steer((0, 0), (10, 0), 3.0)
    check("steer clamps to max step", abs(_dist((0, 0), p) - 3.0) < 1e-9, f"{p}")
    p2 = _steer((0, 0), (1, 0), 3.0)
    check("steer reaches close targets", p2 == (1, 0))

    # ---- 2. obstacle-free world: path near straight-line optimum ----------------------
    world = World(100, 100, [])
    start, goal = (5, 5), (95, 95)
    path, nodes, parents = rrt(world, start, goal, max_step=8, goal_tol=2.0, seed=1)
    check("RRT finds a path in open space", path is not None)
    check("open-space path is valid", path_is_valid(world, path, start, goal))
    straight = _dist(start, goal)
    check("open-space path within 1.6x of straight line", path_length(path) < 1.6 * straight,
          f"len={path_length(path):.1f} straight={straight:.1f}")

    # ---- 3. cluttered world: collision-free path --------------------------------------
    obstacles = [(30, 30, 12), (60, 60, 15), (50, 20, 10), (25, 70, 12), (75, 35, 10)]
    world = World(100, 100, obstacles)
    start, goal = (5, 5), (95, 95)
    for seed in (1, 7, 42, 100):
        path, nodes, parents = rrt(world, start, goal, max_step=6, goal_bias=0.1,
                                   max_iters=8000, goal_tol=2.5, seed=seed)
        check(f"RRT path is collision-free (seed {seed})",
              path is not None and path_is_valid(world, path, start, goal, goal_tol=2.5),
              "no path or invalid")

    # ---- 4. tree edges never cross an obstacle ----------------------------------------
    path, nodes, parents = rrt(world, start, goal, max_step=6, goal_bias=0.1,
                               max_iters=8000, goal_tol=2.5, seed=3)
    bad_edges = 0
    for i, par in enumerate(parents):
        if par is not None and not world.segment_free(nodes[par], nodes[i]):
            bad_edges += 1
    check("no tree edge crosses an obstacle", bad_edges == 0, f"{bad_edges} bad edges")

    # ---- 5. RRT* path is no longer than RRT (usually shorter) -------------------------
    lengths_rrt = []
    lengths_star = []
    for seed in (1, 2, 3, 4, 5):
        pr, _, _ = rrt(world, start, goal, max_step=6, goal_bias=0.1, max_iters=8000,
                       goal_tol=2.5, seed=seed)
        ps, _, _ = rrt_star(world, start, goal, max_step=6, goal_bias=0.1, max_iters=8000,
                            goal_tol=2.5, radius=12, seed=seed)
        if pr and ps:
            lengths_rrt.append(path_length(pr))
            lengths_star.append(path_length(ps))
    avg_rrt = sum(lengths_rrt) / len(lengths_rrt)
    avg_star = sum(lengths_star) / len(lengths_star)
    check("RRT* average path shorter than RRT", avg_star <= avg_rrt + 1e-6,
          f"star={avg_star:.1f} rrt={avg_rrt:.1f}")
    check("RRT* paths valid", all(path_is_valid(world, rrt_star(world, start, goal, max_step=6,
          goal_bias=0.1, max_iters=8000, goal_tol=2.5, radius=12, seed=s)[0], start, goal,
          goal_tol=2.5) for s in (1, 2, 3)))

    # ---- 6. walled-off goal -> failure -------------------------------------------------
    # a wall of overlapping circles splits the space; goal is unreachable
    wall = [(50, y, 6) for y in range(0, 105, 8)]
    walled = World(100, 100, wall)
    path, _, _ = rrt(walled, (10, 50), (90, 50), max_step=5, goal_bias=0.1,
                     max_iters=3000, goal_tol=2.0, seed=1)
    check("walled-off goal reports failure", path is None, "returned a path through a wall!")

    # ---- 7. reproducibility ------------------------------------------------------------
    p1, _, _ = rrt(world, start, goal, max_step=6, goal_bias=0.1, max_iters=8000,
                   goal_tol=2.5, seed=99)
    p2, _, _ = rrt(world, start, goal, max_step=6, goal_bias=0.1, max_iters=8000,
                   goal_tol=2.5, seed=99)
    check("same seed -> identical path", p1 == p2)

    # ---- 8. world collision primitives -------------------------------------------------
    w = World(10, 10, [(5, 5, 2)])
    check("point inside obstacle not free", not w.point_free((5, 5)))
    check("point outside obstacle free", w.point_free((1, 1)))
    check("out of bounds not free", not w.point_free((-1, 5)))
    check("segment through obstacle blocked", not w.segment_free((5, 0), (5, 10)))
    check("segment around obstacle clear", w.segment_free((0, 0), (0, 10)))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
