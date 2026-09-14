"""Iterative Closest Point: aligning two point clouds when you don't know which point matches which.

The Kabsch algorithm finds the optimal rotation aligning two point sets -- but only when you already
know the CORRESPONDENCE, which point maps to which. In the real problem -- registering two 3-D scans,
matching a laser sweep to a map, fitting a model to sensor data -- you have two unordered clouds and no
correspondence at all. ITERATIVE CLOSEST POINT (Besl-McKay, 1992) solves it by alternating two steps
until it converges:

  1. MATCH: for each source point, find its nearest neighbour in the target cloud (the "closest
     point" that gives the algorithm its name).
  2. ALIGN: with those tentative correspondences, solve the Kabsch/Procrustes problem for the rigid
     transform (rotation + translation, optionally scale) that best maps source onto target.

Apply the transform, re-match, re-align. Each iteration cannot increase the mean-squared alignment
error (matching lowers it, then the optimal transform lowers it again), so the process converges
monotonically to a LOCAL optimum. The catch, and it is fundamental to ICP: from a poor initial pose --
a large unknown rotation -- the nearest-neighbour matching locks onto the wrong correspondences and ICP
settles into a wrong basin. It needs a reasonable starting alignment (a coarse global registration, or
a small enough misalignment), which is why real pipelines seed ICP with feature matching. Within its
basin of convergence it is exact and fast, and it is the backbone of LiDAR SLAM, medical-image
registration, and 3-D model stitching.

This module implements ICP over the repo's Kabsch/Umeyama solver, with a brute-force nearest-neighbour
matcher, monotone-decreasing RMSD tracking, and support for rotation-only or rotation+scale. It is
validated exactly: applied to a cloud transformed by a known rotation and translation with SHUFFLED
point order, ICP recovers the transform and drives the RMSD to zero; the alignment error decreases
monotonically every iteration; it handles a scale change with the Umeyama option; it converges from a
perturbed initial guess; and on already-aligned clouds it is a no-op. Pure stdlib; the
correspondence-free registration companion to the Kabsch, Umeyama, and affine-alignment tools."""

from __future__ import annotations

import math

from kabsch import kabsch, umeyama, apply_transform, rmsd


def _nearest_neighbours(source, target):
    """For each source point, the index of its nearest target point (brute force)."""
    matches = []
    for p in source:
        best_j = 0
        best_d = float("inf")
        for j, q in enumerate(target):
            d = sum((p[k] - q[k]) ** 2 for k in range(len(p)))
            if d < best_d:
                best_d = d
                best_j = j
        matches.append(best_j)
    return matches


def _mean_sq_error(source, target, matches):
    """Mean squared distance from each source point to its matched target point."""
    total = 0.0
    for i, j in enumerate(matches):
        total += sum((source[i][k] - target[j][k]) ** 2 for k in range(len(source[i])))
    return total / len(source)


def icp(source, target, max_iter=50, tol=1e-9, with_scale=False):
    """Register `source` onto `target` by Iterative Closest Point.

    Returns a dict: 'R' (rotation), 't' (translation), 'scale', 'aligned' (transformed source),
    'rmsd', 'iterations', 'history' (RMSD per iteration).
    """
    dim = len(source[0])
    # seed with a centroid alignment: translate the source so its centroid matches the target's.
    # This removes the (arbitrarily large) translation up front so ICP's nearest-neighbour matching
    # starts in a far better basin -- without it a big offset guarantees wrong correspondences.
    cs = [sum(p[k] for p in source) / len(source) for k in range(dim)]
    ct = [sum(q[k] for q in target) / len(target) for k in range(dim)]
    shift = [ct[k] - cs[k] for k in range(dim)]
    current = [[p[k] + shift[k] for k in range(dim)] for p in source]
    history = []
    prev_err = float("inf")
    it = 0
    for it in range(1, max_iter + 1):
        matches = _nearest_neighbours(current, target)
        matched_target = [target[j] for j in matches]
        # solve the optimal transform from current -> matched_target (kabsch/umeyama return dicts)
        if with_scale:
            sol = umeyama(current, matched_target, with_scale=True)
        else:
            sol = kabsch(current, matched_target)
        current = sol["transformed"]
        err = _mean_sq_error(current, target, _nearest_neighbours(current, target))
        history.append(math.sqrt(err))
        # converged if the alignment error is tiny, or barely changed since the last iteration
        if err < tol or abs(prev_err - err) < tol:
            break
        prev_err = err

    # recover the overall transform from the original source to the final aligned cloud
    if with_scale:
        sol = umeyama(source, current, with_scale=True)
        R, t, s = sol["R"], sol["t"], sol["scale"]
    else:
        sol = kabsch(source, current)
        R, t, s = sol["R"], sol["t"], 1.0
    return {
        "R": R,
        "t": t,
        "scale": s,
        "aligned": current,
        "rmsd": history[-1] if history else 0.0,
        "iterations": it,
        "history": history,
    }


def apply_rigid(R, t, points, scale=1.0):
    """Apply rotation R, scale, and translation t to a list of points."""
    return apply_transform(R, t, points, scale=scale)


def rotation_z(theta):
    """A 3-D rotation about the z-axis by theta (for building test transforms)."""
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def rotation_2d(theta):
    """A 2-D rotation matrix."""
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]
