"""Quaternions -- the elegant, gimbal-lock-free algebra of 3D rotation.

How do you represent the orientation of a spacecraft, a phone, a game camera, or a robot arm? Euler
angles (roll/pitch/yaw) are intuitive but suffer GIMBAL LOCK -- at certain orientations two axes align
and a degree of freedom vanishes -- and interpolating between two of them wobbles. Rotation matrices are
correct but carry nine numbers with six redundant constraints, and drift away from orthogonality as you
compose thousands of them. QUATERNIONS, discovered by Hamilton in 1843, are the answer the aerospace and
graphics worlds settled on: four numbers, one simple constraint (unit length), no gimbal lock, cheap to
compose, and -- uniquely -- they interpolate along the shortest arc at constant angular velocity.

A quaternion q = w + xi + yj + zk extends the complex numbers with THREE imaginary units obeying
Hamilton's relations i^2 = j^2 = k^2 = ijk = -1. A UNIT quaternion encodes a rotation: the rotation by
angle theta about a unit axis (ax, ay, az) is q = (cos(theta/2), sin(theta/2) * axis). Rotating a vector
v is the sandwich product q v q^-1, and -- the property that makes quaternions sing -- COMPOSING two
rotations is just multiplying their quaternions. Because the half-angle appears, each 3D rotation
corresponds to two quaternions (q and -q), the famous double cover of SO(3) by the unit sphere in 4D,
which is exactly why quaternions dodge the singularities that trap Euler angles.

The headline operation is SLERP -- spherical linear interpolation. To animate smoothly from one
orientation to another, you cannot just linearly blend the components (that cuts through the sphere and
speeds up in the middle); slerp walks the great-circle arc between the two unit quaternions at constant
angular speed, the mathematically correct "shortest rotation" that every animation system and attitude
controller uses.

This module implements quaternion arithmetic (Hamilton product, conjugate, inverse, norm), vector
rotation, slerp, and conversions to and from axis-angle, 3x3 rotation matrices, and Euler angles.
Everything is pure standard library -- ``math`` only.

Validation. (1) Rotating a vector by a quaternion equals multiplying it by the equivalent rotation
matrix, for hundreds of random axis/angle pairs. (2) Every conversion ROUND-TRIPS: quaternion ->
axis-angle -> quaternion, quaternion -> matrix -> quaternion, and quaternion -> Euler -> quaternion all
return the original rotation (up to the q/-q sign). (3) Composition is a homomorphism: rotating by
q1 then q2 equals rotating by the product q2*q1, and the product of two unit quaternions is a proper
rotation matrix (orthogonal, determinant +1). (4) Slerp endpoints are exact, its midpoint bisects the
angle, and it traverses at CONSTANT angular velocity -- equal parameter steps give equal rotation angles.
(5) Known rotations match by hand (90 degrees about z sends x to y). (6) A rotation preserves vector
lengths and the rotation matrix is orthonormal."""

import math


# ---------------------------------------------------------------------------
# core arithmetic (quaternions are 4-tuples (w, x, y, z))
# ---------------------------------------------------------------------------

def multiply(a, b):
    """Hamilton product a * b (rotation composition: 'apply b, then a')."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


def norm(q):
    return math.sqrt(sum(c * c for c in q))


def normalize(q):
    n = norm(q)
    if n == 0:
        raise ValueError("cannot normalize the zero quaternion")
    return tuple(c / n for c in q)


def inverse(q):
    """Inverse quaternion: conjugate / norm^2 (equals the conjugate for unit quaternions)."""
    n2 = sum(c * c for c in q)
    if n2 == 0:
        raise ValueError("zero quaternion has no inverse")
    w, x, y, z = conjugate(q)
    return (w / n2, x / n2, y / n2, z / n2)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# rotations
# ---------------------------------------------------------------------------

def from_axis_angle(axis, angle):
    """Unit quaternion for a rotation of ``angle`` radians about ``axis`` (need not be normalized)."""
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n == 0:
        return (1.0, 0.0, 0.0, 0.0)          # identity for a zero axis
    s = math.sin(angle / 2) / n
    return (math.cos(angle / 2), ax * s, ay * s, az * s)


def to_axis_angle(q):
    """Recover (axis, angle) from a unit quaternion. Axis is unit; angle in [0, pi]."""
    w, x, y, z = normalize(q)
    if w > 1.0:
        w = 1.0
    if w < -1.0:
        w = -1.0
    angle = 2 * math.acos(abs(w))            # abs -> pick the [0, pi] representative
    s = math.sqrt(1 - w * w)
    if s < 1e-12:
        return ((1.0, 0.0, 0.0), 0.0)        # no rotation; axis arbitrary
    sign = 1.0 if w >= 0 else -1.0           # keep axis consistent with the positive-w hemisphere
    return ((sign * x / s, sign * y / s, sign * z / s), angle)


def rotate_vector(q, v):
    """Rotate 3-vector v by unit quaternion q via the sandwich product q v q^-1."""
    qv = (0.0, v[0], v[1], v[2])
    w, x, y, z = multiply(multiply(q, qv), inverse(q))
    return (x, y, z)


# ---------------------------------------------------------------------------
# matrix conversions
# ---------------------------------------------------------------------------

def to_matrix(q):
    """3x3 rotation matrix (list of rows) from a unit quaternion."""
    w, x, y, z = normalize(q)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def from_matrix(m):
    """Unit quaternion from a 3x3 rotation matrix (Shepperd's method, numerically robust)."""
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2][1] - m[1][2]) * s
        y = (m[0][2] - m[2][0]) * s
        z = (m[1][0] - m[0][1]) * s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    return normalize((w, x, y, z))


# ---------------------------------------------------------------------------
# Euler angles (intrinsic Z-Y-X: yaw, pitch, roll)
# ---------------------------------------------------------------------------

def from_euler(roll, pitch, yaw):
    """Quaternion from roll (x), pitch (y), yaw (z) using the Z-Y-X intrinsic convention."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def to_euler(q):
    """Recover (roll, pitch, yaw) from a unit quaternion (Z-Y-X convention)."""
    w, x, y, z = normalize(q)
    # roll (x-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis), clamped for gimbal-lock safety
    sinp = 2 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    # yaw (z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


# ---------------------------------------------------------------------------
# spherical linear interpolation
# ---------------------------------------------------------------------------

def slerp(q0, q1, t):
    """Spherical linear interpolation between unit quaternions q0 and q1 at fraction t in [0, 1]."""
    q0 = normalize(q0)
    q1 = normalize(q1)
    d = dot(q0, q1)
    # take the shorter arc: if the dot is negative, negate one end (q and -q are the same rotation)
    if d < 0:
        q1 = tuple(-c for c in q1)
        d = -d
    if d > 0.9995:
        # nearly parallel: fall back to normalized linear interpolation to avoid dividing by ~0
        result = tuple(q0[i] + t * (q1[i] - q0[i]) for i in range(4))
        return normalize(result)
    theta0 = math.acos(d)
    theta = theta0 * t
    sin_theta0 = math.sin(theta0)
    s0 = math.sin(theta0 - theta) / sin_theta0
    s1 = math.sin(theta) / sin_theta0
    return tuple(s0 * q0[i] + s1 * q1[i] for i in range(4))


def angle_between(q0, q1):
    """The rotation angle (radians) needed to get from orientation q0 to q1, in [0, pi]."""
    q0 = normalize(q0)
    q1 = normalize(q1)
    d = abs(dot(q0, q1))
    d = min(1.0, d)
    return 2 * math.acos(d)


IDENTITY = (1.0, 0.0, 0.0, 0.0)
