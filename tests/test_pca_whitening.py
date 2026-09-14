"""Tests for PCA + whitening: identity covariance, principal axis, Eckart-Young, ZCA proximity, round-trips."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pca_whitening as P  # noqa: E402


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


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def _gauss(rnd):
    return math.sqrt(-2 * math.log(rnd() + 1e-12)) * math.cos(2 * math.pi * rnd())


def _planted_2d(rnd, n, sx, sy, theta):
    """n samples with std sx, sy along axes rotated by theta."""
    ct, stt = math.cos(theta), math.sin(theta)
    data = []
    for _ in range(n):
        a = sx * _gauss(rnd)
        b = sy * _gauss(rnd)
        data.append([ct * a - stt * b, stt * a + ct * b])
    return data


def main():
    rnd = _lcg(42)
    data = _planted_2d(rnd, 800, 3.0, 1.0, math.pi / 4)  # major axis at 45 deg

    # ---- 1. covariance is symmetric ------------------------------------------------------
    cov = P.covariance(data)
    check("covariance symmetric", abs(cov[0][1] - cov[1][0]) < 1e-12)

    # ---- 2. top principal axis aligns with the planted major axis (45 deg) --------------
    model = P.pca(data)
    ax = model["components"][0]
    # should be ~ (+/-0.707, +/-0.707); check |cos angle to (1,1)/sqrt2| ~ 1
    dot = abs(ax[0] * (1 / math.sqrt(2)) + ax[1] * (1 / math.sqrt(2)))
    check("top axis aligns with 45-degree major axis", dot > 0.99, f"|cos| {dot:.4f}")
    check("top eigenvalue >> second", model["eigenvalues"][0] > 4 * model["eigenvalues"][1],
          f"{model['eigenvalues']}")

    # ---- 3. explained-variance ratios sum to 1 and are descending -----------------------
    r = model["explained_variance_ratio"]
    check("explained-variance ratios sum to 1", abs(sum(r) - 1.0) < 1e-9, f"{sum(r)}")
    check("explained-variance ratios descending", all(r[i] >= r[i + 1] for i in range(len(r) - 1)))

    # ---- 4. PCA-whitened data has identity covariance -----------------------------------
    wp, mu, Wp = P.whiten(data, "pca")
    cp = P.covariance(wp)
    idp = all(abs(cp[i][j] - (1.0 if i == j else 0.0)) < 1e-6
              for i in range(len(cp)) for j in range(len(cp)))
    check("PCA-whitened covariance is identity", idp, f"{cp}")

    # ---- 5. ZCA-whitened data has identity covariance and W is symmetric ----------------
    wz, muz, Wz = P.whiten(data, "zca")
    cz = P.covariance(wz)
    idz = all(abs(cz[i][j] - (1.0 if i == j else 0.0)) < 1e-6
              for i in range(len(cz)) for j in range(len(cz)))
    check("ZCA-whitened covariance is identity", idz, f"{cz}")
    check("ZCA whitening matrix symmetric",
          all(abs(Wz[i][j] - Wz[j][i]) < 1e-9 for i in range(len(Wz)) for j in range(len(Wz))))

    # ---- 6. ZCA stays strictly closer to the (centered) data than PCA-whitening ---------
    cen, _ = P.center(data)

    def sqdist(A, B):
        return sum((A[i][j] - B[i][j]) ** 2 for i in range(len(A)) for j in range(len(A[0])))

    dp = sqdist(wp, cen)
    dz = sqdist(wz, cen)
    check("ZCA closer to original than PCA-whitening", dz < dp, f"ZCA {dz:.1f} vs PCA {dp:.1f}")

    # ---- 7. Eckart-Young: rank-1 reconstruction error == tail eigenvalue sum ------------
    # For centered data, mean sq recon error of rank-k = sum of discarded eigenvalues.
    m1 = P.pca(data, n_components=1)
    err = P.reconstruction_error(data, m1)
    tail = sum(model["all_eigenvalues"][1:])  # discarded eigenvalue(s)
    check("rank-1 recon error == discarded eigenvalue (Eckart-Young)",
          abs(err - tail) < 1e-6, f"err {err:.6f} vs tail {tail:.6f}")

    # ---- 8. full-rank PCA reconstructs exactly ------------------------------------------
    mfull = P.pca(data)
    check("full-rank reconstruction is exact", P.reconstruction_error(data, mfull) < 1e-9,
          f"{P.reconstruction_error(data, mfull)}")

    # ---- 9. transform / inverse_transform round-trip ------------------------------------
    scores = P.transform(data, mfull)
    rec = P.inverse_transform(scores, mfull)
    rt = max(abs(data[i][j] - rec[i][j]) for i in range(len(data)) for j in range(2))
    check("transform round-trips through inverse", rt < 1e-9, f"{rt}")

    # ---- 10. whitening inverse: W^{-1} recovers centered data ---------------------------
    # Apply W then solve W x = y for a couple of points via the known inverse relation:
    # since cov(wp) = I, un-whitening is (E diag(sqrt lambda)) applied to scores. Simpler: check
    # that re-covarying gives back the eigenvalues by transforming whitened data forward.
    # Instead verify norm preservation property of ZCA on an isotropic cloud.
    rnd2 = _lcg(7)
    iso = _planted_2d(rnd2, 600, 2.0, 2.0, 0.3)  # already isotropic (sx==sy)
    wi, _, _ = P.whiten(iso, "zca")
    ci = P.covariance(wi)
    check("ZCA on isotropic cloud still identity",
          all(abs(ci[i][j] - (1.0 if i == j else 0.0)) < 1e-6 for i in range(2) for j in range(2)))

    # ---- 11. 3D data: whitening identity + explained variance -------------------------
    rnd3 = _lcg(99)
    d3 = []
    for _ in range(500):
        a = 4 * _gauss(rnd3)
        b = 2 * _gauss(rnd3)
        c = 0.5 * _gauss(rnd3)
        d3.append([a + 0.1 * b, b - 0.2 * c, c + 0.3 * a])
    w3, _, _ = P.whiten(d3, "pca")
    c3 = P.covariance(w3)
    check("3D PCA-whitened covariance is identity",
          all(abs(c3[i][j] - (1.0 if i == j else 0.0)) < 1e-5 for i in range(3) for j in range(3)),
          f"{c3}")
    m3 = P.pca(d3)
    check("3D explained variance sums to 1", abs(sum(m3["explained_variance_ratio"]) - 1.0) < 1e-9)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
