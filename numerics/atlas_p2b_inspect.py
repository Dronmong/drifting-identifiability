"""Inspect the degenerate configuration that drives sigma -> 0 in P2B (d=2).
Reproduces the deterministic DE run and prints the minimizing geometry."""
import numpy as np
from numpy.linalg import eigvals, norm, svd
from scipy.optimize import differential_evolution, root

MASTER = 20260718
tau = 1.0
d, m_atoms = 2, 4
nn = m_atoms * d - (d - 1)
bounds = [(-6, 6)] * nn + [(-3, 3)] * m_atoms


def mean_shift(X, pts, w):
    X = np.atleast_2d(X)
    D = X[:, None, :] - pts[None, :, :]
    r = norm(D, axis=2)
    K = w[None, :] * np.exp(-r / tau)
    Z = K.sum(axis=1)
    return (K[:, :, None] * (-D)).sum(axis=1) / Z[:, None]


def unpack(v):
    first = np.zeros(d); first[0] = abs(v[0])
    rest = np.array(v[1:nn]).reshape(m_atoms - 1, d)
    pts = np.vstack([first, rest])
    w = np.exp(v[nn:]); w = w / w.sum()
    w = 0.96 * w + 0.04 / m_atoms
    return pts, w


def jac(pts, w, c, h):
    f = lambda x: mean_shift(x[None, :], pts, w)[0]
    J = np.zeros((d, d))
    for i in range(d):
        e = np.zeros(d); e[i] = h
        J[:, i] = (f(c + e) - f(c - e)) / (2 * h)
    return J


def obj(v):
    pts, w = unpack(v)
    f = lambda x: mean_shift(x[None, :], pts, w)[0]
    pen = norm(f(np.zeros(d))) / tau
    Dm = jac(pts, w, np.zeros(d), 1e-6)
    return np.max(eigvals(np.eye(d) + Dm).real) + 80.0 * pen


de = differential_evolution(obj, bounds, seed=MASTER + 40, maxiter=60,
                            popsize=18, tol=1e-10, polish=True)
pts, w = unpack(de.x)
f = lambda x: mean_shift(x[None, :], pts, w)[0]
sol = root(f, np.zeros(d), method="hybr", tol=1e-13)
c = sol.x
print("atoms:\n", np.round(pts, 4))
print("weights:", np.round(w, 4))
print("zero c:", np.round(c, 6), " |m_p(c)| =", norm(f(c)))
print("atom distances from c:", np.round(norm(pts - c, axis=1), 4))
print("pairwise atom distances:",
      np.round([norm(pts[i] - pts[j]) for i in range(4) for j in range(i)], 4))
A = np.eye(d) + jac(pts, w, c, 1e-6)
print("I+Dm eigs:", eigvals(A))
print("singvals:", svd(A, compute_uv=False))
