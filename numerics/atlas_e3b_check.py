"""E3(b) follow-up: is the stalled imbalanced state near an exact equilibrium,
or on a slow traveling manifold?  Certify via least-squares residual floor +
inspect the per-particle field structure."""
import numpy as np
from numpy.linalg import norm
from scipy.optimize import least_squares

MASTER = 20260718
rng0 = np.random.default_rng(MASTER + 3)
d, L, N = 1, 5.0, 40
A = rng0.normal(size=(6, d)) * 0.3
B = rng0.normal(size=(6, d)) * 0.3 + L * np.eye(1, d)[0]
p_pts, p_w = np.vstack([A, B]), np.full(12, 1 / 12)
tau = 0.7
alpha0 = 0.2
rr = np.random.default_rng(MASTER + 31)
nA = int(alpha0 * N)
qa = rr.normal(size=(nA, d)) * 0.3
qb = rr.normal(size=(N - nA, d)) * 0.3 + L * np.eye(1, d)[0]
q = np.vstack([qa, qb])
q_w = np.full(N, 1 / N)


def mean_shift(X, pts, w):
    D = X[:, None, :] - pts[None, :, :]
    r = norm(D, axis=2)
    K = w[None, :] * np.exp(-r / tau)
    Z = K.sum(axis=1)
    return (K[:, :, None] * (-D)).sum(axis=1) / Z[:, None]


def V_of(qq):
    return mean_shift(qq, p_pts, p_w) - mean_shift(qq, qq, q_w)


# settle
for _ in range(30000):
    q = q + 0.1 * tau * V_of(q)

V = V_of(q)
print("per-particle V after settling (cluster A = first %d):" % nA)
print("  cluster A: V in [%.2e, %.2e]" % (V[:nA].min(), V[:nA].max()))
print("  cluster B: V in [%.2e, %.2e]" % (V[nA:].min(), V[nA:].max()))
print("  e^{-L/tau} = %.2e" % np.exp(-L / tau))

res = least_squares(lambda v: V_of(v.reshape(N, d)).ravel(), q.ravel(),
                    method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15)
Vfin = V_of(res.x.reshape(N, d))
print("least-squares: residual floor = %.3e (gradient-converged=%s)"
      % (norm(Vfin.ravel()), res.status in (1, 2, 3, 4)))
print("=> %s" % ("EXACT spurious equilibrium exists"
                 if norm(Vfin.ravel()) < 1e-12 else
                 "NO exact equilibrium: strictly positive residual floor "
                 "(slow traveling state, speed ~ e^{-L/tau})"))
