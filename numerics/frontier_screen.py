"""Screens for the post-converse research frontier (2026-07-18).

[A] The Matérn ladder: for the Matérn-3/2 kernel k(r) = (1+r/tau)e^{-r/tau},
    the companion potential f with f'(r) = -r k(r) closes elliptically:
        Delta f = (1/tau^2) f - (n+3) k        (point source, dimension n)
    mirroring the certified Laplace case  Delta psi = (1/tau^2) psi - (n+1) k.
    We check the point-source identity in n = 3 by finite differences.

[B] Mass-blindness of finite-range kernels: for a compactly supported kernel
    (Wendland-type, support radius rho), two-cluster measures with the SAME
    cluster shapes but DIFFERENT cluster masses have IDENTICAL normalized
    mean-shift fields once clusters are separated by more than 2*rho, so the
    drift V vanishes identically while p != q.  (Convention 0/0 = 0 as in the
    Lean meanShift definition.)
"""
import numpy as np

rng = np.random.default_rng(20260718)

# ---------- [A] Matérn ladder closure ----------
tau = 0.7
n = 3

def k32(r):
    return (1 + r / tau) * np.exp(-r / tau)

def f52(r):
    # companion: f(r) = int_r^inf s k32(s) ds = e^{-r/tau} (r^2 + 3 tau r + 3 tau^2)/tau ... check scale
    # d/dr [e^{-r/tau}(a r^2 + b r + c)] = e^{-r/tau}(2a r + b - (a r^2 + b r + c)/tau)
    # want = -r k32 = -(r + r^2/tau) e^{-r/tau}
    #  => -a/tau = -1/tau  => a = 1
    #     2a - b/tau = -1  => b = tau(2+1) = 3 tau
    #     b - c/tau = 0    => c = 3 tau^2
    return np.exp(-r / tau) * (r**2 + 3 * tau * r + 3 * tau**2)

# radial Laplacian of f at radius r in dimension n: f'' + (n-1)/r f'
def lap_f(r, h=1e-5):
    fp = (f52(r + h) - f52(r - h)) / (2 * h)
    fpp = (f52(r + h) - 2 * f52(r) + f52(r - h)) / h**2
    return fpp + (n - 1) / r * fp

rs = np.linspace(0.05, 6.0, 200)
resid = np.array([lap_f(r) - ((1 / tau**2) * f52(r) - (n + 3) * k32(r)) for r in rs])
print("[A] Matérn-3/2 companion closure: max |Delta f - (f/tau^2 - (n+3)k)| =",
      np.max(np.abs(resid)))

# also re-verify the certified Laplace case as a control
def kL(r):
    return np.exp(-r / tau)

def psiL(r):
    return tau * (r + tau) * np.exp(-r / tau)

def lap_psi(r, h=1e-5):
    fp = (psiL(r + h) - psiL(r - h)) / (2 * h)
    fpp = (psiL(r + h) - 2 * psiL(r) + psiL(r - h)) / h**2
    return fpp + (n - 1) / r * fp

residL = np.array([lap_psi(r) - ((1 / tau**2) * psiL(r) - (n + 1) * tau * kL(r) / 1) for r in rs])
# careful: certified identity is psi - tau^2 Delta psi = (n+1) tau^2 Z with point source Z = k
# => Delta psi = psi/tau^2 - (n+1) k   (after dividing by tau^2; psi has the tau factor built in)
residL2 = np.array([lap_psi(r) - ((1 / tau**2) * psiL(r) - (n + 1) * kL(r)) for r in rs])
print("[A] control (Laplace):        max residual =", np.max(np.abs(residL2)))

# ---------- [B] finite-range mass blindness ----------
def wendland(r, rho):
    # C^2 Wendland-type bump, support [0, rho]
    u = np.clip(r / rho, 0, 1)
    return np.where(u < 1, (1 - u) ** 4 * (4 * u + 1), 0.0)

rho = 1.0
d = 2
# two clusters of 3 atoms each, separated by 10 >> 2*rho
A = rng.normal(size=(3, d)) * 0.2
B = rng.normal(size=(3, d)) * 0.2 + np.array([10.0, 0.0])
atoms = np.vstack([A, B])

def drift(x, weights):
    K = wendland(np.linalg.norm(atoms - x, axis=1), rho)
    w = weights * K
    Z = w.sum()
    if Z == 0.0:
        return np.zeros(d)  # Lean convention (0)⁻¹ • D = 0
    return (w[:, None] * (atoms - x)).sum(axis=0) / Z

# p: cluster masses (0.5, 0.5); q: cluster masses (0.2, 0.8); same within-cluster shape
wp = np.concatenate([np.full(3, 0.5 / 3), np.full(3, 0.5 / 3)])
wq = np.concatenate([np.full(3, 0.2 / 3), np.full(3, 0.8 / 3)])

probes = np.vstack([
    A + rng.normal(size=(3, d)) * 0.3,          # near cluster A
    B + rng.normal(size=(3, d)) * 0.3,          # near cluster B
    rng.normal(size=(6, d)) * 0.3 + np.array([5.0, 0.0]),  # dead zone
    rng.normal(size=(6, d)) * 3.0,              # scattered
])
vmax = max(np.linalg.norm(drift(x, wp) - drift(x, wq)) for x in probes)
print("[B] Wendland two-cluster: max |V(p,q)| over probes =", vmax,
      " (p != q: cluster masses 0.5/0.5 vs 0.2/0.8)")

# control: the same configuration under the LAPLACE kernel must NOT have zero drift
def driftL(x, weights):
    K = np.exp(-np.linalg.norm(atoms - x, axis=1) / tau)
    w = weights * K
    Z = w.sum()
    return (w[:, None] * (atoms - x)).sum(axis=0) / Z

vmaxL = max(np.linalg.norm(driftL(x, wp) - driftL(x, wq)) for x in probes)
print("[B] control (Laplace kernel): max |V| =", vmaxL, " (must be > 0)")
