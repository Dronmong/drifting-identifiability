"""The Collapse Atlas: fixed points and stability of Laplace drifting dynamics.

Spec: numerics/CollapseAtlas.md.  Results append to CollapseAtlasResults.md.

Usage:
    uv run --with numpy --with scipy python numerics/collapse_atlas.py [E1|E2|E3|E4|E5|E6|all]
"""
import sys
import numpy as np
from numpy.linalg import norm, eigvals
from scipy.optimize import root, minimize

MASTER = 20260718
OUT = "numerics/CollapseAtlasResults.md"
_log_lines = []


def log(s=""):
    print(s)
    _log_lines.append(s)


def flush(section):
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(f"\n## {section}\n\n```\n" + "\n".join(_log_lines) + "\n```\n")
    _log_lines.clear()


# ---------------- core fields (exact population formulas) ----------------

def mean_shift(X, pts, w, tau):
    """m_mu at probes X (k,d) for atomic mu = (pts (m,d), w (m,))."""
    X = np.atleast_2d(X)
    D = X[:, None, :] - pts[None, :, :]          # (k,m,d)
    r = norm(D, axis=2)                          # (k,m)
    K = w[None, :] * np.exp(-r / tau)            # (k,m)
    Z = K.sum(axis=1)                            # (k,)
    num = (K[:, :, None] * (-D)).sum(axis=1)     # (k,d): (y - x)
    return num / Z[:, None], Z


def drift(X, p_pts, p_w, q_pts, q_w, tau):
    mp, _ = mean_shift(X, p_pts, p_w, tau)
    mq, _ = mean_shift(X, q_pts, q_w, tau)
    return mp - mq


def jac_fd(f, x, h):
    """Finite-difference Jacobian of f: R^d -> R^d at x."""
    d = x.size
    J = np.zeros((d, d))
    for i in range(d):
        e = np.zeros(d); e[i] = h
        J[:, i] = (f(x + e) - f(x - e)) / (2 * h)
    return J


# ---------------- diagnostics ----------------

def energy_dist(p_pts, p_w, q_pts, q_w):
    def cross(A, a, B, b):
        return (a[:, None] * b[None, :] *
                norm(A[:, None, :] - B[None, :, :], axis=2)).sum()
    return 2 * cross(p_pts, p_w, q_pts, q_w) - cross(p_pts, p_w, p_pts, p_w) \
        - cross(q_pts, q_w, q_pts, q_w)


def mmd2(p_pts, p_w, q_pts, q_w, tau):
    def kk(A, a, B, b):
        return (a[:, None] * b[None, :] *
                np.exp(-norm(A[:, None, :] - B[None, :, :], axis=2) / tau)).sum()
    return kk(p_pts, p_w, p_pts, p_w) + kk(q_pts, q_w, q_pts, q_w) \
        - 2 * kk(p_pts, p_w, q_pts, q_w)


def z_of(X, pts, w, tau):
    r = norm(X[:, None, :] - pts[None, :, :], axis=2)
    return (w[None, :] * np.exp(-r / tau)).sum(axis=1)


def psi_of(X, pts, w, tau):
    r = norm(X[:, None, :] - pts[None, :, :], axis=2)
    return (w[None, :] * tau * (r + tau) * np.exp(-r / tau)).sum(axis=1)


def probe_grid(all_pts, tau, d, n1=241, n2=27):
    lo = all_pts.min(axis=0) - 3 * tau
    hi = all_pts.max(axis=0) + 3 * tau
    if d == 1:
        return np.linspace(lo[0], hi[0], n1)[:, None]
    gx = np.linspace(lo[0], hi[0], n2)
    gy = np.linspace(lo[1], hi[1], n2)
    return np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)


def diagnostics(p_pts, p_w, q_pts, q_w, tau, G):
    Zp, Zq = z_of(G, p_pts, p_w, tau), z_of(G, q_pts, q_w, tau)
    Pp, Pq = psi_of(G, p_pts, p_w, tau), psi_of(G, q_pts, q_w, tau)
    ratio = Pp / Pq
    return {
        "ED": energy_dist(p_pts, p_w, q_pts, q_w),
        "MMD2": mmd2(p_pts, p_w, q_pts, q_w, tau),
        "Zgap": np.abs(Zp - Zq).max(),
        "oscRatio": ratio.max() - ratio.min(),
        "defect": np.abs(Zq * Pp - Zp * Pq).max(),
        "resid": norm(drift(q_pts, p_pts, p_w, q_pts, q_w, tau), axis=1).max(),
    }


# ---------------- target families ----------------

def family(name, rng, d=1, L=5.0):
    """Return (pts, w) for the target p."""
    if name == "P1":  # two equal clusters
        A = rng.normal(size=(3, d)) * 0.3
        B = rng.normal(size=(3, d)) * 0.3
        B[:, 0] += L
        pts = np.vstack([A, B])
        w = np.full(6, 1 / 6)
    elif name == "P2":  # two clusters, masses .3/.7
        A = rng.normal(size=(3, d)) * 0.3
        B = rng.normal(size=(3, d)) * 0.3
        B[:, 0] += L
        pts = np.vstack([A, B])
        w = np.concatenate([np.full(3, 0.3 / 3), np.full(3, 0.7 / 3)])
    elif name == "P3":  # three collinear clusters
        pts = np.vstack([rng.normal(size=(2, d)) * 0.2 + np.eye(1, d)[0] * s
                         for s in (0.0, L / 2, L)])
        w = np.full(6, 1 / 6)
    elif name == "P4":  # generic 5-atom cloud
        pts = rng.normal(size=(5, d)) * L / 3
        w = rng.dirichlet(np.ones(5))
    return pts, w


# ---------------- E1: fixed-point census ----------------

def find_zeros(p_pts, p_w, tau, rng, n_seeds=250):
    d = p_pts.shape[1]
    lo = p_pts.min(axis=0) - 2 * tau
    hi = p_pts.max(axis=0) + 2 * tau
    seeds = rng.uniform(lo, hi, size=(n_seeds, d))
    seeds = np.vstack([seeds, p_pts])
    zeros = []
    f = lambda x: mean_shift(x[None, :], p_pts, p_w, tau)[0][0]
    for s in seeds:
        sol = root(f, s, method="hybr", tol=1e-12)
        if sol.success and norm(f(sol.x)) < 1e-9:
            if all(norm(sol.x - z) > 1e-4 * tau for z in zeros):
                zeros.append(sol.x)
    out = []
    for z in zeros:
        J = jac_fd(f, z, 1e-6 * tau)
        ev = eigvals(J)
        kind = "sink" if np.all(ev.real < 0) else \
               ("source" if np.all(ev.real > 0) else "saddle")
        out.append((z, ev, kind))
    return out


def E1():
    rng = np.random.default_rng(MASTER + 1)
    log("E1: fixed-point census of m_p (collapse candidates)")
    for d in (1, 2):
        for name in ("P1", "P2", "P3", "P4"):
            pts, w = family(name, np.random.default_rng(MASTER + 10), d=d)
            for tau in (0.5, 1.0, 2.5, 6.0):
                zs = find_zeros(pts, w, tau, rng)
                sinks = [z for z in zs if z[2] == "sink"]
                log(f"  d={d} {name} tau={tau:4.1f}: zeros={len(zs)} "
                    f"sinks={len(sinks)} "
                    f"kinds={[k for _, _, k in zs]}")
    flush("E1 fixed-point census")


# ---------------- E2: splitting index ----------------

def splitting_index(p_pts, p_w, tau, c):
    f = lambda x: mean_shift(x[None, :], p_pts, p_w, tau)[0][0]
    Dm = jac_fd(f, c, 1e-6 * tau)
    A = np.eye(len(c)) + Dm
    return np.max(eigvals(A).real), Dm


def verify_pair_linearization(p_pts, p_w, tau, c, rng):
    """Simulate a tiny symmetric pair and regress its growth matrix."""
    d = len(c)
    _, Dm = splitting_index(p_pts, p_w, tau, c)
    A_pred = np.eye(d) + Dm
    errs = []
    for _ in range(6):
        u = rng.normal(size=d)
        u *= 1e-6 * tau / norm(u)
        q_pts = np.vstack([c + u, c - u])
        q_w = np.array([0.5, 0.5])
        V = drift(q_pts, p_pts, p_w, q_pts, q_w, tau)
        du_obs = (V[0] - V[1]) / 2
        errs.append(norm(du_obs - A_pred @ u) / norm(u))
    return max(errs)


def E2():
    rng = np.random.default_rng(MASTER + 2)
    log("E2: splitting index sigma = max Re eig(I + Dm_p) at sinks")
    worst = np.inf
    for d in (1, 2):
        for name in ("P1", "P2", "P3", "P4"):
            pts, w = family(name, np.random.default_rng(MASTER + 10), d=d)
            for tau in (0.5, 1.0, 2.5, 6.0):
                for z, ev, kind in find_zeros(pts, w, tau, rng):
                    if kind != "sink":
                        continue
                    sig, _ = splitting_index(pts, w, tau, z)
                    lin_err = verify_pair_linearization(pts, w, tau, z, rng)
                    worst = min(worst, sig)
                    log(f"  d={d} {name} tau={tau:4.1f} sink={np.round(z,3)} "
                        f"sigma={sig:+.4f} linerr={lin_err:.2e}")
    log(f"  minimum sigma over census: {worst:+.5f}")
    # adversarial search in 2-d: minimize sigma over configs with m_p(c)=0
    log("  adversarial search (2-d, 4 atoms + c, 24 restarts):")
    best = np.inf
    for r in range(24):
        rr = np.random.default_rng(MASTER + 100 + r)
        x0 = np.concatenate([rr.normal(size=8) * 3, rr.normal(size=4),
                             rr.normal(size=2)])
        tau = 1.0

        def obj(v):
            pts = v[:8].reshape(4, 2)
            w = np.exp(v[8:12]); w = w / w.sum()
            c = v[12:14]
            f = lambda x: mean_shift(x[None, :], pts, w, tau)[0][0]
            pen = norm(f(c)) / tau
            Dm = jac_fd(f, c, 1e-6)
            sig = np.max(eigvals(np.eye(2) + Dm).real)
            return sig + 60.0 * pen

        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": 4000, "fatol": 1e-10, "xatol": 1e-10})
        v = res.x
        pts = v[:8].reshape(4, 2)
        w = np.exp(v[8:12]); w = w / w.sum()
        c = v[12:14]
        f = lambda x: mean_shift(x[None, :], pts, w, tau)[0][0]
        sol = root(f, c, method="hybr", tol=1e-12)  # polish the zero
        if sol.success and norm(f(sol.x)) < 1e-9:
            sig, _ = splitting_index(pts, w, tau, sol.x)
            best = min(best, sig)
    log(f"  adversarial minimum sigma at exact zeros: {best:+.5f}")
    flush("E2 splitting index")


# ---------------- dynamics ----------------

def run_dynamics(p_pts, p_w, q_pts, q_w, tau, eta, steps,
                 rng=None, batch=None, track=None, track_every=25):
    """Deterministic population dynamics; optional minibatch noise.

    batch=None: exact field.  batch=B: at each step estimate m_p and m_q from
    B atoms resampled by weight (the stochastic mean-shift analogue)."""
    q = q_pts.copy()
    hist = []
    for t in range(steps):
        if batch is None:
            V = drift(q, p_pts, p_w, q, q_w, tau)
        else:
            ip = rng.choice(len(p_pts), size=batch, p=p_w)
            iq = rng.choice(len(q), size=batch, p=q_w)
            mp, _ = mean_shift(q, p_pts[ip], np.full(batch, 1 / batch), tau)
            mq, _ = mean_shift(q, q[iq], np.full(batch, 1 / batch), tau)
            V = mp - mq
        q = q + eta * V
        if track is not None and t % track_every == 0:
            hist.append(track(q, t))
    return q, hist


# ---------------- E3: metastability / spurious equilibria ----------------

def E3():
    log("E3: mass-imbalance relaxation (deterministic transport law)")
    d, L, N = 1, 5.0, 40
    rng0 = np.random.default_rng(MASTER + 3)
    A = rng0.normal(size=(6, d)) * 0.3
    B = rng0.normal(size=(6, d)) * 0.3 + L * np.eye(1, d)[0]
    p_pts, p_w = np.vstack([A, B]), np.full(12, 1 / 12)
    alpha0 = 0.2

    def make_q(rng):
        nA = int(alpha0 * N)
        qa = rng.normal(size=(nA, d)) * 0.3
        qb = rng.normal(size=(N - nA, d)) * 0.3 + L * np.eye(1, d)[0]
        return np.vstack([qa, qb]), np.full(N, 1 / N)

    def alpha_of(q):
        return (q[:, 0] < L / 2).mean()

    # (a) deterministic halving time in flow time (steps * eta) vs L/tau
    log("  (a) deterministic halving flow-time T vs L/tau "
        "(hypothesis: log T linear in L/tau):")
    ratios, logTs = [], []
    for tau in (2.5, 2.0, 1.7, 1.4, 1.2, 1.0, 0.85):
        eta = 0.1 * tau
        q, q_w = make_q(np.random.default_rng(MASTER + 30))
        T = None
        for t in range(120000):
            q = q + eta * drift(q, p_pts, p_w, q, q_w, tau)
            if abs(alpha_of(q) - 0.5) < 0.5 * abs(alpha0 - 0.5):
                T = t * eta
                break
        ratios.append(L / tau)
        logTs.append(np.log(T) if T else np.nan)
        log(f"    L/tau={L/tau:5.2f}: flow-time to halve = "
            f"{(f'{T:9.2f}' if T else '   >12000*eta (stalled)')}")
    ok = ~np.isnan(logTs)
    if ok.sum() >= 3:
        slope = np.polyfit(np.array(ratios)[ok], np.array(logTs)[ok], 1)[0]
        log(f"    fitted slope d(log T)/d(L/tau) = {slope:.3f} "
            "(pure exponential law would be ~1)")

    # (b) hunt an exact spurious equilibrium where the dynamics stalls
    log("  (b) spurious-equilibrium hunt at large separation (L/tau ~ 7):")
    tau = 0.7
    q, q_w = make_q(np.random.default_rng(MASTER + 31))
    q, _ = run_dynamics(p_pts, p_w, q, q_w, tau, 0.1 * tau, 30000)
    res0 = norm(drift(q, p_pts, p_w, q, q_w, tau), axis=1).max()
    log(f"    after 30000 steps: alpha={alpha_of(q):.3f} resid={res0:.2e}")

    def stacked(v):
        qq = v.reshape(N, d)
        return drift(qq, p_pts, p_w, qq, q_w, tau).ravel()

    sol = root(stacked, q.ravel(), method="hybr", tol=1e-13)
    qs = sol.x.reshape(N, d)
    res = norm(drift(qs, p_pts, p_w, qs, q_w, tau), axis=1).max()
    spurious = res < 1e-9 and abs(alpha_of(qs) - 0.5) > 0.05
    log(f"    Newton: success={sol.success} resid={res:.2e} "
        f"alpha={alpha_of(qs):.3f} -> "
        f"{'EXACT SPURIOUS EQUILIBRIUM (wrong masses, V=0 on supp q)' if spurious else 'no exact spurious equilibrium'}")
    if spurious:
        eta = 0.1 * tau
        h = 1e-7

        def step(v):
            qq = v.reshape(N, d)
            return (qq + eta * drift(qq, p_pts, p_w, qq, q_w, tau)).ravel()

        v0 = qs.ravel()
        J = np.zeros((N * d, N * d))
        for i in range(N * d):
            e = np.zeros(N * d); e[i] = h
            J[:, i] = (step(v0 + e) - step(v0 - e)) / (2 * h)
        ev = np.abs(eigvals(J))
        log(f"    spectral radius at spurious eq: {ev.max():.8f} "
            f"({'STABLE (trapped)' if ev.max() <= 1 + 1e-9 else 'unstable (escapable)'})")
    flush("E3 metastability and spurious equilibria")


# ---------------- E4: basins of attraction ----------------

def E4():
    log("E4: basin fractions over (tau, N, spread); 60 inits each")
    d, L = 2, 5.0
    p_pts, p_w = family("P1", np.random.default_rng(MASTER + 10), d=d, L=L)
    for tau in (1.0, 2.5, 6.0):
        for N in (8, 32):
            for spread in (1.0, 4.0):
                counts = {"converged": 0, "collapsed": 0,
                          "metastable": 0, "wandering": 0}
                for repi in range(60):
                    rr = np.random.default_rng(MASTER + 4000 +
                                               hash((tau, N, spread, repi)) % 100000)
                    q0 = rr.normal(size=(N, d)) * spread \
                        + p_pts.mean(axis=0)
                    q_w = np.full(N, 1 / N)
                    q, _ = run_dynamics(p_pts, p_w, q0, q_w, tau,
                                        0.1 * tau, 1500)
                    res = norm(drift(q, p_pts, p_w, q, q_w, tau),
                               axis=1).max()
                    ed = energy_dist(p_pts, p_w, q, q_w)
                    diam = norm(q - q.mean(axis=0), axis=1).max()
                    alpha = (q[:, 0] < L / 2).mean()
                    if ed < 0.05 * L:
                        counts["converged"] += 1
                    elif diam < 0.05 * L:
                        counts["collapsed"] += 1
                    elif res < 1e-4 and abs(alpha - 0.5) > 0.1:
                        counts["metastable"] += 1
                    else:
                        counts["wandering"] += 1
                log(f"  tau={tau} N={N} spread={spread}: {counts}")
    flush("E4 basins")


# ---------------- E5: stability at the truth ----------------

def E5():
    log("E5: spectral radius of the update map at q = p, vs eta/tau")
    d, L = 2, 5.0
    p_pts, p_w = family("P1", np.random.default_rng(MASTER + 10), d=d, L=L)
    N = len(p_pts)
    for tau in (1.0, 2.5):
        for eta_over in (0.05, 0.2, 0.5, 1.0, 2.0):
            eta = eta_over * tau

            def step(v):
                qq = v.reshape(N, d)
                return (qq + eta * drift(qq, p_pts, p_w, qq, p_w, tau)).ravel()

            v0 = p_pts.ravel()
            h = 1e-7
            J = np.zeros((N * d, N * d))
            for i in range(N * d):
                e = np.zeros(N * d); e[i] = h
                J[:, i] = (step(v0 + e) - step(v0 - e)) / (2 * h)
            rho = np.abs(eigvals(J)).max()
            log(f"  tau={tau} eta={eta_over}*tau: rho(J)={rho:.5f} "
                f"({'stable' if rho <= 1 + 1e-9 else 'UNSTABLE'})")
    flush("E5 stability at truth")


# ---------------- E6: Lyapunov candidate screen ----------------

def E6():
    log("E6: monotonicity violation rates of Lyapunov candidates")
    d, L = 1, 5.0
    rng0 = np.random.default_rng(MASTER + 10)
    p_pts, p_w = family("P1", rng0, d=d, L=L)
    G = probe_grid(p_pts, 1.0, d)
    cands = ["ED", "MMD2", "Zgap", "oscRatio", "defect"]
    viol = {c: 0 for c in cands}
    tot = 0
    for rep in range(12):
        rr = np.random.default_rng(MASTER + 600 + rep)
        N = 24
        q = rr.normal(size=(N, d)) * 3.0 + p_pts.mean(axis=0)
        q_w = np.full(N, 1 / N)
        tau = float(rr.choice([1.0, 2.5]))
        prev = None
        for t in range(800):
            V = drift(q, p_pts, p_w, q, q_w, tau)
            q = q + 0.1 * tau * V
            if t % 10 == 0:
                cur = diagnostics(p_pts, p_w, q, q_w, tau, G)
                if prev is not None:
                    tot += 1
                    for c in cands:
                        if cur[c] > prev[c] + 1e-12:
                            viol[c] += 1
                prev = cur
    for c in cands:
        log(f"  {c:9s}: increased on {viol[c]}/{tot} intervals "
            f"({100*viol[c]/tot:.1f}%)")
    flush("E6 Lyapunov screen")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(f"\n# Collapse Atlas run ({which}), seed {MASTER}\n")
    todo = [E1, E2, E3, E4, E5, E6] if which == "all" else [globals()[which]]
    for fn in todo:
        fn()
