"""Corrected headroom check: select tau AND eta to maximize RESOLVE (the actual
objective; you cannot resolve a mode you never reached). Compare best single
bandwidth vs best additive two-scale. Genuine headroom iff best single resolve
is low AND two-scale beats it."""
import sys, numpy as np
sys.path.insert(0,".")
from lowdim_drift import drift_paper, controlled_means
from mode_recovery import coverage, basin_radius, sigma_radius

def evalrun(fieldfn,eta,modes,rb,rs,w,d,sigma,K,N,steps,seeds):
    def target(n,rng): i=rng.integers(0,K,n); return modes[i]+rng.normal(size=(n,d))*sigma
    def init(rng): return modes[0]+rng.normal(size=(N,d))*sigma
    rc,rz=[],[]
    for s in range(seeds):
        rng=np.random.default_rng(100+s); drng=np.random.default_rng(200+s); q=init(rng)
        for _ in range(steps):
            q=q+eta*fieldfn(q,target(N,drng))
            if not np.all(np.isfinite(q)): break
        if not np.all(np.isfinite(q)): return None
        rc.append(coverage(q,modes,rb,w)["unweighted"]); rz.append(coverage(q,modes,rs,w)["unweighted"])
    return float(np.median(rc)),float(np.median(rz))

def run(K,d,L,sigma,N=128,steps=800,seeds=3):
    modes=controlled_means(K,d,L); w=np.full(K,1.0/K)
    rb,rs=basin_radius(modes),sigma_radius(sigma,K=K); tf=3*sigma
    def best(cands):
        b=None
        for fn in cands:
            for eta in (0.05,0.1,0.2):
                r=evalrun(fn,eta,modes,rb,rs,w,d,sigma,K,N,steps,seeds)
                if r and (b is None or r[1]>b[1]): b=r
        return b
    single=best([(lambda q,dt,t=t: drift_paper(q,dt,t,True)) for t in (0.2,0.4,0.6,0.8,1.2)])
    twos=best([(lambda q,dt,tc=tc,a=a: drift_paper(q,dt,tc,True)+a*drift_paper(q,dt,tf,True))
               for tc in (0.4,0.6,0.8,1.2) for a in (0.5,1.0)])
    print(f"K={K:3d} d={d:2d} L/s={L/sigma:>3.0f} | single(reach,resolve)=({single[0]:.2f},{single[1]:.2f})"
          f" | 2scale=({twos[0]:.2f},{twos[1]:.2f}) | headroom={'YES' if single[1]<0.8 else 'no'}"
          f" 2scale_gain={twos[1]-single[1]:+.2f}")

for reg in [(16,2,1.0,0.03),(32,2,1.0,0.02),(64,2,1.0,0.015),
            (16,5,1.0,0.03),(32,5,1.0,0.02),(16,10,1.0,0.03)]:
    run(*reg)
