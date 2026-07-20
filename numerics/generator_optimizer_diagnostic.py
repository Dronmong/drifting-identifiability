"""Diagnostic: does the no-freeze advantage return under plain SGD?
Mechanism prediction: with a NON-adaptive optimizer, the paper field (V=P*Q*Delta)
freezes off-support (tiny magnitude -> tiny steps) while the constant-gain field
does not. If so, normalized beats paper on 'far' inits under SGD -- confirming
that Adam (not the field) is what neutralized the E4 advantage in E5.
"""
import sys, math, warnings, numpy as np
from numpy.linalg import norm
sys.path.insert(0,".")
import run_identifiability_generator as G
from identifiability_drift import compute_field
from run_identifiability_improvement import (energy_distance2_with_reference_self,
    mean_pairwise_distance)

class SGDMLP(G.TanhMLP):
    LR = 0.05  # plain SGD on the field displacement (magnitude matters here)
    def stopgrad_step(self, cache, field):
        z,h1,h2=cache; p=self.params
        dx=-field/len(field)
        g={}; g["W3"]=h2.T@dx; g["b3"]=dx.sum(0)
        dh2=(dx@p["W3"].T)*(1-h2*h2); g["W2"]=h1.T@dh2; g["b2"]=dh2.sum(0)
        dh1=(dh2@p["W2"].T)*(1-h1*h1); g["W1"]=z.T@dh1; g["b1"]=dh1.sum(0)
        if not all(np.all(np.isfinite(v)) for v in g.values()):
            raise FloatingPointError
        for n in self.names: self.params[n]-=self.LR*g[n]

def run(target, init, arm, seed, steps=400, batch=64):
    base=G.seed_base(777, 0, {"broad":0,"far":2}[init], seed)
    m=SGDMLP(target, init, base+1)
    qrng=np.random.default_rng(base+2); rrng=np.random.default_rng(base+3)
    drng=np.random.default_rng(base+4)
    reff=target.sample(1024,np.random.default_rng(base+7))
    reffs=mean_pairwise_distance(reff,reff)
    gain,cross,mask,clip = arm
    for step in range(steps):
        zq=qrng.normal(size=(batch,m.latent_dim)); x,cache=m.forward(zq,True)
        neg=m.forward(rrng.normal(size=(batch,m.latent_dim))) if cross else None
        pos=target.sample(batch,drng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r=compute_field(x,pos,neg,tau=0.35,gain=gain,mask=mask,
                            jitter_sigma=0.0,on_degenerate="zero")
        V=G.clip_vectors(r.V,clip)
        try: m.stopgrad_step(cache,V)
        except FloatingPointError: return float("inf")
        if not m.finite(): return float("inf")
    zf=np.random.default_rng(base+6).normal(size=(1024,m.latent_dim))
    return max(0.0,energy_distance2_with_reference_self(m.forward(zf),reff,reffs))

_,targets=G.load_targets()
sel=[t for t in targets if t.name in
     ("NCJG-gaussian-1d","NCJG-moons-127","NCJG-gmix-2d-K5-uneq")]
paper=("paper",False,True,None); normx=("constant",True,False,2.0)
print("SGD diagnostic: final ED2 (median over 3 seeds), lower=better")
print(f"{'target':22s}{'init':8s}{'paper':>10s}{'norm-x':>10s}{'ratio':>8s}")
for t in sel:
    for init in ("broad","far"):
        pv=np.median([run(t,init,paper,s) for s in range(3)])
        nv=np.median([run(t,init,normx,s) for s in range(3)])
        print(f"{t.name:22s}{init:8s}{pv:10.4f}{nv:10.4f}{nv/max(pv,1e-9):8.3f}")
