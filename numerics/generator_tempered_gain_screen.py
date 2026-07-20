"""H1 screening: does any tempered gain (P*Q)^gamma beat paper under Adam?
Validation SCREEN (not a gate). Exact E5 Adam/MLP semantics, reused/masked
reference, no clip -> isolates the gamma reliability knob. Rows saved to disk.
"""
import sys, csv, pathlib, warnings, numpy as np
from collections import defaultdict
sys.path.insert(0,".")
import run_identifiability_generator as G
from identifiability_drift import compute_field
from run_identifiability_improvement import (energy_distance2_with_reference_self,
    mean_pairwise_distance, hierarchical_stats)

STEPS, SEEDS, BATCH = 800, 6, 64
GAMMAS = [0.0, 0.5, 1.0, 1.5, 2.0]
INITS = ("broad","missing","far","concentrated")
CSVPATH = str(pathlib.Path(__file__).resolve().parent / "h1_screen_rows.csv")

def trial(target, ti, init, ii, gain, gamma, seed):
    base = G.seed_base(31337, ti, ii, seed)
    m = G.TanhMLP(target, init, base+1)
    qr=np.random.default_rng(base+2); dr=np.random.default_rng(base+4)
    reff=target.sample(1024,np.random.default_rng(base+7)); reffs=mean_pairwise_distance(reff,reff)
    for _ in range(STEPS):
        zq=qr.normal(size=(BATCH,m.latent_dim)); x,c=m.forward(zq,True)
        pos=target.sample(BATCH,dr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r=compute_field(x,pos,None,tau=0.35,gain=gain,gamma=gamma,mask=True,
                            jitter_sigma=0.0,on_degenerate="zero")
        try: m.stopgrad_step(c, r.V)
        except FloatingPointError: return float("inf")
        if not m.finite(): return float("inf")
    zf=np.random.default_rng(base+6).normal(size=(1024,m.latent_dim))
    return max(0.0,energy_distance2_with_reference_self(m.forward(zf),reff,reffs))

_,targets=G.load_targets()
arms=[("paper","paper",1.0)]+[(f"pow-g{g}","power",g) for g in GAMMAS]
rows=[]
with open(CSVPATH,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["arm","cell","seed","ed2","family","init"])
    for ti,t in enumerate(targets):
        for ii,init in enumerate(INITS):
            for label,gain,gamma in arms:
                for s in range(SEEDS):
                    ed=trial(t,ti,init,ii,gain,gamma,s)
                    row=[label,f"{t.name}/{init}",s,ed,t.family,init]
                    w.writerow(row); f.flush()
                    rows.append({"arm":label,"cell":f"{t.name}/{init}","seed":s,
                                 "ed2":ed,"family":t.family,"init":init})
        print(f"done {t.name}", flush=True)

def geo(vals):
    v=np.maximum(np.asarray(list(vals),dtype=float),1e-12); return float(np.exp(np.mean(np.log(v))))
def cellmed(arm):
    g=defaultdict(list)
    for r in rows:
        if r["arm"]==arm and np.isfinite(r["ed2"]): g[r["cell"]].append(r["ed2"])
    return {c:float(np.median(v)) for c,v in g.items()}

print("\n=== H1 screen: aggregate ED2 and ratio vs paper (Adam, reused/masked) ===")
pa=geo(cellmed("paper").values())
print(f"{'arm':10s}{'aggED2':>10s}{'ratio':>9s}{'hierCI':>20s}{'win%':>7s}")
print(f"{'paper':10s}{pa:10.5f}{1.0:9.3f}{'baseline':>20s}")
for label,gain,gamma in arms:
    if label=="paper": continue
    agg=geo(cellmed(label).values())
    st=hierarchical_stats(rows,label,seed=99+int(gamma*10))
    win=np.mean(np.asarray(list(st["cell_ratios"].values()))<1.0); ci=st["hierarchical_ci"]
    print(f"{label:10s}{agg:10.5f}{agg/pa:9.3f}   [{ci[0]:.3f},{ci[1]:.3f}]{win:7.2f}")

print("\nper-init geo-ratio vs paper:")
for label,gain,gamma in arms:
    if label=="paper": continue
    byinit=defaultdict(list); pair=defaultdict(dict)
    for r in rows:
        if r["arm"] in ("paper",label) and np.isfinite(r["ed2"]):
            pair[(r["cell"],r["seed"])][r["arm"]]=r["ed2"]
    for (cell,seed),d in pair.items():
        if "paper" in d and label in d and d["paper"]>0:
            byinit[cell.split("/")[1]].append(np.log(max(d[label],1e-12)/max(d["paper"],1e-12)))
    print(f"  {label:10s} "+" ".join(f"{i}={np.exp(np.mean(byinit[i])):.3f}" for i in INITS))
