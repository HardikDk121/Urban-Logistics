#!/usr/bin/env python3
"""
explain_decision.py - DECISION-level explanation for ONE route of a trained checkpoint.
Includes Captum Integrated Gradients for baseline-relative feature attribution.
"""
import argparse, copy, csv, os, sys
from pathlib import Path
import numpy as np
import torch

try:
    import captum
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "captum"])
from captum.attr import IntegratedGradients

sys.path.insert(0, os.environ.get("LASTMILE_DIR", "/content"))
import lastmile_rl as L

FEATS = ["t_cand/tau", "extra_vs_nearest", "same_zone", "reentry", "passby"]
COSTS = list(L.COST_NAMES)

def clone_env(env):
    e = copy.copy(env)
    e.visited, e.zone_left = env.visited.copy(), env.zone_left.copy()
    e.open_exited, e.seq = set(env.open_exited), list(env.seq)
    return e

class IGWrapper(torch.nn.Module):
    """Wrapper to isolate the specific pre-softmax logit for Captum IG."""
    def __init__(self, policy, h_t, mask_t, unv_t, chosen_idx):
        super().__init__()
        self.policy = policy
        self.h_t = h_t
        self.mask_t = mask_t
        self.unv_t = unv_t
        self.chosen_idx = chosen_idx
        
    def forward(self, dyn, static_inp):
        logits, _ = self.policy.heads(static_inp, dyn, self.h_t, self.mask_t, self.unv_t)
        return logits[:, self.chosen_idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--route-id", default="")
    ap.add_argument("--route-index", type=int, default=0)
    ap.add_argument("--step", type=int, default=-1)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--list-steps", action="store_true")
    ap.add_argument("--compare-eval", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    dev = torch.device(a.device)
    L.set_seed(0)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg = L.EnvConfig(**ck["env_cfg"])
    policy = L.Policy().to(dev).eval()
    policy.load_state_dict(ck["policy"])
    lam = np.array(ck["lam"], float)
    run_name = f"{ck['args'].get('name', 'run')}_s{ck['args'].get('seed', '?')}"
    
    print(f"ckpt: {a.ckpt}\n  run={run_name} update={ck['step']} penalty={ck['args'].get('penalty')} env={ck['env_cfg']}")
    
    store = L.RouteStore(a.cache_dir, cfg, dev)
    ids = store.split(a.split)
    rid = a.route_id or ids[a.route_index]
    route = store.get(rid)
    
    with np.load(Path(a.cache_dir) / "routes" / f"{rid}.npz") as z:
        node_ids = [str(s) for s in z["node_ids"]]

    ep = L.run_episode(route, cfg, "greedy", policy)
    with torch.no_grad():
        logits, values, _ = policy.evaluate([route], [ep])
    logits, values = logits[0].cpu().numpy(), values[0].cpu().numpy()
    S, N = logits.shape

    env = L.RouteEnv(route, cfg)
    rows_by_step, summary = [], []
    for t in range(S):
        valid = np.flatnonzero(ep.mask[t])
        ph = env.phase()
        recs = []
        for j in valid:
            tt, c, _ = clone_env(env).step(int(j))
            recs.append((int(j), float(tt), c, float((c * lam[ph]).sum())))
        rows_by_step.append((ph, recs))
        nearest_valid = int(valid[np.argmin(route.T[env.cur, valid])])
        lp = torch.log_softmax(torch.tensor(logits), -1).numpy()[np.arange(S), ep.actions]
        probs = np.exp(lp[t])
        summary.append({"step": t, "phase": ph + 1, "cur": node_ids[env.cur], "chosen": node_ids[int(ep.actions[t])],
                        "n_valid": int(valid.size), "chosen_is_nearest_valid": int(ep.actions[t]) == nearest_valid,
                        "p_chosen": float(probs), "critic_value": float(values[t])})
        env.step(int(ep.actions[t]))

    t = a.step if a.step >= 0 else next((s["step"] for s in summary if not s["chosen_is_nearest_valid"]), 0)
    ph, recs = rows_by_step[t]
    valid = [r[0] for r in recs]
    lg = logits[t][valid]
    pr = np.exp(lg - lg.max()); pr /= pr.sum()
    order = np.argsort(-lg)
    chosen = int(ep.actions[t])
    show = list(order[: a.top])
    cpos = valid.index(chosen)
    if cpos not in show: show.append(cpos)

    print(f"\nDECISION step {t} (phase {ph+1}/{cfg.n_phases}): at {node_ids[int(ep.cur[t])]}, {len(valid)} valid candidates.")
    print(f"CHOSEN (greedy = max logit): node {chosen} / stop {node_ids[chosen]}")
    
    # --- CAPTUM INTEGRATED GRADIENTS ATTRIBUTION ---
    print("\n--- CAPTUM INTEGRATED GRADIENTS ATTRIBUTION ---")
    static = policy.statics([route]).to(dev)
    t_prog = torch.tensor(ep.prog).to(dev)
    t_cur = torch.tensor(ep.cur).to(dev)
    
    x = torch.cat([static[0, t_cur], t_prog], -1)
    h_all, _ = policy.lstm(x.unsqueeze(1))
    h_t = h_all[t].unsqueeze(0)
    
    mask_t = torch.tensor(ep.mask[t]).unsqueeze(0).to(dev)
    unv_t = torch.tensor(ep.unv[t]).unsqueeze(0).to(dev)
    dyn_t = torch.tensor(ep.dyn[t]).unsqueeze(0).to(dev).requires_grad_(True)
    static_t = static.clone().requires_grad_(True)
    
    ig_model = IGWrapper(policy, h_t, mask_t, unv_t, chosen).to(dev)
    ig = IntegratedGradients(ig_model)
    
    attr, delta = ig.attribute(
        inputs=(dyn_t, static_t),
        baselines=(torch.zeros_like(dyn_t), torch.zeros_like(static_t)),
        return_convergence_delta=True
    )
    
    dyn_attr = attr[0][0, chosen].detach().cpu().numpy()
    static_attr_sum = attr[1][0, chosen].detach().cpu().numpy().sum()
    
    print(f"Convergence Delta: {delta.item():.2e}")
    print("Baseline-relative attributions (numeric contribution to pre-softmax logit vs neutral 0-tensor baseline):")
    for name, val in zip(FEATS, dyn_attr):
        print(f"  {name:>18}: {val:>8.3f}")
    print(f"  {'static_embed_sum':>18}: {static_attr_sum:>8.3f}")

    out_rows = []
    for i in show:
        j, tt, c, pen = recs[i]
        d = ep.dyn[t, j]
        rank = int(np.where(order == i)[0][0]) + 1
        
        # Isolate attributions for the chosen node only
        attr_strs = [f"{val:.5f}" for val in dyn_attr] + [f"{static_attr_sum:.5f}"] if j == chosen else [""] * 6
        
        out_rows.append([rank, j, node_ids[j], f"{lg[i]:.5f}", f"{pr[i]:.5f}", *[f"{x:.5f}" for x in d],
                         f"{tt:.1f}", *[f"{x:.4f}" for x in c], f"{pen:.2f}", int(route.zone[j]), j == chosen, *attr_strs])
                         
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{run_name}_{a.split}_{rid}"
    
    with open(out / f"{stem}_step{t}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "node", "stop_id", "logit", "prob", *FEATS, "travel_s", *[f"cost_{n}" for n in COSTS],
                    "penalty_s", "zone", "chosen", *[f"attr_{f}" for f in FEATS], "attr_static_sum"])
        w.writerows(out_rows)
    print(f"\nwrote -> {out}/{stem}_step{t}.csv")

if __name__ == "__main__":
    main()
