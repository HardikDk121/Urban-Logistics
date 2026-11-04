#!/usr/bin/env python3
"""
explain_time.py - Quantifies time-of-day impact via controlled ablation.
Intercepts the environment's progress tensor to zero out sin/cos time inputs 
and compares the resulting route sequence against the nominal timeline.
"""
import argparse, csv, os, sys
from pathlib import Path
import torch

sys.path.insert(0, os.environ.get("LASTMILE_DIR", "/content"))
import lastmile_rl as L

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-routes", type=int, default=50)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    dev = torch.device("cpu")
    L.set_seed(0)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg = L.EnvConfig(**ck["env_cfg"])
    policy = L.Policy().to(dev).eval()
    policy.load_state_dict(ck["policy"])
    run_name = f"{ck['args'].get('name', 'run')}_s{ck['args'].get('seed', '?')}"
    
    store = L.RouteStore(a.cache_dir, cfg, dev)
    ids = store.split(a.split)[:a.max_routes]
    
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    original_progress = L.RouteEnv.progress
    
    for rid in ids:
        route = store.get(rid)
        
        # 1. Nominal Time Rollout
        ep_orig = L.run_episode(route, cfg, "greedy", policy)
        seq_orig = ep_orig.sequence
        time_orig = ep_orig.total_time
        
        # 2. Ablated Time Rollout (Neutral Reference)
        def ablated_progress(self):
            prog = original_progress(self)
            prog[2] = 0.0 # sin=0
            prog[3] = 0.0 # cos=0
            return prog
        
        L.RouteEnv.progress = ablated_progress
        ep_ablated = L.run_episode(route, cfg, "greedy", policy)
        seq_ablated = ep_ablated.sequence
        time_ablated = ep_ablated.total_time
        
        L.RouteEnv.progress = original_progress
        
        results.append({
            "route_id": rid,
            "original_time": time_orig,
            "ablated_time": time_ablated,
            "is_identical": (seq_orig == seq_ablated),
            "time_diff": time_ablated - time_orig
        })
        
    csv_path = out_dir / f"time_ablation_{run_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["route_id", "original_time", "ablated_time", "is_identical", "time_diff"])
        writer.writeheader()
        writer.writerows(results)
        
    changed = sum(1 for r in results if not r["is_identical"])
    print(f"Time-of-Day Ablation Results ({run_name}):")
    print(f"Routes evaluated: {len(ids)} | Sequences altered by zeroing sin/cos: {changed}")
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()
