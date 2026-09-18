#!/usr/bin/env python3
"""Run Phase 1–3 on the Memristor PINN."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pinn28l import HERE, Inc28lPINN, load_s04, make_physics
from train_phases import (
    train_phase1_individual_pretrain,
    train_phase2_pde_residual,
    train_phase3_datafit,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Train the Memristor PINN (paper Phase 1–3).")
    p.add_argument("--phases", default="1,2,3", help="Comma list, e.g. 1 or 1,2,3")
    p.add_argument("--epochs-p1", type=int, default=2000)
    p.add_argument("--epochs-p2", type=int, default=5000)
    p.add_argument("--epochs-p3", type=int, default=4000)
    p.add_argument("--sweep", choices=("first", "second"), default="first")
    p.add_argument("--init-checkpoint", type=Path, default=None,
                   help="Warm-start from a Phase-3 .pt (default: random init)")
    p.add_argument("--out", type=Path, default=HERE / "output" / "checkpoint_trained.pt")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    physics = make_physics()
    if args.init_checkpoint:
        from pinn28l import load_model
        model = load_model(device, args.init_checkpoint)
        for q in model.parameters():
            q.requires_grad_(True)
    else:
        model = Inc28lPINN(physics=physics).to(device)
    phases = {int(s.strip()) for s in args.phases.split(",") if s.strip()}
    if 1 in phases:
        train_phase1_individual_pretrain(model, device, physics, args.epochs_p1)
    if 2 in phases:
        train_phase2_pde_residual(model, device, physics, args.epochs_p2)
    if 3 in phases:
        data = load_s04(args.sweep, device)
        train_phase3_datafit(model, data, device, physics, args.epochs_p3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "hidden_dim": model.cfg.hidden_dim,
                "phases": sorted(phases)}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
