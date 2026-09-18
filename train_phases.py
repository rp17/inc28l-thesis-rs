#!/usr/bin/env python3
"""Phase 1–3 trainers on the paper four-subnet PINN.

Phase 1: per-subnet pretrain on analytical targets (Adam).
Phase 2: log-Poisson + vacancy drift-diffusion (SOAP if available).
Phase 3: S04 I–V fit with PCGrad (SOAP if available).

Hysteresis is scored at the Si/STO junction (thesis LRS), not at x=0.
Poisson ρ includes holes. compute_current is the Memristor PINN I path (no φ→I).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    from soap import SOAP
    SOAP_AVAILABLE = True
except ImportError:
    SOAP_AVAILABLE = False

from pinn28l import Inc28lPINN, PhysicsParameters  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it


def analytical_potential(x, V, physics: PhysicsParameters):
    p = physics
    phi_0 = p.Phi_B_norm + V / p.U_t
    x_intf = p.x_interface
    is_STO = (x < x_intf).float()
    x_rel_sto = torch.clamp(x / x_intf, 0, 1)
    phi_sto = phi_0 * (1 - x_rel_sto)
    x_rel_si = torch.clamp((x - x_intf) / (1 - x_intf + 1e-6), 0, 1)
    V_effect = 0.1 * V / p.U_t
    phi_si = -p.delta_Ec_norm + V_effect + x_rel_si * (p.delta_Ec_norm - V_effect)
    return is_STO * phi_sto + (1 - is_STO) * phi_si


def analytical_vacancy(x, t, physics: PhysicsParameters):
    """Thesis LRS: vacancies pile up toward Si/STO (x → x_interface)."""
    x_intf = physics.x_interface
    is_STO = (x < x_intf).float()
    Cv = 1.0 + 0.2 * t * (x / x_intf) * is_STO
    return torch.clamp(Cv * is_STO + (1 - is_STO), 0.3, 3.0)


def analytical_carriers(phi, Cv, x, V, physics: PhysicsParameters):
    p = physics
    is_STO = (x < p.x_interface).float()
    phi_0 = p.Phi_B_norm + V / p.U_t
    phi_rel = torch.clamp(phi - phi_0, -40, 40)
    n_sto = Cv * torch.exp(-phi_rel)
    ni_ratio = p.n_i_Si / p.N_D_STO
    p_sto = ni_ratio ** 2 / torch.clamp(n_sto, min=1e-15)
    n_si = p.N_D_Si / p.N_D_STO
    p_si = ni_ratio ** 2 / n_si
    n = is_STO * n_sto + (1 - is_STO) * n_si
    p_out = is_STO * p_sto + (1 - is_STO) * p_si
    return torch.clamp(n, 1e-15), torch.clamp(p_out, 1e-20)


def analytical_current(V, physics: PhysicsParameters):
    n_ideality = 1.5
    V_phys = V * 5.0
    exp_arg = torch.clamp(V_phys / (n_ideality * physics.U_t), -20, 20)
    I = torch.exp(exp_arg) - 1.0
    I_max = np.exp(5.0 / (n_ideality * physics.U_t)) - 1.0
    return torch.clamp(I / I_max, -1, 1)


def _x_lrs(n, device, physics):
    return torch.full((n, 1), float(physics.x_interface), device=device)


def train_phase1_individual_pretrain(model: Inc28lPINN, device: str,
                                     physics: PhysicsParameters,
                                     epochs_per_subnet: int = 2000) -> Inc28lPINN:
    print("\n" + "=" * 60)
    print("PHASE 1: Individual subnet pretraining")
    print("=" * 60)
    n_points = 150

    print("\n  [1/4] Pretraining phi_net...")
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.phi_net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(model.phi_net.parameters(), lr=2e-3)
    for epoch in range(epochs_per_subnet):
        x = torch.rand(n_points, 1, device=device)
        t = torch.rand(n_points, 1, device=device)
        V = torch.rand(n_points, 1, device=device) * 2 - 1
        Cv = torch.ones(n_points, 1, device=device)
        opt.zero_grad()
        loss = torch.mean((model.phi_net(x, t, V, Cv) - analytical_potential(x, V, physics)) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.phi_net.parameters(), 1.0)
        opt.step()
        if epoch % 100 == 0 or epoch + 1 == epochs_per_subnet:
            print(f"    Epoch {epoch:4d}: loss = {loss.item():.4e}")

    print("\n  [2/4] Pretraining cv_net (two-stage)...")
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.cv_net.parameters():
        p.requires_grad_(True)
    epochs_a = epochs_per_subnet // 2
    epochs_b = epochs_per_subnet - epochs_a
    for p in model.cv_net.hyst_net.parameters():
        p.requires_grad_(False)
    model.cv_net.delta_Cv.requires_grad_(False)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.cv_net.parameters()), lr=2e-3)
    print("    Stage 2a: analytical fit (base_net)...")
    for epoch in range(epochs_a):
        x = torch.rand(n_points, 1, device=device) * physics.x_interface
        t = torch.rand(n_points, 1, device=device)
        V = torch.rand(n_points, 1, device=device) * 2 - 1
        opt.zero_grad()
        Cv = model.cv_net(x, t, V, torch.zeros(n_points, 1, device=device))
        loss = torch.mean((Cv - analytical_vacancy(x, t, physics)) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.cv_net.parameters(), 1.0)
        opt.step()
        if epoch % 100 == 0 or epoch + 1 == epochs_a:
            print(f"    Epoch {epoch:4d}: loss = {loss.item():.4e}")

    print("    Stage 2b: hysteresis at Si/STO (thesis LRS)...")
    for p in model.cv_net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam([
        {"params": model.cv_net.base_net.parameters(), "lr": 5e-4},
        {"params": model.cv_net.hyst_net.parameters(), "lr": 2e-3},
        {"params": [model.cv_net.delta_Cv], "lr": 1e-3},
    ])
    for epoch in range(epochs_b):
        x = torch.rand(n_points, 1, device=device) * physics.x_interface
        t = torch.rand(n_points, 1, device=device)
        V = torch.rand(n_points, 1, device=device) * 2 - 1
        opt.zero_grad()
        Cv_tr = model.cv_net(x, t, V, torch.zeros(n_points, 1, device=device))
        loss_base = 0.5 * torch.mean((Cv_tr - analytical_vacancy(x, t, physics)) ** 2)
        x_si = _x_lrs(n_points, device, physics)
        dlt = (model.cv_net(x_si, t, V, torch.zeros(n_points, 1, device=device))
               - model.cv_net(x_si, t, V, torch.ones(n_points, 1, device=device)))
        loss = loss_base + 2.0 * torch.mean(torch.relu(0.25 * t - dlt))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.cv_net.parameters(), 1.0)
        opt.step()
        if epoch % 100 == 0 or epoch + 1 == epochs_b:
            print(f"    Epoch {epoch:4d}: loss = {loss.item():.4e}, Δ_si = {dlt.mean().item():.4f}")

    print("\n  [3/4] Pretraining carrier_net...")
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.carrier_net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(model.carrier_net.parameters(), lr=2e-3)
    for epoch in range(epochs_per_subnet):
        x = torch.rand(n_points, 1, device=device)
        t = torch.rand(n_points, 1, device=device)
        V = torch.rand(n_points, 1, device=device) * 2 - 1
        with torch.no_grad():
            phi = model.phi_net(x, t, V, torch.ones_like(x))
            Cv = model.cv_net(x, t, V, torch.zeros_like(x))
        opt.zero_grad()
        n_p, p_p = model.carrier_net(x, t, V, phi, Cv)
        n_t, p_t = analytical_carriers(phi, Cv, x, V, physics)
        loss_n = torch.mean((torch.log(n_p + 1e-15) - torch.log(n_t + 1e-15)) ** 2)
        loss_p = torch.mean((torch.log(p_p + 1e-15) - torch.log(p_t + 1e-15)) ** 2)
        (loss_n + loss_p).backward()
        torch.nn.utils.clip_grad_norm_(model.carrier_net.parameters(), 1.0)
        opt.step()
        if epoch % 100 == 0 or epoch + 1 == epochs_per_subnet:
            print(f"    Epoch {epoch:4d}: loss_n = {loss_n.item():.4e}, loss_p = {loss_p.item():.4e}")

    print("\n  [4/4] Pretraining current_net...")
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.current_net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(model.current_net.parameters(), lr=2e-3)
    for epoch in tqdm(range(epochs_per_subnet), desc="  Phase 1.4 (current_net)"):
        t = torch.ones(n_points, 1, device=device)
        V = torch.linspace(-1, 1, n_points, device=device).unsqueeze(1)
        for bval in (0.0, 1.0):
            branch = torch.full((n_points, 1), bval, device=device)
            Cv_if = 1.5 - 0.8 * branch
            opt.zero_grad()
            I_base = model.current_net.net(torch.cat([t, V, branch], dim=-1))
            barrier = torch.clamp(
                torch.exp(torch.abs(model.current_net.barrier_strength) * (Cv_if - 1.0)),
                0.001, 1000.0)
            fwd = 1.0 + 2.0 * torch.relu(V)
            I_pred = I_base * barrier * fwd
            I_tgt = analytical_current(V, physics) * torch.exp(0.1 * (Cv_if - 1.0) / physics.U_t)
            I_base_tgt = I_tgt / (barrier * fwd + 1e-6)
            d2 = I_base[2:] - 2 * I_base[1:-1] + I_base[:-2]
            loss = (torch.mean((I_base - I_base_tgt) ** 2)
                    + 0.1 * torch.mean((I_pred - I_tgt) ** 2)
                    + 0.05 * torch.mean(d2 ** 2)
                    + 0.01 * torch.mean(torch.exp(-torch.abs(I_base) * 10.0)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.current_net.parameters(), 1.0)
            opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    print("\n  ✓ Phase 1 complete")
    return model


def train_phase2_pde_residual(model: Inc28lPINN, device: str,
                              physics: PhysicsParameters,
                              epochs: int = 5000) -> Tuple[Inc28lPINN, Dict]:
    print("\n" + "=" * 60)
    print("PHASE 2: PDE residual (log-Poisson + vacancy)")
    print(f"  λ² = {physics.lambda_squared:.6f}")
    print("=" * 60)
    model.cv_net.delta_Cv.requires_grad_(False)
    for p in model.cv_net.hyst_net.parameters():
        p.requires_grad_(False)
    for p in model.phi_net.parameters():
        p.requires_grad_(False)
    if SOAP_AVAILABLE:
        print("  SOAP optimizer")
        opt = SOAP(filter(lambda p: p.requires_grad, model.parameters()),
                   lr=3e-3, weight_decay=1e-4, precondition_frequency=10)
    else:
        print("  Adam (SOAP not imported)")
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = {k: [] for k in ("poisson", "vacancy", "ic", "hysteresis", "total", "delta_si")}
    n_col, n_ic, n_h = 100, 50, 50
    lam2 = physics.lambda_squared
    for epoch in tqdm(range(epochs), desc="Phase 2"):
        opt.zero_grad()
        x = torch.rand(n_col, 1, device=device, requires_grad=True) * physics.x_interface
        t = torch.rand(n_col, 1, device=device, requires_grad=True)
        V = torch.rand(n_col, 1, device=device) * 2 - 1
        b = torch.zeros(n_col, 1, device=device)
        phi, n_e, p_h, Cv = model(x, t, V, b)
        dphi = torch.autograd.grad(phi.sum(), x, create_graph=True, retain_graph=True)[0]
        d2phi = torch.autograd.grad(dphi.sum(), x, create_graph=True, retain_graph=True)[0]
        rho = physics.Z_v * Cv - n_e + p_h + 1e-10
        res_p = (torch.log10(torch.abs(lam2 * d2phi) + 1e-10)
                 - torch.log10(torch.abs(rho) + 1e-10))
        loss_p = torch.mean(res_p ** 2)
        dCv_dt = torch.autograd.grad(Cv.sum(), t, create_graph=True, retain_graph=True)[0]
        dCv_dx = torch.autograd.grad(Cv.sum(), x, create_graph=True, retain_graph=True)[0]
        d2Cv = torch.autograd.grad(dCv_dx.sum(), x, create_graph=True, retain_graph=True)[0]
        res_v = dCv_dt - physics.D_v_norm * d2Cv + physics.Z_v * physics.mu_v_norm * (
            dCv_dx * dphi + Cv * d2phi)
        loss_v = torch.mean(torch.log10(torch.abs(res_v) + 1e-10) ** 2)
        x_ic = torch.rand(n_ic, 1, device=device) * physics.x_interface
        Cv_ic = model.cv_net(x_ic, torch.zeros(n_ic, 1, device=device),
                             torch.rand(n_ic, 1, device=device) * 2 - 1,
                             torch.zeros(n_ic, 1, device=device))
        loss_ic = torch.mean((Cv_ic - 1.0) ** 2)
        t_h = torch.ones(n_h, 1, device=device)
        V_h = torch.rand(n_h, 1, device=device) * 2 - 1
        x_si = _x_lrs(n_h, device, physics)
        dlt = (model.cv_net(x_si, t_h, V_h, torch.zeros(n_h, 1, device=device))
               - model.cv_net(x_si, t_h, V_h, torch.ones(n_h, 1, device=device)))
        loss_h = torch.mean(torch.relu(0.25 - dlt))
        x0 = torch.zeros(50, 1, device=device)
        t_bc = torch.rand(50, 1, device=device)
        V_bc = torch.rand(50, 1, device=device) * 2 - 1
        b0 = torch.zeros(50, 1, device=device)
        loss_bc0 = torch.mean((model.phi_net(x0, t_bc, V_bc, model.cv_net(x0, t_bc, V_bc, b0))
                               - (physics.Phi_B_norm + V_bc / physics.U_t)) ** 2)
        xi = _x_lrs(50, device, physics)
        loss_bci = torch.mean(model.phi_net(xi, t_bc, V_bc, model.cv_net(xi, t_bc, V_bc, b0)) ** 2)
        loss_reg = 0.1 * torch.mean((phi - analytical_potential(x, V, physics)) ** 2)
        loss = (loss_p + 0.5 * loss_v + 10.0 * loss_ic + 3.0 * loss_h
                + 5.0 * loss_bc0 + 5.0 * loss_bci + loss_reg)
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        hist["poisson"].append(float(loss_p.detach()))
        hist["vacancy"].append(float(loss_v.detach()))
        hist["ic"].append(float(loss_ic.detach()))
        hist["hysteresis"].append(float(loss_h.detach()))
        hist["total"].append(float(loss.detach()))
        hist["delta_si"].append(float(dlt.mean().detach()))
    model.cv_net.delta_Cv.requires_grad_(True)
    for p in model.cv_net.hyst_net.parameters():
        p.requires_grad_(True)
    for p in model.phi_net.parameters():
        p.requires_grad_(True)
    print(f"  Phase 2 done  Poisson={hist['poisson'][-1]:.3e}  Δ_si={hist['delta_si'][-1]:.3f}")
    return model, hist


def pcgrad_project(g1: List[torch.Tensor], g2: List[torch.Tensor]) -> List[torch.Tensor]:
    f1 = torch.cat([g.flatten() for g in g1])
    f2 = torch.cat([g.flatten() for g in g2])
    dot = torch.dot(f1, f2)
    if dot >= 0:
        return g2
    coeff = dot / (torch.dot(f1, f1) + 1e-12)
    f2p = f2 - coeff * f1
    out, off = [], 0
    for g in g2:
        n = g.numel()
        out.append(f2p[off:off + n].view_as(g))
        off += n
    return out


def train_phase3_datafit(model: Inc28lPINN, data: Dict, device: str,
                         physics: PhysicsParameters,
                         epochs: int = 4000) -> Tuple[Inc28lPINN, Dict]:
    print("\n" + "=" * 60)
    print("PHASE 3: S04 I–V fit (SOAP + PCGrad)")
    print("=" * 60)
    if SOAP_AVAILABLE:
        opt = SOAP(model.parameters(), lr=1e-3, weight_decay=1e-4, precondition_frequency=10)
        print("  SOAP optimizer")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        print("  Adam (SOAP not imported)")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    V = data["V"]
    I_tr, I_re = data["I_trace"], data["I_retrace"]
    n = int(data["n_points"])
    hist = {"trace_r2": [], "retrace_r2": [], "delta_si": []}
    for epoch in tqdm(range(epochs), desc="Phase 3"):
        opt.zero_grad()
        t = torch.ones(n, 1, device=device)
        I_tp = model.compute_current(t, V, torch.zeros(n, 1, device=device))
        I_rp = model.compute_current(t, V, torch.ones(n, 1, device=device))
        if torch.isnan(I_tp).any() or torch.isnan(I_rp).any():
            continue
        Vabs = torch.abs(V)
        w = 1.0 + 3.0 * torch.exp(-((Vabs - 0.5) / 0.15) ** 2) + 2.0 * torch.exp(-Vabs / 0.2)
        w = w / w.mean()
        loss_tr = torch.mean(w * (I_tp - I_tr) ** 2)
        loss_re = torch.mean(w * (I_rp - I_re) ** 2)
        if n > 2:
            _, idx = torch.sort(V.squeeze())
            d2t = I_tp.squeeze()[idx][2:] - 2 * I_tp.squeeze()[idx][1:-1] + I_tp.squeeze()[idx][:-2]
            d2r = I_rp.squeeze()[idx][2:] - 2 * I_rp.squeeze()[idx][1:-1] + I_rp.squeeze()[idx][:-2]
            loss_sm = 5.0 * (torch.mean(d2t ** 2) + torch.mean(d2r ** 2))
        else:
            loss_sm = V.new_zeros(())
        x_si = _x_lrs(50, device, physics)
        t_h = torch.ones(50, 1, device=device)
        V_h = torch.rand(50, 1, device=device) * 2 - 1
        dlt = (model.cv_net(x_si, t_h, V_h, torch.zeros(50, 1, device=device))
               - model.cv_net(x_si, t_h, V_h, torch.ones(50, 1, device=device)))
        loss_h = 0.5 * torch.mean(torch.relu(0.1 - dlt))
        params = [p for p in model.parameters() if p.requires_grad]
        g_tr = torch.autograd.grad(loss_tr, params, retain_graph=True, allow_unused=True)
        g_re = torch.autograd.grad(loss_re, params, retain_graph=True, allow_unused=True)
        g_tr = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_tr, params)]
        g_re = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_re, params)]
        g_re = pcgrad_project(g_tr, g_re)
        g_sh = torch.autograd.grad(loss_sm + loss_h, params, allow_unused=True)
        g_sh = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_sh, params)]
        for p, a, b, c in zip(params, g_tr, g_re, g_sh):
            p.grad = a + b + c
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            r2t = 1 - torch.sum((I_tp - I_tr) ** 2) / (torch.sum((I_tr - I_tr.mean()) ** 2) + 1e-10)
            r2r = 1 - torch.sum((I_rp - I_re) ** 2) / (torch.sum((I_re - I_re.mean()) ** 2) + 1e-10)
        hist["trace_r2"].append(float(r2t))
        hist["retrace_r2"].append(float(r2r))
        hist["delta_si"].append(float(dlt.mean().detach()))
        if epoch % 200 == 0:
            print(f"    Epoch {epoch}: R² {float(r2t):.4f}/{float(r2r):.4f}  Δ_si={hist['delta_si'][-1]:.3f}")
    print(f"  Phase 3 done  R² {hist['trace_r2'][-1]:.4f}/{hist['retrace_r2'][-1]:.4f}")
    return model, hist
