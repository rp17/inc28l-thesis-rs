#!/usr/bin/env python3
"""
INCREMENT 28l: CurrentNet + contact-routed Cv + VOFF shift + physical R

Vacancy polarity according to the thesis by Patrick Kollias:
    Patrick Kollias, Resistive Switching in Epitaxial SrTiO3 on Silicon,
    Ph.D. thesis, Texas State University, 2022.

PINN paper:
    Podorozhny, Theodoropoulou, Tesic,
    arXiv:2609.02966  https://arxiv.org/abs/2609.02966

TARGET VERIFICATION: Match S04 experimental Log I(V)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reference ASCII (S04N First Dataset):
                  S04N First Dataset - Log I(V) (Logarithmic Scale)
     ┌─────────────────────────────────────────────────────────────────────────┐
 1.39┤ ++ Trace (+)   **********++++++                  +++++++****************│
     │ ** Retrace (*)          **    ++                ++   ****               │
     │                          *     +               +    **                  │
 0.68┤                           *    ++             ++    *                   │
     │                           *     +            ++    *                    │
     │                            *    +           ++    **                    │
-0.03┤                            *     +          +     *                     │
     │                            **    +         ++    **                     │
-0.74┤                             *    +        ++     *                      │
     │                             **    +      ++     **                      │
     │                              *    ++++++++      *                       │
-1.45┤                              **   ++            *                       │
     │                               **************   **                       │
     │                                   +        *****                        │
-2.16┤                                   +           **                        │
     │                                                *                        │
-2.87┤                                                *                        │
     └┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬┘
    -5.0              -2.5               0.0               2.5              5.0

Key features to verify:
    1. ONE downward spike (minimum) near V ≈ 0
    2. Flat, high current (~10^+1 A) at negative V
    3. Trace (+) rises FIRST at lower positive V (~0.5)
    4. Retrace (*) rises LATER at higher positive V (~1.5)
    5. Hysteresis gap visible between V = 0 and V = 2.5
    6. ~4 orders of magnitude dynamic range
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
WEIGHTS = HERE / "weights" / "checkpoint_phase3.pt"

I_MAX_A = 2e-9
V_MAX = 5.0
ROUTING_K = 20.0
X_INTERFACE = 0.5
HIDDEN = 192
ALPHA_SI = 0.51392275
ALPHA_PT = 0.0289
GAMMA = 0.20
R_OHM = 100.0
DELTA_EC = 0.35

SWEEPS = (
    ("first", "S04N dif I sens 160 first 2nA linear.dat"),
    ("second", "S04N dif I sens 160 second 2nA linear.dat"),
)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, activation: nn.Module):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            activation,
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = activation

    def forward(self, x):
        return self.act(x + self.net(x))


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 6) -> nn.Module:
    act = nn.GELU()
    layers: list = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), act]
    for _ in range(n_layers - 2):
        layers.append(ResidualBlock(hidden_dim, nn.GELU()))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class CurrentNet(nn.Module):
    """6-layer residual GELU net(t, V, b). Extra scalars match the Phase-3 checkpoint."""

    def __init__(self, hidden_dim: int = HIDDEN):
        super().__init__()
        self.net = build_mlp(3, hidden_dim, 1, n_layers=6)
        self.R_series = nn.Parameter(torch.tensor(100.0))
        self.n_trace = nn.Parameter(torch.tensor(1.5))
        self.n_retrace = nn.Parameter(torch.tensor(1.8))
        self.barrier_strength = nn.Parameter(torch.tensor(3.0))


class VacancyNet(nn.Module):
    """Thesis polarity: LRS / trace peaks at Si/STO; HRS / retrace at Pt/STO."""

    def __init__(self, hidden_dim: int = HIDDEN, x_interface: float = X_INTERFACE):
        super().__init__()
        self.x_interface = float(x_interface)
        self.base_net = build_mlp(2, hidden_dim, 1, n_layers=4)
        self.hyst_net = build_mlp(3, hidden_dim, 1, n_layers=4)
        self.delta_Cv = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, t, V, branch):
        base_mod = torch.sigmoid(self.base_net(torch.cat([t, V], dim=-1))) * 0.2
        hyst = torch.tanh(self.hyst_net(torch.cat([x, V, branch], dim=-1))) * 0.15
        delta = torch.abs(self.delta_Cv) * t
        x_sto = torch.clamp(x / self.x_interface, 0.0, 1.0)
        trace = torch.exp(-5.0 * (1.0 - x_sto))
        retrace = torch.exp(-5.0 * x_sto)
        Cv = (1.0 - branch) * (1.0 + delta * trace + base_mod + hyst)
        Cv = Cv + branch * (1.0 + delta * retrace + base_mod + hyst)
        return torch.clamp(Cv, 0.3, 5.0)


@dataclass
class Inc28lConfig:
    alpha_si: float = ALPHA_SI
    alpha_pt: float = ALPHA_PT
    gamma: float = GAMMA
    R_ohm: float = R_OHM
    n_tr: float = 1.0
    n_re: float = 1.0
    kappa: float = 0.0
    hidden_dim: int = HIDDEN


class Inc28lPINN(nn.Module):
    def __init__(self, cfg: Optional[Inc28lConfig] = None):
        super().__init__()
        self.cfg = cfg or Inc28lConfig()
        self.cv_net = VacancyNet(self.cfg.hidden_dim)
        self.current_net = CurrentNet(self.cfg.hidden_dim)

    def compute_current(self, t, V, branch):
        cfg = self.cfg
        x_pt = torch.zeros_like(V)
        x_si = torch.full_like(V, X_INTERFACE)
        Cv_pt = self.cv_net(x_pt, t, V, branch)
        Cv_si = self.cv_net(x_si, t, V, branch)
        w = torch.sigmoid(ROUTING_K * V)
        gate = torch.exp(cfg.alpha_si * w * (Cv_si - 1.0)
                         + cfg.alpha_pt * (1.0 - w) * (Cv_pt - 1.0))
        gate = torch.clamp(gate, 0.001, 1000.0)
        n_eff = torch.clamp((1.0 - branch) * cfg.n_tr + branch * cfg.n_re, min=0.5)
        V_net = V / n_eff + (cfg.gamma / V_MAX) * w + cfg.kappa * (DELTA_EC / V_MAX) * w
        I_base = self.current_net.net(torch.cat([t, V_net, branch], dim=-1))
        fwd = 1.0 + 2.0 * torch.relu(V)
        I = I_base * gate * fwd
        if cfg.R_ohm > 0.0:
            drop = (cfg.R_ohm * I * I_MAX_A) / V_MAX
            V_net2 = (V - drop) / n_eff + (cfg.gamma / V_MAX) * w + cfg.kappa * (DELTA_EC / V_MAX) * w
            I = self.current_net.net(torch.cat([t, V_net2, branch], dim=-1)) * gate * fwd
        return I


def load_model(device: Optional[str] = None, ckpt: Optional[Path] = None) -> Inc28lPINN:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(ckpt) if ckpt is not None else WEIGHTS
    model = Inc28lPINN().to(device)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    cv = {k[len("cv_net."):]: v for k, v in state.items() if k.startswith("cv_net.")}
    cn = {k[len("current_net."):]: v for k, v in state.items() if k.startswith("current_net.")}
    model.cv_net.load_state_dict(cv)
    model.current_net.load_state_dict(cn)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_s04(name: str, device: str) -> Dict:
    aliases = {tag: fname for tag, fname in SWEEPS}
    fname = aliases.get(name, name)
    path = Path(fname)
    if not path.is_file():
        path = DATA_DIR / Path(fname).name
    if not path.is_file():
        raise FileNotFoundError(f"S04 sweep not found: {name}")
    data = np.loadtxt(path, delimiter="\t", skiprows=1)
    n_cols = data.shape[1]
    if n_cols > 160:
        split = n_cols // 2
        V_tr, I_tr = data[:, 0], data[:, 1:split].mean(axis=1)
        V_re, I_re = data[:, split], data[:, split + 1:].mean(axis=1)
        order_t = np.argsort(V_tr)
        V = V_tr[order_t]
        I_trace = I_tr[order_t]
        order_r = np.argsort(V_re)
        I_retrace = np.interp(V, V_re[order_r], I_re[order_r])
    else:
        V = data[:, 0]
        I_trace = data[:, 1]
        I_retrace = data[:, 2] if n_cols >= 3 else I_trace
    V_max = float(np.max(np.abs(V)))
    I_max = float(np.max(np.abs(np.concatenate([I_trace, I_retrace]))))
    return {
        "V": torch.tensor(V / V_max, dtype=torch.float32, device=device).unsqueeze(-1),
        "I_trace": torch.tensor(I_trace / I_max, dtype=torch.float32, device=device).unsqueeze(-1),
        "I_retrace": torch.tensor(I_retrace / I_max, dtype=torch.float32, device=device).unsqueeze(-1),
        "V_phys": V.astype(np.float64),
        "V_max": V_max,
        "I_max": I_max,
    }


def predict_iv(model: Inc28lPINN, data: Dict) -> Dict:
    V = data["V"]
    n = V.shape[0]
    t = torch.ones(n, 1, device=V.device)
    with torch.no_grad():
        I_t = model.compute_current(t, V, torch.zeros(n, 1, device=V.device))
        I_r = model.compute_current(t, V, torch.ones(n, 1, device=V.device))
    I_t_n = I_t.cpu().numpy().flatten()
    I_r_n = I_r.cpu().numpy().flatten()
    I_t_d = data["I_trace"].cpu().numpy().flatten()
    I_r_d = data["I_retrace"].cpu().numpy().flatten()
    V_p = np.asarray(data["V_phys"]).flatten()
    order = np.argsort(V_p)
    V_p, I_t_n, I_r_n, I_t_d, I_r_d = (a[order] for a in (V_p, I_t_n, I_r_n, I_t_d, I_r_d))

    def r2(pred, tgt):
        ss_res = float(np.sum((pred - tgt) ** 2))
        ss_tot = float(np.sum((tgt - tgt.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    pos, neg = V_p > 0.0, V_p < 0.0
    return {
        "V_phys": V_p,
        "I_trace_norm": I_t_n,
        "I_retrace_norm": I_r_n,
        "I_trace_data_norm": I_t_d,
        "I_retrace_data_norm": I_r_d,
        "r2_trace": r2(I_t_n, I_t_d),
        "r2_retrace": r2(I_r_n, I_r_d),
        "r2_posV_trace": r2(I_t_n[pos], I_t_d[pos]),
        "r2_posV_retrace": r2(I_r_n[pos], I_r_d[pos]),
        "r2_negV_trace": r2(I_t_n[neg], I_t_d[neg]),
        "r2_negV_retrace": r2(I_r_n[neg], I_r_d[neg]),
    }
