#!/usr/bin/env python3
"""
Memristor PINN: four-subnet CurrentNet + contact-routed Cv + VOFF shift + physical R

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
from typing import Dict, Optional, Tuple

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


def get_activation(name: str) -> nn.Module:
    return {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "mish": nn.Mish(),
    }.get(name, nn.GELU())


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int,
              n_layers: int = 6, residual: bool = True,
              activation: str = "gelu", use_residual: Optional[bool] = None) -> nn.Module:
    """Residual+LayerNorm+GELU (paper MLPs) or a plain GELU stack."""
    if use_residual is not None:
        residual = use_residual
    act_fn = get_activation(activation)
    if residual and n_layers >= 3:
        layers: list = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), act_fn]
        for _ in range(n_layers - 2):
            layers.append(ResidualBlock(hidden_dim, get_activation(activation)))
        layers.append(nn.Linear(hidden_dim, output_dim))
        net = nn.Sequential(*layers)
    else:
        layers = [nn.Linear(input_dim, hidden_dim), act_fn]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), get_activation(activation)])
        layers.append(nn.Linear(hidden_dim, output_dim))
        net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return net


@dataclass
class PhysicsParameters:
    """Material constants (paper Table I / Kollias thesis). ΔE_c = 0.35 eV."""
    L_STO: float = 20e-9
    L_Si_substrate: float = 380e-6
    x_interface: float = X_INTERFACE
    E_g_STO: float = 3.3
    E_g_Si: float = 1.12
    chi_STO: float = 4.4
    chi_Si: float = 4.05
    delta_Ec: float = DELTA_EC
    Phi_Pt: float = 5.7
    Phi_B_Pt_STO: float = 1.3
    N_D_STO: float = 1e18
    N_D_Si: float = 5e15
    n_i_Si: float = 1e10
    n_i_STO: float = 1e8
    eps_STO: float = 30.0
    eps_Si: float = 11.7
    eps_0: float = 8.854e-14
    mu_n_STO: float = 6.0
    mu_n_Si: float = 1400.0
    D_v_norm: float = 0.01
    Z_v: float = 2.0
    mu_v_norm: float = 0.05
    T: float = 300.0
    k_B: float = 8.617e-5
    q: float = 1.602e-19
    R_series_init: float = 100.0

    @property
    def U_t(self) -> float:
        return self.k_B * self.T

    @property
    def delta_Ec_norm(self) -> float:
        return self.delta_Ec / self.U_t

    @property
    def Phi_B_norm(self) -> float:
        return self.Phi_B_Pt_STO / self.U_t

    @property
    def Debye_STO(self) -> float:
        return np.sqrt(self.eps_0 * self.eps_STO * self.U_t / (self.q * self.N_D_STO))

    @property
    def Debye_Si(self) -> float:
        return np.sqrt(self.eps_0 * self.eps_Si * self.U_t / (self.q * self.N_D_Si))

    @property
    def lambda_squared(self) -> float:
        return (self.Debye_STO * 1e7 / (self.L_STO * 1e9)) ** 2

    @property
    def mu_p_STO(self) -> float:
        return 0.5

    @property
    def D_n_STO(self) -> float:
        return self.mu_n_STO * self.U_t

    @property
    def D_p_STO(self) -> float:
        return self.mu_p_STO * self.U_t

    @property
    def R_Si_substrate(self) -> float:
        N_D_Si_m3 = self.N_D_Si * 1e6
        mu_n_Si_m2 = self.mu_n_Si * 1e-4
        rho_Si = 1.0 / (self.q * mu_n_Si_m2 * N_D_Si_m3)
        R_target = 500.0
        A_estimate = rho_Si * self.L_Si_substrate / R_target
        return rho_Si * self.L_Si_substrate / A_estimate


def make_physics() -> PhysicsParameters:
    return PhysicsParameters(delta_Ec=DELTA_EC)


class ContinuousVacancyNet(nn.Module):
    """Oxygen-vacancy field C_v(x). Thesis polarity: LRS / trace at Si/STO, HRS / retrace at Pt/STO."""

    def __init__(self, physics: Optional[PhysicsParameters] = None, hidden_dim: int = HIDDEN):
        super().__init__()
        self.p = physics or make_physics()
        self.x_interface = float(self.p.x_interface)
        self.base_net = build_mlp(2, hidden_dim, 1, n_layers=4, residual=True)
        self.hyst_net = build_mlp(3, hidden_dim, 1, n_layers=4, residual=True)
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

    def get_interface_Cv(self, t, V, branch, x=None):
        if x is None:
            x = torch.zeros_like(t)
        return self.forward(x, t, V, branch)


class TwoRegionPotentialNet(nn.Module):
    """Electrostatic potential φ(x): STO residual 5-layer 192, Si plain 4-layer 96."""

    def __init__(self, physics: Optional[PhysicsParameters] = None, hidden_dim: int = HIDDEN):
        super().__init__()
        self.p = physics or make_physics()
        self.sto_net = build_mlp(4, hidden_dim, 1, n_layers=5, residual=True)
        self.si_net = build_mlp(4, hidden_dim // 2, 1, n_layers=4, residual=False)
        self.interface_sharpness = nn.Parameter(torch.tensor(200.0))

    def forward(self, x, t, V, Cv):
        p = self.p
        phi_0 = p.Phi_B_norm + V / p.U_t
        phi_intf_sto = 0.0
        phi_intf_si = -p.delta_Ec_norm
        x_intf = p.x_interface
        is_Si = torch.sigmoid(self.interface_sharpness * (x - x_intf))
        is_STO = 1.0 - is_Si
        inputs = torch.cat([x, t, V, Cv], dim=-1)
        x_rel_sto = torch.clamp(x / x_intf, 0, 1)
        phi_sto = phi_0 * (1 - x_rel_sto) + phi_intf_sto * x_rel_sto
        phi_sto = phi_sto + x_rel_sto * (1 - x_rel_sto) * self.sto_net(inputs)
        phi_sto = phi_sto - 0.5 * (Cv - 1.0)
        x_rel_si = torch.clamp((x - x_intf) / (1 - x_intf), 0, 1)
        V_effect = 0.1 * V / p.U_t
        phi_si = phi_intf_si + V_effect + x_rel_si * (-phi_intf_si - V_effect)
        phi_si = phi_si + x_rel_si * (1 - x_rel_si) * self.si_net(inputs) * 0.1
        return is_STO * phi_sto + is_Si * phi_si


class DDNetStyleCarrierNet(nn.Module):
    """Log-space n, p. φ from TwoRegionPotentialNet is already in U_t units (no second /U_t)."""

    def __init__(self, physics: Optional[PhysicsParameters] = None, hidden_dim: int = HIDDEN):
        super().__init__()
        self.p = physics or make_physics()
        self.log_n_net = build_mlp(5, hidden_dim, 1, n_layers=4, residual=False)
        self.log_p_net = build_mlp(5, hidden_dim, 1, n_layers=4, residual=False)
        self.log_n_scale = nn.Parameter(torch.tensor(3.0))
        self.log_p_scale = nn.Parameter(torch.tensor(3.0))

    def forward(self, x, t, V, phi, Cv) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.p
        is_Si = (x >= p.x_interface).float()
        is_STO = 1.0 - is_Si
        inp = torch.cat([x, t, V, phi, Cv], dim=-1)
        log_n_c = self.log_n_scale * torch.tanh(self.log_n_net(inp))
        log_p_c = self.log_p_scale * torch.tanh(self.log_p_net(inp))
        phi_0 = p.Phi_B_norm + V / p.U_t
        phi_rel = torch.clamp(phi - phi_0, -40, 40)
        log_n_eq_sto = torch.log(Cv + 1e-15) - phi_rel
        ni_ratio = torch.tensor(p.n_i_Si / p.N_D_STO, device=x.device, dtype=x.dtype)
        log_p_eq_sto = torch.log(ni_ratio ** 2 + 1e-15) - log_n_eq_sto
        n_si_ratio = torch.tensor(p.N_D_Si / p.N_D_STO, device=x.device, dtype=x.dtype)
        log_n_eq_si = torch.full_like(x, torch.log(n_si_ratio + 1e-15))
        log_p_eq_si = torch.full_like(x, torch.log(ni_ratio ** 2 / n_si_ratio + 1e-15))
        log_n_eq = is_STO * log_n_eq_sto + is_Si * log_n_eq_si
        log_p_eq = is_STO * log_p_eq_sto + is_Si * log_p_eq_si
        return (torch.exp((log_n_eq + log_n_c).clamp(-30, 30)),
                torch.exp((log_p_eq + log_p_c).clamp(-30, 30)))


class CurrentNet(nn.Module):
    """6-layer residual GELU net(t, V, b) plus Schottky scalars from the Phase-3 checkpoint."""

    def __init__(self, physics: Optional[PhysicsParameters] = None, hidden_dim: int = HIDDEN):
        super().__init__()
        self.p = physics or make_physics()
        self.net = build_mlp(3, hidden_dim, 1, n_layers=6, residual=True)
        self.R_series = nn.Parameter(torch.tensor(float(self.p.R_series_init)))
        self.n_trace = nn.Parameter(torch.tensor(1.5))
        self.n_retrace = nn.Parameter(torch.tensor(1.8))
        self.barrier_strength = nn.Parameter(torch.tensor(3.0))

    def forward(self, t, V, branch, Cv_interface):
        I_base = self.net(torch.cat([t, V, branch], dim=-1))
        barrier = torch.exp(torch.abs(self.barrier_strength) * (Cv_interface - 1.0))
        barrier = torch.clamp(barrier, 0.001, 1000.0)
        return I_base * barrier * (1.0 + 2.0 * torch.relu(V))


VacancyNet = ContinuousVacancyNet
PotentialNet = TwoRegionPotentialNet
CarrierNet = DDNetStyleCarrierNet


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
    """Four-subnet paper PINN. Live I uses contact-routed C_v; φ is not an input to I."""

    def __init__(self, cfg: Optional[Inc28lConfig] = None,
                 physics: Optional[PhysicsParameters] = None):
        super().__init__()
        self.cfg = cfg or Inc28lConfig()
        self.physics = physics or make_physics()
        h = self.cfg.hidden_dim
        self.cv_net = ContinuousVacancyNet(self.physics, h)
        self.phi_net = TwoRegionPotentialNet(self.physics, h)
        self.carrier_net = DDNetStyleCarrierNet(self.physics, h)
        self.current_net = CurrentNet(self.physics, h)

    def forward(self, x, t, V, branch):
        Cv = self.cv_net(x, t, V, branch)
        phi = self.phi_net(x, t, V, Cv)
        n, p = self.carrier_net(x, t, V, phi, Cv)
        return phi, n, p, Cv

    def freeze_hysteresis(self):
        self.cv_net.delta_Cv.requires_grad_(False)

    def unfreeze_hysteresis(self):
        self.cv_net.delta_Cv.requires_grad_(True)

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
    for name, net in (("cv_net", model.cv_net), ("phi_net", model.phi_net),
                      ("carrier_net", model.carrier_net), ("current_net", model.current_net)):
        sub = {k[len(name) + 1:]: v for k, v in state.items() if k.startswith(name + ".")}
        if sub:
            net.load_state_dict(sub)
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
        "n_points": int(V.size),
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
