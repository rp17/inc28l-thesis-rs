#!/usr/bin/env python3
"""Score the Memristor PINN on each S04 2 nA sweep separately (no combined file)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pinn28l import I_MAX_A, SWEEPS, load_model, load_s04, predict_iv

HERE = Path(__file__).resolve().parent
EPS = 1e-15


def plot_one(iv, title: str, path: Path) -> None:
    V = iv["V_phys"]
    nA = I_MAX_A * 1e9
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    ax.plot(V, np.log10(np.abs(iv["I_trace_data_norm"]) * I_MAX_A + EPS),
            color="0.15", lw=1.7, label="data trace")
    ax.plot(V, np.log10(np.abs(iv["I_retrace_data_norm"]) * I_MAX_A + EPS),
            color="crimson", lw=1.7, label="data retrace")
    ax.plot(V, np.log10(np.abs(iv["I_trace_norm"]) * I_MAX_A + EPS),
            color="seagreen", ls="--", lw=1.4, label="model trace")
    ax.plot(V, np.log10(np.abs(iv["I_retrace_norm"]) * I_MAX_A + EPS),
            color="seagreen", ls=":", lw=1.4, label="model retrace")
    ax.set_ylim(-12.5, -8.0)
    ax.set_xlabel("V (V)")
    ax.set_ylabel(r"log$_{10}$ |I| (A)")
    ax.set_title(f"R² {iv['r2_trace']:.3f}/{iv['r2_retrace']:.3f}  "
                 f"+V re {iv['r2_posV_retrace']:.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.plot(V, iv["I_trace_data_norm"] * nA, color="0.15", lw=1.5, label="data trace")
    ax.plot(V, iv["I_retrace_data_norm"] * nA, color="crimson", lw=1.5, label="data retrace")
    ax.plot(V, iv["I_trace_norm"] * nA, color="seagreen", ls="--", lw=1.3, label="model")
    ax.plot(V, iv["I_retrace_norm"] * nA, color="seagreen", ls=":", lw=1.3)
    ax.set_xlabel("V (V)")
    ax.set_ylabel("I (nA)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Run the Memristor PINN on S04 sweeps separately.")
    p.add_argument("--sweep", choices=("first", "second", "both"), default="both")
    p.add_argument("--out", type=Path, default=HERE / "output")
    args = p.parse_args()
    device = "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        raise SystemExit("PyTorch is required. See requirements.txt.")
    model = load_model(device)
    wanted = SWEEPS if args.sweep == "both" else tuple(s for s in SWEEPS if s[0] == args.sweep)
    report = {"increment": "28l", "combined": None, "sweeps": {}}
    for tag, fname in wanted:
        data = load_s04(fname, device)
        iv = predict_iv(model, data)
        png = args.out / f"thesis_iv_{tag}.png"
        plot_one(iv, f"Memristor PINN R=100 Ω  {tag} only", png)
        feat = {k: float(iv[k]) for k in iv if k.startswith("r2_")}
        feat["n_points"] = int(iv["V_phys"].size)
        report["sweeps"][tag] = {"file": fname, **feat, "overlay": str(png)}
        print(f"{tag:6s}  R² {feat['r2_trace']:.4f}/{feat['r2_retrace']:.4f}  "
              f"+V {feat['r2_posV_trace']:.3f}/{feat['r2_posV_retrace']:.3f}  "
              f"→ {png}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "increment28l_perfile.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
