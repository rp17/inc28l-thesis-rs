# Memristor PINN

**Memristor PINN** is the paper four-subnet PINN: `CurrentNet` (6-layer
residual GELU) with vacancy polarity according to the thesis by Patrick
Kollias (Kollias, *Resistive Switching in Epitaxial SrTiO₃ on Silicon*,
Ph.D. thesis, Texas State University, 2022), two-contact Cv routing, a
VOFF argument shift, and a physical series resistance read in ohms.

Training is the paper three-phase schedule (Phase 1 per-subnet
pretrain, Phase 2 log-Poisson PDE, Phase 3 SOAP+PCGrad I–V). A shipped
Phase-3 checkpoint is enough for inference.

Each S04 conductive-AFM sweep is scored **separately**. There is no
combined +V retrace.

## Paper

This PINN follows the cascaded architecture and S04 / Sample-4 CAFM
protocol in:

> Rodion Podorozhny, Nikoleta Theodoropoulou, Jelena Tešić,
> *Physics-Informed Neural Network Surrogate for Oxygen Vacancy Dynamics
> in epitaxial SrTiO₃ on Si memristors via Dynamic Spectral Optimization*,
> arXiv:2609.02966, 2026.
> https://arxiv.org/abs/2609.02966
> https://doi.org/10.48550/arXiv.2609.02966

## Architecture

The paper PINN is four cascaded GELU subnets, `hidden_dim = 192`.
`pinn28l.Inc28lPINN` constructs all four. **Current uses** `cv_net` and
`current_net` only. `phi_net` / `carrier_net` run in `forward()` for
Phases 1–2. I must not call `phi_net`.

| Subnet | Class | Inputs | Purpose | Size | Parameters |
|--------|-------|--------|---------|------|------------|
| `cv_net` | `ContinuousVacancyNet` | `(x, t, V, b)` | Oxygen-vacancy field \(C_v(x)\). Two residual MLPs (`base_net` 4-layer on `(t,V)`, `hyst_net` 4-layer on `(x,V,b)`). | 4+4 layers × 192 | 302,019 |
| `phi_net` | `TwoRegionPotentialNet` | `(x, t, V, C_v)` | Electrostatic potential (STO 5-layer residual 192, Si 4-layer 96). | not used for I | 245,379 |
| `carrier_net` | `DDNetStyleCarrierNet` | `(x, t, V, φ, C_v)` | Electrons / holes in log space (two 4-layer MLPs). | not used for I | 150,916 |
| `current_net` | `CurrentNet` | `(t, V_net, b)` | CAFM I–V shape. 6-layer residual GELU. | 6 layers × 192 | 300,869 |
| **Full PINN** | `Inc28lPINN` | Four subnets in `checkpoint_phase3.pt` | | **999,183** |
| **I graph** | | `cv_net` + `current_net` | | **602,888** |

Live current (normalized), with `w = sigmoid(20 V)`:

```
V_net = V + (γ/5) w
I = net(t, V_net, b) · exp(α_si w (Cv_si−1) + α_pt (1−w) (Cv_pt−1)) · (1 + 2 relu(V))
```

If `R > 0`, one Picard update uses the ohmic drop
`R · I · 2 nA / 5 V`.

| Symbol | Value | Role |
|--------|-------|------|
| `net` | Phase-3 weights, frozen | CAFM shape |
| `α_si` | 0.514 | Si/CBO vacancy drive on +V |
| `α_pt` | 0.0289 | Pt identity on −V |
| `γ` | 0.20 V | VOFF left-shift on +V |
| `R` | 100 Ω | series resistance; 0.2 μV at 2 nA |
| `n` | 1 | ideality left at identity |
| `ΔE_c` | 0.35 eV | paper Anderson offset; not added to `V_net` (κ = 0) |

## Training

```bash
# full Phase 1 + 2 + 3 (defaults 2000 / 5000 / 4000 epochs)
python train_28l.py --phases 1,2,3 --sweep first --out output/checkpoint_trained.pt

# short smoke
python train_28l.py --phases 1 --epochs-p1 2

# continue from the shipped Phase-3 weights
python train_28l.py --phases 3 --init-checkpoint weights/checkpoint_phase3.pt --epochs-p3 200
```

| Phase | What is trained | Loss |
|-------|-----------------|------|
| 1 | each subnet in turn | analytical φ, \(C_v\), \(n,p\), diode-like I |
| 2 | PDE (`phi` frozen after this phase) | log-Poisson (ρ includes holes), vacancy drift-diffusion, BCs |
| 3 | all parameters, PCGrad on trace vs retrace | S04 I–V + smoothness; SOAP if `soap.py` imports |

Hysteresis terms use \(C_v\) at the Si/STO junction (thesis LRS), not at
`x = 0`. Optimizer: SOAP (`soap.py`, arXiv:2409.11321) with Adam fallback.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch CPU is enough. A GPU is used automatically if available.

## Dataset

`data/` holds the two S04 2 nA linear sweeps used in the paper
(thesis Figure 37 / Sample 4, sensitivity 160):

| File | Sweep |
|------|--------|
| `S04N dif I sens 160 first 2nA linear.dat` | first CAFM loop |
| `S04N dif I sens 160 second 2nA linear.dat` | second CAFM loop |

Tab-separated. Column 0 / 160 are voltage; the remaining columns on each
half are repeated current samples. The loader averages those samples and
interpolates retrace onto the trace voltage grid.

## Usage

Score both sweeps and write overlays (no combined file):

```bash
python run_28l.py
```

One sweep only:

```bash
python run_28l.py --sweep first
python run_28l.py --sweep second --out output
```

From Python:

```python
from pinn28l import load_model, load_s04, predict_iv

model = load_model()                       # CPU or CUDA
data = load_s04("first", "cpu")            # or the .dat filename
iv = predict_iv(model, data)
print(iv["r2_trace"], iv["r2_retrace"], iv["r2_posV_retrace"])
```

`load_s04` accepts `"first"`, `"second"`, or a path under `data/`.

Outputs are saved in `output/`:

- `thesis_iv_first.png`, `thesis_iv_second.png`
- `increment28l_perfile.json`

## Expected scores (this snapshot)

| Sweep | R² trace / retrace | +V retrace |
|-------|--------------------|------------|
| first | 0.995 / 0.969 | 0.878 |
| second | 0.975 / 0.941 | 0.738 |

The second-sweep +V retrace stays off until ~3 V in the experiment;
the model still turns on at the first-sweep voltage. That is
sweep-to-sweep scatter, not a combined-score artifact.

## Citation

```bibtex
@article{podorozhny2026pinn,
  title={Physics-Informed Neural Network Surrogate for Oxygen Vacancy
         Dynamics in epitaxial {SrTiO$_3$} on Si memristors via
         Dynamic Spectral Optimization},
  author={Podorozhny, Rodion and Theodoropoulou, Nikoleta and Te{\v{s}}i{\'c}, Jelena},
  journal={arXiv preprint arXiv:2609.02966},
  year={2026},
  url={https://arxiv.org/abs/2609.02966}
}

@phdthesis{kollias2022rs,
  title={Resistive Switching in Epitaxial {SrTiO$_3$} on Silicon},
  author={Kollias, Patrick},
  year={2022},
  school={Texas State University}
}
```
