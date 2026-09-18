# Memristor PINN

**Memristor PINN** has the cascade architecture of four subnets with
GELU activations (for second-order optimizer evaluation)
(`cv_net`, `phi_net`, `carrier_net`, `current_net`; 999,183 parameters)
with vacancy polarity according to the thesis by Patrick
Kollias (Kollias, *Resistive Switching in Epitaxial SrTiO₃ on Silicon*,
Ph.D. thesis, Texas State University, 2022) and the Journal of Applied
Physics article on oxygen-vacancy-driven resistive switching in
SrTiO₃/Si(001) (Kollias et al., 2025), two-contact Cv routing, a
VOFF argument shift, and a physical series resistance.

Training is done in 3 phases. Phase 1 does pretraining on the known
analytical functions of the four cascade subnets to initialize the
weights. In Phase 2 the PINN is trained on the PDE loss (log-Poisson PDE residual).
In Phase 3 the PINN is trained on the experimental dataset of I-V (current on voltage) dependence.

## Paper

This PINN follows the cascaded architecture and S04 / Sample-4 CAFM
(conductive atomic force microscopy) experiment design in:

> Rodion Podorozhny, Nikoleta Theodoropoulou, Jelena Tešić,
> *Physics-Informed Neural Network Surrogate for Oxygen Vacancy Dynamics
> in epitaxial SrTiO₃ on Si memristors via Dynamic Spectral Optimization*,
> arXiv:2609.02966, 2026.
> https://arxiv.org/abs/2609.02966
> https://doi.org/10.48550/arXiv.2609.02966

## Architecture

```
C_v = cv_net(x, t, V, b)
    → φ = phi_net(x, t, V, C_v)
    → (n, p) = carrier_net(x, t, V, φ, C_v)
    → I = current_net(t, V, b, C_v)
```

| Stage | Subnet | Class | Inner nets | Inputs | Purpose | Parameters |
|-------|--------|-------|------------|--------|---------|------------|
| 1 | `cv_net` | `ContinuousVacancyNet` | `base_net` 4-layer residual 192 on `(t,V)`; `hyst_net` 4-layer residual 192 on `(x,V,b)`; $\delta_{C_v}$ | `(x, t, V, b)` | Oxygen-vacancy field $C_v(x)$ | 302,019 |
| 2 | `phi_net` | `TwoRegionPotentialNet` | `sto_net` 5-layer residual 192; `si_net` 4-layer 96; sharpness $s$ | `(x, t, V, C_v)` | Electrostatic potential $\varphi(x)$ | 245,379 |
| 3 | `carrier_net` | `DDNetStyleCarrierNet` | `log_n_net`, `log_p_net` (4-layer 192) | `(x, t, V, φ, C_v)` | Electrons / holes in log space | 150,916 |
| 4 | `current_net` | `CurrentNet` | 6-layer residual GELU `net`; $R$, $n_{\mathrm{tr}}$, $n_{\mathrm{re}}$, $\beta$ | `(t, V, b, C_v)` | Terminal current $I$ | 300,869 |
| | **PINN** | `Inc28lPINN` | Stages 1–4 | | | **999,183** |

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
| $\Delta E_c$ | 0.35 eV | Anderson Si/STO conduction-band offset inside $\phi_{net}$ |

## Training

```bash
# full Phase 1 + 2 + 3 (defaults 2000 / 5000 / 4000 epochs)
python train_28l.py --phases 1,2,3 --sweep first --out output/checkpoint_trained.pt

# short smoke
python train_28l.py --phases 1 --epochs-p1 2

# continue from the shipped Phase-3 weights
python train_28l.py --phases 3 --init-checkpoint weights/checkpoint_phase3.pt --epochs-p3 200
```

| Phase | Subnets trained | Loss |
|-------|-----------------|------|
| 1 (pretraining) | each cascade subnet in turn: `phi_net`, `cv_net` (2a base, then 2b hysteresis), `carrier_net`, `current_net` | analytical $\varphi$, $C_v$, $n,p$, diode-like $I$ |
| 2 | PDE residual (`phi_net` and `cv_net` hysteresis frozen) | log-Poisson (ρ includes holes), vacancy drift-diffusion, BCs |
| 3 | all parameters, PCGrad (for gradient deconfliction) on trace vs retrace | S04 I–V + smoothness |

Hysteresis terms use $C_v$ at the Si/STO junction (thesis LRS).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

`data/` holds the two S04 2 nA linear sweeps
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

## Expected $R^2$ scores

| Sweep | $R^2$ trace / retrace | +V retrace |
|-------|--------------------|------------|
| first | 0.995 / 0.969 | 0.878 |
| second | 0.975 / 0.941 | 0.738 |

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

@article{kollias2025oxygen,
  title={Oxygen vacancy driven resistive switching in {SrTiO$_3$}/{Si}(001)
         heterostructures due to an electronic mechanism},
  author={Kollias, Patrick and Cottier, Ryan J. and Miracle, John T.
          and Cantrell, Samuel R. and Theodoropoulou, Nikoleta},
  journal={Journal of Applied Physics},
  volume={137},
  number={24},
  pages={244502},
  year={2025},
  doi={10.1063/5.0268554},
  url={https://pubs.aip.org/aip/jap/article/137/24/244502/3351065}
}
```
