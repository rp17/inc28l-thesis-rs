# Increment 28l — thesis resistive-switching PINN on S04

Inference-only release of **increment 28l**: the increment-26 `CurrentNet`
(6-layer residual GELU) with Kollias-thesis vacancy polarity, two-contact
Cv routing, a VOFF argument shift, and a physical series resistance
read in ohms.

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

Vacancy-layer polarity (LRS at Si/STO, HRS at Pt/STO) follows
Patrick Kollias, *Resistive Switching in Epitaxial SrTiO₃ on Silicon*,
Ph.D. thesis, Texas State University, 2022.

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

Outputs land in `output/`:

- `thesis_iv_first.png`, `thesis_iv_second.png`
- `increment28l_perfile.json`

## What 28l is

Live current (normalized), with `w = sigmoid(20 V)`:

```
V_net = V + (γ/5) w
I = net(t, V_net, b) · exp(α_si w (Cv_si−1) + α_pt (1−w) (Cv_pt−1)) · (1 + 2 relu(V))
```

If `R > 0`, one Picard update uses the ohmic drop
`R · I · 2 nA / 5 V`. Promoted values:

| Symbol | Value | Role |
|--------|-------|------|
| `net` | Phase-3 increment-26 weights, frozen | CAFM shape |
| `α_si` | 0.514 | Si/CBO vacancy drive on +V |
| `α_pt` | 0.0289 | Pt identity (increment-26 scale) |
| `γ` | 0.20 V | VOFF left-shift on +V |
| `R` | 100 Ω | S0 series resistance; 0.2 μV at 2 nA |
| `n` | 1 | ideality left at identity |
| `ΔE_c` | 0.35 eV | paper Anderson offset; not added to `V_net` (κ = 0) |

`weights/checkpoint_phase3.pt` is the increment-26 SOAP Phase-3
checkpoint (`hidden_dim = 192`). Only `cv_net` and `current_net` are
loaded. `phi_net` is not used for current (see increment 27p).

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
```
