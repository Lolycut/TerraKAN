# TerraKAN (`Frac-ML-KAT-MoE`): Fractional Kolmogorov-Arnold Transformer with Padé Mittag-Leffler Basis and Biokinetic Mixture-of-Experts for Inverse Soil Bioindication
[![DOI](https://zenodo.org/badge/1455550676.svg)](https://doi.org/10.5281/zenodo.24167982)

**Author:** Ilya Kakhno ([ORCID: 0009-0005-5503-3269](https://orcid.org/0009-0005-5503-3269), Faculty of Biology, Belarusian State University, Minsk, Republic of Belarus)  
**E-mail:** [Sunnywww1701@gmail.com](mailto:Sunnywww1701@gmail.com)  
**Code & Weights:** [GitHub](https://github.com/Lolycut/TerraKAN) | **Core PyTorch Library (`pip install abelkan`):** [PyPI](https://pypi.org/project/abelkan/) · [GitHub](https://github.com/Lolycut/abelkan) (`v1.1.0`, `terrakan_v4_frozen.pt`, Seed `42`)  
**Licenses:** Code & Frozen Weights — [MIT License](https://opensource.org/licenses/MIT) | Preprint & Figures — [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)

## Overview

In this repository, I present **TerraKAN (`Frac-ML-KAT-MoE`)**, an **amortized simulation-based inference** Scientific Machine Learning (SciML) architecture that I designed, mathematically verified, and benchmarked for inverse soil bioindication and long-horizon self-purification forecasting from sparse (6–8 points), incomplete (up to 50% missingness), and noisy (25–40% multiplicative noise) enzymatic and respiratory measurements without direct chemical assays at test time.

My work addresses four gaps across computational mathematics and ecological biophysics:

1. **Extrapolation Divergence of Polynomial KANs:** Orthogonal polynomial KANs (`ChebPIKAN` / `fKAN`; [Aghaei, 2024, arXiv:2406.07456](https://arxiv.org/abs/2406.07456)) diverge catastrophically during blind extrapolation on $t \in (150, 365]\text{ days}$ ($R^2 = -8.132$, $\text{RMSE} = 1.804\,\text{EC}_{50}$).
2. **Failure of Classical Bayesian ODE/FDE Solvers:** Integer-order `MAP-ODE` ignores non-Markovian micropore memory ($\alpha < 1$) and fails to reconstruct unmeasured contaminant load $\tilde{P}(t)$ ($R^2 = -0.191 \pm 0.225$). Even a fractional-order `MAP-FDE` solver (Diethelm ABM with free $\alpha$ + L2 Soil Atlas prior) requires $51.98\text{ ms/field}$ of iterative optimization and achieves lower accuracy on $\tilde{P}(t)$ ($R^2 = 0.802 \pm 0.037$ vs. $0.908 \pm 0.019$).
3. **The Open Fractional Transformer–KAN Challenge:** [Tu et al. (2026, *Fractal and Fractional*, DOI: 10.3390/fractalfract10070454)](https://doi.org/10.3390/fractalfract10070454) identified the integration of rational Padé KANs ([Aghaei, 2024, arXiv:2406.14495](https://arxiv.org/abs/2406.14495)), the Mittag-Leffler function $E_{\alpha,1}(-z)$, and non-Markovian fractional kernels into Transformer attention ([Yang & Wang, 2025, arXiv:2409.10594](https://arxiv.org/abs/2409.10594)) as an open fundamental challenge. Here, I provide an analytical and differentiable PyTorch implementation of this architecture class.
4. **Ecological Ambiguity of Static IBSP Indices:** Classical static Integral Index of Biological State (IBSP / IIBS; Kazeev, Kolesnikov, & Valkov, 2003/2016) triggers false alarms on naturally oligotrophic sandy soils (`Arch_5`, $K_m = 0.44$) and false negatives after rainfall pulses. I introduce a dynamic memory-aware index $\text{IBSP}_\alpha(t)$ coupled with a sparse biokinetic Mixture-of-Experts (`PhysicsExpertBank`).

## Repository Structure

```text
TerraKAN/
├── README.md                          # Repository documentation, benchmarks & Zenodo guide
├── LICENSE                            # Dual license: MIT (code/weights) & CC-BY-4.0 (preprint/figures)
├── CITATION.cff                       # GitHub & Zenodo citation metadata
├── .zenodo.json                       # Automated CERN Zenodo DOI deposit descriptor
├── requirements.txt                   # Minimal standard scientific stack (torch, numpy, scipy, matplotlib)
├── .gitignore                         # Standard Python/LaTeX ignore rules
│
├── pade_ml.py                         # Differentiable [3/4] Padé Mittag-Leffler kernel (MittagLefflerPade34)
├── physics_solver.py                  # Diethelm ABM Caputo fractional solver & 5-archetype Soil Atlas
├── model.py                           # TerraKAN (Frac-ML-KAT-MoE) & baselines (ChebPIKAN, SimpleMLMLP, MAP-ODE)
├── run_tests.py                       # 4 mandatory pre-training numerical & autograd unit tests
├── train_eval.py                      # 160-epoch training, Val Youden threshold calibration & 1000x bootstrap evaluation
├── visualize.py                       # Generator for publication figures (Figs. 1-6, 300 DPI)
├── process_buckets_csv.py             # Wet-lab / microcosm CSV inference pipeline (36 vessels, zero re-training)
│
├── terrakan_v4_frozen.pt              # Frozen verified model weights & calibrated anomaly threshold (d_star = 0.290)
├── unit_test_report.json              # Machine-verified numerical test log
├── eval_summary.json                  # Full evaluation metrics, 1000x bootstrap CIs & 3-seed ablation logs
├── map_fde_benchmark.json             # Classical fractional-order MAP-FDE benchmark & Q_4(z) dense grid check
│
├── blank_lab_journal_36pots.csv       # Printable blank laboratory log sheet for 36 microcosm vessels (days 0..28)
├── dry_run_synthetic_36pots.csv       # Synthetic dry-run test log with vessel-specific missingness & lab reference
├── predictions_report.csv             # Blind 60-day forecast & automated diagnosis table for all 36 vessels
│
├── preprint_en.pdf                    # Compiled two-column English preprint for Zenodo / arXiv / bioRxiv
├── preprint_en.tex                    # Self-contained LaTeX source of the English preprint
├── preprint_en.md                     # Markdown edition of the English preprint
│
├── fig1_pade_verification.png         # Fig. 1: Padé [3/4] accuracy vs Oberhettinger integral & Caputo ABM check
├── fig2_extrapolation_benchmark.png   # Fig. 2: Blind extrapolation (150 -> 365 d) vs MAP-ODE, MLP & ChebPIKAN
├── fig3_hysteresis_and_ibsp.png       # Fig. 3: Haldane hysteresis loop, poor-soil specificity & FN safety audit
├── fig4_soil_atlas_passport.png       # Fig. 4: Soil Atlas KL typing, alpha recovery & Val/Test ROC audit
├── fig5_multkan_symbolic_discovery.png# Fig. 5: MultKAN power-law discovery, MoE routing & symbolic equivalence
├── fig6_ablation_and_noise_robustness.png # Fig. 6: 3-seed ablation study, 40% noise robustness & OOD misspecification
└── fig_dry_run_36pots.png             # Fig. 7: Synthetic dry-run verification on 36 microcosm vessels (0..60 d)
```

## Mathematical Architecture (`Frac-ML-KAT-MoE`)

I formulate soil self-purification and biokinetic recovery in dimensionless half-inhibition units $\tilde{P}(t) = P(t) / K_{\text{tox}}$ ($K_{\text{tox}} \equiv 1.0$, $K_i \equiv 1.0$, $\text{EC}_{50}$ units) under the coupled Caputo fractional dynamical system of order $\alpha \in (0, 1]$:

$$D_t^\alpha \tilde{P}(t) = -\left(k_p + \frac{k_{\text{deg}} \Phi_{\text{env}}(t)\, M(t)}{K_s + \tilde{P}(t) + \tilde{P}(t)^2}\right)\tilde{P}(t)$$

$$D_t^\alpha M(t) = r_m \Phi_{\text{env}}(t)\, M(t)\left(1 - \frac{M(t)}{K_m}\right) - \gamma_p \frac{\tilde{P}(t)\, M(t)}{1 + \tilde{P}(t)} + \mu_{\text{sub}} \frac{\tilde{P}(t)\, E(t)}{K_s + \tilde{P}(t) + \tilde{P}(t)^2}$$

$$D_t^\alpha E(t) = \beta_E \frac{M(t)}{1 + \tilde{P}(t)^2} - \delta_E E(t)$$

### Reference Soil Atlas Archetypes

| Code | Soil Archetype | $\text{pH}_{\text{norm}}$ | $\text{NO}_{3,\text{norm}}$ | $W_{\text{norm}}$ | $K_m$ | $\alpha$ | $k_p$ | $k_{\text{deg}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Arch_1` | Sod-Podzolic Loam | 0.58 | 0.55 | 0.60 | 1.00 | 0.75 | 0.018 | 0.085 |
| `Arch_2` | Peat-Bog Soil | 0.36 | 0.44 | 0.88 | 1.18 | 0.55 | 0.012 | 0.065 |
| `Arch_3` | Sod-Carbonate Soil | 0.68 | 0.80 | 0.70 | 1.38 | 0.86 | 0.026 | 0.125 |
| `Arch_4` | Technogenic Solonchak | 0.88 | 0.30 | 0.32 | 0.68 | 0.66 | 0.014 | 0.055 |
| `Arch_5` | Sandy Forest Soil | 0.50 | 0.22 | 0.36 | 0.44 | 0.80 | 0.022 | 0.070 |

To anchor $\tilde{P}(0)$ without direct chemical assays, I exploit the stationary boundary face of the enzyme-to-biomass equation at $t_1 \to 0$. In the simulator, $\beta_E / \delta_E = 0.09 / 0.075 = 1.20$, so $E_{\text{ref}}/M_{\text{ref}} \equiv 1.20$:

$$\frac{E(t_1)}{M(t_1)} = \frac{E_{\text{ref}} / M_{\text{ref}}}{1 + \tilde{P}(0)^2} \implies \tilde{P}_{\text{boundary}}(0) = \sqrt{\max\!\left(0,\, \frac{1.20}{E(t_1)/M(t_1)} - 1\right)}$$

For real laboratory or field soils whose native $E_{\text{ref}}/M_{\text{ref}}$ differs from $1.20$, per-series normalization against uncontaminated day-0 controls (`is_control_G0 == 1` within each `soil_series`) rescales $M(t)$ and $E(t)$ into $E_{\text{norm}}(0)/M_{\text{norm}}(0) \approx 1.20$ before inference.

### 1. Encoder `Frac-KAT` & Fractional Weight Tying

Given sparse observations $\mathbf{v}_j = [\text{pH}_{\text{norm}}, \text{NO}_{3,\text{norm}}, W_{\text{norm}}, M_{\text{obs}}, E_{\text{obs}}]^\top \in \mathbb{R}^5$ at irregular timestamps $0 \le t_1 < \dots < t_N \le 150\text{ d}$ with binary mask $\mathbf{m}_j \in \{0,1\}^5$ and rainfall flag $r_j \in \{0,1\}$, I construct the 12-dimensional token $\mathbf{u}_j = [\mathbf{v}_j \odot \mathbf{m}_j + 0.35(\mathbf{v}_j - \mathbf{v}_1)\odot \mathbf{m}_j, \mathbf{m}_j, \ln(1+t_j), r_j]^\top$. A single-layer causal Transformer encoder ($d_{\text{model}}=48$, 1 head, GELU input projection, LayerNorm residuals) aggregates the sequence, while Kolmogorov-Arnold functional representations live in the downstream `MultKANEnvModule` and `MLKANTrunk`. Causal attention logits are modulated by an Abel-type power-law decay ($\varepsilon = 1.0$ via $\ln(1+\Delta t_{ij})$):

$$\mathcal{A}_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_{\text{model}}}} - \text{softplus}(\gamma_h)(1 - \hat{\alpha}_i)\ln(1 + \max(0, t_i - t_j)), \quad t_i \ge t_j$$

Strict **Weight Tying** ensures the same scalar $\hat{\alpha} \in [0.40, 0.92]$ governs the attention kernel, `SoilAtlasModule`, `MittagLefflerPade34`, and $\text{IBSP}_{\hat{\alpha}}(t)$.

### 2. Differentiable Global Padé `[3/4]` Mittag-Leffler Kernel (`MittagLefflerPade34`)

To approximate $E_{\alpha,1}(-z)$ globally on $z \in [0, \infty)$ with exact gradients $\partial E_{\alpha,1}/\partial\alpha$, I construct a rational `[3/4]` Padé approximant ([Aghaei, 2024, arXiv:2406.14495](https://arxiv.org/abs/2406.14495); [Abueidda et al., 2025, DOI: 10.1016/j.cma.2024.117699](https://doi.org/10.1016/j.cma.2024.117699)):

$$E_{\alpha,1}(-z) \approx R_{[3/4]}^{(\alpha)}(z) = \frac{1 + p_1 z + p_2 z^2 + p_3 z^3}{1 + q_1 z + q_2 z^2 + q_3 z^3 + q_4 z^4}$$

Matching the Maclaurin expansion as $z \to 0$ and Poincaré asymptotics as $z \to \infty$ via Euler's reflection identity $\frac{1}{\Gamma(1 - k\alpha)} = \frac{\Gamma(k\alpha)\sin(k\pi\alpha)}{\pi}$ eliminates the Gamma pole at $\alpha = 0.50$. Solving the closed-form $5\times 5$ system in `float64` (`torch.linalg.solve`) with a $C^1$ smoothstep bridge to $e^{-z}$ on $\alpha \in (0.90, 1.0]$, dense numerical verification across $61$ values of $\alpha \in [0.40, 0.90]$ and $250$ log-spaced nodes $z \in [10^{-3}, 10^3]$ confirms $Q_4(z) \ge 1.000978 > 1$ across the entire operating regime.

### 3. `MultKANEnvModule`, `PhysicsExpertBank`, and Two-Stage Inference

- **`MultKANEnvModule`:** Takes $\mathbf{e} = [\text{pH}_{\text{norm}}, \text{NO}_{3,\text{norm}}, W_{\text{norm}}]^\top \in \mathbb{R}^3$ and combines a Shelford pH tolerance bell with a two-branch log-exponential power layer ([Liu et al., 2024, arXiv:2408.10205](https://arxiv.org/abs/2408.10205)):

$$\Phi_{\text{KAN}}(\mathbf{e}) = \exp\!\left(-\beta_{\text{pH}}(\text{pH}_{\text{norm}} - \text{pH}_{\text{opt}})^2\right) \sum_{b=1}^2 w_b \exp\!\left(\sum_{c=1}^3 W_{\text{power}, b, c} \ln e_c\right)$$

- **`PhysicsExpertBank` (Sparse Top-2 MoE):** Routes $[\mathbf{h}_{\text{final}}, \Phi_{\text{KAN}}, \text{pH}_{\text{norm}}, \Delta E_{\text{drop}}, \Delta M_{\text{drop}}, \tilde{P}_{\text{boundary}}(0)] \in \mathbb{R}^{53}$ through a temperature-annealed ($\tau: 2.0 \to 0.5$) Top-2 sparse gate across $K_{\text{exp}}=4$ physical experts (Haldane inhibition, Monod saturation, Abiotic clean background, Acidic/osmotic stress) to produce $\mathbf{h}_{\text{ctx}}$.
- **3-Mode Mittag-Leffler Expansions & $\text{IBSP}_\alpha(t)$:** `MLKANTrunk` maps $\mathbf{h}_{\text{ctx}}$ via $\mathbf{W}_{\text{head}}\mathbf{h}_{\text{ctx}} + \mathbf{b}_{\text{head}} \in \mathbb{R}^{18}$ to decay rates $\lambda_{1..3}$, mode amplitudes $(c_P, c_M, c_E) \in \mathbb{R}^{3\times 3}$, and asymptotic equilibria $M_\infty, E_\infty$:

$$\tilde{P}_{\text{bio}}(t) = g(d_{\min})\sum_{k=1}^3 c_{P,k} E_{\hat{\alpha},1}(-\lambda_k t^{\hat{\alpha}}), \quad M(t) = M_\infty + \sum_{k=1}^3 c_{M,k} E_{\hat{\alpha},1}(-\lambda_k t^{\hat{\alpha}}), \quad E(t) = E_\infty + \sum_{k=1}^3 c_{E,k} E_{\hat{\alpha},1}(-\lambda_k t^{\hat{\alpha}})$$

$$\text{IBSP}_\alpha(t) = 100 \cdot \exp\!\left(-\left[0.35\,\left|\ln\frac{M(t)}{M_\infty}\right| + 0.40\,\max\!\left(0, 1 - \frac{E(t)}{E_\infty}\right) + \lambda_{\text{mem}}(1 - \hat{\alpha})\tilde{P}_{\text{bio}}(t)\right]\right), \quad \lambda_{\text{mem}} = 0.25$$

### 4. Training Protocol & Amortized Simulation-Based Inference

TerraKAN has **21,036 trainable parameters** and is trained strictly under an **amortized simulation-based inference** paradigm on $N_{\text{train}} = 3,000$ synthetic Soil Atlas scenarios ($N_{\text{val}} = 500$ for Youden threshold calibration, $N_{\text{test}} = 500$ held out). Training runs for 160 epochs on CPU in `float64` using `Adam` ($\text{lr} = 2.5\times 10^{-3}$, weight decay $10^{-5}$, `CosineAnnealingLR` with $\eta_{\min} = 10^{-5}$, batch size 64, gradient-norm clipping 1.0):

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + 0.15\,\mathcal{L}_{\text{phys}} + 0.15\,\mathcal{L}_{\text{atlas}} + 0.05\,\mathcal{L}_{\text{kan}} + 0.02\,\mathcal{L}_{L1} + 0.05\,\mathcal{L}_{\text{balance}}$$

At epochs $\{8, 24, 55, 95, 135, 160\}$, closed-form ridge regression aligns linear readout heads against simulator ground-truth targets $(\tilde{P}_{\text{true}}, M_{\text{true}}, E_{\text{true}}, \alpha_{\text{true}})$ on `Train (3000)`, while test-time inference uses **zero** optimization and **zero** access to $\tilde{P}_{\text{true}}$ or $\alpha_{\text{true}}$.

## Summary of Empirical Benchmarks (Figures 1–7)

| Model Architecture | Params | Time (ms/field) | Hidden $\tilde{P}(t)$ $R^2$ (RMSE) | Biomass $M(t)$ $R^2$ (RMSE) | Enzyme $E(t)$ $R^2$ (RMSE) | Index $\text{IBSP}_\alpha(t)$ $R^2$ (RMSE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TerraKAN (Mine)** | 21,036 | **0.50** (104x vs FDE) | **$0.908 \pm 0.019$ (0.181)** | $0.894 \pm 0.015$ (0.114) | $0.910 \pm 0.013$ (0.166) | **$0.886 \pm 0.016$ (6.88%)** |
| `MAP-FDE` (Diethelm ABM + Prior) | 5 | 51.98 | $0.802 \pm 0.037$ (0.265) | **$0.934 \pm 0.012$ (0.090)** | **$0.928 \pm 0.015$ (0.149)** | $0.849 \pm 0.028$ (7.93%) |
| `MAP-ODE` (Integer ODE + Prior) | 5 | 5.01 | $-0.191 \pm 0.225$ (0.651) | $0.900 \pm 0.019$ (0.111) | $0.891 \pm 0.020$ (0.183) | $0.769 \pm 0.045$ (9.81%) |
| `Simple-ML-MLP` (No Abel/MoE) | 3,763 | 0.43 | $0.702 \pm 0.040$ (0.326) | $0.900 \pm 0.016$ (0.111) | $0.888 \pm 0.015$ (0.185) | $0.742 \pm 0.032$ (10.37%) |
| `ChebPIKAN` (Poly $T_0\dots T_5$) | 21,036 | 0.57 | $-8.132 \pm 3.014$ (1.804) | $-2.173 \pm 0.586$ (0.626) | $-1.223 \pm 0.305$ (0.827) | $-0.909 \pm 0.341$ (28.23%) |

### Key Verification Milestones

- **Kernel Verification (`fig1_pade_verification.png`):** Against the exact Oberhettinger contour integral on $z \in [10^{-3}, 10^3]$, `MittagLefflerPade34` achieves $\max|\Delta| = 7.67\times 10^{-7}$ at $\alpha=0.30$, $3.85\times 10^{-5}$ at $\alpha=0.50$, and $1.12\times 10^{-3}$ at $\alpha=0.78$, with $\min Q_4(z) = 1.000978$ across $\alpha \in [0.40, 0.90]$, passing `torch.autograd.gradcheck` (`float64`) without `NaN` and matching the Diethelm ABM Caputo solver ([Diethelm et al., 2002](https://doi.org/10.1023/A:1016592219341)) within $0.00331\% < 0.5\%$.
- **Soil Atlas Typing & Anomaly Detector (`fig4_soil_atlas_passport.png`):** Across $N=500$ test fields, `SoilAtlasModule` achieves **Macro-Accuracy $= 88.4\%$** and **Macro-F1 $= 0.890$**, recovers $\alpha$ with **$R^2 = 0.9248$** ($\text{RMSE} = 0.0240$), and maintains a stable Youden threshold $d_{\min}^* = 0.290$ across 5 stratified splits (`seeds = [42, 7, 13, 99, 2026]`, mean $|\Delta\text{TPR}| = 1.85\% \pm 0.95\% < 4.0\%$).
- **Symbolic Serialization & Inference Equivalence (`fig5_multkan_symbolic_discovery.png`):** `MultKAN` discovers nitrate exponent $0.345$ (true $0.350$), moisture exponent $0.394$ (true $0.400$), and Shelford pH bell $\exp(-2.36(\text{pH}-0.66)^2)$ (true $\exp(-2.50(\text{pH}-0.65)^2)$). Serialized analytical formulas match tensor outputs to $\max|\Delta_{\text{exact}}| = 9.96\times 10^{-13}$ in `float64`.
- **Synthetic Dry-Run Microcosm Pipeline (`fig_dry_run_36pots.png`):** On 36 simulated vessels across 3 soil series (`S1_sand`, `S2_loam`, `S3_peat`) with per-series control normalization and zero leakage of `is_control_G0` into diagnosis, all 6 `G0` controls satisfy $\tilde{P}_{\text{bio}}(0) \le 0.001 < 0.5\,\text{EC}_{50}$ and $\text{IBSP}(0) \in [92.1\%, 99.5\%] > 92\%$, all 8 clean nutrient-poor `G1` vessels are classified as substrate-depleted without false contamination alarms ($\max \tilde{P}_{\text{bio}}(0) = 0.016$, $\max d_{\min} = 0.148 \le 0.290$), and all 12 high-load `G3` vessels trigger $d_{\min} \ge 0.593 > 0.290$.

## Limitations & Inverse-Crime Considerations

1. **Shared Structural Priors ("Inverse Crime" Risk):** Although TerraKAN generalizes zero-shot to Two-Compartment Sorption + Weibull dynamics and Out-of-Atlas $\alpha$ (Fig. 6A), its internal biokinetic priors belong to the same equation families used in the Soil Atlas simulator. True out-of-distribution robustness must be verified on real laboratory and field soils exhibiting unmodeled phenomena (secondary toxic metabolites, N/P co-limitation, microbial succession).
2. **Synthetic Dry-Run vs. Wet-Lab Data:** The 36-vessel microcosm check in Fig. 7 is a synthetic dry-run verification of the CSV preprocessing and inference pipeline prior to wet-lab execution.
3. **Fixed Boundary Ratio Assumption:** Without per-series `G0` normalization, the boundary anchor assumes $E_{\text{ref}}/M_{\text{ref}} \approx 1.20$; soils with divergent baseline ratios require control-group normalization.

## Quickstart & Reproducibility

```bash
python3 run_tests.py
python3 train_eval.py
python3 visualize.py
python3 process_buckets_csv.py
```

- Print `blank_lab_journal_36pots.csv` for recording raw measurements in the laboratory (`catalase_O2_ml` as proxy for $E$, `respiration_CO2_mg` as proxy for $M$, plus `pH`, `NO3_mg_l`, `moisture_pct`, `rain_flag`, `is_control_G0`, and optional `lab_toxin_ref_ec50`).

## References

1. Tu, W., Liu, X., Wang, H., Li, X., & Han, C. (2026). Fractional Calculus for Large Language Models: A Survey of Potentials, Mechanisms, and Challenges. *Fractal and Fractional*, 10(7), 454. [https://doi.org/10.3390/fractalfract10070454](https://doi.org/10.3390/fractalfract10070454)
2. Liu, Z., Wang, Y., Vaidya, S., et al. (2024). KAN: Kolmogorov-Arnold Networks. [arXiv:2404.19756](https://arxiv.org/abs/2404.19756); and *KAN 2.0*, [arXiv:2408.10205](https://arxiv.org/abs/2408.10205).
3. Yang, X., & Wang, X. (2025). Kolmogorov-Arnold Transformer. *ICLR 2025* / [arXiv:2409.10594](https://arxiv.org/abs/2409.10594).
4. Aghaei, A. A. (2024). rKAN: Rational Kolmogorov-Arnold Networks, [arXiv:2406.14495](https://arxiv.org/abs/2406.14495); and *fKAN*, [arXiv:2406.07456](https://arxiv.org/abs/2406.07456).
5. Abueidda, D. W., Pantidis, P., & Mobasher, M. E. (2025). DeepOKAN: Deep Operator Network Based on Kolmogorov Arnold Networks for Mechanics Problems. *CMAME*, 436, 117699. [https://doi.org/10.1016/j.cma.2024.117699](https://doi.org/10.1016/j.cma.2024.117699)
6. Kolesnikov, S., Timoshenko, A., Varduni, V., et al. (2021). Influence of Silver Nanoparticles on the Biological Indicators of Haplic Chernozem. *Nanomaterials* / [PMC8161183](https://pmc.ncbi.nlm.nih.gov/articles/PMC8161183/).
7. Minenko, V., et al. (2024). Enzymatic Diagnostics of Soil Health of the European Part of Russia. *Soil Systems*, 8(3), 76. [https://www.mdpi.com/2571-8789/8/3/76](https://www.mdpi.com/2571-8789/8/3/76)
8. Wyszkowska, J., et al. (2024). The Potential for Restoring the Activity of Oxidoreductases in Soils Contaminated with Petroleum Products. *Applied Sciences*, 14(9), 3591. [https://www.mdpi.com/2076-3417/14/9/3591](https://www.mdpi.com/2076-3417/14/9/3591)
9. Diethelm, K., Ford, N. J., & Freed, A. D. (2002). A predictor-corrector approach for the numerical solution of fractional differential equations. *Nonlinear Dynamics*, 29, 3–22. [https://doi.org/10.1023/A:1016592219341](https://doi.org/10.1023/A:1016592219341)
10. Kazeev, K. Sh., Kolesnikov, S. I., & Valkov, V. F. (2003/2016). *Biological Diagnostics and Indication of Soils: Methodology and Research Methods* [in Russian]. Rostov-on-Don: Southern Federal University Press.
