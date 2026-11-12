# TerraKAN (`Frac-ML-KAT-MoE`): Fractional Kolmogorov-Arnold Transformer with Padé Mittag-Leffler Basis and Biokinetic Mixture-of-Experts for Inverse Soil Bioindication

**Author:** Ilya Kakhno ([ORCID: 0009-0005-5503-3269](https://orcid.org/0009-0005-5503-3269), Belarusian State University, Faculty of Biology)  
**Version:** `1.0.0-preprint` (`terrakan_v4_frozen.pt`, Seed `42`)  
**Licenses:** Code & Frozen Weights — [MIT License](https://opensource.org/licenses/MIT) | Preprint & Figures — [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)

## Overview

In this repository, I present **TerraKAN (`Frac-ML-KAT-MoE`)**, a neuro-symbolic Scientific Machine Learning (SciML) architecture that I designed, mathematically verified, and benchmarked for inverse soil bioindication and long-horizon self-purification forecasting from sparse, noisy, and incomplete field or microcosm measurements.

My work addresses two open problems across computational mathematics and ecological biophysics:

1. **Computational Mathematics & SciML:** In July 2026, [Tu et al. (2026, *Fractal and Fractional*, DOI: 10.3390/fractalfract10070454)](https://doi.org/10.3390/fractalfract10070454) identified the integration of rational Padé Kolmogorov-Arnold Networks ([Aghaei, 2024, arXiv:2406.14495](https://arxiv.org/abs/2406.14495)), the Mittag-Leffler function $E_{\alpha,1}(-z)$, and non-Markovian fractional memory kernels into Transformer attention ([Yang & Wang, 2025, arXiv:2409.10594](https://arxiv.org/abs/2409.10594)) as an open fundamental challenge. In this repository, I provide the first complete analytical and differentiable PyTorch implementation of this class of architectures.
2. **Ecological Biophysics & Soil Diagnostics:** Polynomial KAN baselines (`ChebPIKAN`, based on Chebyshev orthogonal expansions; [Aghaei, 2024, arXiv:2406.07456](https://arxiv.org/abs/2406.07456)) diverge catastrophically during blind long-horizon extrapolation ($t \in [150, 365]\text{ days}$, $R^2 = -8.13$), whereas classical Bayesian ODE solvers (`MAP-ODE` with Soil Atlas L2 prior) ignore micropore trapping memory ($\alpha < 1$) and fail to reconstruct unmeasured contaminant load $\tilde{P}(t)$ ($R^2 = -0.191$). Simultaneously, static integral soil biological state indices (ИБСП; Kazeev, Kolesnikov, & Valkov, 2003/2016) suffer from false alarms on naturally nutrient-poor soils and false negatives after rainfall or fertilization pulses. I resolve both limitations by introducing a dynamic memory-aware index $\text{ИБСП}_\alpha(t)$ coupled with a sparse biokinetic Mixture-of-Experts (`PhysicsExpertBank`) that separates smooth Monod nutrient limitation from Haldane substrate-inhibition hysteresis.

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
├── fig4_soil_atlas_passport.png       # Fig. 4: Soil Atlas KL passportization, alpha recovery & Val/Test ROC audit
├── fig5_multkan_symbolic_discovery.png# Fig. 5: MultKAN power-law discovery, MoE routing & symbolic formula audit
├── fig6_ablation_and_noise_robustness.png # Fig. 6: 3-seed ablation study, 40% noise robustness & OOD misspecification
└── fig_dry_run_36pots.png             # Fig. 7: Technical dry-run verification on 36 microcosm vessels (0..60 d)
```

## Mathematical Architecture (`Frac-ML-KAT-MoE`)

I formulate soil self-purification and biokinetic recovery in dimensionless half-inhibition units $\tilde{P}(t) = P(t) / K_{\text{tox}}$ ($K_{\text{tox}} \equiv 1.0$, $K_i \equiv 1.0$, $\text{EC}_{50}$ units) under the coupled Caputo fractional dynamical system of order $\alpha \in (0, 1]$:

$$D_t^\alpha \tilde{P}(t) = -\left(k_p + \frac{k_{\text{deg}} \Phi_{\text{env}}(t)\, M(t)}{K_s + \tilde{P}(t) + \tilde{P}(t)^2}\right)\tilde{P}(t)$$

$$D_t^\alpha M(t) = r_m \Phi_{\text{env}}(t)\, M(t)\left(1 - \frac{M(t)}{K_m}\right) - \gamma_p \frac{\tilde{P}(t)\, M(t)}{1 + \tilde{P}(t)} + \mu_{\text{sub}} \frac{\tilde{P}(t)\, E(t)}{K_s + \tilde{P}(t) + \tilde{P}(t)^2}$$

$$D_t^\alpha E(t) = \beta_E \frac{M(t)}{1 + \tilde{P}(t)^2} - \delta_E E(t)$$

Because direct chemical assays of $\tilde{P}(t)$ are unavailable during routine field monitoring, I anchor the initial contaminant shock $\tilde{P}(0)$ without data leakage directly from the stationary boundary face of the enzyme-to-biomass ratio at the first observation $t_1 \to 0$:

$$\frac{E(t_1)}{M(t_1)} = \frac{E_{\text{ref}} / M_{\text{ref}}}{1 + \tilde{P}(0)^2} \implies \tilde{P}_{\text{boundary}}(0) = \sqrt{\max\!\left(0,\, \frac{1.20}{E(t_1)/M(t_1)} - 1\right)}$$

### 1. Causal Fractional Memory Attention (`FractionalMemoryAttention`)

Given sparse, irregularly sampled observations $\{t_j, \mathbf{x}_j\}_{j=1}^N$ ($N \in [6, 8]$, up to 50% missingness, 25% multiplicative noise) with binary availability mask $\mathbf{m}_j \in \{0,1\}^5$ and rainfall indicator $r_j \in \{0,1\}$, I modulate causal self-attention logits with the singular Abel–Riemann–Liouville fractional integral kernel:

$$\mathcal{A}_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}} - \gamma_h (1 - \alpha)\ln(t_i - t_j + \varepsilon) - \ln\Gamma(\alpha), \quad t_i \ge t_j$$

I enforce strict **Weight Tying** of the predicted fractional order $\hat{\alpha} \in [0.40, 0.92]$ across the attention kernel, the Soil Atlas module, the Padé Mittag-Leffler spectral trunk, and the fractional health index $\text{ИБСП}_\alpha(t)$.

### 2. Differentiable Global Padé `[3/4]` Mittag-Leffler Kernel (`MittagLefflerPade34`)

To eliminate both the infinite power-series instability at large arguments and the polynomial divergence of Chebyshev/Jacobi KANs ([Liu et al., 2024, arXiv:2404.19756](https://arxiv.org/abs/2404.19756); [Abueidda et al., 2025, DOI: 10.1016/j.cma.2024.117699](https://doi.org/10.1016/j.cma.2024.117699)), I construct a rational `[3/4]` Padé approximant for $E_{\alpha,1}(-z)$ on $z \in [0, \infty)$:

$$E_{\alpha,1}(-z) \approx R_{[3/4]}^{(\alpha)}(z) = \frac{1 + p_1 z + p_2 z^2 + p_3 z^3}{1 + q_1 z + q_2 z^2 + q_3 z^3 + q_4 z^4}$$

Matching the Maclaurin expansion $1 - \frac{z}{\Gamma(1+\alpha)} + \frac{z^2}{\Gamma(1+2\alpha)} - \dots$ as $z \to 0$ and the Poincaré power-law tail $\frac{z^{-1}}{\Gamma(1-\alpha)} - \frac{z^{-2}}{\Gamma(1-2\alpha)}$ as $z \to \infty$, I apply Euler's reflection identity $\frac{1}{\Gamma(1 - k\alpha)} = \frac{\Gamma(k\alpha)\sin(k\pi\alpha)}{\pi}$ to remove the Gamma pole at $\alpha = 0.50$ and solve the resulting $5\times 5$ linear system analytically in `float64` via `torch.linalg.solve`, with a $C^1$-smooth transition to $\exp(-z)$ on $\alpha \in (0.90, 1.0]$. Across the entire working domain, the denominator satisfies $Q_4(z) \ge 1.001 > 1$, guaranteeing complete absence of positive real poles.

### 3. `MultKAN` Symbolic Discovery & `PhysicsExpertBank` (Sparse Top-2 MoE)

- **Multiplicative KAN Node (`MultKANEnvModule`):** Following [Liu et al. (2024, arXiv:2408.10205)](https://arxiv.org/abs/2408.10205), I combine a Shelford pH tolerance bell with a log-exponential power-law discovery branch:

$$\Phi_{\text{KAN}}(\text{pH}, \text{NO}_3, W) = \exp\!\left(-\beta_{\text{pH}}(\text{pH} - \text{pH}_{\text{opt}})^2\right) \cdot \sum_{b=1}^2 w_b \exp\!\left(\mathbf{W}_{\text{power}, b} \ln(\mathbf{e}_{\text{env}})\right)$$

- **Biokinetic Mixture-of-Experts (`PhysicsExpertBank`):** I route latent states and curve drop markers $[\Phi_{\text{KAN}}, \text{pH}, \Delta E_{\text{drop}}, \Delta M_{\text{drop}}, \tilde{P}_{\text{boundary}}(0)]$ through a temperature-annealed ($\tau: 2.0 \to 0.5$) Top-2 sparse gate across 4 mechanisms:
  - **Expert 0 (Haldane Inhibition):** $\tilde{P} / (K_s + \tilde{P} + \tilde{P}^2)$ for high-load substrate inhibition;
  - **Expert 1 (Monod Saturation):** $\tilde{P} / (K_s + \tilde{P})$ for moderate bio-assimilation;
  - **Expert 2 (Abiotic / Clean Homeostasis):** $-k_p \tilde{P}$ for uncontaminated background soils;
  - **Expert 3 (Acidic / Osmotic Stress):** $M / (1 + \tilde{P}^2)$ under low-pH peat or saline stress.

### 4. Dynamic Fractional Soil Health Index ($\text{ИБСП}_\alpha(t)$)

To prevent false alarms on low-capacity sandy soils (`Arch_5`, $K_m = 0.44$) and capture long-tail micropore memory, I modernize the classical static index of Kazeev & Kolesnikov into a self-normalized fractional functional:

$$\text{ИБСП}_\alpha(t) = 100 \cdot \exp\!\left(-\left[0.35\,\left|\ln\frac{M(t)}{M_{\text{ref}}}\right| + 0.40\,\max\!\left(0, 1 - \frac{E(t)}{E_{\text{ref}}}\right) + \lambda_{\text{mem}}(1 - \alpha)\tilde{P}_{\text{bio}}(t)\right]\right)$$

## Summary of Empirical Benchmarks (Figures 1–7)

| Evaluation Dimension | Metric / Regime | **TerraKAN (Mine)** | `MAP-ODE` (Atlas Prior) | `Simple-ML-MLP` | `ChebPIKAN` (Poly KAN) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Hidden Load $\tilde{P}(t)$ Extrapolation** | $R^2 \pm \text{CI}_{95\%}$ ($150 \to 365\text{ d}$) | **$0.908 \pm 0.019$** | $-0.191 \pm 0.225$ | $0.702 \pm 0.040$ | $-8.132 \pm 3.014$ |
| **Hidden Load $\tilde{P}(t)$ Error** | $\text{RMSE}$ ($\text{EC}_{50}$, $150 \to 365\text{ d}$) | **$0.181$** | $0.651$ | $0.326$ | $1.804$ |
| **Microbial Biomass $M(t)$** | $R^2 \pm \text{CI}_{95\%}$ ($150 \to 365\text{ d}$) | $0.894 \pm 0.015$ | **$0.900 \pm 0.019$** | $0.900 \pm 0.016$ | $-2.173 \pm 0.586$ |
| **Enzyme Activity $E(t)$** | $R^2 \pm \text{CI}_{95\%}$ ($150 \to 365\text{ d}$) | **$0.910 \pm 0.013$** | $0.891 \pm 0.020$ | $0.888 \pm 0.015$ | $-1.223 \pm 0.305$ |
| **Health Index $\text{ИБСП}_\alpha(t)$** | $R^2 \pm \text{CI}_{95\%}$ ($150 \to 365\text{ d}$) | **$0.886 \pm 0.016$** | $0.769 \pm 0.045$ | $0.742 \pm 0.032$ | $-0.909 \pm 0.341$ |
| **Health Index $\text{ИБСП}_\alpha(t)$ RMSE** | $\text{RMSE}$ ($\%$, $150 \to 365\text{ d}$) | **$6.88\%$** | $9.81\%$ | $10.37\%$ | $28.23\%$ |
| **Two-Compartment + Weibull (OOD)** | Zero-shot $\text{RMSE}$ ($\%$, $150 \to 365\text{ d}$) | **$4.14\%$ ($R^2=0.847$)** | $6.30\%$ ($R^2=0.644$) | — | — |
| **Out-of-Atlas $\alpha \in [0.35, 0.96]$** | Zero-shot $\text{RMSE}$ ($\%$, $150 \to 365\text{ d}$) | **$8.11\%$ ($R^2=0.905$)** | $13.08\%$ ($R^2=0.752$) | — | — |
| **Inference Latency per Field** | Wall-clock time (CPU) | **$0.50\text{ ms}$ ($10.1\times$ faster)** | $5.01\text{ ms}$ | $0.38\text{ ms}$ | $0.45\text{ ms}$ |

### Key Verification Milestones

- **Kernel Verification (`fig1_pade_verification.png`):** Against the exact Oberhettinger contour integral on $z \in [10^{-3}, 10^3]$, `MittagLefflerPade34` achieves $\max|\Delta| = 7.67\times 10^{-7}$ at $\alpha=0.30$, $3.85\times 10^{-5}$ at $\alpha=0.50$ (verified against $\text{erfcx}(z)$), and $1.12\times 10^{-3}$ at $\alpha=0.78$, with $\min Q_4(z) \ge 1.0012$, passing `torch.autograd.gradcheck` (`float64`) without `NaN` and matching the Diethelm ABM Caputo solver ([Diethelm et al., 2002](https://doi.org/10.1023/A:1016592219341)) within $0.00331\% < 0.5\%$.
- **Soil Atlas Passportization & Anomaly Detector (`fig4_soil_atlas_passport.png`):** Across $N=500$ test fields, `SoilAtlasModule` achieves **Macro-Accuracy $= 88.4\%$** and **Macro-F1 $= 0.890$**, recovers fractional memory order $\alpha$ with **$R^2 = 0.9248$** ($\text{RMSE} = 0.0240$), and maintains a stable Youden threshold $d_{\min}^* = 0.290$ across 5 stratified splits (`seeds = [42, 7, 13, 99, 2026]`, mean $|\Delta\text{TPR}| = 1.85\% \pm 0.95\% < 4.0\%$).
- **Neuro-Symbolic Discovery (`fig5_multkan_symbolic_discovery.png`):** From 25% noisy field data, `MultKAN` discovers nitrate exponent $0.345$ (true $0.350$), moisture exponent $0.394$ (true $0.400$), and Shelford pH bell $\exp(-2.36(\text{pH}-0.66)^2)$ (true $\exp(-2.50(\text{pH}-0.65)^2)$). Unpacked analytical formulas match raw tensor outputs to $9.96\times 10^{-13}$ in `float64`.
- **Microcosm Dry-Run Pipeline (`fig_dry_run_36pots.png`):** On 36 vessels across 3 soil series (`S1_sand`, `S2_loam`, `S3_peat`) with per-series control normalization and zero leakage of `is_control_G0` into diagnosis, all 6 `G0` controls satisfy $\tilde{P}_{\text{bio}}(0) \le 0.001 < 0.5\,\text{EC}_{50}$ and $\text{ИБСП}(0) \in [92.1\%, 99.5\%] > 92\%$, all 8 clean nutrient-poor `G1` vessels are classified as substrate-depleted without false contamination alarms ($\max \tilde{P}_{\text{bio}}(0) = 0.016$, $\max d_{\min} = 0.148 \le 0.290$), and all 12 high-load `G3` vessels trigger $d_{\min} \ge 0.593 > 0.290$.

## Quickstart & Reproducibility

### 1. Run Mandatory Numerical & Autograd Unit Tests

```bash
python3 run_tests.py
```

### 2. Run Full Training, 1000x Bootstrap Evaluation & Figure Generation

```bash
python3 train_eval.py
python3 visualize.py
```

### 3. Run Wet-Lab / Microcosm CSV Journal Inference (`terrakan_v4_frozen.pt`)

To process a 36-vessel microcosm log (days `0, 3, 7, 14, 21, 28` -> blind forecast to day `60`):

```bash
python3 process_buckets_csv.py
```

- Print `blank_lab_journal_36pots.csv` for recording raw measurements in the laboratory (`catalase_O2_ml` as proxy for $E$, `respiration_CO2_mg` as proxy for $M$, plus `pH`, `NO3_mg_l`, `moisture_pct`, `rain_flag`, `is_control_G0`, and optional `lab_toxin_ref_ec50`).
- Missing respiration measurements on intermediate days (left blank or `NaN` in CSV) are handled automatically via the binary observation mask `avail_mask`.

## References

1. Tu, W., Liu, X., Wang, H., Li, X., & Han, C. (2026). Fractional Calculus for Large Language Models: A Survey of Potentials, Mechanisms, and Challenges. *Fractal and Fractional*, 10(7), 454. [https://doi.org/10.3390/fractalfract10070454](https://doi.org/10.3390/fractalfract10070454)
2. Liu, Z., Wang, Y., Vaidya, S., et al. (2024). KAN: Kolmogorov-Arnold Networks. *arXiv preprint* [arXiv:2404.19756](https://arxiv.org/abs/2404.19756); and *KAN 2.0: Kolmogorov-Arnold Networks Meet Science*, [arXiv:2408.10205](https://arxiv.org/abs/2408.10205).
3. Yang, X., & Wang, X. (2025). Kolmogorov-Arnold Transformer. *ICLR 2025* / [arXiv:2409.10594](https://arxiv.org/abs/2409.10594).
4. Aghaei, A. A. (2024). rKAN: Rational Kolmogorov-Arnold Networks, [arXiv:2406.14495](https://arxiv.org/abs/2406.14495); and *fKAN: Fractional Orthogonal Kolmogorov-Arnold Networks*, [arXiv:2406.07456](https://arxiv.org/abs/2406.07456).
5. Abueidda, D. W., Pantidis, P., & Mobasher, M. E. (2025). DeepOKAN: Deep Operator Network Based on Kolmogorov Arnold Networks for Mechanics Problems. *Computer Methods in Applied Mechanics and Engineering*, 436, 117699. [https://doi.org/10.1016/j.cma.2024.117699](https://doi.org/10.1016/j.cma.2024.117699)
6. Yan, C., et al. (2026). Soil carbon fractions rather than enzyme stoichiometry govern greenhouse gas reactivity to microplastics and metal oxides in hydrologically dynamic soils. *Journal of Hazardous Materials*, 512, 142317.
7. Diethelm, K., Ford, N. J., & Freed, A. D. (2002). A predictor-corrector approach for the numerical solution of fractional differential equations. *Nonlinear Dynamics*, 29, 3–22. [https://doi.org/10.1023/A:1016592219341](https://doi.org/10.1023/A:1016592219341)
8. Казеев К. Ш., Колесников С. И., Вальков В. Ф. (2003/2016). *Биологическая диагностика и индикация почв: методология и методы исследований*. Ростов н/Д: Изд-во ЮФУ.
