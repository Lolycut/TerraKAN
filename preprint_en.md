# TerraKAN: Fractional Kolmogorov-Arnold Transformer with Padé Mittag-Leffler Basis and Biokinetic Mixture-of-Experts for Inverse Soil Bioindication under Sparse Noisy Regimes

**Author:** Ilya Kakhno ([ORCID: 0009-0005-5503-3269](https://orcid.org/0009-0005-5503-3269))  
**Affiliation:** Faculty of Biology, Belarusian State University, Minsk, Republic of Belarus  
**E-mail:** [Sunnywww1701@gmail.com](mailto:Sunnywww1701@gmail.com)  
**Code & Weights:** [GitHub](https://github.com/Lolycut/TerraKAN) (`v1.0.1`, `terrakan_v4_frozen.pt`)  
**Date:** November 12, 2026 | **Licenses:** CC-BY-4.0 (Preprint & Figures) & MIT (Code & Weights)

---

## Abstract

In this work, I introduce **TerraKAN (`Frac-ML-KAT-MoE`)**, an **amortized simulation-based inference** architecture that I designed, mathematically verified, and calibrated to solve the ill-posed inverse problem of soil bioindication and long-horizon self-purification forecasting from sparse (6–8 points), irregular (up to 50% missingness), and noisy (25–40% multiplicative error) enzymatic and respiratory observations without direct chemical assays of the contaminant at test time. Whereas polynomial Kolmogorov-Arnold Networks (`ChebPIKAN`) suffer from catastrophic divergence during blind extrapolation on $t \in [150, 365]\text{ days}$ ($R^2 = -8.13$) and classical integer-order Bayesian ODE solvers (`MAP-ODE`) fail to reconstruct unmeasured contaminant loads ($R^2 = -0.191$) due to ignoring non-Markovian micropore memory ($\alpha < 1$), I unify four analytical primitives into an end-to-end differentiable model: (1) a causal sequence encoder (`Frac-KAT`) with an Abel-type power-law memory attention kernel and strict weight tying of the inferred memory exponent $\hat{\alpha}$; (2) a pole-free global `[3/4]` Padé Mittag-Leffler spectral trunk (`MittagLefflerPade34`) matching exact Maclaurin ($z \to 0$) and Poincaré ($z \to \infty$) asymptotics with denominator $Q_4(z) \ge 1.000978$ across $\alpha \in [0.40, 0.90]$; (3) a log-exponential `MultKAN` symbolic discovery node paired with a sparse Top-2 biokinetic Mixture-of-Experts (`PhysicsExpertBank`) separating Haldane substrate inhibition from Monod nutrient limitation; and (4) a dynamic memory-aware soil health index $\text{IBSP}_\alpha(t)$. Across $N = 500$ held-out test scenarios on $t \in (150, 365]\text{ days}$, TerraKAN achieves $R^2 = 0.908 \pm 0.019$ ($\text{RMSE} = 0.181\,\text{EC}_{50}$) on the unmeasured contaminant load $\tilde{P}(t)$ and $R^2 = 0.886 \pm 0.016$ ($\text{RMSE} = 6.88\%$) on $\text{IBSP}_\alpha(t)$ at $0.50\text{ ms/field}$—outperforming both integer-order `MAP-ODE` ($10\times$ speedup) and a classical fractional-order `MAP-FDE` solver ($R^2 = 0.802$ on $\tilde{P}(t)$, $51.98\text{ ms/field}$, $104\times$ speedup). Finally, I perform a synthetic dry-run pipeline verification of the frozen checkpoint (`terrakan_v4_frozen.pt`) on a simulated 36-vessel microcosm protocol prior to wet-lab execution, demonstrating 100% separation of healthy controls, nutrient-poor clean soils, and contaminated regimes without control-flag leakage.

**Keywords:** Kolmogorov-Arnold Networks; Fractional Calculus; Mittag-Leffler Function; Padé Approximant; Amortized Simulation-Based Inference; Soil Bioindication; Haldane Kinetics.

---

## 1. Introduction & Gap Analysis

Natural soils subjected to xenobiotic, petroleum hydrocarbon, or heavy-metal loading exhibit long-term non-Markovian memory driven by anomalous subdiffusion and sorption–desorption trapping within heterogeneous organo-mineral micropores [6, 7]. Consequently, the relaxation of bioavailable contaminant load $\tilde{P}(t)$ obeys Caputo fractional differential equations of order $\alpha \in (0, 1)$ rather than memoryless exponential kinetics. In field practice, direct chemical quantification of $\tilde{P}(t)$ over multi-month horizons is prohibitively expensive; investigators typically collect only 6–8 sparse noisy observations (with up to 50% independent per-channel missingness) of microbial basal respiration $M(t)$ ($\text{CO}_2$ evolution) and extracellular enzymatic activity $E(t)$ (catalase $\text{O}_2$ displacement), interrupted by up to 50% missing entries and stochastic rainfall pulses [8].

Positioning my approach against the 2024–2026 SciML and ecological literature reveals four critical gaps that I address in this study:

1. **Extrapolation Divergence of Polynomial KANs:** Orthogonal polynomial bases (`ChebPIKAN` / `fKAN` [4]) fit observations inside $t \in [0, 150]\text{ days}$ ($R^2 = 0.918$), but their unbounded polynomial growth $\propto (t/T)^k$ induces catastrophic divergence during blind extrapolation on $t \in (150, 365]\text{ days}$, collapsing to $R^2 = -8.132$ ($\text{RMSE} = 1.804\,\text{EC}_{50}$) on $\tilde{P}(t)$.
2. **Failure of Classical Bayesian ODE/FDE Solvers:** Classical nonlinear least-squares solvers regularized toward Soil Atlas centroids (`MAP-ODE`) fit observed $M(t)$ ($R^2 = 0.900$) and $E(t)$ ($R^2 = 0.891$), but fail to reconstruct hidden $\tilde{P}(t)$ during extrapolation ($R^2 = -0.191 \pm 0.225$, $\text{RMSE} = 0.651\,\text{EC}_{50}$) because integer-order ODEs cannot capture power-law subdiffusion ($\alpha < 1$). Even a fractional-order `MAP-FDE` solver using the Diethelm ABM integrator requires $51.98\text{ ms/field}$ of iterative per-sample optimization and achieves lower accuracy on $\tilde{P}(t)$ ($R^2 = 0.802$) under 50% missingness.
3. **The Open Fractional Transformer–KAN Challenge:** While Yang & Wang [3] proposed the Kolmogorov-Arnold Transformer (KAT) and Aghaei [4] introduced rational Padé KANs (`rKAN`), Tu et al. [1] (July 2026) identified the integration of rational Padé bases, Mittag-Leffler eigenfunctions, and non-Markovian fractional kernels into attention architectures as an open fundamental challenge due to Gamma-function poles and autograd instability. Here, I present an analytical and differentiable realization of this architecture class.
4. **Ecological Ambiguity of Static IBSP Indices:** The classical static Integral Index of Biological State (IBSP / IIBS; Kazeev, Kolesnikov, & Valkov [6, 10]) averages raw enzymatic percentages relative to a fixed standard. Consequently, it triggers false alarms on naturally oligotrophic sandy soils (`Arch_5`, $K_m = 0.44$) and suffers from false negatives after moisture pulses because it neglects both Haldane hysteresis and the fractional memory penalty $(1 - \alpha)\tilde{P}(t)$.

---

## 2. Mathematical Architecture (`Frac-ML-KAT-MoE`)

### 2.1 Dimensionless Fractional Biokinetics, Soil Atlas, & Boundary Anchor

Because absolute contaminant concentration in mg/kg is unidentifiable without direct chemical input due to scale symmetry with inhibition constants, I formulate the coupled Caputo fractional system of order $\alpha \in (0, 1]$ in dimensionless half-inhibition units $\tilde{P}(t) = P(t)/K_{\text{tox}}$ ($K_{\text{tox}} \equiv 1.0$, $K_i \equiv 1.0$, $\text{EC}_{50}$ units):

$$D_t^\alpha \tilde{P}(t) = -\left(k_p + \frac{k_{\text{deg}}\Phi_{\text{env}}(t)\, M(t)}{K_s + \tilde{P}(t) + \tilde{P}(t)^2}\right)\tilde{P}(t) \qquad (1)$$

$$D_t^\alpha M(t) = r_m \Phi_{\text{env}}(t)\, M(t)\left(1 - \frac{M(t)}{K_m}\right) - \gamma_p \frac{\tilde{P}(t)\, M(t)}{1 + \tilde{P}(t)} + \mu_{\text{sub}} \frac{\tilde{P}(t)\, E(t)}{K_s + \tilde{P}(t) + \tilde{P}(t)^2} \qquad (2)$$

$$D_t^\alpha E(t) = \beta_E \frac{M(t)}{1 + \tilde{P}(t)^2} - \delta_E E(t) \qquad (3)$$

| Code | Soil Archetype | $\text{pH}_{\text{n}}$ | $\text{NO}_{3,\text{n}}$ | $W_{\text{n}}$ | $K_m$ | $\alpha$ | $k_p$ | $k_{\text{deg}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Arch_1` | Sod-Podzolic Loam | 0.58 | 0.55 | 0.60 | 1.00 | 0.75 | 0.018 | 0.085 |
| `Arch_2` | Peat-Bog Soil | 0.36 | 0.44 | 0.88 | 1.18 | 0.55 | 0.012 | 0.065 |
| `Arch_3` | Sod-Carbonate Soil | 0.68 | 0.80 | 0.70 | 1.38 | 0.86 | 0.026 | 0.125 |
| `Arch_4` | Technogenic Solonchak | 0.88 | 0.30 | 0.32 | 0.68 | 0.66 | 0.014 | 0.055 |
| `Arch_5` | Sandy Forest Soil | 0.50 | 0.22 | 0.36 | 0.44 | 0.80 | 0.022 | 0.070 |

To anchor $\tilde{P}(0)$ at test time without chemical measurements, I exploit the stationary boundary face of Eq. (3) at $t_1 \to 0$. In the simulator, $\beta_E / \delta_E = 0.09/0.075 = 1.20$, so $E_{\text{ref}}/M_{\text{ref}} \equiv 1.20$:

$$\frac{E(t_1)}{M(t_1)} = \frac{E_{\text{ref}}/M_{\text{ref}}}{1 + \tilde{P}(0)^2} \implies \tilde{P}_{\text{boundary}}(0) = \sqrt{\max\!\left(0,\, \frac{1.20}{E(t_1)/M(t_1)} - 1\right)} \qquad (4)$$

When deploying on real laboratory or field soils whose native $E_{\text{ref}}/M_{\text{ref}}$ ratio differs from $1.20$, per-series control normalization against uncontaminated day-0 controls (`is_control_G0 == 1` within each `soil_series`) rescales $M(t)$ and $E(t)$ into $E_{\text{norm}}(0)/M_{\text{norm}}(0) \approx 1.20$ before inference.

### 2.2 Encoder `Frac-KAT` & Fractional Weight Tying

Given sparse observations $\mathbf{v}_j = [\text{pH}_{\text{norm}}, \text{NO}_{3,\text{norm}}, W_{\text{norm}}, M_{\text{obs}}, E_{\text{obs}}]^\top \in \mathbb{R}^5$ at irregular timestamps $0 \le t_1 < \dots < t_N \le 150\text{ d}$ with binary availability mask $\mathbf{m}_j \in \{0,1\}^5$ and rainfall flag $r_j \in \{0,1\}$, I construct the 12-dimensional token $\mathbf{u}_j = [\mathbf{v}_j \odot \mathbf{m}_j + 0.35(\mathbf{v}_j - \mathbf{v}_1)\odot \mathbf{m}_j, \mathbf{m}_j, \ln(1+t_j), r_j]^\top$. A single-layer causal Transformer encoder ($d_{\text{model}}=48$, 1 attention head, GELU input projection, LayerNorm residuals) aggregates the sequence; the Kolmogorov-Arnold functional representations are placed in the downstream `MultKANEnvModule` and `MLKANTrunk` rather than replacing attention projections with B-splines. Causal self-attention logits are modulated with a logarithmic Abel-type decay term ($\varepsilon = 1.0$ via $\ln(1+\Delta t_{ij})$):

$$\mathcal{A}_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_{\text{model}}}} - \text{softplus}(\gamma_h)(1 - \hat{\alpha}_i)\ln(1 + \max(0, t_i - t_j)), \quad t_i \ge t_j \qquad (5)$$

The encoder predicts $\hat{\alpha} = 0.40 + 0.52\,\sigma(z_\alpha) \in [0.40, 0.92]$ and epistemic uncertainty $\hat{\sigma}_\alpha = 0.01 + 0.15\,\text{softplus}(z_\sigma)$ from the causal prefix mean. Strict **Weight Tying** enforces that the same scalar $\hat{\alpha}$ governs Eq. (5), the Soil Atlas coordinate, the Padé Mittag-Leffler trunk, and $\text{IBSP}_{\hat{\alpha}}(t)$.

### 2.3 Kernel `MittagLefflerPade34`: Pole-Free `[3/4]` Padé Basis

To approximate $E_{\alpha,1}(-z)$ globally on $z \in [0, \infty)$ with exact gradients $\partial E_{\alpha,1}/\partial\alpha$, I construct the rational Padé `[3/4]` approximant:

$$E_{\alpha,1}(-z) \approx R_{[3/4]}^{(\alpha)}(z) = \frac{1 + p_1 z + p_2 z^2 + p_3 z^3}{1 + q_1 z + q_2 z^2 + q_3 z^3 + q_4 z^4} \qquad (6)$$

Matching the Maclaurin series as $z \to 0$ and Poincaré asymptotics as $z \to \infty$ using Euler's reflection identity $m_k = \frac{1}{\Gamma(1-k\alpha)} = \frac{\Gamma(k\alpha)\sin(k\pi\alpha)}{\pi}$ ($k=1,2$) eliminates the Gamma pole at $\alpha = 0.50$. Solving the closed-form $5\times 5$ linear system in `float64` via `torch.linalg.solve` and applying a $C^1$-smoothstep bridge on $\alpha \in (0.90, 1.0]$ toward $e^{-z}$, dense numerical verification across 61 values of $\alpha \in [0.40, 0.90]$ and 250 log-spaced nodes $z \in [10^{-3}, 10^3]$ confirms $Q_4(z) \ge 1.000978 > 1$ across the entire operating regime.

### 2.4 `PhysicsExpertBank` (Sparse Top-2 MoE) & `MultKAN`

`MultKANEnvModule` takes $\mathbf{e} = [\text{pH}_{\text{norm}}, \text{NO}_{3,\text{norm}}, W_{\text{norm}}]^\top \in \mathbb{R}^3$ and combines a Shelford pH tolerance bell with a two-branch log-exponential power layer [2]:

$$\Phi_{\text{KAN}}(\mathbf{e}) = e^{-\beta_{\text{pH}}(\text{pH}_{\text{norm}} - \text{pH}_{\text{opt}})^2} \sum_{b=1}^2 w_b \exp\!\left(\sum_{c=1}^3 W_{\text{power}, b, c}\ln e_c\right) \qquad (8)$$

Next, `PhysicsExpertBank` routes $[\mathbf{h}_{\text{final}}, \Phi_{\text{KAN}}, \text{pH}_{\text{norm}}, \Delta E_{\text{drop}}, \Delta M_{\text{drop}}, \tilde{P}_{\text{boundary}}(0)] \in \mathbb{R}^{53}$ through a temperature-annealed Top-2 sparse gate ($\tau: 2.0 \to 0.5$) across $K_{\text{exp}}=4$ physical experts (Haldane inhibition, Monod saturation, Abiotic clean background, Acidic/osmotic stress) with load-balancing loss $\mathcal{L}_{\text{balance}}$, producing $\mathbf{h}_{\text{ctx}}$.

### 2.5 Two-Stage Inference, Explicit $M(t)$ / $E(t)$ Expansions, & $\text{IBSP}_\alpha(t)$

`SoilAtlasModule` assigns each field to the 5 Soil Atlas archetypes via KL-divergence against Dirichlet mixture weights and computes $d_{\min} = 0.25\,d_{\text{raw}} + 0.75\tanh(0.42\tilde{P}_{\text{boundary}}(0))$, with optimal Youden threshold $d_{\min}^* = 0.290$ frozen on `Val (500)`. `MLKANTrunk` maps $\mathbf{h}_{\text{ctx}}$ via $\mathbf{W}_{\text{head}}\mathbf{h}_{\text{ctx}} + \mathbf{b}_{\text{head}} \in \mathbb{R}^{18}$ to decay rates $\lambda_{1..3}$, mode amplitudes $(c_P, c_M, c_E) \in \mathbb{R}^{3\times 3}$, and asymptotic equilibria $M_\infty, E_\infty$:

$$\tilde{P}_{\text{bio}}(t) = g(d_{\min})\sum_{k=1}^3 c_{P,k} E_{\hat{\alpha},1}(-\lambda_k t^{\hat{\alpha}}), \quad M(t) = M_\infty + \sum_{k=1}^3 c_{M,k} E_{\hat{\alpha},1}(-\lambda_k t^{\hat{\alpha}}), \quad E(t) = E_\infty + \sum_{k=1}^3 c_{E,k} E_{\hat{\alpha},1}(-\lambda_k t^{\hat{\alpha}})$$

$$\text{IBSP}_\alpha(t) = 100 \exp\!\left(-\left[0.35\left|\ln\frac{M(t)}{M_\infty}\right| + 0.40\max\!\left(0, 1 - \frac{E(t)}{E_\infty}\right) + \lambda_{\text{mem}}(1-\hat{\alpha})\tilde{P}_{\text{bio}}(t)\right]\right), \quad \lambda_{\text{mem}} = 0.25$$

### 2.6 Training Protocol, Composite Loss, & Amortized Simulation-Based Inference

TerraKAN has **21,036 trainable parameters** and is trained strictly under an **amortized simulation-based inference** paradigm on $N_{\text{train}} = 3,000$ synthetic Soil Atlas scenarios ($N_{\text{val}} = 500$, $N_{\text{test}} = 500$). I optimize the model for 160 epochs on CPU in `float64` using `Adam` ($\text{lr} = 2.5\times 10^{-3}$, weight decay $10^{-5}$, `CosineAnnealingLR` with $\eta_{\min} = 10^{-5}$, batch size 64, gradient-norm clipping at 1.0):

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + 0.15\,\mathcal{L}_{\text{phys}} + 0.15\,\mathcal{L}_{\text{atlas}} + 0.05\,\mathcal{L}_{\text{kan}} + 0.02\,\mathcal{L}_{L1} + 0.05\,\mathcal{L}_{\text{balance}}$$

At epochs $\{8, 24, 55, 95, 135, 160\}$, closed-form ridge regression aligns the linear heads against ground-truth simulator targets $(\tilde{P}_{\text{true}}, M_{\text{true}}, E_{\text{true}}, \alpha_{\text{true}})$ on `Train (3000)`. At test time, inference operates purely from sparse $(M, E, \text{env})$ observations with **zero** test-time optimization and **zero** access to $\tilde{P}_{\text{true}}$ or $\alpha_{\text{true}}$.

---

## 3. Numerical Verification & Empirical Benchmarks

| Model Architecture | Params | Time (ms/field) | Hidden $\tilde{P}(t)$ $R^2$ (RMSE) | Biomass $M(t)$ $R^2$ (RMSE) | Enzyme $E(t)$ $R^2$ (RMSE) | Index $\text{IBSP}_\alpha(t)$ $R^2$ (RMSE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TerraKAN (Mine)** | 21,036 | **0.50** (104x vs FDE) | **$0.908 \pm 0.019$ (0.181)** | $0.894 \pm 0.015$ (0.114) | $0.910 \pm 0.013$ (0.166) | **$0.886 \pm 0.016$ (6.88%)** |
| `MAP-FDE` (Diethelm ABM + Prior) | 5 | 51.98 | $0.802 \pm 0.037$ (0.265) | **$0.934 \pm 0.012$ (0.090)** | **$0.928 \pm 0.015$ (0.149)** | $0.849 \pm 0.028$ (7.93%) |
| `MAP-ODE` (Integer ODE + Prior) | 5 | 5.01 | $-0.191 \pm 0.225$ (0.651) | $0.900 \pm 0.019$ (0.111) | $0.891 \pm 0.020$ (0.183) | $0.769 \pm 0.045$ (9.81%) |
| `Simple-ML-MLP` (No Abel/MoE) | 3,763 | 0.43 | $0.702 \pm 0.040$ (0.326) | $0.900 \pm 0.016$ (0.111) | $0.888 \pm 0.015$ (0.185) | $0.742 \pm 0.032$ (10.37%) |
| `ChebPIKAN` (Poly $T_0\dots T_5$) | 21,036 | 0.57 | $-8.132 \pm 3.014$ (1.804) | $-2.173 \pm 0.586$ (0.626) | $-1.223 \pm 0.305$ (0.827) | $-0.909 \pm 0.341$ (28.23%) |

---

## 4. Limitations & Inverse-Crime Considerations

1. **Shared Structural Priors ("Inverse Crime" Risk):** Although TerraKAN generalizes zero-shot to Two-Compartment Sorption + Weibull dynamics and Out-of-Atlas $\alpha$ (Fig. 6A), its internal biokinetic priors belong to the same equation families used in the Soil Atlas simulator. True out-of-distribution robustness must be verified on real laboratory and field soils exhibiting unmodeled phenomena (secondary toxic metabolites, N/P co-limitation, microbial succession).
2. **Synthetic Dry-Run vs. Wet-Lab Data:** The 36-vessel microcosm validation in Fig. 7 is a synthetic dry-run verification of the CSV preprocessing and inference pipeline prior to wet-lab execution.
3. **Fixed Boundary Ratio Assumption:** Without per-series `G0` normalization, Eq. (4) assumes $E_{\text{ref}}/M_{\text{ref}} \approx 1.20$; soils with divergent baseline ratios require control-group normalization.

---

## References

1. Tu, W., Liu, X., Wang, H., Li, X., & Han, C. (2026). Fractional Calculus for Large Language Models: A Survey of Potentials, Mechanisms, and Challenges. *Fractal and Fractional*, 10(7), 454. https://doi.org/10.3390/fractalfract10070454
2. Liu, Z., Wang, Y., Vaidya, S., et al. (2024). KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756.
3. Liu, Z., Ma, P., Wang, Y., Matusik, W., & Tegmark, M. (2024). KAN 2.0: Kolmogorov-Arnold Networks Meet Science. arXiv:2408.10205.
4. Yang, X., & Wang, X. (2025). Kolmogorov-Arnold Transformer. ICLR 2025 / arXiv:2409.10594.
5. Aghaei, A. A. (2024). rKAN: Rational Kolmogorov-Arnold Networks (arXiv:2406.14495); fKAN: Fractional Kolmogorov-Arnold Networks with trainable Jacobi basis functions (arXiv:2406.07456).
6. Abueidda, D. W., Pantidis, P., & Mobasher, M. E. (2025). DeepOKAN: Deep Operator Network Based on Kolmogorov Arnold Networks for Mechanics Problems. *CMAME*, 436, 117699. https://doi.org/10.1016/j.cma.2024.117699
7. Kolesnikov, S., Timoshenko, A., Varduni, V., et al. (2021). Influence of Silver Nanoparticles on the Biological Indicators of Haplic Chernozem. *Nanomaterials* / PMC8161183.
8. Minenko, V., et al. (2024). Enzymatic Diagnostics of Soil Health of the European Part of Russia. *Soil Systems*, 8(3), 76. MDPI:2571-8789/8/3/76
9. Wyszkowska, J., et al. (2024). The Potential for Restoring the Activity of Oxidoreductases in Soils Contaminated with Petroleum Products. *Applied Sciences*, 14(9), 3591. MDPI:2076-3417/14/9/3591
10. Diethelm, K., Ford, N. J., & Freed, A. D. (2002). A predictor-corrector approach for the numerical solution of fractional differential equations. *Nonlinear Dynamics*, 29, 3–22. https://doi.org/10.1023/A:1016592219341
11. Kazeev, K. Sh., Kolesnikov, S. I., & Valkov, V. F. (2003/2016). *Biological Diagnostics and Indication of Soils: Methodology and Research Methods* [in Russian]. Rostov-on-Don: Southern Federal University Press.
