# DeepXDE Neural Operator Extensions

This module extends DeepXDE operator classes to support analytic sensitivity propagation, latent Gram matrix decomposition, and asymmetric quantile loss functions.

---

## Architectural Modules

### 1. `jacobian_deeponet_softplus.py`
The primary architecture used throughout the study.
- **Formulation:** Evaluates prediction intervals as:
  $$y_{lower} = y - \mathrm{softplus}(w_{lower}) \cdot \log(1 + s / \bar{s}) \cdot \sigma$$
  $$y_{upper} = y + \mathrm{softplus}(w_{upper}) \cdot \log(1 + s / \bar{s}) \cdot \sigma$$
  where $s(x) = \sqrt{\mathrm{diag}(T(x) G T(x)^T)}$ is the exact pointwise standard deviation derived from the latent Gram matrix $G = J_{branch} J_{branch}^T$.
- **Scale Normalization:** Supports both reference scale normalisation (via `unc_ref`, calibrated deterministically on the training partition) and exponential moving average tracking (via `scale_mode="ema"`).
- **Asymmetric Pinball Loss:** Jointly trains central predictions (MSE) and upper/lower quantile bounds at level alpha = 0.10.

### 2. `vanilla_pinball_deeponet.py`
Controlled ablation baseline model.
- **Identical Backbone:** Shares the identical branch network, trunk network, activations, layer widths, and scalar parameters (w_lower, w_upper) as the Jacobian-DeepONet.
- **Constant Width:** Omits the Jacobian spatial sensitivity prior, predicting uniform bands across space.
- **Scientific Purpose:** Isolates whether spatial interval quality stems from the physical Jacobian prior or merely from end-to-end Pinball optimization.

### 3. `quantile_deeponet.py`
Spatial quantile regression baselines without Jacobian computation:
- **`quantile_x`**: Prediction interval width depends solely on spatial coordinates $x$ through an unconditioned trunk network head.
- **`quantile_ux`**: Conditional quantile regression where interval widths adapt dynamically across both input instances $u$ and spatial evaluation points $x$ through split branch heads.

### 4. `deterministic_deeponet.py`
Clean Cartesian DeepONet baseline trained purely with Mean Squared Error (MSE), serving as the foundation for post-hoc conformal estimation and Monte Carlo stochastic sampling.

### 5. `jacobian_quantile_deeponet.py`
Hybrid architectures combining the analytic Jacobian prior with high-capacity parameterized quantile heads, serving as negative ablation controls in architectural sweeps.
