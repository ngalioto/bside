# Design notes: efficiency, architecture, and usability

This document records design choices made during the `feat/broad-upgrade` overhaul. It is meant for future development — not end-user documentation. `docs/index.md` and `docs/algorithms.md` are the user-facing summaries; this file explains **why** things look the way they do, where we got burned, and what not to "simplify" away.

---

## Guiding principle: algorithms dictate structure

State-space filtering and system ID are not ordinary CRUD apps. A refactor that is "cleaner" in OOP terms can silently change **the linear algebra done per timestep** — and in long sequences, redundant Cholesky factors or dense covariance materializations dominate runtime.

When reviewing changes, ask:

1. **Does this path materialize a dense covariance when a Cholesky factor already exists?**
2. **Does the smoother re-use cross-covariances from the predict step, or recompute `A Σ Aᵀ`?**
3. **Does resampling use log-space weights (`logsumexp`) or raw weights that underflow?**
4. **Does mutating `init_dist` in-place leak back to the caller?**

Usability wins when defaults match textbook algorithm behaviour; complexity is acceptable when it mirrors the math; efficiency is non-negotiable on hot paths.

---

## Module boundaries (post-refactor)

### `SSM` — dynamics + observations only

`SSM` is a thin `nn.Module` container for `dynamics` and `observations`. It exposes simulation helpers (`predict`, `measure`) and a single `update(params)` that forwards to both submodels.

**Moved out:** encoder, `num_y_hist` / `num_u_hist`, multi-shooting `forward` / `_loss` / `fit`. Those belong to system identification, not to "run a filter on a known model."

**Why:** Filters and MCMC posteriors should work with a plain `SSM` without dragging in training-loop state. The original conflation caused double `update(params)` calls (encoder + dynamics) and made it unclear which object owned learnable parameters.

### `SubspaceEncoder(SSM)` — multi-shooting training

Inherits simulation from `SSM` and adds:

- `encoder` model and history lengths (`num_y_hist`, `num_u_hist`)
- `forward(data)` — batched multi-shooting rollout with trajectory masking
- `_loss` / `fit` — training loop

**Efficiency invariant retained:** `remaining_traj` mask inside the time loop skips finished trajectories instead of rolling out padding to `T` for every batch element.

### `bside.sysid` — composable ID primitives

| Class | Role |
|-------|------|
| `Posterior` | `params → log p(y\|θ) + log p(θ)` callable for `mcmc_samplers.Sampler` |
| `MultiShootingLoss` | Thin wrapper over `SubspaceEncoder._loss` for external optimizers |
| `EM` | Closed-form Shumway–Stoffer EM for linear-Gaussian SSMs |

**Posterior design:** No edits to `mcmc-samplers`. The adapter calls `filter.nlog_marginal_likelihood(..., params=params)` which internally runs `model.update(params)` once per evaluation. Works with `KalmanFilter`, `ParticleFilter`, and any filter that implements the same interface.

**`FilteringDistribution.log_prob` must return a 0-D tensor** for single-point evaluation — `mcmc_samplers` targets are scalars.

---

## Model composition (`AdditiveModel`)

### Before (broken)

`AdditiveModel.__init__` did `self.__dict__.update(model.__dict__)`, which:

- Bypassed PyTorch submodule registration for the inner model's parameters
- Broke autograd / optimizer visibility silently

### After (correct)

`self.model = model` with delegation of `forward`, `update`, and property access (`mat_x`, etc.). `LinearGaussianModel` and `NonlinearAdditiveModel` are single-inheritance subclasses of `AdditiveModel`.

**Do not revert** to dict-merging for "convenience."

---

## Covariance representations (`Matrix` / `PSDMatrix`)

### Lazy caches

`PSDMatrix` tracks `_up_to_date`, `_sqrt_up_to_date`, `_inv_up_to_date`. New code must:

- Read through `.val`, `.sqrt`, `.inv` properties (not stale private fields)
- Flip flags when mutating underlying storage
- Prefer `torch.cholesky_inverse` over generic `linalg.inv` for PSD matrices

`DiagonalMatrix` inverts **diagonals only** — never elementwise reciprocal on a dense mostly-zero matrix.

### When to use sqrt vs dense

| Operation | Prefer | Rationale |
|-----------|--------|-----------|
| Gaussian log-prob / Mahalanobis | Cholesky + `solve_triangular` | No explicit inverse |
| Square-root Kalman predict/update | QR on stacked `[A·L \| L_Q]` factors | Never form dense `A Σ Aᵀ` |
| RTS smoothing gain | Dense `P_pred` + cached `U` from predict | Smoothing reuses `U = Σ Aᵀ` from `kf_predict` |
| Sigma-point generation | Cholesky of filter covariance | UT/GH/Cubature need explicit points |

Users can construct `PSDMatrix(default_sqrt=..., sqrt_only=True)` when they only have a Cholesky factor.

---

## Filtering hot paths

### Adaptive Kalman gain (`kalman_gain`)

Picks among `Sinv @ U`, `linalg.solve`, or two-stage `solve_triangular` depending on which factor is fresh. **All linear-Gaussian update paths must call this helper** — do not open-code a new gain computation in a new filter variant.

### Single `U = Σ Aᵀ` in `kf_predict`

```python
U = model.mat_x @ dist.cov
dist_p.cov = U @ model.mat_x.T + Q   # reuses U
```

The cross-covariance `U` is returned when `crossCov=True` and consumed by:

- Joint Kalman update
- `RTSSmoother` smoothing gain (no extra matmul)

### EnKF predict

Always delegates to `model.sample` — the old `LinearGaussianModel` branch with `pass` was removed. `res_Y` is computed once and reused for both `P_Y` and cross-covariance `U`.

### Sigma points: regenerate vs reuse

**Default (`regenerate_points=True`):** UT/GH/Cubature predict objects re-form sigma points between dynamics and observation steps — standard UKF behaviour.

**Opt-in (`regenerate_points=False`):** Re-use propagated particles for the observation step (faster, slightly less accurate). Exposed on `UnscentedKalmanPredict`, `GaussHermitePredict`, `CubatureKalmanPredict`.

### `Filter.filter` and `init_dist` copying

**Plan originally said shallow `copy.copy`.** We switched to **`copy.deepcopy(init_dist)`** because `kalman_update` mutates `dist.cov` in place via PSDMatrix setters; with `y0=True` and a shallow copy, the user's original `init_dist` was corrupted.

History entries still use `deepcopy` per step. Only the working copy at the top of `filter()` needed the fix.

### Particle filter numerics

- Weights accumulated in **log space**; normalization via `torch.logsumexp`
- Per-step log marginal: `logsumexp(log_w) - log(N)`
- **Systematic resampling** is the default (`O(N)`, lower variance than multinomial)
- Observation likelihood uses `solve_triangular` on the noise Cholesky — same pattern as `FilteringDistribution.log_prob`

### Square-root Kalman filter

`SquareRootKalmanFilter` operates on Cholesky factors via QR (`srkf_predict` / `srkf_update`). Strictly equivalent to `KalmanFilter` in exact arithmetic; preferred when covariances are ill-conditioned.

`FilteringDistribution` accepts `sqrt_cov` directly so SR-KF never needs to densify `Σ` on the predict path.

---

## Smoothing

All smoothers consume a **forward filter history** (`List[FilteringDistribution]`) — they do not re-run the filter.

| Smoother | Dynamics requirement | Key efficiency note |
|----------|---------------------|---------------------|
| `RTSSmoother` | `LinearGaussianModel` | Reuses `U` from `kf_predict` |
| `UnscentedRTSSmoother` | `AdditiveModel` | UT cross-cov via `gaussian_quadrature` |
| `ParticleSmoother` | `AdditiveModel` (FFBS) | Vectorized `log p(x_{t+1}\|x_t)` with `solve_triangular` |

Smoothing gain `_smoothing_gain` uses Cholesky solve on `P_pred` when available.

---

## System identification

### EM (linear-Gaussian only)

E-step: `KalmanFilter` + inline RTS backward pass. Lag-one cross-covariances recovered from smoothing gains in the **same** backward sweep — no second pass.

M-step: closed-form Shumway–Stoffer updates for `A`, `C`, `Q`, `R`, `μ₀`, `P₀`. Mutates matrices on the existing `SSM` in place via `Matrix.val` setters.

**Nonlinear EM is explicitly out of scope** — use `Posterior` + gradient-based optimizer or MCMC.

### MultiShootingLoss

Thin adapter: `loss_fn(trajectories) → ssm._loss(T, trajectories, loss_fn)`.

History length is owned by the `SubspaceEncoder` instance (`num_y_hist`, `num_u_hist`), not duplicated on the loss wrapper — avoids conflicting configuration.

---

## DMD

Reduced-rank branch uses `output @ (V.T / S) @ U.T` — never forms dense `Σ⁻¹`.

`DMDc` has separate `@A_rom.setter` / `@B_rom.setter` (the old `@A.setter` aliasing bug duplicated setters).

---

## Data layer

### Current contract

`Data` assumes **`u` is aligned with `y` at the same time indices** — required for `Filter.filter` loops that index `data.u[t-1]` alongside `data.y[t]`.

### Deferred: asynchronous `y` and `u`

`Data` docstring still carries a TODO for observations and controls at different timesteps. This touches filtering loops, `SSM.measure`, and trajectory partitioning. **Do not half-implement** — either keep the aligned contract (current) or design an explicit index map / hold interpolation policy first.

`DataTrajectories.traj_lengths` setter caches `_max_length` / `_min_length` — new partitioning code must use the setter, not assign `traj_lengths` directly.

---

## Bugs that looked like refactors (don't repeat)

| Issue | Symptom | Root cause |
|-------|---------|------------|
| `Filter.filter(..., y0=True)` | Crash / wrong model | Undefined `t`; used `self.model` instead of `self.model.observations` |
| Shallow `init_dist` copy | User's prior corrupted after filter | In-place cov mutation in Kalman update |
| `AdditiveModel.__dict__.update` | Missing params in optimizer | Submodule not registered |
| `PSDMatrix` init check | Silent wrong behaviour | `torch.isclose` returns tensor, not bool |
| `DiagonalMatrix.compute_inv` | Infinities in off-diagonal | Elementwise reciprocal on dense matrix |
| `DMDc @A.setter` for `A_rom` | Wrong matrix updated | Setter aliasing |
| `srkf_update` missing `u` | Wrong observation prediction | `model(dist_x.mean, None)` |
| `FilteringDistribution` + `sqrt_cov` | Constructor crash | Required `cov` even when `sqrt_cov` provided |
| `log_prob` shape | MCMC target not scalar | Returned 1-D for single point |
| Test slicing `kf_means` vs `x_true` | False failures | Prior at `t=0` in filter history |

When "cleaning up" math-heavy code, **compare against analytical KF on a linear system** before merging.

---

## Testing philosophy

1. **Reference oracles on linear-Gaussian systems** — all Gaussian filters must match `KalmanFilter`; RTS must reduce uncertainty vs filter.
2. **Limit behaviour** — EnKF/PF converge to KF with large ensemble / many particles.
3. **Regression tests for fixed bugs** — `y0=True`, `IdentityModel` observations, deepcopy semantics.
4. **Optional smoke tests** — `Posterior` + `DelayedRejectionAdaptiveMetropolis` when `mcmc-samplers` is installed.

Property-based fuzzing is less valuable than closed-form linear-Gaussian checks.

---

## Explicit non-goals / deferred items

From the upgrade plan §8 and items not shipped in this branch:

| Item | Status | Notes |
|------|--------|-------|
| `SquareRootUnscentedKalmanFilter` | Deferred | SR-KF shipped; SR-UKF needs van der Merwe QR on UT points |
| Auxiliary particle filter | Deferred | Bootstrap PF + systematic/stratified/multinomial resampling shipped |
| Nonlinear EM (gradient M-step) | Deferred | Document EM as linear-only; use `Posterior` for nonlinear |
| HMC/NUTS-specific glue | Deferred | Autograd through filter works but needs benchmarking |
| Variational system ID | Deferred | — |
| GPU/TPU benchmarking | Deferred | — |
| Cached UT/GH/Cubature mean weights per filter instance | Deferred | Small win; profile before adding |
| `Data` with misaligned `y` / `u` timesteps | Deferred | See Data layer section |
| `torch.compile` on filter inner loops | Deferred | Opt-in; avoid in-place Python mutation in kernels first |

---

## Checklist for future algorithms

When adding a filter, smoother, or sysid method:

- [ ] Does it match KF / RTS on a linear-Gaussian toy system?
- [ ] Does it respect PSDMatrix lazy sqrt/inv caches?
- [ ] Are Cholesky solves used instead of explicit inverses where possible?
- [ ] Does it implement `nlog_marginal_likelihood` if it should plug into `Posterior`?
- [ ] Does it deep-copy user-provided distributions when mutating in place?
- [ ] Is there a test in `tests/` with a literature reference or analytical oracle?
- [ ] Is there an example script or notebook cell?
- [ ] Is the deferred-vs-shipped status updated in this file?

---

## Related files

| Topic | Location |
|-------|----------|
| Matrix / PSD lazy caches | `bside/models.py` |
| AdditiveModel composition | `bside/dynamics.py` |
| Kalman / SR-KF / PF kernels | `bside/filtering/functional.py` |
| Filter classes | `bside/filtering/filters.py` |
| Weighted particles / sigma points | `bside/filtering/distributions.py` |
| Smoothers | `bside/filtering/smoothers.py` |
| Posterior / EM / multi-shooting | `bside/sysid/` |
| Subspace encoder training | `bside/subspace_encoder.py` |
| Efficiency regression tests | `tests/test_filters.py`, `tests/test_particle_filter.py` |
| User-facing algorithm list | `docs/algorithms.md` |
| Original upgrade plan | `.cursor/plans/bside_broad_upgrade_*.plan.md` |
| MCMC interop target | `~/Documents/mcmc-samplers/docs/design-notes.md` |
