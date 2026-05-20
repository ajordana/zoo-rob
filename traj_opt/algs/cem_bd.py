from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class CEMBDParams(SamplingParams):
    """Policy parameters for block-diagonal CEM.

    Attributes inherited from SamplingParams:
        tk: knot times.
        mean: knot mean, shape (num_knots, nu).
        rng: PRNG key.

    Attributes:
        cov: block-diagonal covariance, shape (num_knots, nu, nu). One
             nu × nu block per knot timestep; no cross-time correlation.
    """

    cov: jax.Array


class CEM_BD(SamplingBasedController):
    """Cross-entropy method with block-diagonal covariance and cov learning rate.

    Block structure: one (nu × nu) block per knot timestep, so each timestep
    has its own full action-space covariance, but timesteps are uncorrelated.

    - Samples from N(mean[t], cov[t]) independently for each timestep t.
    - Picks the top-num_elites samples by total rollout cost.
    - Mean is fully replaced by the elite empirical mean.
    - Covariance is Polyak-averaged toward the elite empirical per-block
      covariance with learning rate cov_lr:
          cov ← (1 - cov_lr) * cov_old + cov_lr * cov_sample
      No min-covariance cap, no eigenvalue clipping.

    Parameter count: num_knots * nu * (nu+1) / 2, intermediate between
    diagonal (num_knots * nu) and full ((num_knots*nu)*(num_knots*nu+1)/2).
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        num_elites: int,
        sigma_start: float,
        mean_lr: float = 1.0,
        cov_lr: float = 0.1,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
    ) -> None:
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.sigma_start = sigma_start
        self.mean_lr = mean_lr
        self.cov_lr = cov_lr

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> CEMBDParams:
        _params = super().init_params(initial_knots, seed)
        nu = self.task.model.nu
        # One nu × nu identity block per knot, scaled by sigma_start^2.
        cov = jnp.tile(
            jnp.eye(nu) * (self.sigma_start ** 2),
            (self.num_knots, 1, 1),
        )
        return CEMBDParams(
            tk=_params.tk, mean=_params.mean, cov=cov, rng=_params.rng
        )

    def sample_knots(
        self, params: CEMBDParams
    ) -> Tuple[jax.Array, CEMBDParams]:
        rng, sample_rng = jax.random.split(params.rng)
        nu = self.task.model.nu
        # Symmetrize each block, then per-block Cholesky.
        cov_sym = 0.5 * (params.cov + jnp.swapaxes(params.cov, -1, -2))
        L = jnp.linalg.cholesky(cov_sym)                   # (T, nu, nu)
        z = jax.random.normal(
            sample_rng, (self.num_samples, self.num_knots, nu)
        )
        # delta[n, t, i] = sum_j L[t, i, j] * z[n, t, j]
        delta = jnp.einsum("tij,ntj->nti", L, z)
        controls = params.mean + delta
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: CEMBDParams, rollouts: Trajectory
    ) -> CEMBDParams:
        costs = jnp.sum(rollouts.costs, axis=1)            # (N,)
        elites = jnp.argsort(costs)[: self.num_elites]
        K = self.num_elites
        nu = self.task.model.nu

        x = rollouts.knots[elites]                         # (K, T, nu)
        elite_mean = jnp.mean(x, axis=0)                   # (T, nu)
        diff = x - elite_mean[None, ...]                   # (K, T, nu)
        # Per-block sample covariance: (T, nu, nu)
        sample_cov = jnp.einsum("kti,ktj->tij", diff, diff) / K
        sample_cov = 0.5 * (sample_cov + jnp.swapaxes(sample_cov, -1, -2))

        # Polyak averaging on both mean and cov.
        new_mean = (1.0 - self.mean_lr) * params.mean + self.mean_lr * elite_mean
        new_cov  = (1.0 - self.cov_lr)  * params.cov  + self.cov_lr  * sample_cov
        return params.replace(mean=new_mean, cov=new_cov)
