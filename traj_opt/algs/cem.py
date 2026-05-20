from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class CEMParams(SamplingParams):
    """Policy parameters for full-covariance CEM with covariance learning rate.

    Attributes inherited from SamplingParams:
        tk: knot times.
        mean: knot mean, shape (num_knots, nu).
        rng: PRNG key.

    Attributes:
        cov: full covariance over the flat (num_knots * nu)-vector,
             shape (D, D) with D = num_knots * nu.
    """

    cov: jax.Array


class CEM(SamplingBasedController):
    """Cross-entropy method with full covariance and a covariance learning rate.

    - Samples from N(mean, cov) over the flat (num_knots * nu)-vector.
    - Picks the top-num_elites samples by total rollout cost.
    - Mean is fully replaced by the elite empirical mean.
    - Covariance is Polyak-averaged toward the elite empirical covariance with
      learning rate cov_lr:
          cov ← (1 - cov_lr) * cov_old + cov_lr * cov_sample
      No min-covariance cap, no eigenvalue clipping. With cov_lr small (e.g.
      0.1), the well-conditioned initial covariance bleeds out slowly and keeps
      cov positive-definite enough for Cholesky to remain numerically stable.
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

    def _flat_dim(self) -> int:
        return self.num_knots * self.task.model.nu

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> CEMParams:
        _params = super().init_params(initial_knots, seed)
        D = self._flat_dim()
        cov = jnp.eye(D) * (self.sigma_start ** 2)
        return CEMParams(
            tk=_params.tk, mean=_params.mean, cov=cov, rng=_params.rng
        )

    def sample_knots(
        self, params: CEMParams
    ) -> Tuple[jax.Array, CEMParams]:
        rng, sample_rng = jax.random.split(params.rng)
        nu = self.task.model.nu
        D = self._flat_dim()
        cov_sym = 0.5 * (params.cov + params.cov.T)
        L = jnp.linalg.cholesky(cov_sym)                   # (D, D)
        z = jax.random.normal(sample_rng, (self.num_samples, D))
        delta = z @ L.T                                    # (N, D)
        delta = delta.reshape(self.num_samples, self.num_knots, nu)
        controls = params.mean + delta
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: CEMParams, rollouts: Trajectory
    ) -> CEMParams:
        costs = jnp.sum(rollouts.costs, axis=1)            # (N,)
        elites = jnp.argsort(costs)[: self.num_elites]
        K = self.num_elites
        D = self._flat_dim()
        nu = self.task.model.nu

        x = rollouts.knots[elites].reshape(K, D)           # (K, D)
        elite_mean_flat = jnp.mean(x, axis=0)              # (D,)
        diff = x - elite_mean_flat[None, :]                # (K, D)
        sample_cov = (diff.T @ diff) / K                   # (D, D), MLE sample cov
        sample_cov = 0.5 * (sample_cov + sample_cov.T)     # enforce symmetry

        # Polyak averaging on both mean and cov.
        elite_mean = elite_mean_flat.reshape(self.num_knots, nu)
        new_mean = (1.0 - self.mean_lr) * params.mean + self.mean_lr * elite_mean
        new_cov  = (1.0 - self.cov_lr)  * params.cov  + self.cov_lr  * sample_cov

        return params.replace(mean=new_mean, cov=new_cov)
