from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class RandomizedSmoothingParams(SamplingParams):
    """Policy parameters for model-predictive path integral control.

    Same as SamplingParams, but with a different name for clarity.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
    """
    tk: jax.Array
    mean: jax.Array
    rng: jax.Array
    noise: jax.Array


class RandomizedSmoothing(SamplingBasedController):
    """Model-predictive path integral control.

    Implements "MPPI-generic" as described in https://arxiv.org/abs/2409.07563.
    Unlike the original MPPI derivation, this does not assume stochastic,
    control-affine dynamics or a separable cost function that is quadratic in
    control.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        mu: float,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        lr: float = 1.0,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            mu: finite difference granularity 
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
        """
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
        self.mu = mu
        self.num_samples = num_samples
        self.lr = lr

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> RandomizedSmoothingParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)

        noise = jnp.zeros((self.num_samples, self.num_knots, self.task.model.nu)) # num_samples x num_knots x nu
        
        
        return RandomizedSmoothingParams(tk=_params.tk, mean=_params.mean, rng=_params.rng, 
                               noise=noise)

    def sample_knots(self, params: RandomizedSmoothingParams) -> Tuple[jax.Array, RandomizedSmoothingParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal( # noise as vectors
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
                self.task.model.nu
            ),
        ) # num_sample x num_knots x nu

        perturbation = self.mu * noise

        controls = params.mean + perturbation
        controls = jnp.concatenate([controls, params.mean[None, ...]])

        return controls, params.replace(rng=rng,
                                        noise = noise)

    def update_params(
        self, params: RandomizedSmoothingParams, rollouts: Trajectory
    ) -> RandomizedSmoothingParams:
        
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps

        baseline = costs[-1]

        costs_subtract_baseline = costs[:-1] - baseline


        clipped_perturbation = jnp.reshape((rollouts.knots[:-1, ...] - params.mean),
                                    (self.num_samples,
                                    self.num_knots,
                                    self.task.model.nu)
                    )
        
        clipped_noise = clipped_perturbation / self.mu
        params = params.replace(noise = clipped_noise)

        gradient = (1/self.mu) * jnp.mean(costs_subtract_baseline[:, None, None] * params.noise, 
                                          axis=0)

        mean = params.mean - self.lr * gradient

        return params.replace(mean=mean)
    
