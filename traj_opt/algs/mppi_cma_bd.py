from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task

from evosax.algorithms.distribution_based.cma_es import eigen_decomposition



@dataclass
class MPPI_CMA_BD_Params(SamplingParams):
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
    perturbation: jax.Array
    covariance: jax.Array



class MPPI_CMA_BD(SamplingBasedController):
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
        noise_level: float,
        temperature: float,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        mean_lr: float = 1.0,
        cov_lr: float = 0.1
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_level: The scale of Gaussian noise to add to sampled controls.
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
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
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.temperature = temperature
        self.mean_lr = mean_lr
        self.cov_lr = cov_lr
        self.eigen_decomp_vmap = jax.vmap(eigen_decomposition, 
                                          in_axes=(0,), 
                                          out_axes=(0, 0, 0)
                                          )

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> MPPI_CMA_BD_Params:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)

        perturbation = jnp.zeros((self.num_samples, self.num_knots, self.task.model.nu)) # num_samples x num_knots x nu
        
        covariance = jnp.tile(jnp.eye(self.task.model.nu) * (self.noise_level**2), 
                              (self.num_knots, 1, 1))  # num_knots x nu x nu
        
        return MPPI_CMA_BD_Params(tk=_params.tk, mean=_params.mean, rng=_params.rng, 
                               perturbation=perturbation,
                               covariance=covariance)

    def sample_knots(self, params: MPPI_CMA_BD_Params) -> Tuple[jax.Array, MPPI_CMA_BD_Params]:
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

        # Eigen decomposition with tricks for numerical stability (from: https://github.com/RobertTLange/evosax/blob/main/evosax/algorithms/distribution_based/cma_es.py)
        _, B, D = self.eigen_decomp_vmap(params.covariance) # B: (num_knots x nu x nu) D: (num_knots x nu)
        
        D_noise = jnp.einsum("ijk, jk -> ijk", noise, D) # num_samples x num_knots x nu

        perturbation = jnp.einsum("ijk, jkl -> ijl", D_noise, B) # num_samples x num_knots x nu

        controls = params.mean + perturbation

        return controls, params.replace(rng=rng,
                                        perturbation = perturbation)

    def update_params(
        self, params: MPPI_CMA_BD_Params, rollouts: Trajectory
    ) -> MPPI_CMA_BD_Params:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)

        clipped_perturbation = jnp.reshape((rollouts.knots - params.mean),
                                           (self.num_samples,
                                            self.num_knots,
                                            self.task.model.nu)
                )
        params = params.replace(perturbation = clipped_perturbation) # Set to clipped perturbations

        perturbation_outer = jnp.einsum("ijk, ijl-> ijkl", params.perturbation, params.perturbation) # num_samples x num_knots x nu x nu

        covariance = (1 - self.cov_lr * jnp.sum(weights)) * params.covariance + self.cov_lr * jnp.sum(
            weights[:, None, None, None] * perturbation_outer, 
            axis = 0
            ) # num_knots x nu x nu 
        
        mean = params.mean + self.mean_lr * jnp.sum(weights[:, None, None] * params.perturbation,
                axis = 0) # Algorithm 6 line 6
        
        return params.replace(mean=mean,
                              covariance = covariance)
    
