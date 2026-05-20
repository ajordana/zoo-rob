import hydrax
import hydrax.alg_base
import jax
import jax.numpy as jnp
import numpy as np
from hydrax.algs import Evosax, MPPI, PredictiveSampling
from algs.cem import CEM
from algs.cem_bd import CEM_BD
from hydrax.task_base import Task
from hydrax.alg_base import SamplingBasedController
from algs.mppi_cma import MPPI_CMA
from algs.mppi_lr import MPPI_lr
from algs.mppi_cma_bd import MPPI_CMA_BD
from algs.randomized_smoothing import RandomizedSmoothing

from evosax.algorithms.distribution_based import CMA_ES, Open_ES
from evosax.types import Fitness, Params, Population, State

import optax
from functools import partial


def create_algorithm(
        name: str,
        task: Task,
        num_samples: int = 1024,
        horizon: float = 1.0,
        num_knots: int = 100,
        spline: str = "zero",
        temperature: float = "0.1",
        noise: float = 0.2,
    )-> hydrax.alg_base:
    
    if name == "CMA-ES":
        
        algorithm = Evosax(
        task=task,
        optimizer=CMA_ES,
        num_samples=num_samples,
        plan_horizon=horizon,
        spline_type=spline,
        num_knots=num_knots,
        )

        algorithm.es_params = algorithm.es_params.replace(std_init = noise)

    elif name == "CEM lr=(1.0, 0.1)":

        algorithm = CEM(
            task=task,
            num_samples=num_samples,
            num_elites=max(1, num_samples // 10),
            sigma_start=noise,
            mean_lr=1.0,
            cov_lr=0.1,
            plan_horizon=horizon,
            spline_type=spline,
            num_knots=num_knots,
        )

    elif name == "CEM lr=(0.1, 0.1)":

        algorithm = CEM(
            task=task,
            num_samples=num_samples,
            num_elites=max(1, num_samples // 10),
            sigma_start=noise,
            mean_lr=0.1,
            cov_lr=0.1,
            plan_horizon=horizon,
            spline_type=spline,
            num_knots=num_knots,
        )

    elif name == "CEM lr=(1.0, 1.0)":

        algorithm = CEM(
            task=task,
            num_samples=num_samples,
            num_elites=max(1, num_samples // 10),
            sigma_start=noise,
            mean_lr=1.0,
            cov_lr=1.0,
            plan_horizon=horizon,
            spline_type=spline,
            num_knots=num_knots,
        )

    elif name == "CEM_BD lr=(1.0, 0.1)":

        algorithm = CEM_BD(
            task=task,
            num_samples=num_samples,
            num_elites=max(1, num_samples // 10),
            sigma_start=noise,
            mean_lr=1.0,
            cov_lr=0.1,
            plan_horizon=horizon,
            spline_type=spline,
            num_knots=num_knots,
        )

    elif name == "CEM_BD lr=(0.1, 0.1)":

        algorithm = CEM_BD(
            task=task,
            num_samples=num_samples,
            num_elites=max(1, num_samples // 10),
            sigma_start=noise,
            mean_lr=0.1,
            cov_lr=0.1,
            plan_horizon=horizon,
            spline_type=spline,
            num_knots=num_knots,
        )

    elif name == "RandomizedSmoothing lr=1":

        algorithm = RandomizedSmoothing(
            task=task,
            num_samples=num_samples,
            num_knots= num_knots,
            plan_horizon=horizon,
            mu=noise,
            lr = 1
        )

    elif name == "RandomizedSmoothing lr=0.1":

        algorithm = RandomizedSmoothing(
            task=task,
            num_samples=num_samples,
            num_knots= num_knots,
            plan_horizon=horizon,
            mu=noise,
            lr = 0.1
        )


    elif name == "RandomizedSmoothing lr=0.2":

        algorithm = RandomizedSmoothing(
            task=task,
            num_samples=num_samples,
            num_knots= num_knots,
            plan_horizon=horizon,
            mu=noise,
            lr = 0.2
        )

    elif name == "RandomizedSmoothing lr=0.05":

        algorithm = RandomizedSmoothing(
            task=task,
            num_samples=num_samples,
            num_knots= num_knots,
            plan_horizon=horizon,
            mu=noise,
            lr = 0.05
        )

    elif name == "RandomizedSmoothing lr=0.01":


        algorithm = RandomizedSmoothing(
            task=task,
            num_samples=num_samples,
            num_knots= num_knots,
            plan_horizon=horizon,
            mu=noise,
            lr = 0.01
        )

    elif name == "RandomizedSmoothing lr=0.005":

        algorithm = RandomizedSmoothing(
            task=task,
            num_samples=num_samples,
            num_knots= num_knots,
            plan_horizon=horizon,
            mu=noise,
            lr = 0.005
        )

    elif name == "RandomizedSmoothing lr=0.001":

        algorithm = RandomizedSmoothing(
            task=task,
            num_samples=num_samples,
            num_knots= num_knots,
            plan_horizon=horizon,
            mu=noise,
            lr = 0.001
        )

    elif name == "RandomizedSmoothing lr=0.0005":

        algorithm = RandomizedSmoothing(
            task=task,
            num_samples=num_samples,
            num_knots= num_knots,
            plan_horizon=horizon,
            mu=noise,
            lr = 0.0005
        )

    elif name == "RandomizedSmoothing lr=0.0001":

        algorithm = RandomizedSmoothing(
            task=task,
            num_samples=num_samples,
            num_knots= num_knots,
            plan_horizon=horizon,
            mu=noise,
            lr = 0.0001
        )

    elif name == "RandomizedSmoothing lr=5e-05":

        algorithm = RandomizedSmoothing(
            task=task,
            num_samples=num_samples,
            num_knots= num_knots,
            plan_horizon=horizon,
            mu=noise,
            lr = 5e-5
        )
    elif name == "MPPI":
        algorithm = MPPI(
                task,
                num_samples = num_samples,
                temperature = temperature,
                noise_level= noise,
                plan_horizon= horizon,
                spline_type=spline,
                num_knots=num_knots
            )
    
    elif name == "MPPI lr=0.1":
        algorithm = MPPI_lr(
                task,
                num_samples = num_samples,
                temperature = temperature,
                noise_level= noise,
                plan_horizon= horizon,
                spline_type=spline,
                num_knots=num_knots,
                learning_rate= 0.1
            )
    
    elif name == "MPPI_CMA lr=(1.0, 0.1)":

        algorithm = MPPI_CMA(
                task,
                num_samples = num_samples,
                temperature = 0.1,
                noise_level= noise,
                plan_horizon= horizon,
                spline_type=spline,
                num_knots=num_knots,
                mean_lr= 1.0,
                cov_lr= 0.1
            )

    elif name == "MPPI_CMA lr=(0.1, 0.1)":

        algorithm = MPPI_CMA(
                task,
                num_samples = num_samples,
                temperature = 0.1,
                noise_level= noise,
                plan_horizon= horizon,
                spline_type=spline,
                num_knots=num_knots,
                mean_lr= 0.1,
                cov_lr= 0.1
            )

    elif name == "MPPI_CMA lr=(1.0, 1.0)":

        algorithm = MPPI_CMA(
                task,
                num_samples = num_samples,
                temperature = 0.1,
                noise_level= noise,
                plan_horizon= horizon,
                spline_type=spline,
                num_knots=num_knots,
                mean_lr= 1.0,
                cov_lr= 1.0
            )

    elif name == "MPPI_CMA_BD lr=(1.0, 0.1)":

        algorithm = MPPI_CMA_BD(
                task,
                num_samples = num_samples,
                temperature = temperature,
                noise_level= noise,
                plan_horizon= horizon,
                spline_type=spline,
                num_knots=num_knots,
                mean_lr= 1.0,
                cov_lr= 0.1
            )
    
    elif name == "MPPI_CMA_BD lr=(0.1, 0.1)":

        algorithm = MPPI_CMA_BD(
                task,
                num_samples = num_samples,
                temperature = temperature,
                noise_level= noise,
                plan_horizon= horizon,
                spline_type=spline,
                num_knots=num_knots,
                mean_lr= 0.1,
                cov_lr= 0.1
            )
        
    elif name == "PredictiveSampling":
        algorithm = PredictiveSampling(
                task,
                num_samples = num_samples,
                noise_level= noise,
                plan_horizon= horizon,
                spline_type= spline,
                num_knots= num_knots,
            )
    
    elif name == "visualization":
            algorithm = PredictiveSampling(
                task,
                num_samples = num_samples,
                noise_level= noise,
                plan_horizon= horizon,
                spline_type= spline,
                num_knots= num_knots,
            )

    else:
        print(f"{name} is not supported in this benchmark yet")
        raise NotImplementedError
    
    return algorithm
