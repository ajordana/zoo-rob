import hydrax
import hydrax.alg_base
import jax
import jax.numpy as jnp
import numpy as np
from hydrax.algs import Evosax, MPPI, PredictiveSampling
from hydrax.task_base import Task

from algs.mppi_cma import MPPI_CMA
from algs.mppi_lr import MPPI_lr
from algs.mppi import MPPI_copy

from evosax.algorithms.distribution_based import CMA_ES, Open_ES
from evosax.algorithms.distribution_based.cma_es import Params as CMA_Params   
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

    elif name == "MPPI-CMA":
        
        def lse_fitness_shaping_fn(
            population: Population, fitness: jax.Array, state: State, params: Params
            ) -> Fitness:
            
            fitness = jax.nn.softmax(-fitness / temperature, axis=0)

            return fitness

        es_params = CMA_Params(
            std_init=noise,
            std_min=0.0,
            std_max=1e8,
            weights=jnp.ones(task.model.nu * num_knots)/(task.model.nu * num_knots),
            mu_eff=1,
            c_mean=1.0,
            c_mu=0.25,
            c_std=0,
            d_std=1,
            c_c=0,
            c_1=0,
            chi_n=1,
        )

        algorithm = Evosax(
        task=task,
        optimizer=CMA_ES,
        es_params= es_params,
        num_samples=num_samples,
        plan_horizon=horizon,
        spline_type=spline,
        num_knots=num_knots,
        fitness_shaping_fn = lse_fitness_shaping_fn
        )

    elif name == "RandomizedSmoothing":

        def baseline_subtraction_fitness_shaping_fn(
            population: Population, fitness: jax.Array, state: State, params: Params
            ) -> Fitness:

            fitness = fitness - jnp.mean(fitness, axis=0)
            return fitness

        Open_ES_ = partial(
            Open_ES,                         
            optimizer=optax.sgd(0.1),        
            std_schedule=optax.constant_schedule(noise),
            use_antithetic_sampling=False,
            fitness_shaping_fn = baseline_subtraction_fitness_shaping_fn,
        )

        algorithm = Evosax(
            task=task,
            optimizer=Open_ES_,
            num_samples=num_samples,
            plan_horizon=horizon,
            spline_type=spline,
            num_knots=num_knots,
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
        
    elif name == "MPPI_cp":
        algorithm = MPPI_copy(
                task,
                num_samples = num_samples,
                temperature = temperature,
                noise_level= noise,
                plan_horizon= horizon,
                spline_type=spline,
                num_knots=num_knots
            )
    
    elif name == "MPPI_lr":
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
    
    elif name == "MPPI_CMA":

        algorithm = MPPI_CMA(
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
    
    elif name == "MPPI_CMA_lr":

        algorithm = MPPI_CMA(
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
    else:
        print(f"{name} is not supported in this benchmark yet")
        raise NotImplementedError
    
    return algorithm
