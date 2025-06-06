import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx
import copy
from hydrax.alg_base import Trajectory, SamplingBasedController
import equinox
import joblib
import tqdm
from functools import partial
import os
from pathlib import Path

class traj_opt_helper:
    def __init__(
        self,
        controller: SamplingBasedController,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
    ):
        # logging
        print(
            f"Trajectory Optimization with {controller.num_knots} steps "
            f"over a {controller.ctrl_steps * controller.task.dt} "
            f"second horizon."
        )

        self.warm_up = False
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.controller = controller
        mjx_data = mjx.put_data(self.mj_model, self.mj_data)
        mjx_data = mjx_data.replace(mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat)
        self.mjx_data = mjx_data
        self.viewer = None

        # initialize the controller
        jit_optimize = jax.jit(partial(controller.optimize))
        self.jit_optimize = jit_optimize

    def __warm_up(self):
        if self.warm_up:
            return
        # warm-up the controller
        print("Jitting the controller...")
        st = time.time()
        policy_params = self.controller.init_params()
        policy_params, _ = self.jit_optimize(self.mjx_data, policy_params)
        policy_params, _ = self.jit_optimize(self.mjx_data, policy_params)
        print(f"Time to jit: {time.time() - st:.3f} seconds")

        self.warm_up = True

    def load_policy(self):
        self.policy_params = joblib.load("policy_params_latest.pkl")
        self.cost_list = joblib.load("costs_latest.pkl")

    def trails( self,
        max_iteration: int = 100,
        num_trails: int = 6) -> None:

        print(f"Controller dt: {self.controller.dt}")
        print(f"Simulator dt: {self.mj_model.opt.timestep}")

        self.__warm_up()

        controller_name = self.controller.__class__.__name__
        task_name = self.controller.task.__class__.__name__

        base_dir = Path(__file__).parent

        path = os.path.join(base_dir,"data", task_name)
        try:
            os.makedirs(path, exist_ok=True)
            print(f"path created: {path}")
        except Exception as e:
            print(f'failed to crate path: {e}')

        params_list = []
        rollouts_list = []
        cost_list_list = []
        seed_list = list(np.arange(num_trails))
        for seed in seed_list:
            cost_list, last_policy_params, last_rollouts = self.optimize(max_iteration, seed=seed)
            cost_list_list.append(cost_list)
            params_list.append(last_policy_params)
            rollouts_list.append(last_rollouts)
        
        cost_array = np.array(cost_list_list)
        last_costs = cost_array[:, -1]

        cost_array = cost_array.mean(axis = 0)

        best_idx = np.argmin(last_costs)
        best_params = params_list[best_idx]
        best_rollout = rollouts_list[best_idx]
        best_ctrls = best_rollout.controls[-1]
        
        try:
            joblib.dump(cost_array, path + "/" + controller_name + "_costs_trails_average.pkl")
            joblib.dump(best_params, path + "/" + controller_name + "_trails_best_params.pkl")
            joblib.dump(best_rollout, path + "/" + controller_name + "_trails_best_rollouts.pkl")
            joblib.dump(best_ctrls, path + "/" + controller_name + "_trails_best_ctrls.pkl")
            print("Results saved")
        except Exception as e:
            print(f"Failed to save results: {e}")

    def optimize(
        self,
        max_iteration: int = 100,
        seed: int = 1
    ) -> list[list, list, Trajectory]:

        policy_params = self.controller.init_params(seed=seed)
        cost_list = []

        for i in tqdm.tqdm(range(max_iteration)):
            policy_params, rollouts = self.jit_optimize(self.mjx_data, policy_params)
            trajectory_cost = jnp.sum(rollouts.costs[-1, :], axis=-1) # Take the current trajectory costs            
            cost_list.append(trajectory_cost) # Append the cost of the current control trajectory to the list

        print("Optimization done.")

        return cost_list, policy_params, rollouts
    
    def opt(
        self,
        max_iteration: int = 100,
        seed: int = 1
    ) -> list[list, list, Trajectory]:

        cost_list = []
        knots_list = [] 
    
        policy_params = self.controller.init_params(seed=seed)
        mean_knots = policy_params.mean 

        knots_list.append(mean_knots)
        
        # trajectory_cost = self.get_cost(mean_knots[None, ...])
        
        for i in tqdm.tqdm(range(max_iteration)):
            policy_params, rollouts = self.jit_optimize(self.mjx_data, policy_params)

            mean_knots = policy_params.mean
            knots_list.append(mean_knots)

            # trajectory_cost = jnp.sum(rollouts.costs[-1, :], axis=-1) # Take the current trajectory costs            
            # cost_list.append(trajectory_cost) # Append the cost of the current control trajectory to the list
            
        costs_list = self.get_cost_list(knots_list)

        # assert (np.array(costs_list[:-1]) == np.array(cost_list)).all()
        print("Optimization done.")

        return costs_list, policy_params, rollouts
    
    def get_cost_list(
        self,
        knots_list: list,
    ) -> list:

        ctrl = self.controller
        task = self.controller.task

        knots = jnp.array(knots_list)      

        tk = (
            jnp.linspace(0.0, ctrl.plan_horizon, self.num_knots) + self.mjx_data.time
        )

        tq = jnp.linspace(tk[0], tk[-1], ctrl.ctrl_steps)
        controls = ctrl.interp_func(tq, tk, knots)

        print(f'mean controls from get_cost_list:{controls}')

        state = self.mjx_data
        _, rollouts = ctrl.eval_rollouts(task.model, state, controls, knots)

        costs = jnp.sum(rollouts.costs, axis=-1)

        return list(costs)
        

    def get_cost(
        self,
        knots: jax.Array,
    ) -> list:

        ctrl = self.controller
        task = self.controller.task

        tk = jnp.linspace(0.0, ctrl.plan_horizon, ctrl.num_knots)
        tq = jnp.linspace(tk[0], tk[-1], ctrl.ctrl_steps)
        controls = ctrl.interp_func(tq, tk, knots)

        state = self.mjx_data
        _, rollouts = ctrl.eval_rollouts(task.model, state, controls, knots)

        return jnp.sum(rollouts.costs[0, :], axis=-1)
        
        
    def optimize_save_results(
        self,
        max_iteration: int = 100,
        seed: int = 1
    ) -> list:

        self.__warm_up()
        policy_params = self.controller.init_params(seed=seed)
        controller_name = self.controller.__class__.__name__
        task_name = self.controller.task.__class__.__name__
        base_dir = Path(__file__).parent
        path = os.path.join(base_dir,"data", task_name)

        os.makedirs(path, exist_ok=True)

        cost_list = []

        for i in tqdm.tqdm(range(max_iteration)):
            policy_params, rollouts = self.jit_optimize(self.mjx_data, policy_params)
            trajectory_cost = jnp.sum(rollouts.costs[-1, :], axis=-1) # Take the current trajectory costs            
            cost_list.append(trajectory_cost) # Append the cost of the current control trajectory to the list

        print("Optimization done.")

        self.policy_params = policy_params
        self.rollouts = rollouts
        try:
            joblib.dump(policy_params, path + "/" + controller_name + "_policy_params.pkl")
            joblib.dump(rollouts,  path + "/" + controller_name + "_rollouts.pkl")
            joblib.dump(cost_list, path + "/" + controller_name + "_costs.pkl")
            print("Results saved")
        except Exception as e:
            print(f"Failed to save results: {e}")

        return cost_list

    def visualize_rollout(self, task,
                          controller):
        
        self.__create_temporary_viewer()

        controller_name = controller.__class__.__name__
        task_name = task.__class__.__name__

        base_dir = Path(__file__).parent

        path = os.path.join(base_dir,"data", task_name)

        file_name = path + "/" + controller_name + "_trails_best_ctrls.pkl"

        controls = joblib.load(file_name)
        i = 0
        horizon = controls.shape[0]
        dt = float(self.mj_model.opt.timestep)

        i = 0
        while self.viewer.is_running():
            t_start = time.time()

            # apply control and step
            self.tmp_mj_data.ctrl[:] = controls[i]
            mujoco.mj_step(self.mj_model, self.tmp_mj_data)
            self.viewer.sync()

            # sleep the remainder of dt to approximate real time
            elapsed = time.time() - t_start
            to_sleep = dt - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

            i += 1
            if i == horizon:
                i = 0
                self.__reset_tmp_data()

    def get_path(self):
        task_name = self.controller.task.__class__.__name__
        base_dir = Path(__file__).parent
        path = os.path.join(base_dir,"data", task_name) + "/"

        return path
    def __create_temporary_viewer(self):
        if self.viewer is None:
            self.tmp_mj_data = copy.copy(self.mj_data)
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.tmp_mj_data)

    def __reset_tmp_data(self):
        self.tmp_mj_data.qpos[:] = self.mj_data.qpos
        self.tmp_mj_data.qvel[:] = self.mj_data.qvel

    def visualize_all(self):
        with tqdm.tqdm(total=self.controller.num_populations, desc="Visualizing rollouts", ncols=0) as pbar:
            while True:
                for i in range(self.controller.num_populations):
                    self.visualize_rollout(i, loop=False)
                    pbar.update(1)
                pbar.reset()
            # for i in tqdm.tqdm(range(self.controller.num_populations), desc="Visualizing rollouts"):
            #     self.visualize_rollout(i, loop=False)
