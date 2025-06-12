import time
import statistics

import jax
import jax.numpy as jnp
import os
os.environ['MUJOCO_GL'] = 'egl'   # or 'osmesa'
import mujoco

import imageio
from IPython.display import Image as IPyImage
from mujoco import GLContext, MjvScene, MjvOption, MjrContext

import mujoco.viewer
import numpy as np
from mujoco import mjx
import copy
from hydrax.alg_base import Trajectory, SamplingBasedController
import joblib
import tqdm
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt

class TrajectoryOptimizer:

    def __init__(
        self,
        name: str,
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

        print(f'task.dt:{controller.task.dt}; controller.dt:{controller.dt}; '
            f'task.model.opt.timestep: {controller.task.model.opt.timestep}; '
            f'task.mj_model.opt.timestep: {controller.task.mj_model.opt.timestep}; '
            f'simulator mj_model.opt.timestep: {mj_model.opt.timestep}'
            )

        self.warm_up = False
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.controller = controller
        
        
        mjx_data = mjx.put_data(self.mj_model, self.mj_data)
        mjx_data = mjx_data.replace(mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat)
        self.mjx_data = mjx_data
        
        self.viewer = None
        self.controller_name = name

        # initialize the controller
        jit_optimize = jax.jit(partial(controller.optimize))
        self.jit_optimize = jit_optimize

    @staticmethod
    def get_path(task):
        task_name = task.__class__.__name__
        base_dir = Path(__file__).parent
        path = os.path.join(base_dir,"data", task_name) + "/"
        return path   

    def reset_mjx_data(self):
        """Enhanced reset method that ensures complete state reset"""
        # Create a fresh mjx_data from the original mj_data
        mjx_data = mjx.put_data(self.mj_model, self.mj_data)
        mjx_data = mjx_data.replace(
            mocap_pos=self.mj_data.mocap_pos, 
            mocap_quat=self.mj_data.mocap_quat,
            time=self.mj_data.time  # Ensure time is also reset
        )
        self.mjx_data = mjx_data

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
        num_trails: int = 6,
        save_npz:bool = False) -> None:

        print(f"Controller dt: {self.controller.dt}")
        print(f"Simulator dt: {self.mj_model.opt.timestep}")

        self.__warm_up()

        controller_name = self.controller_name
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

        # Find the best final solution
        best_idx = np.argmin(last_costs)
        best_params = params_list[best_idx]
        best_rollout = rollouts_list[best_idx]
        best_knots = best_params.mean

        # Convert the solution from knots to controls
        best_ctrls = self.knots2ctrls(best_knots[None, ...]).squeeze(axis=0)

        # Plot the controls to check their values
        self.plot_controls(best_ctrls)

        figure_path = os.path.join(base_dir,"figures", task_name)

        try:
            os.makedirs(figure_path, exist_ok=True)
            print(f"path created: {figure_path}")
        except Exception as e:
            print(f'failed to crate path: {e}')

        try: 
            stem = str(path + f"/{controller_name}_trails")
            joblib.dump(cost_array,    stem + "_costs.pkl")
            joblib.dump(best_params,   stem + "_best_params.pkl")
            joblib.dump(best_rollout,  stem + "_best_rollouts.pkl")
            joblib.dump(best_ctrls,    stem + "_best_ctrls.pkl")
        
            if save_npz:

                np.savez_compressed(
                    str(figure_path + f"/{controller_name}_trails_costs.npz"), 
                    costs=cost_array,
                )
  
                print(f'.npz saved')

        except Exception as e:
            print(f"Failed to save results: {e}")

    def optimize(
        self,
        max_iteration: int = 100,
        seed: int = 1
    ) -> list[list, list, Trajectory]:

        knots_list = [] 
        
        self.reset_mjx_data()
        
        policy_params = self.controller.init_params(seed=seed)
        mean_knots = policy_params.mean 

        knots_list.append(mean_knots)
        
        for _ in tqdm.tqdm(range(max_iteration)):
            policy_params, rollouts = self.jit_optimize(self.mjx_data, policy_params)
            
            self.reset_mjx_data()
            mean_knots = policy_params.mean
            knots_list.append(mean_knots)

        cost_list = self.get_cost_list(knots_list)

        print("Optimization done.")

        return cost_list, policy_params, rollouts

    def get_cost_list(
        self,
        knots_list: list,
    ) -> list:

        ctrl = self.controller
        task = self.controller.task

        knots = jnp.array(knots_list)      

        controls = self.knots2ctrls(knots)

        state = self.mjx_data
        _, rollouts = ctrl.eval_rollouts(task.model, state, controls, knots)

        costs = jnp.sum(rollouts.costs, axis=-1)

        return list(costs)
        
    def knots2ctrls(self,
                    knots: jax.Array
        )-> jax.Array:
        # This function follow exactly how spline interpolation was done in Hydrax: https://github.com/vincekurtz/hydrax/blob/main/hydrax/alg_base.py/#L208

        ctrl = self.controller

        tk = (
            jnp.linspace(0.0, ctrl.plan_horizon, ctrl.num_knots) + self.mjx_data.time
        )

        tq = jnp.linspace(tk[0], tk[-1], ctrl.ctrl_steps)
        controls = ctrl.interp_func(tq, tk, knots)

        return controls
    
    def plot_controls(self, controls):
        """
        Plot the sequence of control vectors.
        
        Args:
            controls: array of shape (horizon, Nu), where horizon is
                    the number of time steps and Nu is the number of controls.
        """

        ctrls = np.asarray(controls)
        horizon, Nu = ctrls.shape
        
        dt = float(self.controller.dt)
        t = np.arange(horizon) * dt
        
        ub = getattr(self.controller.task, 'ub', None)
        lb = getattr(self.controller.task, 'lb', None)

        # create one subplot per control dimension
        fig, axes = plt.subplots(Nu, 1, sharex=True, figsize=(8, 2*Nu))
        if Nu == 1:
            axes = [axes]  # make it iterable
        
        for i, ax in enumerate(axes):
            ax.plot(t, ctrls[:, i], lw=1.5)

            if ub is not None and lb is not None:
                # plot upper/lower bounds as horizontal dashed lines
                ax.axhline(y=ub[i], color='r', linestyle='--', label="ub" if i==0 else None)
                ax.axhline(y=lb[i], color='r', linestyle='--', label="lb" if i==0 else None)
            
            ax.set_ylabel(f"$u_{i+1}$")
            ax.grid(True)
        
        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(f"Control Trajectories ({self.controller_name})", y=1.00)
        fig.tight_layout()
        plt.show(block=False)
        
    def optimize_save_results(
        self,
        max_iteration: int = 100,
        seed: int = 1
    ) -> list:

        self.__warm_up()
        policy_params = self.controller.init_params(seed=seed)
        controller_name = self.controller_name
        task_name = self.controller.task.__class__.__name__
        base_dir = Path(__file__).parent
        path = os.path.join(base_dir,"data", task_name)

        os.makedirs(path, exist_ok=True)

        knots_list = [] 
        cost_list = []
        policy_params = self.controller.init_params(seed=seed)
        mean_knots = policy_params.mean 
        knots_list.append(mean_knots)


        for i in tqdm.tqdm(range(max_iteration)):
            policy_params, rollouts = self.jit_optimize(self.mjx_data, policy_params)

            mean_knots = policy_params.mean
            knots_list.append(mean_knots)

        cost_list = self.get_cost_list(knots_list)

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

    def visualize_rollout(self, task, controller_name):
        
        self.__create_temporary_viewer()

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


    def __create_temporary_viewer(self):
        if self.viewer is None:
            self.tmp_mj_data = copy.copy(self.mj_data)
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.tmp_mj_data)

    def __reset_tmp_data(self):
        self.tmp_mj_data.qpos[:] = self.mj_data.qpos
        self.tmp_mj_data.qvel[:] = self.mj_data.qvel
            
    def visualize_rollout_gif(
        self,
        task, 
        controller_name: str,
        fps: int = None,
        width: int = 480,
        height: int = 480,
        camera_id: int = -1,
    ) -> IPyImage:
        """
        Renders the rollout for the named controller into a GIF and returns it.
        Controls will be loaded from:
        data/<TaskName>/<controller_name>_trails_best_ctrls.pkl
        GIF is written to:
        data/<TaskName>/<controller_name>.gif
        """
        # locate the folder exactly as visualize_rollout does
        task_name = task.__class__.__name__
        base_dir  = Path(__file__).parent
        path      = os.path.join(base_dir, "data", task_name)

        pkl_file = os.path.join(path, f"{controller_name}_trails_best_ctrls.pkl")
        controls = joblib.load(pkl_file)

        # prepare temporary data for rendering
        self.__create_temporary_viewer()
        data = self.tmp_mj_data
        
        # Reset to initial state
        self.__reset_tmp_data()
        
        # Create renderer using the modern MuJoCo API
        renderer = mujoco.Renderer(self.mj_model, height=height, width=width)
        
        # Set camera (if specified)
        if camera_id >= 0:
            renderer.enable_camera_id = camera_id

        # Calculate proper FPS - default to 30 FPS for smooth playback
        dt = float(self.mj_model.opt.timestep)
        if fps is None:
            # Use 30 FPS by default for smooth visualization
            # If you want real-time speed, use: fps = int(1.0 / dt)
            fps = 60

        frames = []
        
        try:
            for u in controls:
                data.ctrl[:] = u
                mujoco.mj_step(self.mj_model, data)
                
                # Update scene and render
                renderer.update_scene(data)
                img = renderer.render()
                frames.append(img)
                
        finally:
            # Always close the renderer to free resources
            renderer.close()

        # ensure output dir and write GIF with looping
        os.makedirs(path, exist_ok=True) 
        gif_path = os.path.join(path, f"{controller_name}.gif")
        
        # Save GIF with infinite looping and proper duration
        imageio.mimsave(
            gif_path, 
            frames, 
            fps=fps,
            loop=0,  # 0 means infinite loop
            duration=1.0/fps  # Ensure consistent frame timing
        )
        
        print(f"GIF saved to: {gif_path}")
        
        # Display the GIF in Jupyter notebook
        from IPython.display import Image, display
        
        # Method 1: Try loading from file
        try:
            gif_image = Image(filename=gif_path)
            return gif_image
        except:
            # Method 2: Load as base64 data (more reliable)
            with open(gif_path, 'rb') as f:
                gif_data = f.read()
            
            gif_image = Image(data=gif_data, format='gif')
            return gif_image