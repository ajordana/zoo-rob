import numpy as np

import jax.numpy as jnp

import mujoco
from mujoco import mjx
from tasks.cart_pole_unconstrained import CartPoleUnconstrained
from tasks.pusht_unconstrained import PushTUnconstrained
from tasks.pendulum_unconstrained import PendulumUnconstrained
from tasks.double_cart_pole_unconstrained import DoubleCartPoleUnconstrained
from tasks.humanoid_mocap_unconstrained import HumanoidMocapUnconstrained

from hydrax.task_base import Task


def create_task(task_name: str
                )->list[Task, mujoco.MjModel, mujoco.MjData]:
    # Easy tasks
    if task_name == "CartPole":
        # CartPole
        task = CartPoleUnconstrained()
        task.dt = 0.01
        task.mj_model.opt.timestep = task.dt
        task.mj_model.opt.iterations = 1
        task.mj_model.opt.ls_iterations = 5
        task.mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART

        task.model = mjx.put_model(task.mj_model)
        task.model = task.model.replace(
            opt=task.model.opt.replace(
                timestep=0.01,
                iterations=1,
                ls_iterations=5,
                disableflags=task.model.opt.disableflags | mujoco.mjtDisableBit.mjDSBL_WARMSTART
            )
        )

        mj_model = task.mj_model # Model used by the simulator when visualizing results

        mj_data = mujoco.MjData(mj_model) # Data for both simulator and optimizer

    elif task_name == "InvertedPendulum":
        task = PendulumUnconstrained()
        task.dt = 0.01
        task.mj_model.opt.timestep = task.dt
        task.mj_model.opt.iterations = 1
        task.mj_model.opt.ls_iterations = 5
        task.mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART

        task.model = mjx.put_model(task.mj_model)
        task.model = task.model.replace(
            opt=task.model.opt.replace(
                timestep=0.01,
                iterations=1,
                ls_iterations=5,
                disableflags=task.model.opt.disableflags | mujoco.mjtDisableBit.mjDSBL_WARMSTART
            )
        )
        mj_model = task.mj_model # Model used by the simulator when visualizing results

        mj_data = mujoco.MjData(mj_model) # Data for both simulator and optimizer
        mj_data.qpos[:] = np.array([0.0])
        mj_data.qvel[:] = np.array([0.0])

    elif task_name == "DoubleCartPole":
        task = DoubleCartPoleUnconstrained()
        task.dt = 0.01
        task.mj_model.opt.timestep = task.dt
        task.mj_model.opt.iterations = 1
        task.mj_model.opt.ls_iterations = 5
        task.mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART

        task.model = mjx.put_model(task.mj_model)
        task.model = task.model.replace(
            opt=task.model.opt.replace(
                timestep=0.01,
                iterations=1,
                ls_iterations=5,
                disableflags=task.model.opt.disableflags | mujoco.mjtDisableBit.mjDSBL_WARMSTART
            )
        )
        mj_model = task.mj_model # Model used by the simulator when visualizing results

        mj_data = mujoco.MjData(mj_model) # Data for both simulator and optimizer

    # Hard tasks (contact rich)
        
    elif task_name == "PushT":
        # PushT
        task = PushTUnconstrained()
        task.dt = 0.02
        task.mj_model.opt.timestep = task.dt
        task.mj_model.opt.iterations = 2
        task.mj_model.opt.ls_iterations = 5
        task.mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART

        task.model = mjx.put_model(task.mj_model)
        task.model = task.model.replace(
            opt=task.model.opt.replace(
                timestep=0.02,
                iterations=2,
                ls_iterations=5,
            )
        )
        mj_model = task.mj_model # Model used by the simulator when visualizing results

        mj_data = mujoco.MjData(mj_model) # Data for both simulator and optimizer
        mj_data.qpos = [0.1, 0.1, 1.3, 0.0, 0.0]


    elif task_name == "HumanoidBalance":
        #HumanoidMocap
        task = HumanoidMocapUnconstrained(reference_filename="DefaultDatasets/mocap/UnitreeG1/balance.npz", start = 200) # Humanoid balancing!
        task.dt = 0.02
        task.mj_model.opt.timestep = task.dt
        task.mj_model.opt.iterations = 1  
        task.mj_model.opt.ls_iterations = 6
        task.mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
        task.mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART

        # Convert and update MJX model
        task.model = mjx.put_model(task.mj_model)
        task.model = task.model.replace(
            opt=task.model.opt.replace(
                timestep=0.02,
                iterations=1,
                ls_iterations=6,
                o_solimp=jnp.array([0.9, 0.95, 0.001, 0.5, 2])
            )
        )

        mj_model = task.mj_model  # Model used by the simulator when visualizing results

        mj_data = mujoco.MjData(mj_model) # Data for both simulator and optimizer
        mj_data.qpos[:] = task.reference[0]


    else:
        print(f"{task_name} is not implemented")

    return task, mj_model, mj_data
