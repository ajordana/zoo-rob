from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from pathlib import Path

from hydrax.task_base import Task
# ROOT = str(Path(__file__).resolve().parent.parent)
from hydrax import ROOT


class PushTUnconstrained(Task):
    # This task is modified from Hydrax, the control limits are enforced as penalties to avoid clipping
    # https://github.com/vincekurtz/hydrax/tree/main/hydrax/tasks
    """Push a T-shaped block to a desired pose."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/pusht/scene.xml"
        )

        # Get sensor ids
        self.block_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "position"
        )
        self.block_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation"
        )

        self.lb = mj_model.actuator_ctrlrange[:, 0].copy()
        self.ub = mj_model.actuator_ctrlrange[:, 1].copy()

        mj_model.actuator_forcelimited[:] = 0
        mj_model.actuator_ctrllimited[:]  = 0
        mj_model.actuator_ctrlrange[:]    = [-jnp.inf, jnp.inf]

        super().__init__(mj_model, trace_sites=["pusher"])

    def _bound_violation(self, ctrl, ord=2):
        """
        Constrained are enforced using eqn(14) of this paper: https://arxiv.org/pdf/2404.10395
        Return ‖v‖ where v is the element-wise violation of controls
        against [lb, ub].
        """
        lower = jnp.maximum(self.lb - ctrl, 0)   # only where A is below lb
        upper = jnp.maximum(ctrl - self.ub, 0)   # only where A is above ub
        v = lower + upper                # violation vector

        # raw penalty = L_ord norm of the violation
        penalty = 25 * jnp.linalg.norm(v, ord)

        # if penalty != 0 then add 1, else leave as 0: 
        penalty = jnp.where(penalty != 0, penalty + 1, penalty)

        return penalty


    def _get_position_err(self, state: mjx.Data) -> jax.Array:
        """Position of the block relative to the target position."""
        sensor_adr = self.model.sensor_adr[self.block_position_sensor]
        return state.sensordata[sensor_adr : sensor_adr + 3]

    def _get_orientation_err(self, state: mjx.Data) -> jax.Array:
        """Orientation of the block relative to the target orientation."""
        sensor_adr = self.model.sensor_adr[self.block_orientation_sensor]
        block_quat = state.sensordata[sensor_adr : sensor_adr + 4]
        goal_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        return mjx._src.math.quat_sub(block_quat, goal_quat)

    def _close_to_block_err(self, state: mjx.Data) -> jax.Array:
        """Position of the pusher block relative to the block."""
        block_pos = state.qpos[:2]
        pusher_pos = state.qpos[3:] + jnp.array([0.0, 0.1])  # y bias
        return block_pos - pusher_pos

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        position_err = self._get_position_err(state)
        orientation_err = self._get_orientation_err(state)
        close_to_block_err = self._close_to_block_err(state)

        position_cost = jnp.sum(jnp.square(position_err))
        orientation_cost = jnp.sum(jnp.square(orientation_err))
        close_to_block_cost = jnp.sum(jnp.square(close_to_block_err))
        bound_violation_cost = self._bound_violation(control)

        return 5 * position_cost + orientation_cost + 0.01 * close_to_block_cost + bound_violation_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ℓ_T(x_T)."""
        return self.running_cost(state, jnp.zeros(self.model.nu))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the level of friction."""
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.1, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}
