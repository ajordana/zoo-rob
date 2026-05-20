import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task

# Modified from Hydrax: control limits are enforced as a penalty instead of
# hard clipping, matching the convention of the other *_unconstrained tasks.
# https://github.com/vincekurtz/hydrax/tree/main/hydrax/tasks


class WalkerUnconstrained(Task):
    """A planar biped tasked with walking forward (penalty-bound controls)."""

    def __init__(self) -> None:
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/walker/scene.xml"
        )

        self.lb = mj_model.actuator_ctrlrange[:, 0].copy()
        self.ub = mj_model.actuator_ctrlrange[:, 1].copy()

        mj_model.actuator_forcelimited[:] = 0
        mj_model.actuator_ctrllimited[:]  = 0
        mj_model.actuator_ctrlrange[:]    = [-jnp.inf, jnp.inf]

        super().__init__(mj_model, trace_sites=["torso_site"])

        self.torso_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_position"
        )
        self.torso_velocity_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_subtreelinvel"
        )
        self.torso_zaxis_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_zaxis"
        )

        self.target_velocity = 1.5
        self.target_height = 1.2

    def _bound_violation(self, ctrl, ord=2):
        lower = jnp.maximum(self.lb - ctrl, 0)
        upper = jnp.maximum(ctrl - self.ub, 0)
        v = lower + upper
        penalty = 10 * jnp.linalg.norm(v, ord)
        penalty = jnp.where(penalty != 0, penalty + 1, penalty)
        return penalty

    def _get_torso_height(self, state: mjx.Data) -> jax.Array:
        sensor_adr = self.model.sensor_adr[self.torso_position_sensor]
        return state.sensordata[sensor_adr + 2]

    def _get_torso_velocity(self, state: mjx.Data) -> jax.Array:
        sensor_adr = self.model.sensor_adr[self.torso_velocity_sensor]
        return state.sensordata[sensor_adr]

    def _get_torso_deviation_from_upright(self, state: mjx.Data) -> jax.Array:
        sensor_adr = self.model.sensor_adr[self.torso_zaxis_sensor]
        return state.sensordata[sensor_adr + 2] - 1.0

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        state_cost = self.terminal_cost(state)
        control_cost = jnp.sum(jnp.square(control))
        bound_violation_cost = self._bound_violation(control)
        return state_cost + 0.1 * control_cost + bound_violation_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        height_cost = jnp.square(
            self._get_torso_height(state) - self.target_height
        )
        orientation_cost = jnp.square(
            self._get_torso_deviation_from_upright(state)
        )
        velocity_cost = jnp.square(
            self._get_torso_velocity(state) - self.target_velocity
        )
        return 10.0 * height_cost + 3.0 * orientation_cost + 1.0 * velocity_cost
