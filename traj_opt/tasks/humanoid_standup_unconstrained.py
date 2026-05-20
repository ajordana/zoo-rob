from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task

# Modified from Hydrax: control bounds are enforced via a penalty in the
# running cost instead of hard clipping.
# https://github.com/vincekurtz/hydrax/blob/main/hydrax/tasks/humanoid_standup.py


class HumanoidStandupUnconstrained(Task):
    """Standup task for the Unitree G1 (penalty-bound controls)."""

    def __init__(self) -> None:
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/scene.xml")

        self.lb = mj_model.actuator_ctrlrange[:, 0].copy()
        self.ub = mj_model.actuator_ctrlrange[:, 1].copy()

        mj_model.actuator_forcelimited[:] = 0
        mj_model.actuator_ctrllimited[:] = 0
        mj_model.actuator_ctrlrange[:] = [-jnp.inf, jnp.inf]

        super().__init__(
            mj_model,
            trace_sites=["imu_in_torso", "left_foot", "right_foot"],
        )

        self.orientation_sensor_id = mj_model.sensor("imu_in_torso_quat").id
        self.velocity_sensor_id = mj_model.sensor("imu_in_torso_linvel").id
        self.torso_id = mj_model.site("imu_in_torso").id

        self.target_height = 0.9
        self.qstand = jnp.array(mj_model.keyframe("stand").qpos)

    def _bound_violation(self, ctrl, ord=2):
        lower = jnp.maximum(self.lb - ctrl, 0)
        upper = jnp.maximum(ctrl - self.ub, 0)
        v = lower + upper
        penalty = 10 * jnp.linalg.norm(v, ord)
        penalty = jnp.where(penalty != 0, penalty + 1, penalty)
        return penalty

    def _get_torso_height(self, state: mjx.Data) -> jax.Array:
        return state.site_xpos[self.torso_id, 2]

    def _get_torso_orientation(self, state: mjx.Data) -> jax.Array:
        sensor_adr = self.model.sensor_adr[self.orientation_sensor_id]
        quat = state.sensordata[sensor_adr : sensor_adr + 4]
        upright = jnp.array([0.0, 0.0, 1.0])
        return mjx._src.math.rotate(upright, quat)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        orientation_cost = jnp.sum(
            jnp.square(self._get_torso_orientation(state))
        )
        height_cost = jnp.square(
            self._get_torso_height(state) - self.target_height
        )
        nominal_cost = jnp.sum(jnp.square(state.qpos[7:] - self.qstand[7:]))
        bound_violation_cost = self._bound_violation(control)
        return (
            10.0 * orientation_cost
            + 10.0 * height_cost
            + 0.1 * nominal_cost
            + bound_violation_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        return self.running_cost(state, jnp.zeros(self.model.nu))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        rng, q_rng, v_rng = jax.random.split(rng, 3)
        q_err = 0.01 * jax.random.normal(q_rng, (7,))
        v_err = 0.01 * jax.random.normal(v_rng, (6,))
        qpos = data.qpos.at[0:7].set(data.qpos[0:7] + q_err)
        qvel = data.qvel.at[0:6].set(data.qvel[0:6] + v_err)
        return {"qpos": qpos, "qvel": qvel}
