import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class WalkerUnconstrained(Task):
    """A planar biped tasked with walking forward."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/walker/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["torso_site"])

        # Get sensor ids
        self.torso_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_position"
        )
        self.torso_velocity_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_subtreelinvel"
        )
        self.torso_zaxis_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_zaxis"
        )

        self.ub = [1., 1., 1., 1., 1., 1.]
        self.ub = [-1., -1., -1., -1., -1., -1.]

        # Set the target velocity (m/s) and height
        # TODO: make these parameters
        self.target_velocity = 1.5
        self.target_height = 1.2

    def _bound_violation(self, ctrl, ord=2):
        """
        Constrained are enforced using eqn(14) of this paper: https://arxiv.org/pdf/2404.10395
        Return ‖v‖ where v is the element-wise violation of controls
        against [lb, ub].
        """
        lower = jnp.maximum(self.lb - ctrl, 0)   # only where A is below lb
        upper = jnp.maximum(ctrl - self.ub, 0)   # only where A is above ub
        v = lower + upper                # violation vector

        penalty = 50 * jnp.linalg.norm(v, ord)

        # if penalty != 0 then add 1, else leave as 0: 
        penalty = jnp.where(penalty != 0, penalty + 1, penalty)

        return penalty

    def _get_torso_height(self, state: mjx.Data) -> jax.Array:
        """Get the height of the torso above the ground."""
        sensor_adr = self.model.sensor_adr[self.torso_position_sensor]
        return state.sensordata[sensor_adr + 2]  # px, py, pz

    def _get_torso_velocity(self, state: mjx.Data) -> jax.Array:
        """Get the horizontal velocity of the torso."""
        sensor_adr = self.model.sensor_adr[self.torso_velocity_sensor]
        return state.sensordata[sensor_adr]

    def _get_torso_deviation_from_upright(self, state: mjx.Data) -> jax.Array:
        """Get the deviation of the torso from the upright position."""
        sensor_adr = self.model.sensor_adr[self.torso_zaxis_sensor]
        return state.sensordata[sensor_adr + 2] - 1.0

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        state_cost = self.terminal_cost(state)
        control_cost = jnp.sum(jnp.square(control))
        bound_violation_cost = self._bound_violation(control)

        return state_cost + 0.1 * control_cost + bound_violation_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
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
