import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task

from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)

class CartPoleUnconstrained(Task):
    # This task is modified from Hydrax, the control limits are enforced as penalties to avoid clipping
    # https://github.com/vincekurtz/hydrax/tree/main/hydrax/tasks
    """A cart-pole swingup task."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/cart_pole_unconstrained/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["tip"])

        self.ub = jnp.array([1.])
        self.lb = jnp.array([-1.])

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
        penalty = 50 * jnp.linalg.norm(v, ord)

        # if penalty != 0 then add 1, else leave as 0: 
        penalty = jnp.where(penalty != 0, penalty + 1, penalty)

        return penalty
    
    def _distance_to_upright(self, state: mjx.Data) -> jax.Array:
        """Get a measure of distance to the upright position."""
        theta = state.qpos[1] + jnp.pi
        theta_err = jnp.array([jnp.cos(theta) - 1, jnp.sin(theta)])
        return jnp.sum(jnp.square(theta_err))

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        theta_cost = self._distance_to_upright(state)
        centering_cost = jnp.sum(jnp.square(state.qpos[0]))
        velocity_cost = 0.01 * jnp.sum(jnp.square(state.qvel))
        control_cost = 0.01 * jnp.sum(jnp.square(control)) 
        bound_violation_cost = self._bound_violation(control)

        return theta_cost + centering_cost + velocity_cost + control_cost + bound_violation_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        theta_cost = 10 * self._distance_to_upright(state)
        centering_cost = jnp.sum(jnp.square(state.qpos[0]))
        velocity_cost = 0.01 * jnp.sum(jnp.square(state.qvel))
        return theta_cost + centering_cost + velocity_cost
