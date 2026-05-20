"""Hydrax-style Unitree Go2 walking task (trot gait, penalty-bound controls).

Model source: MuJoCo Menagerie's unitree_go2 (vendored under
traj_opt/models/unitree_go2/). Cost is adapted from
signal_mpc/envs/unitree_go2_env.py (Brax-based env in
/home/jianghan/Workspace/mppi) and rewritten in Hydrax style: only
running_cost / terminal_cost, no Brax state-info bookkeeping. Targets
(velocity, height, yaw) are constants and the gait phase is read directly
from state.time.
"""

import os
from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from hydrax.task_base import Task


GO2_SCENE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "unitree_go2", "scene_mjx.xml",
)


def _get_foot_step(
    duty_ratio: float,
    cadence: float,
    amplitude: float,
    phases: jax.Array,
    time: jax.Array,
) -> jax.Array:
    """Target foot z-height per leg at time t. Lifted from mppi.utils."""
    def step_height(t, footphase, duty_ratio):
        angle = (t + jnp.pi - footphase) % (2 * jnp.pi) - jnp.pi
        angle = jnp.where(duty_ratio < 1, angle * 0.5 / (1 - duty_ratio), angle)
        clipped = jnp.clip(angle, -jnp.pi / 2, jnp.pi / 2)
        value = jnp.where(duty_ratio < 1, jnp.cos(clipped), 0.0)
        return jnp.where(jnp.abs(value) >= 1e-6, jnp.abs(value), 0.0)

    h = amplitude * jax.vmap(step_height, in_axes=(None, 0, None))(
        time * 2 * jnp.pi * cadence + jnp.pi, phases, duty_ratio
    )
    return h


class Go2WalkUnconstrained(Task):
    """Unitree Go2 forward trot, position-controlled (penalty-bound)."""

    GAIT_WEIGHT = 0.1  # subclass override to 0 for the "no-gait" variant

    def __init__(self) -> None:
        mj_model = mujoco.MjModel.from_xml_path(GO2_SCENE)

        # ----- physics patches for stable dt=0.02 with MJX -----
        # Menagerie's stock Go2 model (kp=50, joint damping=2, ls_iterations=5,
        # pyramidal friction cone) is tuned for its native dt=0.002 and blows
        # up in mjx at dt=0.02 even with home-pose controls. The patches below
        # match the verified-stable settings used by mppi/dial-mpc/soft-mpc
        # (kp=30, damping=0, ls=20, elliptic cone). Physical identity (mesh,
        # mass, joint topology) is unchanged.
        mj_model.opt.ls_iterations = 20
        # Note: cone="pyramidal" (menagerie default) is more vmap-stable than
        # elliptic on this model + nu=12; elliptic produced cross-sample NaN
        # in mjx batched rollouts at batch=64.
        # Position controller: kp=30, kd=0.5 (mppi-verified at dt=0.02).
        # Tested kp=500+dampratio=1 (Humanoid-style) — too stiff for Go2's
        # small leg joints. Critical damping (kd≈2 via mj_fullM) also failed
        # empirically.
        mj_model.actuator_gainprm[:, 0] = 30.0
        mj_model.actuator_biasprm[:, 1] = -30.0
        mj_model.dof_damping[6:] = 0.0  # skip 6-DOF base free joint

        # The position-controlled actuators in menagerie already have
        # ctrlrange set (per-joint). We keep those bounds for the penalty and
        # then disable hard clipping — same pattern as the other *Unconstrained
        # tasks in this repo.
        self.lb = mj_model.actuator_ctrlrange[:, 0].copy()
        self.ub = mj_model.actuator_ctrlrange[:, 1].copy()
        mj_model.actuator_forcelimited[:] = 0
        mj_model.actuator_ctrllimited[:] = 0
        mj_model.actuator_ctrlrange[:] = [-jnp.inf, jnp.inf]

        # ALSO disable joint hard limits — sampled controls (±3σ from home)
        # can push joint targets past the model's <joint range="...">, and
        # MuJoCo's joint-limit spring is too stiff at dt=0.02 → NaN. The
        # bound-violation penalty in the cost handles bounds softly instead.
        mj_model.jnt_limited[:] = 0

        super().__init__(mj_model, trace_sites=["FL_foot", "FR_foot", "RL_foot", "RR_foot"])

        # Foot sites for the gait reward and torso body for base pose.
        self.foot_site_ids = jnp.array([
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        ])
        self.torso_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )

        # Default standing pose (used as the joint regulariser target AND as
        # the algorithm's initial control mean — position-controlled actuators
        # would drive the joints from home (0.9, -1.8) to zero if the mean
        # started at zero, immediately destabilizing the robot).
        self.qstand = jnp.array(mj_model.keyframe("home").qpos)
        self._default_pose = self.qstand[7:]
        self.default_action = self._default_pose

        # Trot gait parameters (matching unitree_go2_trot.yaml).
        self.duty_ratio = 0.45
        self.cadence = 2.0
        self.amplitude = 0.08
        self.phases = jnp.array([0.0, 0.5, 0.5, 0.0])  # FL, FR, RL, RR (trot)

        # Locomotion targets.
        self.target_height = 0.28
        self.target_vx = 1.0
        self.target_vy = 0.0
        self.target_yaw = 0.0

    def _bound_violation(self, ctrl, ord: int = 2) -> jax.Array:
        lower = jnp.maximum(self.lb - ctrl, 0.0)
        upper = jnp.maximum(ctrl - self.ub, 0.0)
        v = lower + upper
        penalty = 10.0 * jnp.linalg.norm(v, ord)
        penalty = jnp.where(penalty != 0, penalty + 1.0, penalty)
        return penalty

    def _torso_quat(self, state: mjx.Data) -> jax.Array:
        # Free joint quaternion is qpos[3:7] in (w, x, y, z) convention.
        return state.qpos[3:7]

    def _torso_z_world(self, state: mjx.Data) -> jax.Array:
        # Rotate body z-axis into world frame and return it.
        return mjx._src.math.rotate(jnp.array([0.0, 0.0, 1.0]), self._torso_quat(state))

    def _torso_yaw(self, state: mjx.Data) -> jax.Array:
        q = self._torso_quat(state)
        w, x, y, z = q[0], q[1], q[2], q[3]
        return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _stage_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        # Height tracking.
        z = state.qpos[2]
        height_cost = jnp.square(z - self.target_height)

        # Upright (body z vs world z).
        body_z = self._torso_z_world(state)
        upright_cost = jnp.sum(jnp.square(body_z - jnp.array([0.0, 0.0, 1.0])))

        # Body-frame linear velocity. world-frame qvel rotated into body frame.
        vel_world = state.qvel[0:3]
        quat = self._torso_quat(state)
        # rotate by inverse quat = conjugate: (w, -x, -y, -z)
        quat_inv = quat * jnp.array([1.0, -1.0, -1.0, -1.0])
        vel_body = mjx._src.math.rotate(vel_world, quat_inv)
        vel_cost = jnp.square(vel_body[0] - self.target_vx) + \
                   jnp.square(vel_body[1] - self.target_vy)

        # Yaw tracking.
        yaw = self._torso_yaw(state)
        d_yaw = yaw - self.target_yaw
        # wrap to [-pi, pi]
        yaw_cost = jnp.square(jnp.arctan2(jnp.sin(d_yaw), jnp.cos(d_yaw)))

        # Joint regulariser to the home pose.
        joint_reg_cost = jnp.sum(jnp.square(state.qpos[7:] - self._default_pose))

        # Gait reward: track per-foot z target derived from the trot phase.
        z_feet = state.site_xpos[self.foot_site_ids, 2]
        z_feet_tar = _get_foot_step(
            self.duty_ratio, self.cadence, self.amplitude, self.phases, state.time
        )
        # Same scaling as in the source env: (z*20)^2 = 400 * z^2.
        gait_cost = 400.0 * jnp.sum(jnp.square(z_feet - z_feet_tar))

        return (
            5.0 * height_cost
            + 0.5 * upright_cost
            + 1.0 * vel_cost
            + 0.3 * yaw_cost
            + 0.1 * joint_reg_cost
            + self.GAIT_WEIGHT * gait_cost
        )

    _NAN_REPLACEMENT = 1.0e4

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        c = self._stage_cost(state, control) + self._bound_violation(control)
        # If the physics went unstable for this sample (state NaN), replace
        # the resulting NaN cost with a large finite value so the MPPI
        # softmax-weighting is not poisoned across samples.
        return jnp.where(jnp.isnan(c), self._NAN_REPLACEMENT, c)

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        c = self._stage_cost(state, jnp.zeros(self.model.nu))
        return jnp.where(jnp.isnan(c), self._NAN_REPLACEMENT, c)

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


class Go2WalkNoGaitUnconstrained(Go2WalkUnconstrained):
    """Same as Go2WalkUnconstrained but with gait reward disabled (gait_weight=0).
    Tests whether the algorithm can discover walking purely from velocity +
    height + upright + yaw + joint regularization, without phase guidance."""
    GAIT_WEIGHT = 0.0
