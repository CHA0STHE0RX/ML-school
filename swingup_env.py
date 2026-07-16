"""SwingupDoublePendulum-v0 — fixed-base, both-hinges-actuated double pendulum
swing-up. Starts hanging; the goal is to raise the tip. Weak-wrist actuation:
the shoulder motor's gear is shoulder_ratio x the elbow's (ratio 0 ~ Acrobot,
ratio 1 ~ fully actuated, ~0.2 = a weak wrist that nudges but cannot hoist).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import gymnasium as gym
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

_XML = str(Path(__file__).resolve().parent / "assets" / "swingup_double_pendulum.xml")
ELBOW_GEAR = 100.0
DEFAULT_SHOULDER_RATIO = 0.2


class SwingupDoublePendulumEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"]}

    def __init__(self, xml_file: str = _XML, frame_skip: int = 5,
                 shoulder_ratio: float = DEFAULT_SHOULDER_RATIO,
                 reset_noise_scale: float = 0.02,
                 ctrl_cost: float = 1e-4, vel_cost: float = 0.0,
                 default_camera_config: dict | None = None, **kwargs):
        # Reward = tip_height - ctrl_cost*|a|^2 - vel_cost*|w|^2. A swing-up must
        # build high joint velocities to pump energy upward, so vel_cost defaults
        # to 0 (a velocity penalty fights swing-up); calibrate_swingup.py confirmed
        # PPO holds the tip up ~68% of the time at these defaults.
        self._shoulder_ratio = float(shoulder_ratio)
        self._reset_noise_scale = float(reset_noise_scale)
        self._ctrl_cost = float(ctrl_cost)
        self._vel_cost = float(vel_cost)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        MujocoEnv.__init__(
            self, xml_file, frame_skip, observation_space=observation_space,
            default_camera_config=default_camera_config or {}, **kwargs,
        )
        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self._tip = self.model.site("tip").id
        self._shoulder_act = self.model.actuator("m_shoulder").id
        self._elbow_act = self.model.actuator("m_elbow").id
        # weak wrist: shoulder torque authority = shoulder_ratio x elbow's.
        self.model.actuator_gear[self._elbow_act, 0] = ELBOW_GEAR
        self.model.actuator_gear[self._shoulder_act, 0] = self._shoulder_ratio * ELBOW_GEAR
        # Disable MuJoCo's auto-reset-on-instability where the flag exists
        # (mujoco >= 3.x): the auto-reset restores qpos0 = BOTH POLES UPRIGHT,
        # i.e. the goal state, so an exploded episode would be teleported to
        # the goal. With the flag set, the state goes NaN instead and the
        # step() guard below ends the episode at the reward floor. The
        # warning-counter guard stays as the fallback on older mujoco.
        if hasattr(mujoco.mjtDisableBit, "mjDSBL_AUTORESET"):
            self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET
        self._warn_count = self._warning_total()

    def _warning_total(self) -> int:
        """Cumulative MuJoCo warning count on this mjData (BADQPOS/BADQVEL/...)."""
        return int(sum(w.number for w in self.data.warning))

    def _get_obs(self) -> np.ndarray:
        q, v = self.data.qpos, self.data.qvel
        return np.concatenate([np.sin(q), np.cos(q), np.clip(v, -20.0, 20.0)]).ravel()

    def _get_rew(self, action) -> tuple[float, dict]:
        tip_z = float(self.data.site_xpos[self._tip][2])
        ctrl_cost = self._ctrl_cost * float(np.sum(np.square(action)))
        vel_cost = self._vel_cost * float(np.sum(np.square(self.data.qvel)))
        reward = tip_z - ctrl_cost - vel_cost
        info = {"tip_height": tip_z, "reward_height": tip_z,
                "ctrl_cost": -ctrl_cost, "vel_cost": -vel_cost}
        return reward, info

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward, info = self._get_rew(np.asarray(action, dtype=np.float64))
        if self.render_mode == "human":
            self.render()
        # Instability guard. When MuJoCo's integrator detects NaN/huge values
        # (mjWARN_BADQPOS/BADQVEL/BADQACC — see MUJOCO_LOG.TXT, triggered under
        # heavy PHYSICS_SHIFT / delayed high-torque actions) it silently
        # auto-resets mjData to defaults — which for this model is BOTH POLES
        # UPRIGHT. Without this guard an exploded episode teleports to the goal
        # state and collects the maximum height reward. Detect the warning-
        # counter delta, score the step at the physical floor (tip fully
        # hanging — no bonus for breaking the simulator), and end the episode:
        # the measurement past this point is meaningless. Termination does let
        # a policy escape future negative reward, but reaching an explosion
        # requires extreme states a passive policy cannot cheaply produce.
        warn_now = self._warning_total()
        exploded = warn_now > self._warn_count
        self._warn_count = warn_now
        terminated = bool(exploded or not np.isfinite(self.state_vector()).all())
        if terminated:
            reward = -1.2  # tip-height floor: worst legitimate per-step reward
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            info["sim_exploded"] = True
        # truncation handled by the TimeLimit wrapper added in gym.make.
        return obs, reward, terminated, False, info

    def reset_model(self) -> np.ndarray:
        n = self._reset_noise_scale
        qpos = self.init_qpos.copy()
        qpos[0] = np.pi   # shoulder down -> hanging
        qpos[1] = 0.0     # elbow inline with pole1
        qpos = qpos + self.np_random.uniform(-n, n, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * n
        self.set_state(qpos, qvel)
        self._warn_count = self._warning_total()  # fresh episode, fresh baseline
        return self._get_obs()


def register_swingup() -> None:
    """Idempotently register SwingupDoublePendulum-v0 with gymnasium."""
    if "SwingupDoublePendulum-v0" not in gym.registry:
        gym.register(
            id="SwingupDoublePendulum-v0",
            entry_point="swingup_env:SwingupDoublePendulumEnv",
            max_episode_steps=1000,
        )


register_swingup()
