"""All wrappers. Single `strength` axis in [0, 1]. strength=0 == identity."""
from __future__ import annotations
from collections import deque
import gymnasium as gym
import numpy as np
from records import ModType, ClassicAttrs, MujocoGravity, DEFAULT_PHYSICS_ATTRS

class FlickerWrapper(gym.ObservationWrapper):
    """P(obs := 0) = strength."""

    def __init__(self, env: gym.Env, strength: float = 0.0):
        super().__init__(env)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        self.strength = strength

    def observation(self, observation):
        if self.strength > 0.0 and self.np_random.random() < self.strength:
            return np.zeros_like(observation)
        # return self.np_random.uniform(low=obs_low, high=obs_high)  # random instead of zero
        # return self._last_obs  # freeze last obs
        return observation

class GaussianObsNoise(gym.ObservationWrapper):
    """obs += N(0, strength * per-dim std). The per-dim std is a property of the env,
    not of the episode, so it is estimated once from a fixed-seed random rollout and
    cached (deterministic, reproducible run-to-run)."""

    def __init__(self, env: gym.Env, strength: float = 0.0, warmup_steps: int = 200,
                 scale_seed: int = 0):
        super().__init__(env)
        self.strength = float(strength)
        self._scale: np.ndarray | None = None
        self._warmup_steps = warmup_steps
        self._scale_seed = scale_seed

    def _estimate_scale(self):
        # Seed BOTH the env reset and the action sampler. env.reset(seed=) seeds
        # env.np_random but NOT action_space's separate RNG, so action_space.sample()
        # is otherwise an unseeded side-channel that makes the scale drift run-to-run.
        self.env.action_space.seed(self._scale_seed)
        buf = []
        obs, _ = self.env.reset(seed=self._scale_seed)
        buf.append(obs)
        for _ in range(self._warmup_steps):
            obs, _, term, trunc, _ = self.env.step(self.env.action_space.sample())
            buf.append(obs)
            if term or trunc:
                obs, _ = self.env.reset(); buf.append(obs)
        arr = np.asarray(buf, dtype=np.float64)
        # Drop exploded-sim rows: one NaN observation (e.g. MuJoCo instability
        # during the random warmup) would otherwise poison the scale for the
        # whole probe. All-finite rollouts are untouched by this filter.
        finite = np.isfinite(arr).all(axis=1)
        arr = arr[finite] if finite.any() else np.zeros((1, arr.shape[1]))
        s = arr.std(axis=0)
        s[s < 1e-6] = 1.0
        self._scale = s.astype(np.float32)
        # self._scale = (env.observation_space.high - env.observation_space.low) / 6

    def reset(self, **kw):
        # Estimate the scale BEFORE the episode, then let super().reset() re-reset to
        # the caller's seed. Otherwise the warmup rollout (which resets and steps the
        # live env) leaves the episode desynced: the agent sees the seeded reset obs
        # while the env actually sits at the warmup end-state.
        if self.strength > 0.0 and self._scale is None:
            self._estimate_scale()
        return super().reset(**kw)

    def observation(self, observation):
        if self.strength <= 0.0:
            return observation
        if self._scale is None:
            self._estimate_scale()
        noise = self.np_random.normal(0.0, self.strength, size=observation.shape)
        return (observation + noise * self._scale).astype(observation.dtype)

class ActionDelayWrapper(gym.Wrapper):
    """Execute action from k = round(strength * max_delay) steps ago.

    k is an integer, so the strength axis is quantized to max_delay+1 levels,
    and Python's round() is banker's rounding (0.05 * 10 -> k=0, not 1). This
    mapping is part of the recorded measurement semantics — do not change it
    silently, or every stored ACTION_DELAY curve is re-labeled. Envs probed
    with a small s_max (e.g. swing-up at 0.2 -> k in {0, 1, 2}) resolve only a
    few distinct delays; prefer raising max_delay over shrinking s_max further.
    """

    def __init__(self, env: gym.Env, strength: float = 0.0, max_delay: int = 10):
        super().__init__(env)
        self.k = int(round(strength * max_delay))
        self._buf: deque = deque(maxlen=self.k + 1)

    def reset(self, **kw):
        self._buf.clear()
        zero = np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)
        for _ in range(self.k):
            self._buf.append(zero)
        return self.env.reset(**kw)

    def step(self, action):
        if self.k == 0:
            return self.env.step(action)
        self._buf.append(action)
        return self.env.step(self._buf.popleft())

class PhysicsShiftWrapper(gym.Wrapper):
    """Scale dynamics params by (1 +/- strength). Sign alternates per reset.

    Dispatches on the declared knob (the env's profile is the source of truth):
      - ClassicAttrs(names): scale those python attrs on env.unwrapped.
      - MujocoGravity():      scale env.unwrapped.model.opt.gravity[2].
    Default knob is ClassicAttrs(DEFAULT_PHYSICS_ATTRS) -> legacy behavior.
    """

    def __init__(self, env: gym.Env, strength: float = 0.0, knob=None):
        super().__init__(env)
        self.strength = float(strength)
        self._flip = False
        self._knob = knob if knob is not None else ClassicAttrs(DEFAULT_PHYSICS_ATTRS)
        u = self.env.unwrapped
        if isinstance(self._knob, MujocoGravity):
            if not hasattr(u, "model"):
                raise RuntimeError(
                    f"PhysicsShiftWrapper(MujocoGravity) requires a MuJoCo env with "
                    f"env.unwrapped.model, but {type(u).__name__} has none."
                )
            self._default_gravity_z = float(u.model.opt.gravity[2])
        else:  # ClassicAttrs
            self._defaults = {a: getattr(u, a) for a in self._knob.names
                              if hasattr(u, a) and isinstance(getattr(u, a), (int, float))}
            if not self._defaults:
                raise RuntimeError(
                    f"PhysicsShiftWrapper has no known physics parameters to mutate on env "
                    f"{type(u).__name__}. Add the env's param names to the knob, "
                    f"or do not run PHYSICS_SHIFT perturbations on this env."
                )

    def reset(self, **kw):
        self._flip = not self._flip
        factor = 1.0 + (1.0 if self._flip else -1.0) * self.strength
        u = self.env.unwrapped
        if isinstance(self._knob, MujocoGravity):
            u.model.opt.gravity[2] = self._default_gravity_z * factor
        else:
            for attr, default in self._defaults.items():
                setattr(u, attr, default * factor)
        # factor = 1.0 + self.np_random.uniform(-strength, strength)  # random sign each reset
        return self.env.reset(**kw)

def apply_mod(env: gym.Env, mod_type: ModType, strength: float,
              physics_knob=None) -> gym.Env:
    if mod_type == ModType.NONE or strength == 0.0:
        return env
    if mod_type == ModType.FLICKER:
        return FlickerWrapper(env, strength)
    if mod_type == ModType.GAUSSIAN_NOISE:
        return GaussianObsNoise(env, strength)
    if mod_type == ModType.ACTION_DELAY:
        return ActionDelayWrapper(env, strength)
    if mod_type == ModType.PHYSICS_SHIFT:
        return PhysicsShiftWrapper(env, strength, physics_knob)
    raise ValueError(f"Unknown mod type: {mod_type}")
