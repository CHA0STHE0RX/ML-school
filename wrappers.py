"""All wrappers. Single `strength` axis in [0, 1]. strength=0 == identity."""
from __future__ import annotations
from collections import deque
import gymnasium as gym
import numpy as np
from records import ModType

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
        s = np.asarray(buf, dtype=np.float64).std(axis=0)
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
    """Execute action from k = round(strength * max_delay) steps ago."""

    def __init__(self, env: gym.Env, strength: float = 0.0, max_delay: int = 10):
        super().__init__(env)
        self.k = int(round(strength * max_delay))
        self._buf: deque = deque(maxlen=max(self.k, 1))

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
    """Scale dynamics params by (1 +/- strength). Sign alternates per reset."""

    def __init__(self, env: gym.Env, strength: float = 0.0):
        super().__init__(env)
        self.strength = float(strength)
        self._flip = False
        u = self.env.unwrapped
        attrs = ("gravity", "masscart", "masspole", "length", "force_mag", "m", "l", "g")
        self._defaults = {a: getattr(u, a) for a in attrs
                          if hasattr(u, a) and isinstance(getattr(u, a), (int, float))}
        if not self._defaults:
            raise RuntimeError(
                f"PhysicsShiftWrapper has no known physics parameters to mutate on env "
                f"{type(u).__name__}. Add the env's param names to the `attrs` tuple, "
                f"or do not run PHYSICS_SHIFT perturbations on this env."
            )

    def reset(self, **kw):
        self._flip = not self._flip
        factor = 1.0 + (1.0 if self._flip else -1.0) * self.strength
        for attr, default in self._defaults.items():
            setattr(self.env.unwrapped, attr, default * factor)
        # factor = 1.0 + self.np_random.uniform(-strength, strength)  # random sign each reset
        return self.env.reset(**kw)

def apply_mod(env: gym.Env, mod_type: ModType, strength: float) -> gym.Env:
    if mod_type == ModType.NONE or strength == 0.0:
        return env
    if mod_type == ModType.FLICKER:
        return FlickerWrapper(env, strength)
    if mod_type == ModType.GAUSSIAN_NOISE:
        return GaussianObsNoise(env, strength)
    if mod_type == ModType.ACTION_DELAY:
        return ActionDelayWrapper(env, strength)
    if mod_type == ModType.PHYSICS_SHIFT:
        return PhysicsShiftWrapper(env, strength)
    raise ValueError(f"Unknown mod type: {mod_type}")
