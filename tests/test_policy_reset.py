"""Policy.reset() must clear stateful internals between episodes."""
from __future__ import annotations
import numpy as np
from agents.base import Policy, StatelessPolicy


def test_stateless_policy_is_callable():
    p = StatelessPolicy(lambda obs: obs * 2)
    out = p(np.array([1.0, 2.0]))
    np.testing.assert_array_equal(out, np.array([2.0, 4.0]))


def test_stateless_policy_reset_is_noop():
    p = StatelessPolicy(lambda obs: obs * 2)
    p.reset()
    out = p(np.array([1.0]))
    np.testing.assert_array_equal(out, np.array([2.0]))


def test_stateful_policy_reset_clears_state():
    class Counter(Policy):
        def __init__(self):
            self.count = 0
        def reset(self) -> None:
            self.count = 0
        def __call__(self, obs):
            self.count += 1
            return np.array([self.count], dtype=np.float32)

    c = Counter()
    c(np.zeros(1)); c(np.zeros(1)); c(np.zeros(1))
    assert c.count == 3
    c.reset()
    assert c.count == 0
    out = c(np.zeros(1))
    np.testing.assert_array_equal(out, np.array([1.0]))


def test_stateful_leak_isolated_by_reset(tmp_path):
    """Two episodes on same seed produce identical returns ONLY if reset() is called between."""
    import gymnasium as gym
    from critic import _eval_policy

    class StatefulAdder(Policy):
        """Action = constant + accumulated step count. Leaks across episodes if not reset."""
        def __init__(self):
            self.steps = 0
        def reset(self) -> None:
            self.steps = 0
        def __call__(self, obs):
            self.steps += 1
            return np.array([np.tanh(self.steps * 0.01)], dtype=np.float32)

    def env_fn():
        return gym.make("Pendulum-v1")

    p = StatefulAdder()
    m1, _ = _eval_policy(env_fn, p, n_episodes=2, seed=42)
    m2, _ = _eval_policy(env_fn, p, n_episodes=2, seed=42)
    assert abs(m1 - m2) < 1e-6, f"Reset failed to isolate state: {m1} vs {m2}"
