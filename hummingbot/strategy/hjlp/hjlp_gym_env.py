import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple

# Placeholder import for the hJLPSystem orchestrator
# from .hjlp_system import hJLPSystem

class HJLPEnv(gym.Env):
    """
    Custom Gym environment for the hJLP RL Open Games system.
    Wires in the modular agent orchestrator and exposes RL hooks.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, hjlp_system: Any):
        super().__init__()
        self.hjlp = hjlp_system
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # Match observation space to actual observation size (15 features)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.state = None
        self.current_step = 0
        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Resets the environment and returns the initial observation."""
        super().reset(seed=seed)
        self.current_step = 0
        self.hjlp.reset()
        self.state = self.hjlp.get_observation()
        return self.state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executes one step in the environment using the provided action.
        Returns (obs, reward, terminated, truncated, info).
        """
        result = self.hjlp.step_with_action(action)
        self.state = self.hjlp.get_observation()
        reward = result['reward_signal'].total_reward
        terminated = result.get('done', False)
        truncated = self.current_step >= self.hjlp.max_steps
        info = result.get('info', {})
        self.current_step += 1
        return self.state, reward, terminated, truncated, info

    def render(self):
        # Optional: implement visualization of environment state
        print(f"Step: {self.current_step}, State: {self.state}")

    def close(self):
        # Optional: cleanup
        pass 