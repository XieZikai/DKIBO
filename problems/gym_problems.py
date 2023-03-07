import gymnasium as gym
import numpy as np
from typing import Optional
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder


class Swimmer:
    def __init__(
        self,
        seed=None,
        num_rollouts: int = 3,
        render: bool = False,
    ):
        render_mode = "human" if render else None
        self.env = gym.make(
            "Swimmer-v4",
            render_mode=render_mode,
        )
        self.policy_shape = (
            self.env.action_space.shape[0],
            self.env.observation_space.shape[0],
        )
        self.dim = np.prod(
            self.policy_shape,
            dtype=int,
        )
        self.seed = seed
        self.counter = 0
        self.num_rollouts = num_rollouts

        self.lb = -1 * np.ones(self.dim)
        self.ub = 1 * np.ones(self.dim)
        self.input_columns = [str(i) for i in range(self.dim)]

    def __call__(self, x):
        self.counter += 1
        x = np.array(x)
        if len(x.shape) == 2:
            x = x[0]
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        M = x.reshape(self.policy_shape)

        returns = []
        observations = []
        actions = []

        for i in range(self.num_rollouts):
            obs, info = self.env.reset(seed=self.seed)
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0

            while not (terminated or truncated):
                action = np.dot(M, obs)
                observations.append(obs)
                actions.append(action)
                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.env.step(action)
                total_reward += reward
                steps += 1
                if self.env.render_mode is not None:
                    self.env.render()
            returns.append(total_reward)

        return np.mean(returns) * -1


class LunarLander:
    """This environment is a classic rocket trajectory optimization problem."""

    def __init__(
        self,
        render: bool = False,
        seed: Optional[int] = None,
    ):
        self.id = "LunarLander-v2"
        render_mode = "human" if render else None
        self.env = gym.make(
            self.id,
            render_mode=render_mode,
        )
        self.seed = seed
        self.env.action_space.seed(seed=seed)
        self.dim = 12
        self.lb = np.zeros(12)
        self.ub = 2 * np.ones(12)
        self.counter = 0
        self.input_columns = [str(i) for i in range(self.dim)]

    def heuristic_Controller(self, s, w):
        angle_targ = s[0] * w[0] + s[2] * w[1]
        if angle_targ > w[2]:
            angle_targ = w[2]
        if angle_targ < -w[2]:
            angle_targ = -w[2]
        hover_targ = w[3] * np.abs(s[0])

        angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
        hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

        if s[6] or s[7]:
            angle_todo = w[8]
            hover_todo = -(s[3]) * w[9]

        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
            a = 2
        elif angle_todo < -w[11]:
            a = 3
        elif angle_todo > +w[11]:
            a = 1
        return a

    def visualise(
        self,
        x,
        path: str,
    ):
        temp_env = gym.make(
            self.id,
            render_mode="rgb_array",
        )
        temp_env.action_space.seed(seed=self.seed)
        video_recorder = VideoRecorder(
            env=temp_env,
            path=path,
            enabled=True,
        )
        state, info = temp_env.reset(seed=self.seed)

        while True:
            temp_env.unwrapped.render()
            video_recorder.capture_frame()
            received_action = self.heuristic_Controller(
                state,
                x,
            )
            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = temp_env.step(received_action)
            state = next_state
            if terminated or truncated:
                break

        video_recorder.close()
        video_recorder.enabled = False
        temp_env.close()

    def __call__(self, x):
        self.counter += 1
        x = np.array(x)
        if len(x.shape) == 2:
            x = x[0]
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        total_rewards = []
        for i in range(0, 3):  # controls the number of episode/plays per trial
            state, info = self.env.reset(seed=self.seed)
            rewards_for_episode = []
            num_steps = 2000

            for step in range(num_steps):
                if self.env.render_mode is not None:
                    self.env.render()

                received_action = self.heuristic_Controller(
                    state,
                    x,
                )
                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.env.step(received_action)
                rewards_for_episode.append(reward)
                state = next_state
                if terminated or truncated:
                    break

            rewards_for_episode = np.array(rewards_for_episode)
            total_rewards.append(np.sum(rewards_for_episode))
        total_rewards = np.array(total_rewards)
        mean_rewards = np.mean(total_rewards)

        return mean_rewards * -1
