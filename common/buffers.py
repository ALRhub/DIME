from typing import Optional

import numpy as np
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples, ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from triton.language import dtype
from common.type_aliases import ReplayBufferSamplesMTNP,ReplayBufferSamplesNp

####### This class overwrites the DictReplayBuffer from stable baselines. It throws an exception when running the DMC's
# humanoid tasks because of the head height observation. Either shimmy or dmc returns it as a 1-dim
# observation, which is not aligned with the general framework. This class only takes care of dimensonality issues of
# observations during sampling from the buffer and doesn't change anything else


class DMCCompatibleDictReplayBuffer(DictReplayBuffer):
    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        # type: ignore[signature-mismatch]
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: np.atleast_3d(obs)[batch_inds, env_indices, :] for key, obs in self.observations.items()},
                                   env)
        next_obs_ = self._normalize_obs(
            {key: np.atleast_3d(obs)[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
        )

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(
                self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
        )
    
class MTReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size: int,
        observation_space,
        action_space,
        device,
        num_tasks: int,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,):
        super().__init__(buffer_size,
        observation_space,
        action_space,
        device,
        n_envs,
        optimize_memory_usage,
        handle_timeout_termination)
        self.num_tasks = num_tasks
        tasks = np.array(range(num_tasks)).repeat(int(n_envs / num_tasks), 0)
        self.task_ids = np.expand_dims(tasks, 0).repeat(self.buffer_size, 0)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamplesMTNP:
        # Sample randomly the env idx, no need to include all tasks at once(as in BRC)
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
            self.task_ids[batch_inds, env_indices].reshape(-1, 1)
        )

        return ReplayBufferSamplesMTNP(*tuple(map(self.to_torch, data)))

