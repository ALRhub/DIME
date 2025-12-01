from typing import Union, Dict, Optional, Tuple

from stable_baselines3.common.type_aliases import Schedule
import jax
import numpy as np
import jax.numpy as jnp

from functools import partial

import optax
from gymnasium import spaces
from models.critic import MTVectorCriticConcat
from diffusion.common.utils import get_sampler_init
from diffusion.od.od_integrators import get_integrator as get_integrator_od
from diffusion.od.od_sampling import sample as sample_od
from common.policies import BaseJaxPolicy
from common.type_aliases import MTRLTrainState
from stable_baselines3.common.type_aliases import Schedule

from models.utils import activation_fn
from diffusion.diffusion_policy import DiffPol
from common.task_embedder import TaskEmbedding
from hydra.utils import get_method

class MT_DiffPol(DiffPol):
    def __init__(self,observation_space: spaces.Space,
                 action_space: spaces.Box,
                 cfg,
                 squash_output: bool = True,
                 **kwargs,):
        super().__init__(observation_space,
                 action_space,
                 cfg,
                 squash_output,
                 **kwargs,)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        task_ids=None
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if not task_ids:
            task_ids = jnp.zeros(observation.shape[0])
        task_embeddings = self.qf.get_task_embeddings(task_ids)

        observation, vectorized_env = self.prepare_obs(observation)

        actions = self._predict(observation, deterministic=deterministic, task_embeddings=task_embeddings)

        # Convert to numpy, and reshape to the original action shape
        actions = np.array(actions).reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Clip due to numerical instability
                actions = np.clip(actions, -1, 1)
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)  # type: ignore[call-overload]

        return actions, state


    def _predict(self, observation: np.ndarray, task_embeddings, deterministic: bool = False) -> np.ndarray:
        if not self.use_sde:
            self.reset_noise()
        actions, *_ = MT_DiffPol.sample_action(self.actor_state, self.actor_state.params, observation, self.noise_key,
                                            self.sampler, task_embeddings=task_embeddings)
        return actions

    def build(self, key, lr_schedule: Schedule, qf_learning_rate: float):
        key, score_key, stat_distr_key, qf_key, dropout_key, stat_distr_bn_key, bn_key = jax.random.split(key, 7)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array([spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        a_dim = self.action_space.shape[0]
        obs_dim = obs.shape[1]
        task_ids = jnp.zeros(1, dtype=jnp.int8)

        # Initialize actor
        key, diff_key = jax.random.split(key, 2)
        if self.cfg.task_embedding_incorporation == 'concat':
            # initialize Q-function
            self.qf = MTVectorCriticConcat(
                dropout_rate=self.cfg.alg.critic.dropout_rate,
                use_layer_norm=self.cfg.alg.critic.use_layer_norm,
                use_batch_norm=self.cfg.alg.optimizer.bn,
                bn_warmup=self.cfg.alg.optimizer.bn_warmup,
                batch_norm_momentum=self.cfg.alg.optimizer.bn_momentum,
                batch_norm_mode=self.cfg.alg.optimizer.bn_mode,
                net_arch=self.cfg.alg.critic.hs,
                activation_fn=activation_fn[self.cfg.alg.critic.activation],
                n_critics=self.cfg.alg.critic.n_critics,
                n_atoms=self.cfg.alg.critic.n_atoms,
                n_tasks=self.cfg.n_tasks,
                task_embed_dim=self.cfg.task_embedding_dim,
            )

            qf_init_variables = self.qf.init(
                {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
                obs,
                action,
                task_ids,
                train=False,
            )
            target_qf_init_variables = self.qf.init(
                {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
                obs,
                action,
                task_ids,
                train=False,
            )

            self.qf_state = MTRLTrainState.create(
                apply_fn=self.qf.apply,
                params=qf_init_variables["params"],
                batch_stats=qf_init_variables["batch_stats"],
                target_params=target_qf_init_variables["params"],
                target_batch_stats=target_qf_init_variables["batch_stats"],
                tx=optax.adam(
                    learning_rate=qf_learning_rate,  # type: ignore[call-arg]
                    **dict({
                        'b1': self.cfg.alg.optimizer.b1,
                        'b2': 0.999  # default
                    }),
                ),
                task_embedding_incorporation=self.cfg.task_embedding_incorporation,
            )

            self.qf.apply = jax.jit(  # type: ignore[method-assign]
                self.qf.apply,
                static_argnames=("dropout_rate", "use_layer_norm",
                                 "use_batch_norm", "batch_norm_momentum", "bn_mode"),
            )
            task_embedding_dim = self.cfg.task_embedding_dim
            a_dim = a_dim + task_embedding_dim
            obs_dim = obs_dim + task_embedding_dim
            self.actor_model, self.actor_state = get_sampler_init(self.cfg.sampler.name)(diff_key, self.cfg, a_dim, obs_dim)
            target_model_state = get_sampler_init(self.cfg.sampler.name)(diff_key, self.cfg, a_dim, obs_dim)
            self.actor_target_model, self.target_actor_state = target_model_state
            self.integrator = get_integrator_od(self.cfg, self.actor_model)
            self.target_integrator = get_integrator_od(self.cfg, self.actor_target_model)
            sampler = get_method(self.cfg.sampler)
            self.sampler = partial(sampler, integrator=self.integrator, diffusion_model=self.actor_model)
            self.target_sampler = partial(sampler, integrator=self.target_integrator,
                                          diffusion_model=self.actor_target_model)
        else:
            raise NotImplementedError(f'This incorporation not implemented, see MT_DiffPol')
        self.task_embedder = TaskEmbedding(self.cfg.n_tasks, self.cfg.task_embedding_dim)
        return key

    @staticmethod
    @partial(jax.jit, static_argnames=["sampler", "return_logprob"])
    def sample_action(actor_state, actor_params, observations, key, sampler, task_embeddings, return_logprob=False):
        out = sampler(key, actor_state, actor_params, observations, stop_grad=False, task_embeddings=task_embeddings)
        # terminal costs = prior log prob loss for od and prior log prob loss - momentum loss for ud
        final_action, running_costs, stochastic_costs, terminal_costs, a_t, v_t = out
        return final_action, running_costs, stochastic_costs, terminal_costs, a_t, v_t

