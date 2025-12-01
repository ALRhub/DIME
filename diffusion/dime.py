import os
import jax
import flax
import optax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from gymnasium import spaces
from functools import partial

from diffusion.diffusion_policy import DiffPol
from flax.training.train_state import TrainState
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from common.off_policy_algorithm import OffPolicyAlgorithmJax
from common.type_aliases import ReplayBufferSamplesNp, RLTrainState
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self, step) -> jnp.ndarray:
        log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)))
        return jnp.exp(log_ent_coef)


class ConstantEntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self, step) -> float:
        # Hack to not optimize the entropy coefficient while not having to use if/else for the jit
        self.param("dummy_param", init_fn=lambda key: jnp.full((), self.ent_coef_init))
        return jax.lax.stop_gradient(self.ent_coef_init)


class DIME(OffPolicyAlgorithmJax):
    policy_aliases: ClassVar[Dict[str, Type[DiffPol]]] = {  # type: ignore[assignment]
        "MlpPolicy": DiffPol,
        # Minimal dict support using flatten()
        "MultiInputPolicy": DiffPol,
    }

    policy: DiffPol
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(self,
                 policy,
                 env: Union[GymEnv, str],
                 model_save_path: str,
                 save_every_n_steps: int,
                 cfg,
                 train_freq: Union[int, Tuple[int, str]] = 1,
                 action_noise: Optional[ActionNoise] = None,
                 replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
                 replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 use_sde_at_warmup: bool = False,
                 tensorboard_log: Optional[str] = None,
                 verbose: int = 0,
                 _init_setup_model: bool = True,
                 stats_window_size: int = 100,
                 ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=cfg.alg.optimizer.lr_actor,
            qf_learning_rate=cfg.alg.optimizer.lr_critic,
            buffer_size=cfg.alg.buffer_size,
            learning_starts=cfg.alg.learning_starts,
            batch_size=cfg.alg.batch_size,
            tau=cfg.alg.tau,
            gamma=cfg.alg.gamma,
            train_freq=train_freq,
            gradient_steps=cfg.alg.utd,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            policy_kwargs=None,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=cfg.seed,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
            stats_window_size=stats_window_size,
        )
        self.cfg = cfg
        self.policy_delay = self.cfg.alg.policy_delay
        self.ent_coef_params = self.cfg.alg.ent_coef
        self.crossq_style = True
        self.use_bnstats_from_live_net = False
        self.policy_q_reduce_fn = jax.numpy.mean
        self.save_every_n_steps = save_every_n_steps
        self.model_save_path = model_save_path
        self.policy_tau = self.cfg.alg.policy_tau
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self, reset=False) -> None:
        if not reset:
            super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None or reset:
            # pytype: disable=not-instantiable
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.action_space,
                self.cfg
            )
            # pytype: enable=not-instantiable

            assert isinstance(self.qf_learning_rate, float)

            self.key = self.policy.build(self.key, self.lr_schedule, self.qf_learning_rate)

            self.key, ent_key = jax.random.split(self.key, 2)

            self.qf = self.policy.qf  # type: ignore[assignment]

            # The entropy coefficient or entropy can be learned automatically
            # see Automating Entropy Adjustment for Maximum Entropy RL section
            # of https://arxiv.org/abs/1812.05905
            if self.ent_coef_params["type"] == "auto":
                    ent_coef_init = self.ent_coef_params['init']
                    # Note: we optimize the log of the entropy coeff which is slightly different from the paper
                    # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
                    self.ent_coef = EntropyCoef(ent_coef_init)
            elif self.ent_coef_params["type"] == "const":
                # This will throw an error if a malformed string (different from 'auto') is passed
                assert isinstance(
                    self.ent_coef_params["init"], float
                ), f"Entropy coef must be float when not equal to 'auto', actual: {self.ent_coef_params['init']}"
                self.ent_coef = ConstantEntropyCoef(self.ent_coef_params["init"])  # type: ignore[assignment]
            else:
                raise NotImplementedError(f"Entropy coefficient type {self.ent_coef_params['type']} not supported")

            self.ent_coef_state = TrainState.create(
                apply_fn=self.ent_coef.apply,
                params=self.ent_coef.init({"params": ent_key}, 0.0)["params"],
                tx=optax.adam(
                    # learning_rate=self.learning_rate,
                    learning_rate=1.0e-3,
                ),
            )

            # automatically set target entropy if needed
            self.target_entropy = self.action_space.shape[0] * 4.0

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def train(self, batch_size, gradient_steps):
        # Sample all at once for efficiency (so we can jit the for loop)
        data = self.replay_buffer.sample(batch_size * gradient_steps, env=self._vec_normalize_env)
        # Pre-compute the indices where we need to update the actor
        # This is a hack in order to jit the train loop
        # It will compile once per value of policy_delay_indices
        policy_delay_indices = {i: True for i in range(gradient_steps) if
                                ((self._n_updates + i + 1) % self.policy_delay) == 0}
        policy_delay_indices = flax.core.FrozenDict(policy_delay_indices)

        if isinstance(data.observations, dict):
            keys = list(self.observation_space.keys())
            obs = np.concatenate([data.observations[key].numpy() for key in keys], axis=1)
            next_obs = np.concatenate([data.next_observations[key].numpy() for key in keys], axis=1)
        else:
            obs = data.observations.numpy()
            next_obs = data.next_observations.numpy()

        # Convert to numpy
        data = ReplayBufferSamplesNp(
            obs,
            data.actions.numpy(),
            next_obs,
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
        )

        (
            self.policy.qf_state,
            self.policy.actor_state,
            self.policy.target_actor_state,
            self.ent_coef_state,
            self.key,
            log_metrics,
        ) = self._train(
            self.crossq_style,
            self.use_bnstats_from_live_net,
            self.gamma,
            self.tau,
            self.policy_tau,
            self.target_entropy,
            gradient_steps,
            data,
            policy_delay_indices,
            self.policy.qf_state,
            self.policy.actor_state,
            self.policy.target_actor_state,
            self.ent_coef_state,
            self.key,
            self.num_timesteps,
            self.policy_q_reduce_fn,
            self.policy.sampler,
            self.policy.target_sampler,
            self.cfg.alg.critic.v_min,
            self.cfg.alg.critic.v_max,
            self.cfg.alg.critic.entr_coeff,
            self.cfg.alg.critic.n_atoms
        )
        self._n_updates += gradient_steps

        if self.model_save_path is not None:
            if (self.num_timesteps % self.save_every_n_steps == 0) or (self.num_timesteps == (self.learning_starts+1)):
                self._save_model()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for k, v in log_metrics.items():
            try:
                log_val = v.item()
            except:
                log_val = v
            self.logger.record(f"train/{k}", log_val)

    @staticmethod
    @partial(jax.jit, static_argnames=["crossq_style", "use_bnstats_from_live_net", "sampler", "num_atoms", "z_atoms",
                                       "v_min", "v_max", "entr_coeff"])
    def update_critic(
            crossq_style: bool,
            use_bnstats_from_live_net: bool,
            gamma: float,
            actor_state: TrainState,
            qf_state: RLTrainState,
            ent_coef_state: TrainState,
            observations: np.ndarray,
            actions: np.ndarray,
            next_observations: np.ndarray,
            rewards: np.ndarray,
            dones: np.ndarray,
            n_env_interacts: int,
            num_atoms: int,
            z_atoms: jnp.ndarray,
            v_min: int,
            v_max: int,
            entr_coeff: float,
            key,
            sampler
    ):
        key, noise_key, dropout_key_target, dropout_key_current, redq_key = jax.random.split(key, 5)
        # sample action from the actor

        out = DiffPol.sample_action(actor_state, actor_state.params, next_observations, noise_key, sampler)
        all_actions, next_run_costs, next_sto_costs, next_terminal_costs, latents, v_t = out
        next_state_actions = jax.lax.stop_gradient(all_actions)
        next_run_costs = jax.lax.stop_gradient(next_run_costs)
        next_sto_costs = jax.lax.stop_gradient(next_sto_costs)
        next_terminal_costs = jax.lax.stop_gradient(next_terminal_costs)

        ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params}, n_env_interacts)

        def ce_loss(params, batch_stats, dropout_key):
            if not crossq_style:
                next_q_values = qf_state.apply_fn(
                    {
                        "params": qf_state.target_params,
                        "batch_stats": qf_state.target_batch_stats if not use_bnstats_from_live_net else batch_stats
                    },
                    next_observations, next_state_actions,
                    rngs={"dropout": dropout_key_target},
                    train=False
                )

                # shape is (n_critics, batch_size, 1)
                current_q_values, state_updates = qf_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    observations, actions,
                    rngs={"dropout": dropout_key},
                    mutable=["batch_stats"],
                    train=True,
                )

            else:
                # ----- CrossQ's One Weird Trick™ -----
                # concatenate current and next observations to double the batch size
                # new shape of input is (n_critics, 2*batch_size, obs_dim + act_dim)
                # apply critic to this bigger batch
                catted_q_values, state_updates = qf_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    jnp.concatenate([observations, next_observations], axis=0),
                    jnp.concatenate([actions, next_state_actions], axis=0),
                    rngs={"dropout": dropout_key},
                    mutable=["batch_stats"],
                    train=True,
                )
                current_q_values, next_q_values = jnp.split(catted_q_values, 2, axis=1)

            if next_q_values.shape[0] > 2:  # only for REDQ
                # REDQ style subsampling of critics.
                m_critics = 2
                next_q_values = jax.random.choice(redq_key, next_q_values, (m_critics,), replace=False, axis=0)

            next_q_values_q1 = next_q_values[0]
            next_q_values_q2 = next_q_values[1]

            current_q1 = current_q_values[0]
            current_q2 = current_q_values[1]

            def projection(next_dist, rewards, dones, gamma, v_min, v_max, num_atoms, support):
                delta_z = (v_max - v_min) / (num_atoms - 1)
                batch_size = rewards.shape[0]

                entr_bon = - (1 - dones[:, None]) * gamma * ent_coef_value * (next_run_costs + next_sto_costs + next_terminal_costs)

                # Compute target_z
                target_z = jnp.clip(rewards[:,None] + entr_bon + (1 - dones[:, None]) * gamma * support, a_min=v_min, a_max=v_max)
                b = (target_z - v_min) / delta_z
                l = jnp.floor(b).astype(jnp.int32)
                u = jnp.ceil(b).astype(jnp.int32)

                # Adjust l and u to ensure they remain within valid bounds
                l = jnp.where((u > 0) & (l == u), l - 1, l)
                u = jnp.where((l < (num_atoms - 1)) & (l == u), u + 1, u)

                # Create the projected distribution
                proj_dist = jnp.zeros_like(next_dist)

                # Offset calculation for batch indexing
                offset = jnp.arange(batch_size)[:, None] * num_atoms
                # offset = jnp.tile(offset, (1, num_atoms))  # Repeat along the second axis

                # Index updates for proj_dist
                l_idx = (l + offset).ravel()
                u_idx = (u + offset).ravel()

                # Flattened updates
                l_update = (next_dist * (u.astype(jnp.float32) - b)).ravel()
                u_update = (next_dist * (b - l.astype(jnp.float32))).ravel()

                # Flatten proj_dist for updates
                proj_dist_flat = proj_dist.ravel()

                # Add values to proj_dist
                proj_dist_flat = proj_dist_flat.at[l_idx].add(l_update)
                proj_dist_flat = proj_dist_flat.at[u_idx].add(u_update)

                # Reshape back to [batch_size, num_atoms]
                proj_dist = proj_dist_flat.reshape(batch_size, num_atoms)

                return proj_dist

            target_q1_projected = projection(next_dist=next_q_values_q1, rewards=rewards, dones=dones, gamma=gamma,
                                             v_min=v_min, v_max=v_max, num_atoms=num_atoms, support=z_atoms)
            target_q2_projected = projection(next_dist=next_q_values_q2, rewards=rewards, dones=dones, gamma=gamma,
                                             v_min=v_min, v_max=v_max, num_atoms=num_atoms, support=z_atoms)

            next_q_values = jax.lax.stop_gradient(jnp.mean(
                jnp.stack([target_q1_projected, target_q2_projected], axis=0), axis=0))

            def binary_cross_entropy(pred, target):
                return -jnp.mean(jnp.sum(target * jnp.log(pred + 1e-15), axis=-1)) + entr_coeff*jnp.mean(jnp.sum(pred*jnp.log(pred + 1e-15), axis=-1)) #+ (1 - target) * jnp.log(1 - pred + 1e-15))

            loss = binary_cross_entropy(current_q1, next_q_values) + binary_cross_entropy(current_q2, next_q_values)
            qf_pi1 = jnp.sum(current_q1 * z_atoms, axis=-1)
            qf_pi2 = jnp.sum(current_q2 * z_atoms, axis=-1)
            entr_1 = -jnp.mean(jnp.sum(current_q1 * jnp.log(current_q1 + 1e-15), axis=-1))
            entr_2 = -jnp.mean(jnp.sum(current_q2 * jnp.log(current_q2 + 1e-15), axis=-1))
            min_qf_pi = jax.lax.stop_gradient(jnp.min(jnp.stack([qf_pi1, qf_pi2], axis=0), axis=0).squeeze())
            return loss, (state_updates, min_qf_pi, next_q_values, entr_1, entr_2)

        (qf_loss_value, (state_updates, current_q_values, next_q_values, entr_1, entr_2)), grads = \
            jax.value_and_grad(ce_loss, has_aux=True)(qf_state.params, qf_state.batch_stats, dropout_key_current)

        qf_state = qf_state.apply_gradients(grads=grads)
        qf_state = qf_state.replace(batch_stats=state_updates["batch_stats"])

        metrics = {
            'critic_loss': qf_loss_value,
            'ent_coef': ent_coef_value,
            'current_q_values': current_q_values.mean(),
            'next_q_values': next_q_values.mean(),
            'entrQ_1': entr_1,
            'entrQ_2': entr_2,
        }
        return qf_state, metrics, key

    @staticmethod
    @partial(jax.jit, static_argnames=["q_reduce_fn", "sampler"])
    def update_actor(
            actor_state: TrainState,
            qf_state: RLTrainState,
            ent_coef_state: TrainState,
            observations: np.ndarray,
            n_env_interacts: int,
            key,
            z_atoms: jnp.ndarray,
            sampler,
            q_reduce_fn,
    ):
        key, dropout_key, noise_key = jax.random.split(key, 3)

        def actor_loss(actor_state_in, actor_params):
            out = DiffPol.sample_action(actor_state_in, actor_params, observations, noise_key, sampler)
            actions, run_costs, sto_costs, terminal_costs, latents, v_t = out
            qf_pi = qf_state.apply_fn(
                {
                    "params": qf_state.params,
                    "batch_stats": qf_state.batch_stats
                },
                observations,
                actions,
                rngs={"dropout": dropout_key}, train=False
            )

            qf_pi1 = jnp.sum(qf_pi[0] * z_atoms, axis=-1)
            qf_pi2 = jnp.sum(qf_pi[1] * z_atoms, axis=-1)
            min_qf_pi = q_reduce_fn(jnp.stack([qf_pi1, qf_pi2], axis=0), axis=0).squeeze()
            ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params}, n_env_interacts)
            actor_loss = (- min_qf_pi + ent_coef_value * (run_costs.squeeze() + sto_costs.squeeze() + terminal_costs.squeeze())).mean()

            max_actions = jnp.max(jnp.max(latents, axis=0), axis=1)
            min_actions = jnp.min(jnp.min(latents, axis=0), axis=1)
            mean_actions = jnp.mean(jnp.mean(latents, axis=0), axis=1)

            latent_acts = {'max_la': max_actions, 'min_la': min_actions, 'mean_la': mean_actions}

            return actor_loss, (run_costs.mean(), sto_costs.mean(), terminal_costs.mean(), latent_acts)

        outs = jax.value_and_grad(actor_loss, has_aux=True, argnums=1)(actor_state, actor_state.params)
        (act_loss_value, (run_costs_mean, sto_costs, terminal_costs, latent_acts)), grads = outs
        actor_state = actor_state.apply_gradients(grads=grads)
        metrics = {"entropy": 0.0,
                   "run_costs": run_costs_mean,
                   "sto_costs": sto_costs,
                   "terminal_costs": terminal_costs,
        }
        return actor_state, qf_state, act_loss_value, key, [metrics, latent_acts]

    @staticmethod
    @jax.jit
    def soft_update(tau: float, qf_state: RLTrainState):
        qf_state = qf_state.replace(
            target_params=optax.incremental_update(qf_state.params, qf_state.target_params, tau))
        qf_state = qf_state.replace(
            target_batch_stats=optax.incremental_update(qf_state.batch_stats, qf_state.target_batch_stats, tau))
        return qf_state

    @staticmethod
    @jax.jit
    def soft_update_target_actor(tau: float, actor_state: TrainState, target_actor_state: TrainState):
        target_actor_state = target_actor_state.replace(
            params=optax.incremental_update(actor_state.params, target_actor_state.params, tau))
        return target_actor_state

    @staticmethod
    @jax.jit
    def update_temperature(target_entropy: np.ndarray, ent_coef_state: TrainState, entropy: float):
        def temperature_loss(temp_params):
            ent_coef_value = ent_coef_state.apply_fn({"params": temp_params}, 0)
            ent_coef_loss = -ent_coef_value * (entropy - target_entropy).mean()
            return ent_coef_loss

        ent_coef_loss, grads = jax.value_and_grad(temperature_loss)(ent_coef_state.params)
        ent_coef_state = ent_coef_state.apply_gradients(grads=grads)

        return ent_coef_state, ent_coef_loss

    @classmethod
    @partial(jax.jit,
             static_argnames=["cls", "crossq_style", "use_bnstats_from_live_net", "gradient_steps", "q_reduce_fn",
                              "sampler", "target_sampler", "v_min", "v_max", "num_atoms", "entr_coeff"])
    def _train(
            cls,
            crossq_style: bool,
            use_bnstats_from_live_net: bool,
            gamma: float,
            tau: float,
            policy_tau: float,
            target_entropy: np.ndarray,
            gradient_steps: int,
            data: ReplayBufferSamplesNp,
            policy_delay_indices: flax.core.FrozenDict,
            qf_state: RLTrainState,
            actor_state: TrainState,
            target_actor_state: TrainState,
            ent_coef_state: TrainState,
            key,
            n_env_interacts,
            q_reduce_fn,
            sampler,
            target_sampler,
            v_min,
            v_max,
            entr_coeff,
            num_atoms
    ):
        actor_loss_value = jnp.array(0)
        actor_metrics = [{}]
        for i in range(gradient_steps):

            def slice(x, step=i):
                assert x.shape[0] % gradient_steps == 0
                batch_size = x.shape[0] // gradient_steps
                return x[batch_size * step: batch_size * (step + 1)]

            z_atoms = jnp.linspace(v_min,  v_max, num_atoms)

            (
                qf_state,
                log_metrics_critic,
                key,
            ) = cls.update_critic(
                crossq_style,
                use_bnstats_from_live_net,
                gamma,
                target_actor_state,
                qf_state,
                ent_coef_state,
                slice(data.observations),
                slice(data.actions),
                slice(data.next_observations),
                slice(data.rewards),
                slice(data.dones),
                n_env_interacts,
                num_atoms,
                z_atoms,
                v_min,
                v_max,
                entr_coeff,
                key,
                target_sampler
            )
            qf_state = DIME.soft_update(tau, qf_state)
            target_actor_state = target_actor_state
            # hack to be able to jit (n_updates % policy_delay == 0)
            # a = False
            if i in policy_delay_indices:  # and a:
                (actor_state, qf_state, actor_loss_value, key, actor_metrics) = cls.update_actor(
                    actor_state,
                    qf_state,
                    ent_coef_state,
                    slice(data.observations),
                    n_env_interacts,
                    key,
                    z_atoms,
                    sampler,
                    q_reduce_fn,
                )
                ent_coef_state, _ = DIME.update_temperature(target_entropy, ent_coef_state,
                                                           actor_metrics[0]['run_costs'])

                target_actor_state = DIME.soft_update_target_actor(policy_tau, actor_state, target_actor_state)
        log_metrics = {'actor_loss': actor_loss_value, **actor_metrics[0], **log_metrics_critic}
        return qf_state, actor_state, target_actor_state, ent_coef_state, key, log_metrics

    def predict_critic(self, observation, action):
        return self.policy.predict_critic(observation, action)

    def current_entropy_coeff(self):
        return self.ent_coef_state.apply_fn({"params": self.ent_coef_state.params})

    def _save_model(self):
        save_model_state(self.policy.actor_state, self.model_save_path, "actor_state", self.num_timesteps)
        save_model_state(self.policy.qf_state, self.model_save_path, "critic_state", self.num_timesteps)

    def load_model(self, path, n_steps_actor, n_steps_critic):
        self.policy.actor_state = load_state(path, "actor_state", n_steps_actor, train_state=self.policy.actor_state)
        self.policy.qf_state = load_state(path, "critic_state", n_steps_critic, train_state=self.policy.qf_state)


# Save and load model
def save_model_state(train_state, path, name, n_steps):
    # Serialize the model parameters, 'hb/h1hand-run-v0'
    serialized_state = flax.serialization.to_bytes(train_state)
    os.makedirs(path, exist_ok=True)
    extended_path = os.path.join(path, f'{name}_{n_steps}.msgpack')
    # Save the serialized parameters to a file
    with open(extended_path, 'wb') as f:
        f.write(serialized_state)


def load_state(path, name, n_steps, train_state=None):
    extended_path = os.path.join(path, f'{name}_{n_steps}.msgpack')
    # Load the serialized parameters from a file
    with open(extended_path, 'rb') as f:
        train_state_loaded = f.read()
    return flax.serialization.from_bytes(train_state, train_state_loaded)