import jax.numpy as jnp
import jax

import numpyro.distributions as npdist
import wandb
import matplotlib.pyplot as plt
from flax import traverse_util


def inverse_softplus(x):
    # Numerically stable implementation of inverse softplus
    # Threshold above which the approximation log(e^x - 1) ≈ x is used
    threshold = 20.0
    return jnp.where(x > threshold, x, jnp.log(jnp.expm1(x)))


def check_stop_grad(expression, stop_grad):
    return jax.lax.stop_gradient(expression) if stop_grad else expression


def sample_kernel(rng_key, mean, scale):
    eps = jax.random.normal(rng_key, shape=(mean.shape[0],))
    return mean + scale * eps


def log_prob_kernel(x, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(x)


def avg_list_entries(list, num):
    assert len(list) >= num
    print(range(0, len(list) - num))
    return [sum(list[i:i + num]) / float(num) for i in range(0, len(list) - num + 1)]


def reverse_transition_params(transition_params):
    flattened_params, tree = jax.tree_util.tree_flatten(transition_params, is_leaf=None)
    reversed_flattened_params = list(map(lambda w: jnp.flip(w, axis=0), flattened_params))
    return jax.tree_util.tree_unflatten(tree, reversed_flattened_params)


def interpolate_values(values, X):
    # Compute the interpolated values
    interpolated_values = [X] + [X + (X / 2 - X) * t for t in values[1:-1]] + [X / 2]
    return interpolated_values


def flattened_traversal(fn):
    def mask(data):
        flat = traverse_util.flatten_dict(data)
        return traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})

    return mask


def plot_annealing(model_state, cfg):
    if cfg.use_wandb:
        fig, ax = plt.subplots()
        b = jax.nn.softplus(model_state.params['params']['betas'])
        b = jnp.cumsum(b) / jnp.sum(b)

        ax.plot(b)
        return {"figures/annealing": [wandb.Image(fig)]}
    else:
        return {}


def plot_timesteps(diffusion_model, model_state, cfg):
    if cfg.use_wandb:
        dt_fn = lambda step: diffusion_model.delta_t_fn(step, model_state.params)
        dts = jax.vmap(dt_fn)(jnp.arange(cfg.algorithm.num_steps))
        fig, ax = plt.subplots()
        ax.plot(dts)
        return {"figures/timesteps": [wandb.Image(fig)]}
    else:
        return {}


def init_dt(cfg):
    if cfg.per_step_dt:
        dt_schedule = cfg.sampler.dt_schedule
        return inverse_softplus(jnp.ones(cfg.alg.actor.diff_steps) * cfg.dt * dt_schedule(jnp.arange(cfg.alg.actor.diff_steps)))
    else:
        return jnp.ones(1) * inverse_softplus(cfg.dt)


def get_sampler_init(alg_name):

    if alg_name == 'dis':
        from diffusion.od.dis import init_dis
        return init_dis
    if alg_name == 'mtdis_concat':
        from diffusion.od.dis import init_dis
        return init_dis

    else:
        raise ValueError(f'No sampler named {alg_name}.')
