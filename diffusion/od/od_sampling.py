import jax
import distrax
import jax.numpy as jnp


def single_sample(seed, model_state, params, obs, integrator, diffusion_model, stop_grad=False):
    key, key_gen = jax.random.split(seed)

    init_x = diffusion_model.prior_sampler(params, key, 1)
    key, key_gen = jax.random.split(key_gen)
    init_x = jnp.squeeze(init_x, 0)
    if stop_grad:
        init_x = jax.lax.stop_gradient(init_x)
    key, key_gen = jax.random.split(key_gen)
    aux = (init_x, jnp.zeros(1), key)
    integrate = integrator(model_state, params, obs, stop_grad)
    aux, per_step_output = jax.lax.scan(integrate, aux, jnp.arange(0, diffusion_model.num_steps))
    final_x, log_ratio, _ = aux

    terminal_costs = diffusion_model.prior_log_prob(init_x, params)
    running_cost = -(log_ratio + distrax.Tanh().forward_log_det_jacobian(final_x).sum())
    # running_cost = -log_ratio

    final_x = distrax.Tanh().forward(final_x)
    x_t = per_step_output
    x_t = jnp.concatenate([jnp.expand_dims(init_x, 0), x_t])
    x_t = x_t.at[-1].set(final_x)
    stochastic_costs = jnp.zeros_like(running_cost)
    return final_x, running_cost, stochastic_costs, terminal_costs.reshape(running_cost.shape), x_t, None


def sample(key, model_state, params, obs, integrator, diffusion_model, stop_grad=False):
    keys = jax.random.split(key, num=obs.shape[0])
    in_tuple = (keys, model_state, params, obs, integrator, diffusion_model, stop_grad)
    in_axes = (0, None, None, 0, None, None, None)
    rnd_result = jax.vmap(single_sample, in_axes=in_axes)(*in_tuple)
    x_0, running_costs, stochastic_costs, terminal_costs, x_t, _ = rnd_result

    return x_0, running_costs, stochastic_costs, terminal_costs, x_t, None

def mt_sample_concat(key, model_state, params, obs, integrator, diffusion_model, task_embeddings, stop_grad=False,):
    obs = jnp.concat((obs, task_embeddings), axis=-1)
    return sample(key,model_state,params,obs,integrator,diffusion_model,stop_grad)

