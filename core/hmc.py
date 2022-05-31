import jax
import jax.numpy as jnp
from .utils import ifelse


def leapfrog(params, momentum, log_prob_val_and_grad_fn, step_size, n_steps):
    
    # define a single step
    def step(i, args):
        params, momentum = args
        
        # update momentum
        _, grad = log_prob_val_and_grad_fn(params)
        momentum = momentum + 0.5 * step_size * grad

        # update params
        params = params + momentum * step_size

        # update momentum
        _, grad = log_prob_val_and_grad_fn(params)
        momentum = momentum + 0.5 * step_size * grad
        
        return params, momentum

    # do 'n_steps'
    args = (params, momentum)
    args = jax.lax.fori_loop(0, n_steps, step, args)
    # for i in range(n_steps): args = step(i, args)
    params, momentum = args

    return params, momentum


def hmc_sampler(log_prob_val_and_grad_fn, params, key, n_steps, n_leapfrog_steps, step_size):

    # define a single step
    def step(i, args):
        params, params_history, total_accept_prob, key = args
        key, normal_key, uniform_key = jax.random.split(key, 3)

        # generate random momentum
        momentum = jax.random.normal(normal_key, params.shape)

        # leapfrog
        new_params, new_momentum = leapfrog(params, momentum, log_prob_val_and_grad_fn, step_size, n_leapfrog_steps)

        # MH correction
        potentaial_energy_diff = log_prob_val_and_grad_fn(new_params)[0] - log_prob_val_and_grad_fn(params)[0]
        kinetic_energy_diff = 0.5*(momentum**2 - new_momentum**2).sum()
        log_accept_prob = potentaial_energy_diff + kinetic_energy_diff
        accept_prob = jnp.minimum(1, jnp.exp(log_accept_prob))
        total_accept_prob += accept_prob
        accept = jax.random.uniform(uniform_key) < accept_prob
        params = ifelse(accept, new_params, params)
        
        # store history
        params_history = params_history.at[i].set(params)
        
        return params, params_history, total_accept_prob, key
    
    # do 'n_steps'
    params_history = jnp.zeros([n_steps, len(params)])
    args = (params, params_history, 0, key)
    args = jax.lax.fori_loop(0, n_steps, step, args)
    # for i in range(n_steps): args = step(i, args)
    _, params_history, total_accept_prob, key = args
    avg_accept_prob = total_accept_prob/n_steps
    
    # print(f'Avg. accept. prob.: {(total_accept_prob/n_steps):.2%}')
    return params_history, avg_accept_prob
