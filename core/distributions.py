import jax
import jax.numpy as jnp


def make_xent_log_likelihood(predict_fn):

    def out_fn(params, x, y):
        logits = predict_fn(x, params)
        log_prob = jnp.sum(y * jax.nn.log_softmax(logits))
        return log_prob

    return out_fn


def make_gaussian_log_prior(std):
    
    def out_fn(params, x):
        n_params = len(params)
        dy2 = (params**2).sum()
        log_prob = -0.5 * n_params * jnp.log(std**2 * 2 * jnp.pi) - 0.5 * dy2/std**2
        return log_prob
    
    return out_fn


def make_input_sensitivity_fn(predict_fn, T=1):

    def out_fn(params, x):
        def f(params):
            y = predict_fn(x, params) # [batch_size, out_dim]
            y = jnp.concatenate(y) # [batch_size*out_dim]
            return y
        jacobian = jax.jacfwd(f)(params) # [batch_size*out_dim, n_params]
        jacobian_norm = jnp.abs(jacobian).mean() # [1]
        return -T*jacobian_norm

    return out_fn


def make_log_posterior_fn(log_likelihood_fn, log_prior_fn, n):

    def out_fn(params, x, y):
        log_likelihood = log_likelihood_fn(params, x, y)
        log_prior = log_prior_fn(params, x)
        return log_likelihood + log_prior/n

    return out_fn
