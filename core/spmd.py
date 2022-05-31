import jax
import jax.numpy as jnp
from functools import partial
from .utils import ifelse, split_into_batches


def spmd(f, x, y, params, log_prob_fn, n_dev=1, batch_size=-1, *args):
    # f: (log_prob_val_and_grad_fn, params, *args) -> (chain, *logs)

    # batch x, y, params across devices
    x_batched = split_into_batches(x, n_dev)
    y_batched = split_into_batches(y, n_dev)
    params_batched = jnp.repeat(params[None], n_dev, axis=0)
    
    # run each batch on a separate device
    @partial(jax.pmap, axis_name='i')
    def g(x, y, params):
        log_prob_val_and_grad_fn = make_epoch_log_prob_val_and_grad_fn(log_prob_fn, x, y, batch_size)
        return f(log_prob_val_and_grad_fn, params, *args)
    out_batched = g(x_batched, y_batched, params_batched)
    
    # check that each chain is the same, then return just the first output
    assert jnp.allclose(out_batched[0][0], out_batched[0][1])
    out_single = [out[0] for out in out_batched]
    
    return out_single


def make_epoch_log_prob_val_and_grad_fn(log_prob_fn, x, y, batch_size):
    val_and_grad_fn = jax.value_and_grad(log_prob_fn)

    def out_fn(params):

        # split the samples into batches
        n_batches = len(x) // batch_size
        x_batched = split_into_batches(x, n_batches)
        y_batched = split_into_batches(y, n_batches)

        # iterate through batches, accumulating value and grad
        def step_batch(i, args):
            val_epoch, grad_epoch = args
            x = x_batched[i]
            y = y_batched[i]
            val_batch, grad_batch = val_and_grad_fn(params, x, y)
            val_batch = jax.lax.psum(val_batch, axis_name='i')
            grad_batch = jax.lax.psum(grad_batch, axis_name='i')
            val_epoch += val_batch
            grad_epoch += grad_batch
            return val_epoch, grad_epoch
        args = (0, jnp.zeros_like(params))
        args = jax.lax.fori_loop(0, n_batches, step_batch, args)
        val, grad = args

        return val, grad

    return out_fn
