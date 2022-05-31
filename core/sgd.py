import jax
import jax.numpy as jnp
from functools import partial
from .utils import split_into_batches


def train_sgd(key, log_prob_fn, params_init, x, y, n_epochs, n_dev, batch_size, lr):

    @partial(jax.pmap, axis_name='i', out_axes=None)
    def train(x, y, params, key):
        # iterate through epochs
        def step_epoch(i, args):
            params, val_history, key = args
            key, _ = jax.random.split(key)

            # batch x, y
            idx = jax.random.permutation(key, len(x), 0)
            x_shuffled = x[idx]
            y_shuffled = y[idx]
            n_batches = len(x) // batch_size
            x_batched = split_into_batches(x_shuffled, n_batches)
            y_batched = split_into_batches(y_shuffled, n_batches)

            # iterate through batches
            def step_batch(i, args):
                params, val_epoch = args
                x = x_batched[i]
                y = y_batched[i]
                val_batch, grads_batch = jax.value_and_grad(log_prob_fn)(params, x, y)
                grads_batch = jax.lax.psum(grads_batch, axis_name='i')
                val_batch = jax.lax.psum(val_batch, axis_name='i')
                val_epoch += val_batch
                params += lr*grads_batch
                return params, val_epoch
            params, val = jax.lax.fori_loop(0, n_batches, step_batch, (params, 0))
            val_history = val_history.at[i].set(val)

            return params, val_history, key
        val_history = jnp.zeros(n_epochs)
        args = params_init, val_history, key
        args = jax.lax.fori_loop(0, n_epochs, step_epoch, args)
        params, val_history, key = args
        return params, val_history

    # batch x, y, params across devices
    x_batched = split_into_batches(x, n_dev)
    y_batched = split_into_batches(y, n_dev)
    params_batched = jnp.repeat(params_init[None], n_dev, axis=0)
    key_batched = jnp.repeat(key[None], n_dev, axis=0)

    # train on each device
    params, val_history = train(x_batched, y_batched, params_batched, key_batched)
    
    return params, val_history
