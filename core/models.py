import jax
import haiku as hk
from jax.flatten_util import ravel_pytree
from .utils import ravel_pytree_ as ravel_fn


def make_mlp_fn():
    """Returns a forward function for an MLP of given dimensions."""

    def forward(x):
        """
        Input: [B, C, L]
        Output: [B, output_dim]
        """
        x = hk.Reshape(output_shape=[28, 28, 1])(x) # [B, 784] -> [B, 28, 28, 1]
        x = hk.Conv2D(10, 5, 5)(x)
        x = jax.nn.relu(x)
        x = hk.Conv2D(10, 5, 5)(x)
        x = jax.nn.relu(x)
        x = hk.Conv2D(5, 5, 5)(x)
        x = jax.nn.relu(x)
        x = hk.Flatten()(x)
        x = hk.Linear(10)(x)

        return x

    return forward


def make_flattened_predict_fn(net, params_sample):
    _, unravel_fn = ravel_pytree(params_sample)

    @jax.jit
    def predict_fn(x, params):
        params = unravel_fn(params)
        y_hat, _ = net.apply(params, None, None, x)
        return y_hat
    
    return predict_fn


def make_nn(key, x):
    # create NN
    net_fn = make_mlp_fn()
    net = hk.transform_with_state(net_fn)
    params, _ = net.init(key, x)
    
    # use arrays as params instead of pytrees
    predict_fn = make_flattened_predict_fn(net, params)
    params = ravel_fn(params)

    return predict_fn, params
