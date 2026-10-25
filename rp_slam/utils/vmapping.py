from jax import vmap
from functools import wraps


def auto_vmap(param_name, expected_ndim):
    """
    Augmented function wrapper that acts as vmap, but automatically
    computes the number of vmaps that need to be applied, by
    computing the difference between the expected input dimension
    (`expected_ndim`) and the actual input dimension.

    The number of extra dimensions is taken from a designated param
    given by `param_name`. E.g. if the input is a Gaussian distparam,
    can use `param_name=mean` and `expected_ndim=1`, because the
    precision-weighted mean is a 1D array.
    """
    def decorator(f):
        @wraps(f)
        def wrapper(x):
            extra_dims = x.params[param_name].ndim - expected_ndim

            vmapped_f = f
            for _ in range(extra_dims):
                vmapped_f = vmap(vmapped_f)

            return vmapped_f(x)
        return wrapper
    return decorator

