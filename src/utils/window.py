import jax.numpy as jnp


def temporal_features(x, window=11):
    """
    Convert (..., T, D) latents to (..., T, window * D) features.

    window is the total number of frames. window=11 equals radius=5.
    Boundary frames are repeated separately within each sequence.
    """
    if not isinstance(window, int) or window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer.")
    
    x = jnp.asarray(x)
    radius = window // 2
    padding = [(0, 0)] * x.ndim
    padding[-2] = (radius, radius)
    padded = jnp.pad(x, padding, mode="edge")
    return jnp.concatenate([padded[..., i:i + x.shape[-2], :] for i in range(window)], axis=-1)
