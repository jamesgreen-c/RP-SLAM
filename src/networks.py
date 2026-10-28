import flax.linen as nn

import jax.numpy as jnp
from jax import Array
from typing import Sequence


class MLP(nn.Module):
    """Simple MLP with no final Dense layer."""
    features : Sequence[int]

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        return x


class TanhMLP(nn.Module):
    """Simple MLP with tanh activation and no final Dense layer."""
    features : Sequence[int]

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for feat in self.features:
            x = nn.Dense(feat)(x)
            # x = nn.LayerNorm()(x)
            x = nn.tanh(x)
        return x


class CNN(nn.Module):
    """Simple CNN with no final Dense layer."""
    cnn_features : Sequence[dict]

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for feat in self.cnn_features:
            x = nn.Conv(**feat)(x)
            x = nn.relu(x)
        return x.flatten()


class DenseCNN(nn.Module):
    """ CNN with optional fully connected layers and no output layer. """
    cnn_features: Sequence[dict]
    mlp_features: Sequence[int] = ()

    @nn.compact
    def __call__(self, x: Array) -> Array:

        # dm_control returns uint8 pixels
        # x = x.astype(jnp.float32) / 255.0

        for feat in self.cnn_features:
            x = nn.Conv(**feat)(x)
            x = nn.relu(x)

        x = x.reshape(-1)

        for feat in self.mlp_features:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)

        return x