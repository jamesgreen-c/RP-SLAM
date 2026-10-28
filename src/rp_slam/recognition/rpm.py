from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn

from jax import vmap, Array

from src.distmaps import DistMap
from src.rp_slam.config import Config
from src.distributions import LGStationaryParam, AllParams, NetworkParams
from src.utils.kalman import parallel_filter


INITIALIZER = jax.nn.initializers.variance_scaling(
    scale=0.1, mode='fan_in', distribution='truncated_normal'
)

class GaussianRecognition(nn.Module):
    network: nn.Module
    dist_map: DistMap
    compressor: nn.Module | None = None
    kernel_init: Callable = INITIALIZER
    bias_init: Callable = jax.nn.initializers.zeros
    constant_cov: bool = (
        False  # if True, recognition covariance is constant across all data
    )

    def encode(self, x: Array) -> Array:
        """Shared observation features, before the Gaussian recognition head."""
        return self.compressor(x) if self.compressor is not None else x

    @nn.compact
    def __call__(self, x: Array, return_features: bool = False):

        features = self.encode(x)
        x = self.network(features)
        
        if self.constant_cov:
            mean_dim = self.dist_map.latent_dim
            cov_dim = self.dist_map.input_dim - mean_dim
            cov_flat = self.variable(
                'params', 'cov', jnp.zeros, (cov_dim,)
            )
            x = nn.Dense(mean_dim, kernel_init=INITIALIZER)(x)
            x = jnp.concatenate((x, cov_flat.value))
        else:
            x = nn.Dense(
                self.dist_map.input_dim,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init
            )(x)

        factor = self.dist_map(x)
        return (factor, features) if return_features else factor


class RPSSM:
    def __init__(
            self,
            prior: LGStationaryParam,
            recognition: list[GaussianRecognition],
    ):
        self.prior = prior
        self.recognition = recognition
        self.latent_dim = self.prior.latent_dim

    def _data(self, data):
        data = tuple(data) if isinstance(data, (tuple, list)) else (data,)

        if not data or len(data) != len(self.recognition):
            raise ValueError('Expected one observation array per recogniser.')
        if any(x.shape[:2] != data[0].shape[:2] for x in data):
            raise ValueError('Observation arrays must share batch and time axes.')
        if data[0].shape[1] < 1:
            raise ValueError('At least one observation is required.')
        
        return data

    def get_prior(self, prior_params):
        """Construct the linear-Gaussian prior from its learnable parameters."""
        return self.prior.update(prior_params)

    def init(
            self,
            key: Array,
            data: tuple[Array] | Array,
            config: Config
        ) -> AllParams:
        data = self._data(data)
        prior_key, *rec_keys = jr.split(key, len(data) + 1)

        # prior_params = self.prior.init(prior_key, data)
        # for now just init prior at whatever parameters you pass it
        prior_params = dict(self.prior.opt_param)  # returns only learnable parameters
        rec_params = [enc.init(k, x[0,0]) for enc, k, x in zip(self.recognition, rec_keys, data)]

        return {"prior": prior_params, "rpm": rec_params}
    
    def get_factors(self, rec_params: list[NetworkParams], data):
        """
        Apply the observation recognisers over batch and time.

        Parameters
        ----------
        rec_params: (J,). Observation-network Flax variable trees, one per recogniser.
        data:       (J, ...) or (...) if J=1. Observation arrays shaped (B,T,*O_j).

        Returns
        -------
        output: Observation ratios with pwm=(J,B,T,N), p=(J,B,T,N,N).
        """
        data = self._data(data)

        if len(rec_params) != len(self.recognition):
            raise ValueError('Recognition parameter count does not match the model.')
        
        # apply recognition to data, stack outputted statistics into a single NatParam for all factors
        outs = [vmap(vmap(lambda x: rec.apply(p, x)))(d) for rec, p, d in zip(self.recognition, rec_params, data)]
        factors = type(outs[0])(**{k: jnp.stack([out.params[k] for out in outs]) for k in outs[0].params})
        return factors

