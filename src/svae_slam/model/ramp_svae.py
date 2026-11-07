"""RAMP node inference with one generative decoder per observation modality."""

from typing import Sequence

import jax.numpy as jnp
import jax.random as jr

from jax import Array, vmap

from src.distributions import AllParams, NetworkParams
from src.rp_slam.recognition.ramp import RAMPSSM
from src.rp_slam.recognition.rpm import GaussianRecognition
from src.svae_slam.config import Config


class RAMPSVAE(RAMPSSM):
    """Use RAMP's Gaussian node beliefs with SVAE-style observation decoders.

    The first three constructor arguments are those of RAMPSSM. ``generation``
    adds one decoder per recognition modality, in the same order as the data.
    Each decoder maps a latent vector (N,) to a flattened observation Gaussian.
    """

    def __init__(
        self,
        latent_dim: int,
        recognition: Sequence[GaussianRecognition],
        generation: Sequence[GaussianRecognition],
        message_network: GaussianRecognition,
        action_conditioned: bool = False
    ):
        super().__init__(latent_dim, recognition, message_network, action_conditioned)
        self.generation = tuple(generation)

        if len(self.generation) != len(self.recognition):
            raise ValueError('Expected one decoder per observation modality.')

    def init(
            self,
            key: Array,
            data: tuple[Array, ...] | Array,
            config: Config,
            actions=None
        ) -> AllParams:
        """Initialise RAMP messages/recognisers and independent decoder trees."""
        data = self._data(data)
        actions = self._actions(actions, data)
        J = len(data)

        prior_key, rec_key, gen_key = jr.split(key, 3)
        prior_keys = jr.split(prior_key, 2)
        rec_keys = jr.split(rec_key, J)
        gen_keys = jr.split(gen_key, J)

        # encoder params
        rec_params = [enc.init(k, x[0, 0]) for enc, k, x in zip(self.recognition, rec_keys, data)]

        # nodewise rpm params
        input_dim = self.latent_dim + self.latent_dim * (self.latent_dim + 1) // 2
        action_dim = 0 if actions is None else actions.shape[-1]
        dummy = jnp.zeros(input_dim + action_dim)
        prior_params = {
            'forward': self.message_network.init(prior_keys[0], dummy),
            'backward': self.message_network.init(prior_keys[1], dummy),
        }

        # generative params
        latent_example = jnp.zeros((self.latent_dim,))
        gen_params = [dec.init(k, latent_example) for dec, k in zip(self.generation, gen_keys)]

        return {'prior': prior_params, 'rec': rec_params, 'gen': gen_params}

    def get_rec_factors(self, rec_params: list[NetworkParams], data):
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
        return self.get_factors(rec_params, data)

    def get_gen_factors(self, gen_params: list[NetworkParams], latents):
        """
        Apply the decoder to latents over batch and time.

        Parameters
        ----------
        gen_params: (J,).  Generative-network Flax variable trees, one per decoder.
        latents:    (...). Latent arrays shaped (B,T, N).

        Returns
        -------
        output: (J,) Gaussian NatParams with pwm=(B,T,D_j), p=(B,T,D_j, D_j).
        """
        if len(gen_params) != len(self.generation):
            raise ValueError('Generation parameter count does not match the model.')

        # apply generation to latents for NatParams over all data modalities
        return tuple(
            vmap(vmap(lambda z: decoder.apply(params, z)))(latents)
            for decoder, params in zip(self.generation, gen_params)
        )
