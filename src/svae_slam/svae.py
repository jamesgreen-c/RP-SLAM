import jax.numpy as jnp
import jax.random as jr

from jax import vmap, Array

from src.distmaps import DistMap
from src.svae_slam.config import Config
from src.rp_slam.recognition.rpm import GaussianRecognition
from src.distributions import LGStationaryParam, AllParams, NetworkParams


class SVAE:
    def __init__(
            self,
            prior: LGStationaryParam,
            recognition: list[GaussianRecognition],
            generation: list[GaussianRecognition]
    ):
        self.prior = prior
        self.recognition = recognition
        self.generation = generation
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
        J = len(data)

        if len(self.generation) != J:
            raise ValueError("Expected one decoder per observation modality.")

        rec_key, gen_key = jr.split(key)
        rec_keys = jr.split(rec_key, J)
        gen_keys = jr.split(gen_key, J)

        latent_example = jnp.zeros((self.latent_dim,))

        prior_params = dict(self.prior.opt_param)
        rec_params = [enc.init(k, x[0, 0]) for enc, k, x in zip(self.recognition, rec_keys, data)]
        gen_params = [dec.init(k, latent_example) for dec, k in zip(self.generation, gen_keys)]

        return {"prior": prior_params, "rec": rec_params, "gen": gen_params}
    
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
        data = self._data(data)

        if len(rec_params) != len(self.recognition):
            raise ValueError('Recognition parameter count does not match the model.')
        
        # apply recognition to data, stack outputted statistics into a single NatParam for all factors
        outs = [vmap(vmap(lambda x: rec.apply(p, x)))(d) for rec, p, d in zip(self.recognition, rec_params, data)]
        factors = type(outs[0])(**{k: jnp.stack([out.params[k] for out in outs]) for k in outs[0].params})
        return factors

    def get_gen_factors(self, gen_params: list[NetworkParams], latents):
        """
        Apply the decoder to latents over batch and time.

        Parameters
        ----------
        gen_params: (J,).  Generative-network Flax variable trees, one per decoder.
        latents:    (...). Latent arrays shaped (B,T,*O_j).

        Returns
        -------
        output: (J,) Gaussian NatParams with pwm=(B,T,N), p=(B,T,N,N).
        """

        if len(gen_params) != len(self.generation):
            raise ValueError('Generation parameter count does not match the model.')
        
        # apply generation to latents for NatParams over all data modalities
        return tuple(vmap(vmap(lambda z: decoder.apply(params, z)))(latents) for decoder, params in zip(self.generation, gen_params))
