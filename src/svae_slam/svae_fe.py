"""SVAE ELBO with structured Gaussian inference and Gaussian decoders."""

import optax
import jax.numpy as jnp
import jax.random as jr

from typing import Any

from jax import vmap, Array

from src.svae_slam.config import Config
from src.svae_slam.svae import SVAE
from src.utils.kalman import parallel_smoother
from src.distributions import AllParam


class FreeEnergy:
    def __init__(self, model: SVAE):
        self.model = model

    def configure(self, config: Config):
        """Configure the number of observation recognition factors."""
        self.num_factors = len(self.model.recognition)

    def init(
            self,
            key: Array,
            data: tuple[Array, ...] | Array,
            config: Config
        ) -> tuple[dict, dict[str, optax.OptState], dict[str, optax.GradientTransformation]]:
        self.configure(config)

        params = self.model.init(key, data, config)

        opts = {
            "prior": config.prior.build(), 
            "rec": config.recognition.build(),
            "gen": config.generation.build()
        }
        opt_states = {name: opts[name].init(params[name]) for name in opts}
        return params, opt_states, opts

    def get_posterior(self, prior_params, rec_params, observations):

        """Return (prior_chains, factors, smoothed).

        Prior and smoothed chains have means=(B,T,N), covs=(B,T,N,N),
        and cross_covs=(B,T-1,N,N), oriented as Cov(z_{t+1}, z_t).
        Factors have pwm=(J,B,T,N), p=(J,B,T,N,N).
        """
        factors_nat = self.model.get_rec_factors(rec_params, observations)
        factors = AllParam(factors_nat.sum(axis=0)).dist_param
        batch, times = factors.params['mean'].shape[:2]

        prior = self.model.get_prior(prior_params)
        prior_chains = vmap(lambda _: prior.to_chain(times))(jnp.arange(batch))
        posterior = vmap(lambda f: parallel_smoother(prior, f))(factors)
        return prior_chains, factors_nat, posterior

    def loss(
            self,
            params: dict,
            observations: tuple[Array, ...] | Array,
            key: Array,
            beta: float = 1.0,
            num_samples: int = 1,
        ) -> tuple[Array, Any]:
        """
        Negative SVAE objective, averaged over batch and time.

        beta=1 gives an unbiased Monte Carlo estimate of the negative ELBO.
        Keep gradients through the posterior to train recognition parameters.
        num_samples must be a positive, static integer under JIT.
        """
        observations = self.model._data(observations)

        prior_chains, factors_nat, posterior = self.get_posterior(
            params["prior"], params["rec"], observations
        )

        # sample latents for Monte Carlo estimation of non-linear generative likelihoods
        keys = jr.split(key, num_samples)
        latents = vmap(posterior.sample)(keys)  # (S,B,T,N)
        gen_factors = vmap(lambda z: self.model.get_gen_factors(params["gen"], z))(latents)

        # calculate loss
        log_likelihood, kl_qp = self.get_loss_terms(prior_chains, gen_factors, posterior, observations)

        batch, times = posterior.params["means"].shape[:2]
        obs_dim = observations[0].reshape(batch, times, -1).shape[-1]
        Z = batch * times * obs_dim
        loss = -(log_likelihood - beta * kl_qp) / Z

        aux = {
            "posterior": posterior,
            "factors_nat": factors_nat,
            "log_likelihood": log_likelihood / Z,
            "kl_qp": kl_qp / Z,
        }

        return loss, aux

    def get_loss_terms(
            self,
            prior_chains,
            gen_factors,
            posterior,
            observations,
        ) -> tuple[Array, Array]:
        """Return summed expected log likelihood and joint trajectory KL."""
        observations = self.model._data(observations)

        if len(gen_factors) != len(observations):
            raise ValueError("Expected one generative factor per observation modality.")

        kl_qp = vmap(lambda q, p: q.kl(p))(posterior, prior_chains).sum()
        batch, times = posterior.params["means"].shape[:2]

        one_log_prob = lambda _f, _x: _f.dist_param.log_prob(_x)
        log_prob = lambda _factors, _obs: vmap(vmap(one_log_prob))(_factors, _obs).sum()
        log_likelihood = sum(
            vmap(lambda _sample_factors: log_prob(_sample_factors, obs))(factors).mean()
            for factors, obs in zip(gen_factors, (x.reshape(batch, times, -1) for x in observations))
        )

        return log_likelihood, kl_qp

    def apply(self, prior_params, rec_params, observations):
        """Return the joint Gaussian smoothing posterior as LGChainDistParam.

        Parameters have shapes means=(B,T,N), covs=(B,T,N,N), and
        cross_covs=(B,T-1,N,N), oriented as Cov(z_{t+1}, z_t).
        """
        return self.get_posterior(prior_params, rec_params, observations)[2]
