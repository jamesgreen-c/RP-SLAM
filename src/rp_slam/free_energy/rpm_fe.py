"""RPSSM interior free energy and exact Gaussian smoothing inference.

Recognition ratios act as Gaussian observation potentials on the linear-
Gaussian prior. Kalman smoothing returns the joint posterior chain; filtering
returns causal beliefs. Inputs are one observation array per recogniser.
"""

import optax
import jax.numpy as jnp

from typing import Any

from jax import lax, vmap, Array
from jax.scipy.special import logsumexp

from src.rp_slam.config import Config
from src.distributions import AllParam
from src.rp_slam.recognition.rpm import RPSSM
from src.utils.kalman import parallel_smoother


class ConstrainedIVFreeEnergy:
    def __init__(self, model: RPSSM):
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

        opts = {"prior": config.prior.build(), "rpm": config.recognition.build()}
        opt_states = {name: opts[name].init(params[name]) for name in opts}
        return params, opt_states, opts

    def get_posterior(self, prior_params, rec_params, observations):

        """Return (prior_chains, factors, smoothed).

        Prior and smoothed chains have means=(B,T,N), covs=(B,T,N,N),
        and cross_covs=(B,T-1,N,N), oriented as Cov(z_{t+1}, z_t).
        Factors have pwm=(J,B,T,N), p=(J,B,T,N,N).
        """
        factors_nat = self.model.get_factors(rec_params, observations)
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
            beta: float = 1.0, 
            em: bool = False
        ) -> tuple[Array, Any]:

        prior_chains, factors_nat, posterior = self.get_posterior(
            params["prior"], params["rpm"], observations
        )

        # in EM mode hold the E-step belief fixed, but still train the messages.
        posterior = lax.cond(em, lax.stop_gradient, lambda q: q, posterior)

        # Calculate the RPSSM objective.
        kl_qf, log_Gamma, kl_qp = self.get_loss_terms(prior_chains, factors_nat, posterior)
        batch, times = posterior.params['means'].shape[:2]
        Z = batch * times
        loss = -(log_Gamma - kl_qf - beta * kl_qp) / Z

        aux = {
            'posterior': posterior,
            'factors_nat': factors_nat,
            'kl_qp': kl_qp / Z,
            'kl_qf': kl_qf / Z,
            'log_Gamma': log_Gamma / Z
        }

        return loss, aux

    def get_loss_terms(self, prior_chains, factors_nat, posterior):
        """Sum recognition and prior KL terms and empirical normalisers.

        Chains carry batch/time axes (B,T); factors carry (J,B,T).
        Return scalar sums (kl_qf, log_Gamma, kl_qp).
        """
        kl_qp = vmap(lambda qtk, ptk: qtk.kl(ptk))(posterior, prior_chains) # B

        prior_chains = prior_chains.all_param
        posterior = posterior.all_param

        kl_qf = vmap(lambda fntk: vmap(vmap(lambda qtk, ftk: qtk.kl(qtk+ftk)))(posterior.nat_param, fntk))(factors_nat) # JxBxT
        log_gammas = vmap(lambda fntk:
                          vmap(lambda qnk, fnk, pnk:
                               vmap(lambda qk, pk:
                                    vmap(lambda fk: (fk + qk).lognormalizer - (fk + pk).lognormalizer)(fnk)
                                )(qnk, pnk),
                               in_axes=(1,1,1)
                            )(posterior.nat_param, fntk, prior_chains.nat_param)
                         )(factors_nat) # JxTxBxB
        
        log_Gamma = vmap(vmap(lambda G: jnp.diag(G) - logsumexp(G, axis=1)))(log_gammas) # JxTxB
        return kl_qf.sum(), log_Gamma.sum(), kl_qp.sum()

    def apply(self, prior_params, rec_params, observations):
        """Return the joint Gaussian smoothing posterior as LGChainDistParam.

        Parameters have shapes means=(B,T,N), covs=(B,T,N,N), and
        cross_covs=(B,T-1,N,N), oriented as Cov(z_{t+1}, z_t).
        Use get_posterior(...)[3] for causal filtered beliefs.
        """
        return self.get_posterior(prior_params, rec_params, observations)[2]
