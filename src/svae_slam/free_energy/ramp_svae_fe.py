"""Hybrid nodewise RAMP free energy with generative reconstruction."""

from typing import Any

import jax.numpy as jnp
import jax.random as jr
import optax

from jax import Array, lax, vmap
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map

from src.distributions import AllParam
from src.utils.math import gaussian_kl
from src.svae_slam.config import Config
from src.svae_slam.model.ramp_svae import RAMPSVAE


class FreeEnergy:
    def __init__(self, model: RAMPSVAE):
        self.model = model

    def configure(self, config: Config):
        """Count all inference branches; only the last two have RPM losses."""
        self.batch_size = config.batch_size
        self.num_factors = len(self.model.recognition) + 2

    def init(
            self,
            key: Array,
            data: tuple[Array, ...] | Array,
            config: Config,
            actions: Array | None = None
        ) -> tuple[dict, dict[str, optax.OptState], dict[str, optax.GradientTransformation]]:
        self.configure(config)
        params = self.model.init(key, data, config, actions)
        opts = {
            'prior': config.prior.build(),
            'rec': config.recognition.build(),
            'gen': config.generation.build(),
        }
        opt_states = {name: opts[name].init(params[name]) for name in opts}
        return params, opt_states, opts

    def get_posterior(self, prior_params, rec_params, observations, actions=None):
        """
        Form smoothed and causal Gaussian node marginals from message sums.

        q_t = f0 + observations_t + left_t + right_t in natural parameters.
        The filtered belief excludes right_t, so it cannot use future data.
        The posterior has mean/covariance, without cross-time covariance blocks.

        Parameters
        ----------
        prior_params:  Model params: 'forward' and 'backward' message-network parameters.
        rec_params:    sequence of length J. Observation-network Flax variable trees.
        observations:  (B,T,*O_1) or sequence of J arrays (B,T,*O_j). 
                            Observations with common batch and time axes.

        Returns
        -------
        output: tuple (prior, messages, posterior)
            prior:     GaussianNatParam,  pwm=(N,),      p=(N,N), shared across nodes.
            messages:  GaussianNatParam,  pwm=(K,B,T,N), p=(K,B,T,N,N).
            posterior: GaussianDistParam, mean=(B,T,N),  cov=(B,T,N,N).
        """
        # apply recognition to data to get factors over latent nodes. Apply BP to find messages.
        factors_nat = self.model.get_rec_factors(rec_params, observations)
        messages = self.model.get_messages(prior_params, factors_nat, actions)
        
        # calculate posterior 
        prior = self.model.prior
        posterior = (prior + messages.sum(axis=0)).dist_param
        return prior, messages, posterior

    def loss(
            self,
            params: dict,
            key: Array,
            observations: tuple[Array, ...] | Array,
            actions: Array | None = None,
            beta: float = 1.0,
            num_samples: int = 1,
            em: bool = False,
        ) -> tuple[Array, Any]:

        if num_samples < 1:
            raise ValueError('num_samples must be a positive, static integer.')
        
        # calculate posterior
        # observations = self.model._data(observations)
        prior, messages, posterior = self.get_posterior(
            params['prior'], params['rec'], observations, actions
        )
        posterior = lax.cond(em, lax.stop_gradient, lambda q: q, posterior)

        # sample latents for Monte Carlo estimation of non-linear generative likelihoods
        keys = jr.split(key, num_samples)
        latents = vmap(lambda k: posterior.sample(k, None))(keys)
        gen_factors = vmap(lambda z: self.model.get_gen_factors(params['gen'], z))(latents)

        # calculate loss terms
        log_likelihood, kl_qf, log_Gamma, kl_qp = self.get_loss_terms(
            prior, messages, gen_factors, posterior, observations
        )

        batch, times = posterior.params['mean'].shape[:2]
        obs_dim = observations[0].reshape(batch, times, -1).shape[-1]
        Z = batch * times * obs_dim
        loss = -(log_likelihood + log_Gamma - kl_qf - beta * kl_qp) / Z

        return loss, {
            'posterior': posterior,
            'log_likelihood': log_likelihood / Z,
            'kl_qp': kl_qp / Z,
            'kl_qf': kl_qf / Z,
            'log_Gamma': log_Gamma / Z,
        }

    def get_loss_terms(
            self,
            prior,
            messages,
            gen_factors,
            posterior,
            observations,
        ) -> tuple[Array, Array, Array, Array]:
        """
        Return scalar sums (log_likelihood, KL_qf, log_Gamma, KL_qp).

        Parameters
        ----------
        prior:        GaussianNatParam. Fixed node prior with pwm=(N,), p=(N,N).
        messages:     GaussianNatParam. All incoming ratios with pwm=(K,B,T,N), p=(K,B,T,N,N).
        gen_factors:  GaussianNatParam. Generative distribution
        posterior:    Tuple[GaussianDistParam], with shape (S,B,T, ...) 
                        Node beliefs with mean=(B,T,N), cov=(B,T,N,N). B must be at least two.

        Returns
        -------
        output: tuple (log_likelihood, kl_qf, log_Gamma, kl_qp)
            log_likelihood estimates the expected log likelihood.
            kl_qf and log_Gamma sum over branches, sequences and times.
            kl_qp sums over sequences and times only.
        """
        observations = self.model._data(observations)
        batch, times = posterior.params['mean'].shape[:2]

        if len(gen_factors) != len(observations):
            raise ValueError('Expected one generative factor per observation modality.')

        # nodewise RPM loss terms without data RPM nodes
        temporal_messages = tree_map(lambda x: x[-2:], messages)
        kl_qf, log_Gamma, kl_qp = self._get_temporal_loss_terms(prior, temporal_messages, posterior)

        # generative loss terms
        one_log_prob = lambda f, x: f.dist_param.log_prob(x)
        log_prob = lambda factors, obs: vmap(vmap(one_log_prob))(factors, obs).sum()
        log_likelihood = sum(
            vmap(lambda _sample_factors: log_prob(_sample_factors, obs))(factors).mean()
            for factors, obs in zip(gen_factors, (x.reshape(batch, times, -1) for x in observations))
        )

        return log_likelihood, kl_qf, log_Gamma, kl_qp

    def _get_temporal_loss_terms(self, prior, messages, posterior):
        """
        Return scalar sums (KL_qf, log_Gamma, KL_qp), as in model_fe.py.

        Parameters
        ----------
        prior:        GaussianNatParam. Fixed node prior with pwm=(N,), p=(N,N).
        messages:     GaussianNatParam. Only temporal ratios with pwm=(K,B,T,N), p=(K,B,T,N,N).
        posterior:    GaussianDistParam 
                        Node beliefs with mean=(B,T,N), cov=(B,T,N,N). B must be at least two.

        Returns
        -------
        output: tuple (kl_qf, log_Gamma, kl_qp)
            Three scalar arrays ().
            kl_qf and log_Gamma sum over branches, sequences and times.
            kl_qp sums over sequences and times only.
        """
        q = AllParam(posterior).nat_param
        if q.params['pwm'].shape[0] < 2:
            raise ValueError('RAMP learning needs at least two observation sequences.')
        messages_by_time = tree_map(lambda x: jnp.moveaxis(x, 2, 0), messages)
        q_by_time = tree_map(lambda x: jnp.swapaxes(x, 0, 1), q)

        # evaluate pairwise Gaussian products separately at each time node.
        def node_terms(inputs):
            messages, node_q = inputs
            return self._node_loss_terms(prior, messages, node_q)

        terms = vmap(node_terms)((messages_by_time, q_by_time))
        return tuple(term.sum() for term in terms)

    @staticmethod
    def _node_loss_terms(prior, messages, q):
        """
        Evaluate all incoming branches at one time node using unbatched KL APIs.

        The batch indices n and m select sequences, not time nodes.
        Each auxiliary Gaussian q_n+r_jn is used only to evaluate the bound.

        Parameters
        ----------
        prior:     GaussianNatParam. Fixed node prior with pwm=(N,), p=(N,N).
        messages:  GaussianNatParam. Incoming ratios at this time, pwm=(K,B,N), p=(K,B,N,N).
        q:         GaussianNatParam. Node beliefs at this time, pwm=(B,N), p=(B,N,N).

        Returns
        -------
        output: tuple (kl_qf, log_Gamma, kl_qp)
            Three scalar arrays ().
            kl_qf and log_Gamma sum over K branches and B sequences.
            kl_qp sums over B sequences once.
        """
        # kl_qp = vmap(lambda qn: qn.kl(prior))(q)
        kl_qp = vmap(lambda qn: gaussian_kl(qn, prior))(q)
        batch_size = q.params['pwm'].shape[0]

        def factor_terms(ratios):
            # kl_qf = vmap(lambda qn, rn: qn.kl(qn + rn))(q, ratios)
            kl_qf = vmap(lambda qn, rn: gaussian_kl(qn, qn + rn))(q, ratios)
            phi_f = vmap(lambda rm: (prior + rm).lognormalizer)(ratios)
            
            # G[n,m]: query belief n, candidate subtree m, at this time/branch.
            G = vmap(
                lambda qn: vmap(lambda rm: (qn + rm).lognormalizer)(ratios)
            )(q) - phi_f[None]
            # Phi(q_n) cancels between the diagonal and its row normaliser.
            # +log(B) retains the empirical mixture's 1/B. model_fe.py omits
            # this constant; retaining it makes a zero boundary message neutral.
            log_Gamma = jnp.diag(G) - logsumexp(G, axis=1) + jnp.log(batch_size)
            return kl_qf.sum(), log_Gamma.sum()

        kl_qf, log_Gamma = vmap(factor_terms)(messages)
        return kl_qf.sum(), log_Gamma.sum(), kl_qp.sum()

    def apply(self, prior_params, rec_params, observations, actions=None):
        """Return RAMP's smoothed Gaussian node marginals."""
        return self.get_posterior(prior_params, rec_params, observations, actions)[2]
