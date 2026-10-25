"""RAMP interior free energy and Gaussian node inference.

The posterior combines observation ratios with learned messages from both
neighbours. Filtered beliefs omit messages from future observations.
"""

from typing import Any

import jax
import jax.numpy as jnp

from jax import Array, lax, vmap
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map

from rp_slam.config import Config
from rp_slam.distributions import AllParam
from rp_slam.recognition.ramp import RAMPSSM


class ConstrainedIVFreeEnergy:

    def __init__(self, model: RAMPSSM):
        self.model = model

    def configure(self, config: Config):
        self.batch_size = config.batch_size
        self.num_factors = len(self.model.recognition) + 2

    def init(self, key: Array, data: tuple[Array, ...] | Array, config: Config):
        self.configure(config)
        params = self.model.init(key, data, config)
        opts = {'prior': config.prior.build(), 'rpm': config.recognition.build()}
        opt_states = {name: opts[name].init(params[name]) for name in opts}
        return params, opt_states, opts

    def get_posterior(self, prior_params, rec_params, observations):
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
        output: tuple (prior, messages, posterior, filtered, features)
            prior:     GaussianNatParam,  pwm=(N,),      p=(N,N), shared across nodes.
            messages:  GaussianNatParam,  pwm=(K,B,T,N), p=(K,B,T,N,N).
            posterior: GaussianDistParam, mean=(B,T,N),  cov=(B,T,N,N).
        """
        # apply recognition to data to get factors over latent nodes. Apply BP to find messages.
        factors_nat = self.model.get_factors(rec_params, observations)
        messages = self.model.get_messages(prior_params, factors_nat)

        # calculate posterior 
        prior = self.model.prior
        posterior = (prior + messages.sum(axis=0)).dist_param
        return prior, messages, posterior

    def loss(
            self, 
            params: dict, 
            observations: tuple[Array, ...] | Array, 
            beta: float = 1.0, 
            em: bool = False, 
        ) -> tuple[Array, Any]:
        
        prior, messages, posterior = self.get_posterior(
            params['prior'], params['rpm'], observations
        )

        # In EM mode hold the E-step belief fixed, but still train the messages.
        posterior = lax.cond(em, lax.stop_gradient, lambda q: q, posterior)

        # calculate loss
        kl_qf, log_Gamma, kl_qp = self.get_loss_terms(prior, messages, posterior)

        batch, times = posterior.params['mean'].shape[:2]
        Z = batch * times  # One objective per node, not per incoming edge.
        loss = -(log_Gamma - kl_qf - beta * kl_qp) / Z

        return loss, {
            'kl_qp': kl_qp / Z,
            'kl_qf': kl_qf / Z,
            'log_Gamma': log_Gamma / Z,
        }

    def get_loss_terms(self, prior, messages, posterior):
        """
        Return scalar sums (KL_qf, log_Gamma, KL_qp), as in model_fe.py.

        Parameters
        ----------
        prior:        GaussianNatParam. Fixed node prior with pwm=(N,), p=(N,N).
        messages:     GaussianNatParam. All incoming ratios with pwm=(K,B,T,N), p=(K,B,T,N,N).
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

        # Evaluate pairwise Gaussian products separately at each time node.
        def node_terms(inputs):
            messages, node_q = inputs
            return self._node_loss_terms(prior, messages, node_q)

        terms = jax.vmap(node_terms)((messages_by_time, q_by_time))
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
        kl_qp = vmap(lambda qn: qn.kl(prior))(q)
        batch_size = q.params['pwm'].shape[0]

        def factor_terms(ratios):
            kl_qf = vmap(lambda qn, rn: qn.kl(qn + rn))(q, ratios)
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


    def apply(self, prior_params, rec_params, observations):
        """Return smoothed Gaussian node marginals for the observations.

        Returns GaussianDistParam with mean=(B,T,N) and cov=(B,T,N,N).
        These are node marginals, not a joint trajectory distribution.
        Use get_posterior(...)[3] for the causal filtered marginals.
        """
        return self.get_posterior(prior_params, rec_params, observations)[2]
