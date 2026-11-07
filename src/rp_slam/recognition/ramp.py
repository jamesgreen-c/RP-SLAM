"""
Single-chain Gaussian RAMP model with observation recognition.

There is one latent z_t per time, observation branches at each node, and
learned forward/backward messages on the chain. There is no LGSSM transition
density: the fixed node prior is f0 = N(0, I). Training lives in ramp_fe.py.

Recognition outputs are ratio natural parameters
r_j = eta(f_j) - eta(f0), so f_j has parameters prior + r_j.
Positive message precisions restrict the family, but ensure normalisability.

The parameter groups match rpm.py: 'rpm' contains observation recognisers;
'prior' contains the forward and backward message networks,
not parameters of f0.

Inference supplies Gaussian node marginals, not a joint trajectory posterior
or a generative transition kernel. Marginal sampling does not reproduce an
LGSSM path sampler.

Shape notation
--------------
B: number of sequences in the batch.
T: number of observation times.
J: number of observation recognisers.
N: latent_dim.
O_j: observation event shape for recogniser j.
F = N + N*(N+1)//2: message-network input feature dimension.

GaussianNatParam shapes describe its 'pwm' vector and 'p' precision matrix.
GaussianDistParam shapes describe its 'mean' vector and 'cov' matrix.
Network parameters are Flax variable trees.
"""

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp
import jax.random as jr

from jax import Array, lax, vmap
from jax.tree_util import tree_map

from src.distributions import GaussianDistParam, GaussianNatParam
from src.rp_slam.recognition.rpm import GaussianRecognition
from src.rp_slam.config import Config


class RAMPSSM:
    """
    Time-shared messages on one SSM-shaped factor tree.

    Observation arrays share batch and time axes. Each node receives its local
    observation messages and messages from the two neighbouring nodes.
    Absent boundary edges send zero natural parameters.

    Parameters
    ----------
    latent_dim:  Latent dimension N.
    recognition: Sequence of length J. GaussianRecognition modules mapping 
                    observations with event shape O_j to ratio parameters pwm=(N,), p=(N,N).
    message_network: GaussianRecognition module mapping cavity features of
                     length F to outgoing Gaussian ratio parameters.
    """

    def __init__(
        self,
        latent_dim: int,
        recognition: Sequence[GaussianRecognition],
        message_network: GaussianRecognition,
        action_conditioned: bool = False
    ):
        if latent_dim < 1 or not recognition:
            raise ValueError('Use positive dimensions and at least one recogniser.')
        
        self.latent_dim = latent_dim
        self.recognition = tuple(recognition)
        self.message_network = message_network
        self.prior = GaussianNatParam(p=jnp.eye(latent_dim), pwm=jnp.zeros(latent_dim))
        self.action_conditioned = action_conditioned

    def _data(self, data) -> tuple[Array, ...]:
        data = tuple(data) if isinstance(data, (tuple, list)) else (data,)

        if len(data) != len(self.recognition):
            raise ValueError('Supply one observation array per recogniser.')
        if any(x.ndim < 2 for x in data):
            raise ValueError('Observation arrays need batch and time axes.')
        
        shape = data[0].shape[:2]
        if min(shape) < 1 or any(x.shape[:2] != shape for x in data):
            raise ValueError('Observation arrays must share nonempty (B,T) axes.')
        
        return data

    def _actions(self, actions, data):
        """Validate actions shaped (B, T - 1, action_dim), or return None."""
        if actions is None:
            if self.action_conditioned:
                raise ValueError("Actions are required when action conditioning is enabled.")
            return None

        if not self.action_conditioned:
            raise ValueError("Actions were supplied, but action conditioning is disabled.")

        actions = jnp.asarray(actions)
        batch, times = data[0].shape[:2]

        if (actions.ndim != 3 or actions.shape[:2] != (batch, times - 1) or actions.shape[-1] < 1):
            raise ValueError(
                "Expected actions with shape "
                f"({batch}, {times - 1}, action_dim), "
                f"with action_dim >= 1; got {actions.shape}."
            )

        return actions

    def init(self, key: Array, data, config: Config, actions=None) -> dict:
        """Use the same model-init signature and optimizer groups as RPSSM."""

        data = self._data(data)
        actions = self._actions(actions, data)
        count = len(self.recognition)
        keys = jr.split(key, count + 2)

        # encoder params
        rec_params = [rec.init(k, x[0, 0]) for rec, k, x in zip(self.recognition, keys[:count], data)]

        # nodewise rpm params
        input_dim = self.latent_dim + self.latent_dim * (self.latent_dim + 1) // 2
        action_dim = 0 if actions is None else actions.shape[-1]
        dummy = jnp.zeros(input_dim + action_dim)
        prior_params = {
            'forward': self.message_network.init(keys[count], dummy),
            'backward': self.message_network.init(keys[count + 1], dummy),
        }

        return {'prior': prior_params, 'rpm': rec_params}

    def get_factors(self, rec_params, data):
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
        factors = type(outs[0])(**{k: jnp.stack([out.params[k] for out in outs]) for k in outs[0].params.keys()})
        return factors

    def _message(self, params, ratio: GaussianNatParam, action: Array | None = None):
        """
        Encode a batch of cavity ratios and apply the existing Gaussian head.

        The prior is implicit and fixed. The cavity belief is prior + ratio.
        Only the precision's lower triangle is needed because it is symmetric.
        DistMap acts on one example, hence the vmap over the batch.

        Parameters
        ----------
        params: Parameters of one directed message network.
        ratio:  Cavity ratio with pwm=(B,N), p=(B,N,N).

        Returns
        -------
        output: Outgoing ratios with pwm=(B,N), p=(B,N,N). 
                    The intermediate network features have shape (B,F).
        """
        rows, cols = jnp.tril_indices(self.latent_dim)

        pwm = ratio.params["pwm"]
        p = ratio.params['p'][..., rows, cols]
        feats = (pwm, p, action) if self.action_conditioned else (pwm, p) 
        features = jnp.concatenate(feats, axis=-1)
        return vmap(lambda x: self.message_network.apply(params, x))(features)

    def _pass(self, params, local, reverse=False, actions: Array | None = None) -> GaussianNatParam:
        """
        Send messages along the chain without feeding a message back to its sender.

        At each source node the cavity is its observation ratio plus the message
        from the other side. The destination's message is excluded.

        Parameters
        ----------
        params:   Parameters of the forward or backward message network.
        local:    Summed observation ratios with pwm=(B,T,N), p=(B,T,N,N).
        reverse:  False for a forward scan, True for a backward scan.

        Returns
        -------
        output: GaussianNatParam
            Incoming chain ratios at their destination nodes, with pwm=(B,T,N), p=(B,T,N,N).
            The first time is zero for a forward pass.
            The last time is zero for a backward pass.
        """
        zero = tree_map(lambda x: jnp.zeros_like(x[:, 0]), local)

        # a single node has no incoming chain messages.
        if local.params["pwm"].shape[1] == 1:
            return tree_map(lambda x: x[:, None], zero)

        source_slice = slice(1, None) if reverse else slice(None, -1)
        sites = tree_map(lambda x: jnp.swapaxes(x[:, source_slice], 0, 1), local)
        actions = None if actions is None else jnp.swapaxes(actions, 0, 1)

        def step(incoming, inputs):
            site, action = inputs
            outgoing = self._message(params, incoming + site, action)
            return outgoing, outgoing

        _, outgoing = lax.scan(step, zero, (sites, actions), reverse=reverse)
        outgoing = tree_map(lambda x: jnp.swapaxes(x, 0, 1), outgoing)

        def pad(message, empty):
            parts = ((message, empty[:, None]) if reverse else (empty[:, None], message))
            return jnp.concatenate(parts, axis=1)

        return tree_map(pad, outgoing, zero)

    def get_messages(self, prior_params, factors_nat, actions=None) -> GaussianNatParam:
        """
        Stack observation, left and right ratios at every latent node.

        The penultimate branch contains messages from the past.
        The final branch contains messages from the future.
        Their absent endpoint messages are exactly zero natural parameters.

        Parameters
        ----------
        prior_params:  Model params. 'forward' and 'backward' message-network parameters.
        factors_nat:   Observation ratios with pwm=(J,B,T,N), p=(J,B,T,N,N).

        Returns
        -------
        output: GaussianNatParam
            All incoming ratios with pwm=(J+2,B,T,N), p=(J+2,B,T,N,N).
            The first J branches are observations, followed by left and right.
        """
        local = factors_nat.sum(axis=0)
        left = self._pass(prior_params['forward'], local, actions=actions)
        right = self._pass(prior_params['backward'], local, reverse=True, actions=actions)
        return tree_map(
            lambda obs, past, future: jnp.concatenate((obs, past[None], future[None]), axis=0),
            factors_nat, left, right,
        )
