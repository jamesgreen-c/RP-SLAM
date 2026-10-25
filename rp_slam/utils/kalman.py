"""Dynamax filtering and smoothing for Gaussian recognition factors.

Each factor mean is a pseudo-observation with its covariance as observation
noise and an identity emission matrix. Dynamics are z_{t+1}=A z_t+b+noise.
Inputs describe one sequence; callers can vmap over the batch dimension.
"""

import jax.numpy as jnp
from jax import vmap

from dynamax.linear_gaussian_ssm import parallel_lgssm_filter, parallel_lgssm_smoother
from dynamax.linear_gaussian_ssm.inference import make_lgssm_params
from dynamax.utils.utils import symmetrize

from rp_slam.distributions import LGChainDistParam, GaussianDistParam
from rp_slam.utils.math import inv_quad_form


def _lgssm_params(prior, factors):
    """Build an unbatched Dynamax model with constant transition bias b.

    factors has mean=(T,D), cov=(T,D,D). The prior supplies m1, Q1, A, Q, b.
    A vector bias is shared across time by Dynamax.
    """
    _, dim = factors.params['mean'].shape
    dtype = factors.params['mean'].dtype
    return make_lgssm_params(
        prior.params['m1'],
        prior.params['Q1'],
        prior.params['A'],
        prior.params['Q'],
        jnp.eye(dim, dtype=dtype),
        factors.params['cov'],
        dynamics_bias=prior.params['b'],
        emissions_bias=jnp.zeros(dim, dtype=dtype),
    )


def _filter_distribution(prior, factors, output):
    """Return Dynamax's filtered Gaussian node marginals.

    prior and factors are retained in this helper's interface. The result has
    mean=(T,D), cov=(T,D,D), without a cross-time sampling coupling.
    """
    return GaussianDistParam(
        dist_map=None,
        mean=output.filtered_means,
        cov=symmetrize(output.filtered_covariances),
    )


def parallel_filter(prior, factors):
    """Return causal GaussianDistParam marginals for one sequence.

    factors has mean=(T,D) and cov=(T,D,D). The returned mean and covariance
    have the same shapes and use observations only up to each time.
    """
    params = _lgssm_params(prior, factors)
    output = parallel_lgssm_filter(params, factors.params['mean'])
    return _filter_distribution(prior, factors, output)


def parallel_smoother(prior, factors):
    """Return (LGChainDistParam posterior, GaussianDistParam filtered).

    The smoothing chain has means=(T,D), covs=(T,D,D), and
    cross_covs=(T-1,D,D), oriented as Cov(z_{t+1},z_t).
    Filtered marginals reuse the forward output from Dynamax's smoother.
    """
    params = _lgssm_params(prior, factors)
    output = parallel_lgssm_smoother(params, factors.params['mean'])

    A = prior.params['A']
    Q = prior.params['Q']
    
    filtered_covs = symmetrize(output.filtered_covariances)
    smoothed_covs = symmetrize(output.smoothed_covariances)
    
    cross_covs = vmap(
        lambda S, F: inv_quad_form(Q + A @ F @ A.T, S.T, A @ F)
    )(smoothed_covs[1:], filtered_covs[:-1])
    
    posterior = LGChainDistParam(
        means=output.smoothed_means,
        covs=smoothed_covs,
        cross_covs=cross_covs,
    )
    return posterior
