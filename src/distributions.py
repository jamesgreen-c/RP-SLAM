import jax
import jax.numpy as jnp
import flax.linen as nn

from jax import vmap
from jax.random import PRNGKey, multivariate_normal
from src.utils import vmapping
from jax.scipy.linalg import solve_triangular
from typing import Any, Union, Optional
from flax.core.frozen_dict import FrozenDict
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN


from typing import TYPE_CHECKING, Sequence

from src.utils import math
if TYPE_CHECKING:
    from src.distmaps import DistMap


NetworkParams = Union[FrozenDict, dict[str, Any]]
AllParams = dict[str, Any]


class NatParam:
    """
    Class for storing and manipulating natural
    parameters of an exponential family distribution.

    Initialize the class by passing in parameters
    with names, e.g. for a Gaussian distribution,
    dist = NatParam(precision=..., precision_weighted_mean=...).

    The class then saves the parameters as a dict, e.g.
    in the Gaussian case above,
    dist.params = {'precision': ..., 'precision_weighted_mean': ...}.

    When defining a custom NatParam class, the functions that need to
    be specified are lognormalizer, expsuffstat_dot, and flatten
    (and optionally dist_param).

    Every subclass of NatParam is registered as a PyTree node,
    which allows the use of vmap and jit on functions with
    signature Array -> (subclass of NatParam).
    """
    
    def __init__(self, dist_map: Optional["DistMap"] = None, **kwargs):
        self.dist_map = dist_map
        # TODO: implement this
        # for key, value in kwargs.items():
        #     setattr(self, key, value)
        self.params = kwargs

    def __add__(self, other):
        return type(self)(self.dist_map, **{k: v + other.params[k] for k, v in self.params.items()})

    def __sub__(self, other):
        return type(self)(self.dist_map, **{k: v - other.params[k] for k, v in self.params.items()})
    
    def sum(self, axis):
        return type(self)(self.dist_map, **{k: jnp.sum(v, axis=axis) for k, v in self.params.items()})

    @property
    def lognormalizer(self):
        pass

    def expsuffstat_dot(self, v):
        """Compute the dot product <t(x)>*v, where v is a NatParam of the same type."""
        pass

    def kl(self, other):
        return (self.expsuffstat_dot(self - other)
                - self.lognormalizer + other.lognormalizer)
    
    def update(self, params):
        flattened_params = self.flatten(params)
        return self.dist_map(flattened_params)

    @property
    def latent_dim(self):
        pass

    @property
    def dist_param(self):
        """Turn natural parameters into corresponding distribution parameters"""
        pass

    ##### register subclasses of NatParam as PyTree nodes for JAX operations
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node_class(cls)

    def tree_flatten(self):
        leaves = list(self.params.values())
        aux_data = (list(self.params.keys()), self.dist_map)
        return leaves, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        param_keys, dist_map = aux_data
        return cls(dist_map, **dict(zip(param_keys, leaves)))
    #####
    

class DistParam:
    """
    Generic class for standard distribution parameters
    (e.g. mean and covariance for a Gaussian). Contains
    a method self.nat_param that returns the corresponding
    NatParam object.
    """
    def __init__(self, dist_map: Optional["DistMap"] = None, **kwargs):
        self.dist_map = dist_map
        self.params = kwargs

    # TODO: can remove--just for debugging
    def __add__(self, other):
        return type(self)(self.dist_map, **{k: v + other.params[k] for k, v in self.params.items()})

    @property
    def nat_param(self):
        pass

    ##### register subclasses of NatParam as PyTree nodes for JAX operations
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node_class(cls)

    def tree_flatten(self):
        leaves = list(self.params.values())
        aux_data = (list(self.params.keys()), self.dist_map)
        return leaves, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        param_keys, dist_map = aux_data
        return cls(dist_map, **dict(zip(param_keys, leaves)))
    #####


class AllParam:
    """
    Class storing both natural and distribution parameters
    of a distribution. In the RP-SSM code, the prior, factors,
    and posterior all require both natural and distribution
    parameters to compute the loss. This class is initialized
    with either the natural or distribution parameters, and
    automatically computes the other on initialization.
    """
    nat_param: NatParam
    dist_param: DistParam

    def __init__(self, param):
        if isinstance(param, NatParam):
            self.nat_param = param
            self.dist_param = param.dist_param
        elif isinstance(param, DistParam):
            self.nat_param = param.nat_param
            self.dist_param = param
        elif isinstance(param, tuple):
            self.nat_param = param[0]
            self.dist_param = param[1]

    def kl(self, other):
        return self.nat_param.kl(other.nat_param)

    @property
    def lognormalizer(self):
        return self.nat_param.lognormalizer

    def __add__(self, other):
        return AllParam((self.nat_param + other.nat_param, self.dist_param + other.dist_param))


class GaussianNatParam(NatParam):
    """
    Gaussian natural parameters, given by ['p', 'pwm'] = [A, A*m],
    where A is the precision matrix and m is the mean. The sufficient
    statistics are [-0.5*xx.T, x].
    """
    # p: Array
    # pwm: Array

    def flatten(self, params):
        assert {'pwm', 'chol'} <= set(params)
        return jnp.concatenate([params['pwm'], params['chol']])

    @property
    def latent_dim(self):
        return self.params['pwm'].shape[-1]

    @property
    @vmapping.auto_vmap('pwm', 1)
    def dist_param(self):
        cov = math.spd_inverse(self.params['p'])
        mean = cov @ self.params['pwm']
        return GaussianDistParam(
            dist_map=None,
            mean=mean,
            cov=cov
        ) # set dist_map=None because it won't be necessary anymore

    @property
    def lognormalizer(self):
        quad, det = math.inv_quad_form_symmetric(self.params['p'], self.params['pwm'])
        return 0.5 * (quad - det)
    
    def expsuffstat_dot(self, v):
        term1 = math.inv_quad_form(self.params['p'], self.params['pwm'], v.params['pwm'])
        L = jnp.linalg.cholesky(self.params['p'])
        first = solve_triangular(
            L.T,
            solve_triangular(L, v.params['p'], lower=True),
            lower=False
        )
        second = solve_triangular(
            L.T,
            solve_triangular(L, jnp.outer(self.params['pwm'], self.params['pwm']), lower=True),
            lower=False
        )
        term2 = -0.5 * jnp.trace(first + second @ first)
        return term1 + term2
    

class GaussianDistParam(DistParam):
    """Contains parameters ['mean', 'cov']"""
    # mean: Array
    # cov: Array

    @property
    @vmapping.auto_vmap('mean', 1)
    def nat_param(self) -> GaussianNatParam:
        p = math.spd_inverse(self.params['cov'])
        pwm = p @ self.params['mean']
        return GaussianNatParam(
            dist_map=None,
            p=p,
            pwm=pwm
        )
    
    def sample(self, key: jax.Array, shape: Sequence[int]):
        return jax.random.multivariate_normal(
            key, self.params["mean"], self.params["cov"], shape
        )

    def log_prob(self, x):
        """
        Compute the log-probability of x under the
        Gaussian distribution `self`.
        """
        assert x.shape == self.params["mean"].shape
        if self.params["cov"].ndim == 2:
            inv_quad_form, logdet = math.inv_quad_form_symmetric(
                self.params["cov"], x - self.params["mean"]
            )
        elif self.params["cov"].ndim == 1:
            logdet = jnp.sum(jnp.log(self.params["cov"]))
            inv_quad_form = jnp.sum((x - self.params["mean"]) ** 2 / self.params["cov"])
        return -0.5 * (
            self.params["mean"].shape[-1] * jnp.log(2.0 * jnp.pi) + logdet + inv_quad_form
        )
    

@jax.tree_util.register_pytree_node_class
class LGStationaryParam:
    """Linear-Gaussian prior with p(z_1)=N(m1,Q1) and transitions N(Az+b,Q).

    stationary=True fixes m1=0 and Q1=I; otherwise m1 and Q1 are learnable.
    In both modes Q=Q1-A Q1 A.T and b=(I-A)m1 are derived to preserve
    the initial marginal. These require a stable A and positive definite Q.
    A supplied as a vector represents diagonal dynamics with sigmoid-mapped
    learnable entries when update() is called.
    """
    # A: Array
    # b: Array

    # TODO: if p(z_1) is invariant dist, don't do the recursion, just copy p(z_1) T times

    def __init__(self, stationary: bool, **kwargs):
        self.params = kwargs
        self.stationary = stationary
        dim = kwargs['A'].shape[0]

        # when initialized at stationary distribution, WLOG assume m1=0, Q1=I
        if stationary:
            self.params.update(
                {
                    'm1': jnp.zeros(dim),
                    'Q1': jnp.eye(dim)
                }
            )
            
            self.opt_param = {'A': self.params['A']} # only learn A
        else:
            self.opt_param = {
                'm1': self.params['m1'],
                'Q1': self.params['Q1'],
                'A': self.params['A']
            }
            
        # if A is given as a vector, reshape it to a diagonal matrix
        if kwargs['A'].ndim == 1:
            self.params.update({'A': jnp.diag(kwargs['A'])})

        # enforce stability
        self.params.update(
            {
                'Q': self.params['Q1'] - self.params['A'] @ self.params['Q1'] @ self.params['A'].T,
                'b': (jnp.eye(dim) - self.params['A']) @ self.params['m1']
            }
        )

    @property
    def latent_dim(self):
        return self.params['A'].shape[0]
    
    def update(self, params):
        """Update an arbitrary number of parameters."""
        new_params = {**self.params, **params} # jax-compatible update
        # TODO: when not stationary, turn Q1 param into covariance
        if new_params['A'].ndim == 1:
            new_params['A'] = nn.sigmoid(new_params['A'])
        return LGStationaryParam(stationary=self.stationary, **new_params)
    
    def to_chain(self, num_timesteps):
        """Return prior chain moments for a positive number of timesteps."""
        if num_timesteps < 1:
            raise ValueError('At least one timestep is required.')
        # TODO: adjust if using p instead of Q
        # NOTE: check older commits for a more general version of this function
        # that allows for nonstationary and returning dynamics params
        if self.stationary:
            means = jnp.tile(self.params['m1'][None], (num_timesteps, 1))
            covs = jnp.tile(self.params['Q1'][None], (num_timesteps, 1, 1)) # TODO: check that this is the correct shape
            cross_covs = self.params['A'] @ covs[:-1] # Cov(z_t+1,z_t) = A @ Sigma_t
        else:
            As = jnp.tile(self.params['A'][None], (num_timesteps, 1, 1))
            bs = jnp.concatenate(
                [
                    self.params['m1'][None],
                    jnp.tile(self.params['b'][None], (num_timesteps-1, 1))
                ]
            )
            Qs = jnp.concatenate(
                [
                    self.params['Q1'][None],
                    jnp.tile(self.params['Q'][None], (num_timesteps-1, 1, 1))
                ]
            )

            @vmap
            def recursion(elem1, elem2):
                A1, b1, Q1 = elem1
                A2, b2, Q2 = elem2
                return A2 @ A1, A2 @ b1 + b2, A2 @ Q1 @ A2.T + Q2

            init_elems = (As, bs, Qs)
            _, means, covs = jax.lax.associative_scan(recursion, init_elems)
            cross_covs = As[1:] @ covs[:-1] # Cov(z_t+1,z_t) = A_t+1 @ Sigma_t
        
        return LGChainDistParam(means=means, covs=covs, cross_covs=cross_covs)

    def tree_flatten(self):
        leaves = list(self.params.values())
        aux_data = (list(self.params.keys()), self.stationary)
        return leaves, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        param_keys, stationary = aux_data
        return cls(stationary=stationary, **dict(zip(param_keys, leaves)))
    

class LGChainDistParam(DistParam):
    """
    Class containing the full distribution of a linear-Gaussian chain.

    Contains parameters ['means', 'covs', 'cross_covs'].

    Is generated from LGStationaryParam.to_chain.

    Contains methods to compute KL divergences between chains and
    compute a corresponding AllParam object.
    """
    # means: Array
    # covs: Array
    # cross_covs: Array

    def kl(self, other):
        """Compute KL(self||other)"""
        def kl_t(muq, mup, Sq, Sp):
            return MVN(muq, Sq).kl_divergence(MVN(mup, Sp))

        if self.params['means'].shape[0] == 1:
            return kl_t(
                self.params['means'][0], other.params['means'][0],
                self.params['covs'][0], other.params['covs'][0],
            )
        
        def kl_tt(
                muqt, muqtt,
                mupt, muptt,
                Sqt, Sqtt,
                Spt, Sptt,
                Sqx, Spx
            ):

            muq = jnp.concatenate((muqt, muqtt))
            mup = jnp.concatenate((mupt, muptt))
            Sq = jnp.block(
                [[Sqt, Sqx.T],
                 [Sqx,  Sqtt]]
            )
            Sp = jnp.block(
                [[Spt, Spx.T],
                 [Spx, Sptt]]
            )
            return kl_t(muq, mup, Sq, Sp)
        
        marginal = vmap(kl_t)(
            self.params['means'][1:-1],
            other.params['means'][1:-1],
            self.params['covs'][1:-1],
            other.params['covs'][1:-1]
        )
        pairwise = vmap(kl_tt)(
            self.params['means'][:-1], self.params['means'][1:],
            other.params['means'][:-1], other.params['means'][1:],
            self.params['covs'][:-1], self.params['covs'][1:],
            other.params['covs'][:-1], other.params['covs'][1:],
            self.params['cross_covs'], other.params['cross_covs']
        )

        return jnp.sum(pairwise) - jnp.sum(marginal)
    
    @property
    def all_param(self):
        dist_param = GaussianDistParam(
            dist_map=None,
            mean=self.params['means'],
            cov=self.params['covs']
        )
        return AllParam(dist_param)
    
    @property
    def mean_field(self):
        """Return Gaussian node marginals, omitting cross-time covariances."""
        return GaussianDistParam(mean=self.params["means"], cov=self.params["covs"])

    def sample(self, key: PRNGKey):
        """Sample a joint trajectory using backward conditional sampling."""
        means = self.params["means"]
        covs = self.params["covs"]
        cross_covs = self.params["cross_covs"]

        def sample_trajectory(key, means, covs, cross_covs):
            T, latent_dim = means.shape
            keys = jax.random.split(key, T)
            I = jnp.eye(latent_dim)

            # sample z_{T-1} from its smoothed marginal
            z_T = multivariate_normal(keys[-1], mean=means[-1], cov=covs[-1])

            def backward_step(z_next, inputs):
                key, mean, next_mean, cov, next_cov, cross_cov = inputs

                # cross_cov = Cov(z_{t+1}, z_t)
                gain = jnp.linalg.solve(next_cov + 1e-12 * I, cross_cov).T

                conditional_mean = mean + gain @ (z_next - next_mean)
                conditional_cov = cov - gain @ cross_cov
                conditional_cov = 0.5 * (conditional_cov + conditional_cov.T)
                conditional_cov += 1e-12 * I

                z = multivariate_normal(key, mean=conditional_mean, cov=conditional_cov)
                return z, z

            inputs = (
                keys[:-1][::-1],
                means[:-1][::-1],
                means[1:][::-1],
                covs[:-1][::-1],
                covs[1:][::-1],
                cross_covs[::-1],
            )

            _, reverse_samples = jax.lax.scan(backward_step, z_T, inputs)
            return jnp.concatenate([reverse_samples[::-1], z_T[None]], axis=0)

        # unbatched: (T, D)
        if means.ndim == 2:
            return sample_trajectory(key, means, covs, cross_covs)

        # batched: (B, T, D)
        B = means.shape[0]
        keys = jax.random.split(key, B)
        return vmap(sample_trajectory)(keys, means, covs, cross_covs)