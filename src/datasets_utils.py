import jax
import jax.numpy as np
import numpy
import jax.random as jr

from functools import partial
from jax import vmap, Array, lax


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.datasets import Dataset


class LDSPrior:
    def __init__(self, T):
        self.T = T

    def sample(self, key, stationary_params, num_samples):
        m1 = stationary_params['m1']
        Q1 = stationary_params['Q1']
        A = stationary_params['A']
        b = stationary_params['b']
        Q = stationary_params['Q']

        As = np.tile(A[None], (self.T, 1, 1))
        bs = np.concatenate([
            m1[None],
            np.tile(b[None], (self.T-1, 1))
        ])
        Qs = np.concatenate([
            Q1[None],
            np.tile(Q[None], (self.T-1, 1, 1))
        ])

        @partial(np.vectorize, signature="(n),(t,d,d),(t,d),(t,d,d)->(t,d)")
        def sample_single(key, A, b, Q):
            biases = jr.multivariate_normal(key, b, Q)
            init_elems = (A, biases)

            @vmap
            def recursion(elem1, elem2):
                A1, b1 = elem1
                A2, b2 = elem2
                return A2 @ A1, A2 @ b1 + b2
            
            _, sample = jax.lax.associative_scan(recursion, init_elems)
            return sample

        keys = jr.split(key, num_samples)
        sample = sample_single(keys, As, bs, Qs)

        return sample

def random_rotation(key, n, theta=None):
    if n == 1:
        return np.eye(1) * jr.uniform(key)
    if theta is None:
        key, subkey = jr.split(key)
        theta = 0.5 * np.pi * jr.uniform(subkey)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
    out = np.eye(n)
    out = out.at[:2, :2].set(rot)
    q = np.linalg.qr(jr.uniform(key, shape=(n, n)))[0]
    return q.dot(out).dot(q.T)


def generate_linear_data(
    num_factors: int,
    latent_dim: int,
    emission_dim: int,
    num_sequences: int,
    num_timesteps: int,
    emission_cov: float,
    key: Array
) -> "Dataset":
    from src.datasets import Dataset

    J, K, D, N, T = num_factors, latent_dim, emission_dim, num_sequences, num_timesteps

    key, key_m1, key_A, key_b, key_C, key_d = jr.split(key, 6)

    A = 0.95 * random_rotation(key_A, K, theta=np.pi/5) ### 0.95 ### theta
    Q = np.eye(K) - A @ A.T
    # Q = 0.1 * np.eye(K)

    params = {
        # 'm1': jr.normal(key_m1, shape=(K,)), # for now initialize both data and model at stationary distribution
        # 'Q1': Q,
        'm1': np.zeros(K),
        'Q1': np.eye(K),
        'A': A,
        # 'b': jr.normal(key_b, shape=(K,)),
        'b': np.zeros(K),
        'Q': Q,
        'R': np.eye(D) * np.sqrt(emission_cov) # keep R constant across J for now
    }

    lat_key, obs_key = jr.split(key)

    # sample an additional 25% sequences to use as validation data
    M = N + N // 4

    prior = LDSPrior(T)
    latent_sample = prior.sample(lat_key, params, M) # MxTxK

    emission_params = {
        'C': jr.normal(key_C, shape=(J, D, K)),
        'd': jr.normal(key_d, shape=(J, D)),
    }

    # not a pure function (uses M,T,latent_sample) but should be fine
    def sample_single_factor(emission_params, key):
        C, d, R = emission_params['C'], emission_params['d'], params['R']
        obs_sample = latent_sample @ C.T + jr.multivariate_normal(key, d, R, shape=(M, T)) # MxTxD
        return obs_sample

    obs_samples = vmap(sample_single_factor)(emission_params, jr.split(obs_key, J))

    params.update(emission_params)
    
    return Dataset(
        train_data=tuple([o[:N] for o in obs_samples]),
        train_states=latent_sample[:N],
        val_data=tuple([v[N:] for v in obs_samples]),
        val_states=latent_sample[N:],
        params=params
    )
