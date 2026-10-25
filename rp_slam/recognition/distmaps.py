import jax.numpy as jnp
import flax.linen as nn

from jax import Array
from rp_slam.distributions import NatParam, GaussianNatParam


EPS = 1e-3


class DistMap:
    def __init__(self, latent_dim: int):
        self.latent_dim = latent_dim

    @property
    def input_dim(self):
        """
        Returns the dimension of the input to the distmap.
        E.g. if latent_dim = 3 and then distmap gives
        a Gaussian with diagonal precision, then the input
        dim should be 3+3=6. If the precision were Cholesky-
        parametrized, then the input dim should be 3+3*(3+1)/2=9.
        """
        pass

    def __call__(self, x: Array) -> NatParam:
        pass


class MVNDiag(DistMap):

    def __init__(self, latent_dim: int, max_precision: float | None = None):
        super().__init__(latent_dim)
        if max_precision is not None and max_precision <= 0:
            raise ValueError('max_precision must be positive.')
        self.max_precision = max_precision

    @property
    def input_dim(self):
        return self.latent_dim * 2
    
    def __call__(self, x: Array) -> GaussianNatParam:
        pwm = x[:self.latent_dim]
        diagonal = nn.softplus(x[self.latent_dim:])
        if self.max_precision is not None:
            diagonal = self.max_precision * nn.tanh(diagonal / self.max_precision)
        p = jnp.diag(diagonal)
        
        return GaussianNatParam(p=p, pwm=pwm)


class MVNCholesky(DistMap):

    @property
    def input_dim(self):
        return self.latent_dim + self.latent_dim * (self.latent_dim + 1) // 2
    
    def __call__(self, x: Array) -> GaussianNatParam:
        pwm = x[:self.latent_dim]
        p_lower_tri = x[self.latent_dim:2*-self.latent_dim]
        p_diag = x[-self.latent_dim:]
    
        L = jnp.zeros((self.latent_dim, self.latent_dim))
        L = L.at[jnp.diag_indices(self.latent_dim)].set(p_diag)
        L = L.at[jnp.tril_indices(self.latent_dim, -1)].set(p_lower_tri)
        p = L @ L.T + EPS * jnp.eye(self.latent_dim)

        return GaussianNatParam(p=p, pwm=pwm)


class MVNCholeskySoftplus(DistMap):

    def __init__(self, latent_dim: int, max_chol: float | None = None):
        super().__init__(latent_dim)
        if max_chol is not None and max_chol <= 0:
            raise ValueError('max_chol must be positive.')
        self.max_chol = max_chol

    @property
    def input_dim(self):
        return self.latent_dim + self.latent_dim * (self.latent_dim + 1) // 2
    
    def __call__(self, x: Array) -> GaussianNatParam:
        pwm = x[:self.latent_dim]
        p_lower_tri = x[self.latent_dim:-self.latent_dim]
        p_diag = x[-self.latent_dim:]

        p_diag = nn.softplus(p_diag)
        if self.max_chol is not None:
            p_diag = self.max_chol * nn.tanh(p_diag / self.max_chol)
            p_lower_tri = self.max_chol * nn.tanh(p_lower_tri / self.max_chol)
    
        L = jnp.zeros((self.latent_dim, self.latent_dim))
        L = L.at[jnp.diag_indices(self.latent_dim)].set(p_diag)
        L = L.at[jnp.tril_indices(self.latent_dim, -1)].set(p_lower_tri)
        p = L @ L.T + EPS * jnp.eye(self.latent_dim)

        return GaussianNatParam(p=p, pwm=pwm)


class MVNCholeskyReLU(DistMap):

    @property
    def input_dim(self):
        return self.latent_dim + self.latent_dim * (self.latent_dim + 1) // 2
    
    def __call__(self, x: Array) -> GaussianNatParam:
        pwm = x[:self.latent_dim]
        p_lower_tri = x[self.latent_dim:-self.latent_dim]
        p_diag = x[-self.latent_dim:]
    
        L = jnp.zeros((self.latent_dim, self.latent_dim))
        L = L.at[jnp.diag_indices(self.latent_dim)].set(nn.relu(p_diag))
        L = L.at[jnp.tril_indices(self.latent_dim, -1)].set(p_lower_tri)
        p = L @ L.T + EPS * jnp.eye(self.latent_dim)

        return GaussianNatParam(p=p, pwm=pwm)
