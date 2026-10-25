import jax.numpy as jnp
from jax import vmap

from jax.scipy.linalg import solve_triangular


def inv_quad_form_symmetric(S, m):
    """Compute m.T * inv(S) * m and log(det(S))"""
    L = jnp.linalg.cholesky(S)
    x = solve_triangular(L, m, lower=True)
    logdet = 2 * jnp.sum(jnp.log(jnp.diag(L)))
    return x.T @ x, logdet


def inv_quad_form(S, m1, m2):
    """Compute m1.T * inv(S) * m2"""
    L = jnp.linalg.cholesky(S)
    x1 = solve_triangular(L, m1, lower=True)
    x2 = solve_triangular(L, m2, lower=True)
    return x1.T @ x2


def spd_inverse(S):
    """Invert a spd matrix via Cholesky"""
    L = jnp.linalg.cholesky(S)
    Linv = solve_triangular(L, jnp.eye(S.shape[-1]), lower=True)
    return Linv.T @ Linv


def logdet(S):
    L = jnp.linalg.cholesky(S)
    logdet = 2 * jnp.sum(jnp.log(jnp.diag(L)))
    return logdet


def precision_mean_and_chol(precision, weighted_mean):
    """Unbatched Gaussian mean and Cholesky factor, using triangular solves."""
    precision = 0.5 * (precision + precision.T)
    chol = jnp.linalg.cholesky(precision)
    mean = solve_triangular(
        chol.T, solve_triangular(chol, weighted_mean, lower=True), lower=False
    )
    return mean, chol


def _gaussian_kl_precision(pq, hq, pp, hp):
    """KL(q || p), avoiding subtraction of large Gaussian log normalizers."""
    mq, lq = precision_mean_and_chol(pq, hq)
    mp, lp = precision_mean_and_chol(pp, hp)
    relative = solve_triangular(lq, lp, lower=True)
    diagonal = jnp.diag(relative)
    x = diagonal - 1.0
    # d^2 - 1 - 2 log(d) is nonnegative, but cancels near d=1.
    series = x * x * (2.0 - (2.0 / 3.0) * x + 0.5 * x * x - 0.4 * x**3)
    exact = (diagonal - 1.0) * (diagonal + 1.0) - 2.0 * jnp.log(diagonal)
    diagonal_term = jnp.where(jnp.abs(x) < 1e-3, series, exact)
    off_diagonal = relative - jnp.diag(diagonal)
    covariance_term = jnp.sum(off_diagonal**2) + jnp.sum(diagonal_term)
    mean_term = jnp.sum((lp.T @ (mq - mp))**2)
    return 0.5 * (covariance_term + mean_term)


def gaussian_kl_precision(pq, hq, pp, hp):
    """Stable Gaussian KL with optional broadcast batch dimensions."""
    shape = jnp.broadcast_shapes(pq.shape[:-2], hq.shape[:-1], pp.shape[:-2], hp.shape[:-1])
    if not shape:
        return _gaussian_kl_precision(pq, hq, pp, hp)
    dim = hq.shape[-1]
    arrays = (
        jnp.broadcast_to(pq, shape + (dim, dim)).reshape((-1, dim, dim)),
        jnp.broadcast_to(hq, shape + (dim,)).reshape((-1, dim)),
        jnp.broadcast_to(pp, shape + (dim, dim)).reshape((-1, dim, dim)),
        jnp.broadcast_to(hp, shape + (dim,)).reshape((-1, dim)),
    )
    return vmap(_gaussian_kl_precision)(*arrays).reshape(shape)


def scale_sv(A, eps):
    """
    Given a matrix A, scale its singular values to <=1-EPS.
    Only do so if there is a singular value already >1-EPS.

    Intuitively, this should work better than clipping
    if all singular values are relatively close to 1, as it
    preserves the relative sizes between singular values.
    If there is a single singular value that is very large,
    this will push the others to 0, so clipping may be
    more appropriate.
    """
    
    if len(A.shape) < 2:
        A = jnp.diag(A)  # from vector to diagonal matrix
        _, s, _ = jnp.linalg.svd(A)
        scale = jnp.maximum(jnp.max(s), 1-eps) / (1-eps)
        A = jnp.diag(A / scale)  # from diagonal matrix to vector
    else:
        _, s, _ = jnp.linalg.svd(A)
        scale = jnp.maximum(jnp.max(s), 1-eps) / (1-eps)
        A = A / scale
    return A


def clip_sv(A, eps):
    """
    Clip the SVs of a matrix to be less than 1-EPS.
    """
    u, s, vt = jnp.linalg.svd(A)
    return u @ jnp.diag(jnp.clip(s, 0., 1.-eps)) @ vt
