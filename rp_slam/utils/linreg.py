import jax.numpy as jnp

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import r_regression
from sklearn.kernel_ridge import KernelRidge


def linear_r2(x, y):
    """Compute linear regression R^2 between x, y"""
    x_reshaped = jnp.reshape(x, (-1, x.shape[-1]))
    y_reshaped = jnp.reshape(y, (-1, y.shape[-1]))
    linreg = LinearRegression()
    linreg.fit(x_reshaped, y_reshaped)
    return linreg.score(x_reshaped, y_reshaped)

def krr_r2(x, y):
    """
    Compute kernel ridge regression score between x, y.
    Use RBF kernel with default hyperparameters.
    """
    x_reshaped = jnp.reshape(x, (-1, x.shape[-1]))
    y_reshaped = jnp.reshape(y, (-1, y.shape[-1]))
    krr = KernelRidge(kernel='rbf')
    krr.fit(x_reshaped, y_reshaped)
    return krr.score(x_reshaped, y_reshaped)

def linear_corr(x, y):
    """Compute linear correlation coefficient between x, y"""
    x_reshaped = jnp.reshape(x, (-1, x.shape[-1]))
    y_reshaped = jnp.reshape(y, (-1, y.shape[-1]))
    coef = r_regression(x_reshaped, y_reshaped.ravel())[0]
    return coef

def linear_predict(x, y, cov=None):
    """
    Apply a linear transformation to match x to y.
    Optionally pass a cov array of covariance
    matrices that will also get transformed.
    """
    x_reshaped = jnp.reshape(x, (-1, x.shape[-1]))
    y_reshaped = jnp.reshape(y, (-1, y.shape[-1]))
    linreg = LinearRegression()
    linreg.fit(x_reshaped, y_reshaped)
    transformed_means = linreg.predict(x_reshaped).reshape(x.shape[:-1] + (y.shape[-1],))
    if cov is None:
        return transformed_means
    else:
        transformed_covs = linreg.coef_ @ cov @ linreg.coef_.T
        return transformed_means, transformed_covs

def krr_predict(x, y):
    """
    Match x to y via kernel ridge regression with default RBF kernel.
    """
    x_reshaped = jnp.reshape(x, (-1, x.shape[-1]))
    y_reshaped = jnp.reshape(y, (-1, y.shape[-1]))
    krr = KernelRidge(kernel='rbf')
    krr.fit(x_reshaped, y_reshaped)
    transformed_means = krr.predict(x_reshaped).reshape(x.shape[:-1] + (y.shape[-1],))
    return transformed_means
