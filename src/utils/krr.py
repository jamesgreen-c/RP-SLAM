import math

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.scipy.linalg import cho_solve
from tqdm import tqdm


def _matrix(x):
    """Flatten leading sample axes, preserving the final feature axis."""
    x = jnp.asarray(x)
    if x.ndim < 2:
        raise ValueError("Expected shape (..., features), with at least two axes.")
    return x.reshape(-1, x.shape[-1])


def _scaler(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    # leave constant features well defined.
    return mean, jnp.where(std > 1e-8, std, 1.0)


def _r2(truth, preds):
    truth, preds = jnp.asarray(truth), jnp.asarray(preds)
    if truth.shape != preds.shape:
        raise ValueError(f"Shape mismatch: truth={truth.shape}, predictions={preds.shape}")

    truth, preds = _matrix(truth), _matrix(preds)
    rss = jnp.sum((truth - preds) ** 2, axis=0)
    tss = jnp.sum((truth - truth.mean(axis=0)) ** 2, axis=0)

    # finite scores for constant targets.
    safe_tss = jnp.where(tss > 0, tss, 1.0)
    return jnp.where(tss > 0, 1.0 - rss / safe_tss, (rss == 0).astype(truth.dtype))


class KernelRidge:
    def __init__(
        self,
        X,
        Y,
        params: dict,
        key=None,
        batch_frac: float | None = None,
        standardize: bool = True,
    ):
        X, Y = jnp.asarray(X), jnp.asarray(Y)

        if X.ndim < 2 or Y.ndim < 2:
            raise ValueError("X and Y must have shape (..., features).")
        if X.shape[:-1] != Y.shape[:-1]:
            raise ValueError("X and Y must have matching sample axes.")

        dtype = jnp.result_type(X.dtype, Y.dtype, jnp.float32)
        X, Y = _matrix(X).astype(dtype), _matrix(Y).astype(dtype)
        if X.shape[0] == 0:
            raise ValueError("Training data must contain at least one sample.")

        self.n_features = X.shape[-1]
        self.n_targets = Y.shape[-1]
        self.params = {}

        # Estimate scaling from all training observations before subsampling.
        if standardize:
            self.x_mean, self.x_std = _scaler(X)
            self.y_mean, self.y_std = _scaler(Y)
        else:
            self.x_mean = jnp.zeros((1, self.n_features), dtype=dtype)
            self.x_std = jnp.ones_like(self.x_mean)
            self.y_mean = jnp.zeros((1, self.n_targets), dtype=dtype)
            self.y_std = jnp.ones_like(self.y_mean)

        X = (X - self.x_mean) / self.x_std
        Y = (Y - self.y_mean) / self.y_std

        if batch_frac is not None:
            if not 0 < batch_frac <= 1:
                raise ValueError("batch_frac must lie in (0, 1].")

            if batch_frac < 1:
                key = jr.PRNGKey(1234) if key is None else key
                size = max(1, int(X.shape[0] * batch_frac))
                indices = jr.choice(key, X.shape[0], shape=(size,), replace=False)
                X, Y = X[indices], Y[indices]

        self.x, self.y = X, Y
        self.k = None
        self.update(params)

    def kernel(self, X1, X2):
        """Gaussian kernel on already-standardised feature matrices."""
        dist_sq = (
            jnp.sum(X1**2, axis=1)[:, None]
            + jnp.sum(X2**2, axis=1)[None, :]
            - 2.0 * (X1 @ X2.T)
        )
        # roundoff can make squared distances slightly negative.
        dist_sq = jnp.maximum(dist_sq, 0.0)
        return jnp.exp(-0.5 * dist_sq / self.params["sigma"]**2)

    def fit(self, Y, K):
        diagonal = jnp.diag_indices(K.shape[0])
        regularized = K.at[diagonal].add(self.params["lambda"])
        chol = jnp.linalg.cholesky(regularized)
        return cho_solve((chol, True), Y)

    def update(self, params: dict):
        candidate = {**self.params, **params}

        for name in ("sigma", "lambda"):
            if name not in candidate:
                raise KeyError(f"Missing parameter: {name!r}")
            value = float(candidate[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
            candidate[name] = value

        rebuild_kernel = (
            self.k is None
            or candidate["sigma"] != self.params.get("sigma")
        )
        self.params = candidate

        if rebuild_kernel:
            self.k = self.kernel(self.x, self.x)

        self.alpha = self.fit(self.y, self.k)
        return self

    def predict(self, X, batch_size: int = 500):
        """Return shape (*X.shape[:-1], n_targets), in original target units."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        X = jnp.asarray(X)
        if X.ndim < 2 or X.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected shape (..., {self.n_features}); got {X.shape}."
            )

        output_shape = (*X.shape[:-1], self.n_targets)
        X = (_matrix(X) - self.x_mean) / self.x_std

        if X.shape[0] == 0:
            return jnp.empty(output_shape, dtype=self.y.dtype)

        chunks = []
        for start in range(0, X.shape[0], batch_size):
            batch = X[start:start + batch_size]
            chunks.append(self.kernel(batch, self.x) @ self.alpha)

        preds = jnp.concatenate(chunks, axis=0)
        preds = preds * self.y_std + self.y_mean
        return preds.reshape(output_shape)

    @staticmethod
    def r2(truth, preds):
        """R² per target dimension, pooling all sample axes."""
        return _r2(truth, preds)


def sweep_krr(
    train_X,
    train_Y,
    val_X,
    val_Y,
    sigma_range,
    lambda_range,
    batch_frac: float | None = None,
    batch_size: int = 500,
    seed: int = 1234,
):
    sigmas = [float(s) for s in sigma_range]
    lambdas = [float(lam) for lam in lambda_range]

    if not sigmas or not lambdas:
        raise ValueError("Both parameter ranges must be nonempty.")

    for name, values in (("sigma", sigmas), ("lambda", lambdas)):
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"All {name} values must be finite and positive.")

    model = KernelRidge(
        train_X,
        train_Y,
        params={"sigma": sigmas[0], "lambda": lambdas[0]},
        key=jr.PRNGKey(seed),
        batch_frac=batch_frac,
    )

    scores = np.full((len(sigmas), len(lambdas)), np.nan)
    best_score = -math.inf
    best_params = None

    with tqdm(total=len(sigmas) * len(lambdas), desc="Fitting KRR") as pbar:
        for i, sigma in enumerate(sigmas):
            for j, lam in enumerate(lambdas):
                params = {"sigma": sigma, "lambda": lam}
                if model.params != params:
                    model.update(params)

                preds = model.predict(val_X, batch_size=batch_size)
                score = float(jnp.mean(model.r2(val_Y, preds)))
                scores[i, j] = score

                if math.isfinite(score) and score > best_score:
                    best_score = score
                    best_params = params.copy()

                pbar.set_postfix(best_r2=f"{best_score:.4f}")
                pbar.update()

    if best_params is None:
        raise RuntimeError(
            "Every candidate produced a non-finite score. "
            "Check the data and try stronger regularisation."
        )

    model.update(best_params)
    return model, scores