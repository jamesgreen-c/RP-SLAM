"""Neural pose regression using centred windows of latent states."""

from dataclasses import dataclass, replace
from functools import partial
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from flax import linen as nn
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from jax import Array
from src.networks import MLP
from src.utils.window import temporal_features


def _scaler(x):
    mean = x.mean(axis=(0, 1), keepdims=True)
    std = x.std(axis=(0, 1), keepdims=True)
    return mean, jnp.where(std > 1e-8, std, 1.0)


@dataclass
class Config:
    batch_size: int = 32
    iterations: int = 5000
    seed: int = 1234
    lr: float = 1e-3
    patience: int = 500
    window: int = 11
    use_pbar: bool = True


class DecoderNetwork(nn.Module):
    """ Project any input dimension to target dimension with NN"""
    network: nn.Module
    output_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = self.network(x)
        x = nn.Dense(self.output_dim)(x)
        return x


def mse_loss(params, model, targets, inputs):
    preds = model.apply({"params": params}, inputs)
    return jnp.mean((preds - targets) ** 2)


# def mse_loss(params, model, targets, inputs, temporal_weight=0.1):
#     "Encourage smoothness"
#     preds = model.apply({"params": params}, inputs)

#     position_loss = jnp.mean((preds - targets) ** 2)

#     pred_steps = jnp.diff(preds, axis=1)
#     true_steps = jnp.diff(targets, axis=1)
#     temporal_loss = jnp.mean((pred_steps - true_steps) ** 2)

#     return position_loss + temporal_weight * temporal_loss


class DecoderTrainer:

    def __init__(
            self, 
            model: DecoderNetwork, 
            config: Config
        ):
        self.model = model
        self.config = config

    def train_step(self, params, opt_state, targets, inputs):
        loss_fn = partial(mse_loss, model=self.model, targets=targets, inputs=inputs)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = self.opt.update(grads, opt_state, params)
        return loss, optax.apply_updates(params, updates), opt_state

    def fit(self, x, y, val_x, val_y):
        """
        Fit targets x from latents y, each shaped (sequences, time, features).

        Pass raw latents. Windows and scaling are fitted here. Validation is
        checked every iteration, and the best parameters are restored.
        """
        config = self.config
        train_step = jax.jit(self.train_step)
        evaluate = jax.jit(partial(mse_loss, model=self.model))

        # make windowed sequences and standardise
        y = temporal_features(y, config.window)
        val_y = temporal_features(val_y, config.window)
        x, y, val_x, val_y = self._standardise(x, y, val_x, val_y)

        # initialisation
        key, init_key = jr.split(jr.PRNGKey(config.seed))
        self.params = self.model.init(init_key, y[0, 0])["params"]

        self.opt = optax.adam(config.lr)
        opt_state = self.opt.init(self.params)
        
        # stores
        best_loss = float("inf")
        best_params = self.params
        wait = 0
        self.loss_tot = []
        self.val_loss_tot = []

        # run training        
        batch_size = min(config.batch_size, len(y))
        pbar = tqdm(range(config.iterations), disable=not config.use_pbar)
        for self.itr in pbar:

            # batching
            key, batch_key = jr.split(key)
            indices = jr.choice(batch_key, len(y), (batch_size,), replace=False)
            
            # train step
            loss, self.params, opt_state = train_step(
                self.params, opt_state, x[indices], y[indices]
            )

            # loss logging
            val_loss = float(evaluate(self.params, targets=val_x, inputs=val_y))
            self.loss_tot.append(float(loss))
            self.val_loss_tot.append(val_loss)
            pbar.set_postfix(loss=f"{float(loss):.4f}", val_loss=f"{val_loss:.4f}")

            # checks
            if not np.isfinite(float(loss)) or not np.isfinite(val_loss):
                raise FloatingPointError("Neural regression loss became non-finite.")
            
            if val_loss < best_loss:
                best_loss, best_params, wait = val_loss, self.params, 0
            else:
                wait += 1
            
            if wait >= config.patience:
                break

        # pbar.close()
        self.params = best_params
        return self

    def apply(self, params, y):
        """Decode raw latents shaped (B,T,D) or (T,D) in original pose units."""
        y = temporal_features(y, self.config.window)
        mean = self.input_mean.reshape(-1)
        std = self.input_std.reshape(-1)
        preds = self.model.apply({"params": params}, (y - mean) / std)
        return preds * self.target_std.reshape(-1) + self.target_mean.reshape(-1)

    def predict(self, y):
        return self.apply(self.params, y)

    def _standardise(self, x, y, val_x, val_y):
        
        self.input_mean, self.input_std = _scaler(y)
        self.target_mean, self.target_std = _scaler(jnp.asarray(x))

        y = (y - self.input_mean) / self.input_std
        val_y = (val_y - self.input_mean) / self.input_std

        x = (jnp.asarray(x) - self.target_mean) / self.target_std
        val_x = (jnp.asarray(val_x) - self.target_mean) / self.target_std

        return x, y, val_x, val_y


def neural_regression(
    x,
    val_x,
    train_states,
    val_states,
    config,
    plot=True,
    smoothing_sigma=1.5,
    hidden_dims=(256, 256),
):
    """
    Return (trainer, raw_predictions, smoothed_predictions).

    x and val_x are raw (N,T,D) latents. Targets are (N,T,K) poses.
    window is the total width, including the centre frame.
    smoothing_sigma=None disables the optional Gaussian filter.
    """

    # fit neural regression
    model = DecoderNetwork(MLP(tuple(hidden_dims)), train_states.shape[-1])
    trainer = DecoderTrainer(model, config)
    trainer.fit(train_states, x, val_states, val_x)

    preds_raw = trainer.predict(val_x)
    preds_smooth = preds_raw
    if smoothing_sigma is not None:
        preds_smooth = jnp.asarray(gaussian_filter1d(
            np.asarray(preds_raw), 
            sigma=smoothing_sigma, 
            axis=1
        ))

    if plot:

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(trainer.loss_tot, label="Training")
        ax.plot(trainer.val_loss_tot, label="Validation")
        ax.set(xlabel="Iteration", ylabel="Mean squared error (standardised poses)")
        ax.legend()
        fig.tight_layout()
        plt.show()

    return trainer, preds_raw, preds_smooth
