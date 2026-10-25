import optax

import jax
import jax.numpy as jnp
import jax.random as jr

from typing import Callable, Any
from tqdm import tqdm
from dataclasses import dataclass

from jax import Array, vmap
from flax import linen as nn


###########################
#     helper functions    #
###########################

def sse_loss(params, model, x, y):
    """
    Sum Squared Error from decoder on x (target) to y (data)
    
    Parameters
    ----------
    x: (B, T, D) - batch size, time steps, latent dimension
    y: (B, T, K) - batch size, time steps, data dimension
    
    Returns
    -------
    sse:  Sum of the squared error of the NN to the targets
    """

    batched_apply = vmap(vmap(lambda _y: model.apply({'params': params}, _y)))
    preds = batched_apply(y)
    return jnp.sum((preds - x)**2), preds


###########################
#          config         #
###########################
@dataclass
class Config:
    batch_size: int
    iterations: int
    seed: int

    lr: float = 1e-3
    patience: int = 100

    use_pbar: bool = True


###########################
#         decoder         #
###########################

class DecoderNetwork(nn.Module):
    """ Project any input dimension to target dimension with NN"""
    network: nn.Module
    output_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = self.network(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class DecoderTrainer:

    opt_state: optax.OptState
    opts: optax.GradientTransformation
    itr: int

    def __init__(
            self,
            model: DecoderNetwork,
            config: Config,
            logger: Callable = lambda *x: {}
    ):
        self.model = model
        self.config = config
        self.logger = logger

    def train_step(self, params, opt_state: optax.OptState, x: Array, y: Array) -> tuple[float, Any, optax.OptState]:
        """

        Parameters
        ----------
        x: (B, T, D) - targets
        y: (B, T, K) - data where K >= D

        Returns
        -------
        loss:           sse of the current NN projection error
        new_param:      new NN params after SGD update
        new_opt_state:  new NN optimizer state after SGD update
        pred:           current projections from data to target space
        """
        
        (loss, pred), grad = jax.value_and_grad(sse_loss, has_aux=True)(params, self.model, x, y)
        updates, new_opt_state = self.opt.update(grad, opt_state, params)
        new_param = optax.apply_updates(params, updates)

        return loss, new_param, new_opt_state, pred

    def fit(self, x: Array, y: Array, val_x: Array | None = None, val_y: Array | None = None) -> None:
        """
        Fit the decoder model to the target x and data y.
        
        Parameters
        ----------
            x:      (N, T, D) - targets
            y:      (N, T, K) - data
            val_x:  (S, T, D) - validation targets
            val_y:  (S, T, K) - validation data
        """

        N, T = x.shape[:2]
        print(f'Training Decoder with N={N}, T={T}')

        train_step = jax.jit(self.train_step)

        # initialisation        
        key, subkey = jr.split(jr.PRNGKey(self.config.seed))
        self.params = self.model.init(key, y[0, 0])['params']

        self.opt = optax.adam(self.config.lr)
        self.opt_state = self.opt.init(self.params)
        
        # stores
        best_val_loss = jnp.inf
        best_params = None
        wait = 0
        self.loss_tot = []
        self.val_loss_tot = [] 

        # batching function
        _batcher = self._batching(self.config.batch_size, N)

        # run training
        pbar = tqdm(range(self.config.iterations), disable=not(self.config.use_pbar))
        for self.itr in pbar:

            # batch
            key, subkey = jr.split(key)
            batch_indices = _batcher(subkey)
            batch_x = x[batch_indices]
            batch_y = y[batch_indices]

            # train step
            loss, self.params, self.opt_state, pred = train_step(self.params, self.opt_state, batch_x, batch_y)

            # loss logging
            loss = loss / (N * T)
            self.loss_tot.append(loss)
            to_print = self.logger(self, batch_indices)
            to_print.update({'loss': f'{loss:.3f}'})

            if val_x is not None and val_y is not None:
                val_loss, _ = sse_loss(self.params, self.model, val_x, val_y)
                self.val_loss_tot.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = self.params
                    wait = 0
                
                else:
                    wait += 1

            if wait >= self.config.patience:
                self.params = best_params
                pbar.n = pbar.total
                pbar.refresh()  
                jax.debug.print(f'Early stopping at iteration {self.itr} with best validation loss {best_val_loss:.3f}')
                break

            pbar.set_postfix(**to_print)

    def _batching(self, batch_size, N):
        if batch_size >= N:
            print("Using entire dataset")
            batcher = lambda _: jnp.arange(N)
            return batcher
        
        batcher = lambda _k: jr.randint(_k, (batch_size, ), 0, N)
        return batcher

    def apply(self, params, y: Array) -> Array:
        """
        Apply the decoder model to the input y.

        Parameters
        ---------
        y:       (B, T, D) or (T, D) the set of (or singular) timeseries to apply the decoder to
        params:  model parameters
        
        Returns
        -------
        preds:  Projection of the input y to target space
        """
        assert y.ndim in (2, 3), "Data must be of shape (B, T, D) or (T, D)"

        _apply = vmap(lambda _y: self.model.apply({"params": params}, _y))
        _apply = vmap(_apply) if y.ndim == 3 else _apply
        return _apply(y)
        



# def get_posterier_train_state_means(
#         data,
#         trainer: Trainer,
#         seq_idx: int | None = None
# ):
#     """
    
#     Extract all the posterior means from the training data for 
#     the higher dimensional latent states. If `seq_idx` is provided,
#     it will compute the posterior means for that specific sequence index.
#     """
#     data = data.standardised_data

#     if seq_idx is None:
#         ts_means = jax.vmap(
#             lambda i: trainer.apply(
#                 (data.train_data[0][None, i],)
#             )[1].params["means"][0]
#         )(jnp.arange(data.train_data[0].shape[0]))
        
#     else:
#         sequence = data.train_data[0][None, seq_idx]
#         _, ts_posterior = trainer.apply((sequence,))
#         ts_means = ts_posterior.params['means'][0]
    
#     return ts_means
