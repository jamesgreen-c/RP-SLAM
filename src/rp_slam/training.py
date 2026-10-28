
import optax

import jax
import jax.random as jr
import numpy as np

from tqdm import tqdm
from typing import Callable

from jax import Array
from jax.random import PRNGKey

from src.rp_slam.free_energy import rpm_fe
from src.distributions import AllParams
from src.utils.linreg import linear_r2
from src.rp_slam.config import Config
from src.utils.math import scale_sv, clip_sv
from src.rp_slam.free_energy import ramp_fe


EPS = 1e-3


class Trainer:
    params: AllParams
    opt_states: dict[str, optax.OptState]
    opts: dict[str, optax.GradientTransformation]
    itr: int

    def __init__(
            self,
            model: ramp_fe.ConstrainedIVFreeEnergy | rpm_fe.ConstrainedIVFreeEnergy,
            config: Config,
            logger: Callable = lambda *x: {}
    ):
        
        self.model = model
        self.config = config
        self.logger = logger
        self.itr = 0
        self.r2_history = []

    def train_step(
            self,
            itr: int,
            params: AllParams,
            opt_states: dict[str, optax.OptState],
            data: tuple[Array, ...] | Array,
    ) -> tuple[AllParams, dict[str, optax.OptState], Array]: 
        model_params, model_opt_states, model_loss = self._update_model(
            itr,
            params,
            opt_states,
            data
        )
        new_params = {**params, **model_params}
        new_opt_states = {**opt_states, **model_opt_states}
        return new_params, new_opt_states, model_loss
    
    def fit(self, data: tuple[Array, ...] | Array, use_pbar: bool = True, y: Array | None = None) -> None:
        """Fit observation sequences; optionally record latent regression R²."""
        data = self.model.model._data(data)

        key, init_key = jr.split(jr.PRNGKey(self.config.seed))
        self.params, self.opt_states, self.opts = self.init(init_key, data)

        train_step = jax.jit(self.train_step) if self.config.jit else self.train_step

        # Run model training.
        self.loss_tot = []
        self.r2_history = []

        pbar = tqdm(range(self.config.num_iter), disable=not(use_pbar), desc="Training")
        for self.itr in pbar:
            key, subkey = jr.split(key)

            if self.config.batch_size == data[0].shape[0]:
                if self.itr == 0:
                    jax.debug.print('using entire dataset')
                batch_indices = np.arange(self.config.batch_size)
            else:
                batch_indices = jr.randint(subkey, (self.config.batch_size,), 0, data[0].shape[0])
            batch = [d[batch_indices] for d in data]

            self.params, self.opt_states, loss = train_step(self.itr, self.params, self.opt_states, batch)
            self._stabilise_params()
            
            self.loss_tot.append(loss)
            pbar.set_postfix(loss=float(loss))

            if y is not None and self.itr % 100 == 0:
                _posterior = self.apply(data)
                name = 'mean' if 'mean' in _posterior.params else 'means'
                x = _posterior.params[name]
                r2 = linear_r2(x, y)
                self.r2_history.append(r2)


    def init(self, key: PRNGKey, initial_batch: tuple[Array, ...] | Array):
        """Initialise model parameters and optimiser states from observations."""
        params, opt_states, opts = self.model.init(
            key, 
            initial_batch,
            self.config
        )
        return params, opt_states, opts
    
    def _update_model(self, itr: int, params, opt_states, data):
        """Update prior/message and recognition parameters using the model loss."""

        beta = self.config.beta_schedule(itr)
        em = self.config.em

        _params = {"prior": params["prior"], "rpm": params["rpm"]}
        (loss, aux), grads = jax.value_and_grad(self.model.loss, has_aux=True)(_params, data, beta=beta, em=em)

        new_params = {}
        new_opt_states = {}
        for name in ("prior", "rpm"):
            updates, new_opt_states[name] = self.opts[name].update(grads[name], opt_states[name], params[name])
            new_params[name] = optax.apply_updates(params[name], updates)
        return new_params, new_opt_states, loss


    def apply(self, data: tuple[Array, ...] | Array):
        """Return the model's smoothing posterior for observation sequences.

        RAMPSSM returns GaussianDistParam node marginals (mean, cov).
        RPSSM returns LGChainDistParam (means, covs, cross_covs).
        Both retain the leading batch and time axes.
        """
        return self.model.apply(self.params["prior"], self.params["rpm"], data)

    def _stabilise_params(self):
        """Apply the configured singular-value constraint to RPSSM dynamics."""
        # RAMP uses bounded message heads, not an LGSSM matrix.
        if 'A' not in self.params['prior']:
            return  
        
        # TODO stabilisation would require new Q calculation so needs to go into prior? 
        if self.config.stabilise_A == 'scale':
            self.params["prior"]["A"] = scale_sv(self.params["prior"]["A"], EPS)
        elif self.config.stabilise_A == 'clip':
            self.params["prior"]["A"] = clip_sv(self.params["prior"]["A"], EPS)
    