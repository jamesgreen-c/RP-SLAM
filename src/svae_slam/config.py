import optax

from dataclasses import dataclass, field
from typing import Callable, Union, Optional


LearningRate = Union[float, Callable[[int], float]]


@dataclass
class OptimConfig:
    """Configuration for a single parameter-block optimiser."""

    optimizer: Callable[[LearningRate], optax.GradientTransformation] = optax.adam
    lr: LearningRate = 1e-3
    max_grad_norm: Optional[float] = None

    decay_steps: Optional[int] = None
    decay_rate: Optional[float] = None
    staircase: bool = False

    def schedule(self) -> LearningRate:
        if callable(self.lr):
            return self.lr

        if self.decay_steps is None or self.decay_rate is None:
            return self.lr

        return optax.exponential_decay(
            init_value=self.lr,
            transition_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=self.staircase,
        )

    def build(self) -> optax.GradientTransformation:
        transforms = []

        if self.max_grad_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.max_grad_norm))

        transforms.append(self.optimizer(self.schedule()))
        return optax.chain(*transforms)


@dataclass
class Config:
    batch_size: int = 32
    num_iter: int = 1000
    stabilise_A: str | None = None

    # model
    beta_schedule: LearningRate = lambda i: 1.0
    em: bool = False

    # execution
    seed: int = 0
    jit: bool = True

    prior: OptimConfig = field(default_factory=lambda: OptimConfig(
        optimizer=optax.adam,
        lr=1e-3,
        max_grad_norm=10.0,
    ))

    recognition: OptimConfig = field(default_factory=lambda: OptimConfig(
        optimizer=optax.adam,
        lr=1e-3,
        max_grad_norm=10.0,
    ))

    generation: OptimConfig = field(default_factory=lambda: OptimConfig(
        optimizer=optax.adam,
        lr=1e-3,
        max_grad_norm=10.0,
    ))
