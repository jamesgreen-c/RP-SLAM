import os
import jax.random as jr
import jax.numpy as np
from src import datasets_utils
from flax.struct import dataclass
from jax import Array
from pathlib import Path


ROOT_DIR = '/ceph/scratch/samoh/experiments/ramp'


@dataclass
class Dataset:
    train_data: tuple[Array, ...]
    train_states: Array
    val_data: tuple[Array, ...]
    val_states: Array
    params: dict[str, Array]

    # TODO: standardizing over the correct axes?
    @property
    def standardized_data(self):
        means = tuple(np.mean(d, axis=(0,1), keepdims=True) for d in self.train_data)
        stds = tuple(np.std(d, axis=(0,1), keepdims=True) for d in self.train_data)
        scaled_train_data = tuple((d-m)/s for d, m, s in zip(self.train_data, means, stds))
        scaled_val_data = tuple((d-m)/s for d, m ,s in zip(self.val_data, means, stds))

        return Dataset(
            train_data=scaled_train_data,
            train_states=self.train_states,
            val_data=scaled_val_data,
            val_states=self.val_states,
            params=self.params
        )

    @property
    def flatten(self):
        train_shape = self.train_data[0].shape[:2] + (-1,)
        val_shape = self.val_data[0].shape[:2] + (-1,)
        train_data = tuple(np.reshape(x, train_shape) for x in self.train_data)
        val_data = tuple(np.reshape(x, val_shape) for x in self.val_data)

        return Dataset(
            train_data=train_data,
            train_states=self.train_states,
            val_data=val_data,
            val_states=self.val_states,
            params=self.params
        )

    def __getitem__(self, index):
        """Allow indexing over all J data modalities"""
        return Dataset(
            train_data=tuple(x[index] for x in self.train_data),
            train_states=self.train_states[index],
            val_data=tuple(x[index] for x in self.val_data),     
            val_states=self.val_states[index],
            params=self.params
        )
        

def load_dataset(name: str, seed: int, datadir: str = ROOT_DIR) -> Dataset:
    """
    Load dataset from string (for experiments).
    """
    key = jr.PRNGKey(seed+1) # to avoid correlations with PRNGKey(seed), which trainer.fit uses
    if name == 'linear_small':
        return datasets_utils.generate_linear_data(1, 3, 5, 200, 100, 0.1, key)
    elif name == 'linear_medium':
        return datasets_utils.generate_linear_data(1, 5, 10, 200, 100, 0.1, key)
    elif name == 'linear_large':
        return datasets_utils.generate_linear_data(1, 10, 20, 200, 100, 0.1, key)

    elif name == 'gps':
            
            BASE_DIR = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "slam" / "data").is_dir())
            data_dir = BASE_DIR / "experiments" / "slam" / "data" / "files"
    
            latents = np.load(os.path.join(data_dir, "latents.npy"))

            observations = latents[:, :, :2]
            observations += 0.1 * jr.normal(key, shape=observations.shape)
    
            return Dataset(
                train_data=(observations[:200], ),
                train_states=latents[:200],
                val_data=(observations[200:], ),
                val_states=latents[200:],
                params=None
            )
    
    elif name == 'sensor':
        
        BASE_DIR = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "slam" / "data").is_dir())
        data_dir = BASE_DIR / "experiments" / "slam" / "data" / "files"

        latents = np.load(os.path.join(data_dir, "latents.npy"))
        observations = np.load(os.path.join(data_dir, "sensor.npy"))

        return Dataset(
            train_data=(observations[:200], ),
            train_states=latents[:200],
            val_data=(observations[200:], ),
            val_states=latents[200:],
            params=None
        )
    
    elif name == 'radar':
        
        BASE_DIR = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "slam" / "data").is_dir())
        data_dir = BASE_DIR / "experiments" / "slam" / "data" / "files"

        latents = np.load(os.path.join(data_dir, "latents.npy"))
        observations = np.load(os.path.join(data_dir, "radar.npy"))

        return Dataset(
            train_data=(observations[:200], ),
            train_states=latents[:200],
            val_data=(observations[200:], ),
            val_states=latents[200:],
            params=None
        )
    
    elif name == 'image':
        
        BASE_DIR = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "slam" / "data").is_dir())
        data_dir = BASE_DIR / "experiments" / "slam" / "data" / "files"

        latents = np.load(os.path.join(data_dir, "latents.npy"))
        observations = np.load(os.path.join(data_dir, "images.npy")).astype(np.float32) / 255.0
        # print(observations.shape)
        observations = observations[:, :, 18:42, :]

        return Dataset(
            train_data=(observations[:200], ),
            train_states=latents[:200],
            val_data=(observations[200:], ),
            val_states=latents[200:],
            params=None
        )
    
    elif name == 'greyscale':
        
        BASE_DIR = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "slam" / "data").is_dir())
        data_dir = BASE_DIR / "experiments" / "slam" / "data" / "files"

        latents = np.load(os.path.join(data_dir, "latents.npy"))
        observations = np.load(os.path.join(data_dir, "greyscale_images.npy")).astype(np.float32) / 255.0
        observations = observations[:, :, 18:42, :]

        return Dataset(
            train_data=(observations[:200], ),
            train_states=latents[:200],
            val_data=(observations[200:], ),
            val_states=latents[200:],
            params=None
        )

    elif name == 'corrupted':
        
        BASE_DIR = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "slam" / "data").is_dir())
        data_dir = BASE_DIR / "experiments" / "slam" / "data" / "files"

        latents = np.load(os.path.join(data_dir, "latents.npy"))
        observations = np.load(os.path.join(data_dir, "corrupted_greyscale_images.npy")).astype(np.float32) / 255.0
        observations = observations[:, :, 18:42, :]

        return Dataset(
            train_data=(observations[:200], ),
            train_states=latents[:200],
            val_data=(observations[200:], ),
            val_states=latents[200:],
            params=None
        )

    elif name == 'uninformative':
            
        BASE_DIR = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "slam" / "data").is_dir())
        data_dir = BASE_DIR / "experiments" / "slam" / "data" / "files"

        latents = np.load(os.path.join(data_dir, "latents.npy"))
        observations = np.load(os.path.join(data_dir, "uninformative_greyscale_images.npy")).astype(np.float32) / 255.0
        observations = observations[:, :, 18:42, :]

        return Dataset(
            train_data=(observations[:200], ),
            train_states=latents[:200],
            val_data=(observations[200:], ),
            val_states=latents[200:],
            params=None
        )

    else:
        raise NotImplementedError

