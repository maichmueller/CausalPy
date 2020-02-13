from .noise import Noise
from typing import Optional, Union, Collection
import numpy as np


class DiscreteNoise(Noise):
    def __init__(
        self,
        values: Collection[Union[float, int]],
        probabilities: Optional[Collection[float]] = None,
        seed: Optional[int] = None,
    ):
        self.values = values
        self.probabilities = probabilities
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, size, **kwargs):
        return self.rng.choice(a=self.values, p=self.probabilities, size=size)
