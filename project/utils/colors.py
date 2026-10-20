import distinctipy
import numpy as np


def generate_distinct_colors(n: int, seed: int = 42) -> np.ndarray:
    """Generate n visually distinct colors as an (n, 3) array in [0, 1]."""
    return np.array(distinctipy.get_colors(n, rng=seed))
