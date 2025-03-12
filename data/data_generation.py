import numpy as np


def linsep(d: int, m: int, low: float=0., high: float=1.) -> tuple:
    """Generate a linearly separable dataset.

    Args:
        d (int): Space dimension (>= 2)
        m (int): Dataset dimension (>= 1)
        low (float): Lower bound for random sampling (default: 0)
        high (float): Upper bound for random sampling (default: 1)

    Returns:
        tuple: (X, y), where X is an m × d array of points representing the
               dataset and an y an m-dimensional array of labels (-1 or +1)
    """
    if d < 2:
        raise ValueError("d must be >= 2!")
    if m < 1:
        raise ValueError("m must be >= 1!")
    if low >= high:
        raise ValueError("low must be < high!")

    # Generate random hyperplane
    b = np.random.uniform(low=low, hight=high, size=d)
    b_0 = np.random.uniform(low=low, high=high)

    # Construct the dataset
    X = np.random.uniform(low=low, high=high, size=(m, d))
    y = np.where(X @ b + b_0 >= 0, 1, -1)

    return X, y