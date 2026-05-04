from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from dynamics.hamiltonian import (
    angular_velocities_to_canonical_state,
    double_pendulum_hamiltonian_rhs,
)
from dynamics.solvers import solve_hamiltonian


FEATURE_ORDER = ["theta1", "p1", "theta2", "p2"]
TARGET_ORDER = ["dtheta1_dt", "dp1_dt", "dtheta2_dt", "dp2_dt"]


def _require_torch():
    if torch is None:
        raise ImportError(
            "PyTorch is required for torch dataset assembly or serialization. "
            "Install torch in your environment before using these helpers."
        )


def sample_initial_state(rng, omega_scale=2.0):
    """Sample a random initial condition as angles and angular velocities."""
    theta1 = rng.uniform(-np.pi, np.pi)
    theta2 = rng.uniform(-np.pi, np.pi)
    omega1 = rng.uniform(-omega_scale, omega_scale)
    omega2 = rng.uniform(-omega_scale, omega_scale)
    return theta1, omega1, theta2, omega2


def build_dataset(
    num_trajectories,
    t_max,
    dt,
    seed,
    m1=1.0,
    m2=1.0,
    L1=1.0,
    L2=1.0,
    g=9.81,
    omega_scale=2.0,
):
    """
    Build a canonical dataset from Hamiltonian trajectories.

    Inputs are states [theta1, p1, theta2, p2].
    Targets are the corresponding time derivatives from Hamilton's equations.
    """
    rng = np.random.default_rng(seed)
    states = []
    derivatives = []

    config = {
        "m1": m1,
        "m2": m2,
        "L1": L1,
        "L2": L2,
        "g": g,
        "t_max": t_max,
        "dt": dt,
    }

    for _ in range(num_trajectories):
        theta1, omega1, theta2, omega2 = sample_initial_state(rng, omega_scale=omega_scale)
        y0 = angular_velocities_to_canonical_state(theta1, omega1, theta2, omega2, m1, m2, L1, L2)
        solution = solve_hamiltonian(config=config, y0=y0, method="rk4")

        traj = np.column_stack(
            [
                solution["theta1"],
                solution["p1"],
                solution["theta2"],
                solution["p2"],
            ]
        ).astype(np.float32)
        rhs = np.array(
            [double_pendulum_hamiltonian_rhs(state, m1, m2, L1, L2, g) for state in traj],
            dtype=np.float32,
        )

        states.append(traj)
        derivatives.append(rhs)

    X = np.concatenate(states, axis=0)
    y = np.concatenate(derivatives, axis=0)
    return X, y


def train_val_split(X, y, val_fraction, seed):
    """Random train/validation split at the sample level."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)

    n_val = int(n * val_fraction)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def standardize(train_array, val_array):
    """Compute train statistics and standardize train/validation arrays."""
    mean = train_array.mean(axis=0, keepdims=True)
    std = train_array.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (train_array - mean) / std, (val_array - mean) / std, mean, std


def assemble_dataset_dict(
    X_train,
    y_train,
    X_val,
    y_val,
    dt,
    m1=1.0,
    m2=1.0,
    L1=1.0,
    L2=1.0,
    g=9.81,
    as_torch=True,
):
    """
    Assemble the standard dataset dictionary from raw train/validation arrays.

    By default, tensors are returned because the downstream training code uses PyTorch.
    """
    X_train_std, X_val_std, x_mean, x_std = standardize(X_train, X_val)
    y_train_std, y_val_std, y_mean, y_std = standardize(y_train, y_val)

    dataset = {
        "feature_order": FEATURE_ORDER.copy(),
        "target_order": TARGET_ORDER.copy(),
        "dt": np.array([dt], dtype=np.float32),
        "m1": np.array([m1], dtype=np.float32),
        "m2": np.array([m2], dtype=np.float32),
        "L1": np.array([L1], dtype=np.float32),
        "L2": np.array([L2], dtype=np.float32),
        "g": np.array([g], dtype=np.float32),
        "X_train": X_train_std.astype(np.float32),
        "y_train": y_train_std.astype(np.float32),
        "X_val": X_val_std.astype(np.float32),
        "y_val": y_val_std.astype(np.float32),
        "X_train_raw": X_train.astype(np.float32),
        "y_train_raw": y_train.astype(np.float32),
        "X_val_raw": X_val.astype(np.float32),
        "y_val_raw": y_val.astype(np.float32),
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
    }

    if not as_torch:
        return dataset

    _require_torch()
    return {
        key: torch.from_numpy(value)
        if isinstance(value, np.ndarray)
        else value
        for key, value in dataset.items()
    }


def build_training_dataset(
    num_trajectories,
    t_max,
    dt,
    seed,
    val_fraction=0.2,
    m1=1.0,
    m2=1.0,
    L1=1.0,
    L2=1.0,
    g=9.81,
    omega_scale=2.0,
    as_torch=True,
):
    """Build the full train/validation-ready dataset object in one call."""
    X, y = build_dataset(
        num_trajectories=num_trajectories,
        t_max=t_max,
        dt=dt,
        seed=seed,
        m1=m1,
        m2=m2,
        L1=L1,
        L2=L2,
        g=g,
        omega_scale=omega_scale,
    )
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_fraction, seed)
    return assemble_dataset_dict(
        X_train,
        y_train,
        X_val,
        y_val,
        dt=dt,
        m1=m1,
        m2=m2,
        L1=L1,
        L2=L2,
        g=g,
        as_torch=as_torch,
    )


def save_dataset(dataset, path):
    """Save a torch-based dataset dictionary to disk."""
    _require_torch()
    torch.save(dataset, path)


def load_dataset(path, device="cpu"):
    """Load a serialized torch dataset from disk."""
    _require_torch()
    return torch.load(Path(path), map_location=device)
