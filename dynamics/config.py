import json

import numpy as np


DEFAULT_DOUBLE_PENDULUM_CONFIG = {
    "m1": 10.0,
    "m2": 5.0,
    "L1": 2.0,
    "L2": 1.0,
    "g": 9.81,
    "theta1_0": float(np.pi / 2.0),
    "omega1_0": 0.0,
    "theta2_0": float(np.pi / 2.0 + 0.01),
    "omega2_0": 0.0,
    "t_max": 20.0,
    "dt": 0.005,
}


def build_config(**overrides):
    """Return a simulation config dictionary with user overrides applied."""
    config = DEFAULT_DOUBLE_PENDULUM_CONFIG.copy()
    config.update(overrides)
    validate_config(config)
    return config


def load_config(path, **overrides):
    """Load a JSON config file, apply optional overrides, and validate the result."""
    with open(path, "r", encoding="ascii") as handle:
        user_config = json.load(handle)

    config = DEFAULT_DOUBLE_PENDULUM_CONFIG.copy()
    config.update(user_config)
    config.update(overrides)
    validate_config(config)
    return config


def validate_config(config):
    """Validate that the main physical and numerical parameters are admissible."""
    for key in ("m1", "m2", "L1", "L2", "g", "t_max", "dt"):
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")

    for key in ("m1", "m2"):
        if config[key] <= 0.0:
            raise ValueError(f"{key} must be strictly positive.")

    for key in ("L1", "L2"):
        if config[key] <= 0.0:
            raise ValueError(f"{key} must be strictly positive.")

    if config["g"] <= 0.0:
        raise ValueError("g must be strictly positive.")

    if config["t_max"] <= 0.0:
        raise ValueError("t_max must be strictly positive.")

    if config["dt"] <= 0.0:
        raise ValueError("dt must be strictly positive.")
