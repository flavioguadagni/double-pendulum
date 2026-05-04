import numpy as np
import torch

from dynamics.config import build_config
from dynamics.hamiltonian import (
    angles_to_cartesian,
    canonical_trajectory_to_angular_velocities,
    verlet_step,
)
from dynamics.integrators import integrate_trajectory, rk4_step
from dynamics.solvers import build_default_hamiltonian_state
from neural_networks.hamiltonian_nn import build_hamiltonian_model, hamiltonian_vector_field
from neural_networks.vector_field_nn import build_vector_field_model


def _select_stepper(method):
    steppers = {
        "rk4": rk4_step,
        "verlet": verlet_step,
    }
    try:
        return steppers[method]
    except KeyError as exc:
        raise ValueError(f"Unsupported rollout integration method: {method}") from exc


def _base_solution_metadata(config):
    return {
        "m1": config["m1"],
        "m2": config["m2"],
        "L1": config["L1"],
        "L2": config["L2"],
        "g": config["g"],
    }


def _build_solution_from_canonical_trajectory(config, t, y):
    theta1 = y[:, 0]
    p1 = y[:, 1]
    theta2 = y[:, 2]
    p2 = y[:, 3]
    omega1, omega2 = canonical_trajectory_to_angular_velocities(
        y,
        config["m1"],
        config["m2"],
        config["L1"],
        config["L2"],
    )
    x1, y1, x2, y2 = angles_to_cartesian(theta1, theta2, config["L1"], config["L2"])

    solution = {
        "t": t,
        "theta1": theta1,
        "omega1": omega1,
        "theta2": theta2,
        "omega2": omega2,
        "p1": p1,
        "p2": p2,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }
    solution.update(_base_solution_metadata(config))
    return solution


def load_vector_field_checkpoint(path, device="cpu"):
    """Load a trained vector-field model checkpoint and rebuild the model."""
    checkpoint = torch.load(path, map_location=device)
    model = build_vector_field_model(
        hidden_dim=checkpoint["hidden_dim"],
        num_hidden_layers=checkpoint["num_hidden_layers"],
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_hnn_checkpoint(path, device="cpu"):
    """Load a trained HNN checkpoint and rebuild the model."""
    checkpoint = torch.load(path, map_location=device)
    model = build_hamiltonian_model(
        hidden_dim=checkpoint["hidden_dim"],
        num_hidden_layers=checkpoint["num_hidden_layers"],
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def make_vector_field_rhs(model, normalization, device="cpu"):
    """
    Build an RHS callable from a trained vector-field model.

    The normalization dictionary must provide x_mean, x_std, y_mean, and y_std.
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    x_mean = normalization["x_mean"].float().to(device)
    x_std = normalization["x_std"].float().to(device)
    y_mean = normalization["y_mean"].float().to(device)
    y_std = normalization["y_std"].float().to(device)

    def rhs(state, *args):
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        state_std = (state_t - x_mean) / x_std
        with torch.no_grad():
            pred_std = model(state_std)
            pred = pred_std * y_std + y_mean
        return pred.squeeze(0).cpu().numpy()

    return rhs


def make_hnn_rhs(model, device="cpu"):
    """Build an RHS callable from a trained Hamiltonian neural network."""
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    def rhs(state, *args):
        x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)
        with torch.enable_grad():
            dxdt = hamiltonian_vector_field(model, x)
        return dxdt.detach().squeeze(0).cpu().numpy()

    return rhs


def rollout_vector_field_model(model, normalization, config=None, y0=None, method="rk4", device="cpu"):
    """Integrate the learned vector-field model and return a standard solution dictionary."""
    config = build_config(**(config or {}))
    state0 = build_default_hamiltonian_state(config) if y0 is None else y0
    stepper = _select_stepper(method)
    rhs = make_vector_field_rhs(model, normalization, device=device)
    t, y = integrate_trajectory(stepper, rhs, state0, config["t_max"], config["dt"])
    return _build_solution_from_canonical_trajectory(config, t, y)


def rollout_hnn_model(model, config=None, y0=None, method="rk4", device="cpu"):
    """Integrate the HNN-induced canonical vector field and return a standard solution dictionary."""
    config = build_config(**(config or {}))
    state0 = build_default_hamiltonian_state(config) if y0 is None else y0
    stepper = _select_stepper(method)
    rhs = make_hnn_rhs(model, device=device)
    t, y = integrate_trajectory(stepper, rhs, state0, config["t_max"], config["dt"])
    return _build_solution_from_canonical_trajectory(config, t, y)


def rollout_vector_field_checkpoint(path, config=None, y0=None, method="rk4", device="cpu"):
    """Load a vector-field checkpoint, roll it out, and return the solution dictionary."""
    model, checkpoint = load_vector_field_checkpoint(path, device=device)
    normalization = {
        "x_mean": checkpoint["x_mean"],
        "x_std": checkpoint["x_std"],
        "y_mean": checkpoint["y_mean"],
        "y_std": checkpoint["y_std"],
    }
    return rollout_vector_field_model(
        model,
        normalization,
        config=config,
        y0=y0,
        method=method,
        device=device,
    )


def rollout_hnn_checkpoint(path, config=None, y0=None, method="rk4", device="cpu"):
    """Load an HNN checkpoint, roll it out, and return the solution dictionary."""
    model, _ = load_hnn_checkpoint(path, device=device)
    return rollout_hnn_model(model, config=config, y0=y0, method=method, device=device)


__all__ = [
    "load_hnn_checkpoint",
    "load_vector_field_checkpoint",
    "make_hnn_rhs",
    "make_vector_field_rhs",
    "rollout_hnn_checkpoint",
    "rollout_hnn_model",
    "rollout_vector_field_checkpoint",
    "rollout_vector_field_model",
]
