from .config import build_config
from .hamiltonian import (
    angles_to_cartesian as hamiltonian_angles_to_cartesian,
    angular_velocities_to_canonical_state,
    canonical_trajectory_to_angular_velocities,
    double_pendulum_hamiltonian_rhs,
    verlet_step,
)
from .integrators import integrate_trajectory, rk4_step
from .lagrangian import (
    angles_to_cartesian as lagrangian_angles_to_cartesian,
    double_pendulum_lagrangian_rhs,
)


def build_default_lagrangian_state(config):
    """Build the default [theta1, omega1, theta2, omega2] state from a config dictionary."""
    return [
        config["theta1_0"],
        config["omega1_0"],
        config["theta2_0"],
        config["omega2_0"],
    ]


def build_default_hamiltonian_state(config):
    """Build the default [theta1, p1, theta2, p2] state from a config dictionary."""
    return angular_velocities_to_canonical_state(
        config["theta1_0"],
        config["omega1_0"],
        config["theta2_0"],
        config["omega2_0"],
        config["m1"],
        config["m2"],
        config["L1"],
        config["L2"],
    )


def _base_solution_metadata(config):
    return {
        "m1": config["m1"],
        "m2": config["m2"],
        "L1": config["L1"],
        "L2": config["L2"],
        "g": config["g"],
    }


def solve_lagrangian(config=None, y0=None, method="rk4"):
    """Solve the double pendulum in Lagrangian coordinates and return a plotting-friendly solution dict."""
    config = build_config(**(config or {}))
    if method != "rk4":
        raise ValueError("The Lagrangian solver currently supports only method='rk4'.")

    state0 = build_default_lagrangian_state(config) if y0 is None else y0
    params = (config["m1"], config["m2"], config["L1"], config["L2"], config["g"])
    t, y = integrate_trajectory(
        rk4_step,
        double_pendulum_lagrangian_rhs,
        state0,
        config["t_max"],
        config["dt"],
        *params,
    )

    theta1 = y[:, 0]
    omega1 = y[:, 1]
    theta2 = y[:, 2]
    omega2 = y[:, 3]
    x1, y1, x2, y2 = lagrangian_angles_to_cartesian(theta1, theta2, config["L1"], config["L2"])

    solution = {
        "t": t,
        "theta1": theta1,
        "omega1": omega1,
        "theta2": theta2,
        "omega2": omega2,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }
    solution.update(_base_solution_metadata(config))
    return solution


def solve_hamiltonian(config=None, y0=None, method="rk4"):
    """Solve the double pendulum in canonical coordinates and return a plotting-friendly solution dict."""
    config = build_config(**(config or {}))
    state0 = build_default_hamiltonian_state(config) if y0 is None else y0
    params = (config["m1"], config["m2"], config["L1"], config["L2"], config["g"])

    steppers = {
        "rk4": rk4_step,
        "verlet": verlet_step,
    }
    try:
        stepper = steppers[method]
    except KeyError as exc:
        raise ValueError(f"Unsupported Hamiltonian integration method: {method}") from exc

    t, y = integrate_trajectory(
        stepper,
        double_pendulum_hamiltonian_rhs,
        state0,
        config["t_max"],
        config["dt"],
        *params,
    )

    theta1 = y[:, 0]
    p1 = y[:, 1]
    theta2 = y[:, 2]
    p2 = y[:, 3]
    omega1, omega2 = canonical_trajectory_to_angular_velocities(y, config["m1"], config["m2"], config["L1"], config["L2"])
    x1, y1, x2, y2 = hamiltonian_angles_to_cartesian(theta1, theta2, config["L1"], config["L2"])

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
