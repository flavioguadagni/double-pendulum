import numpy as np


def canonical_to_angular_velocities(state, m1, m2, L1, L2):
    """
    Convert a canonical state to angular velocities.

    State convention:
    [theta1, p1, theta2, p2]
    """
    theta1, p1, theta2, p2 = state
    delta = theta1 - theta2
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)
    mass_term = m1 + m2 * sin_delta**2

    omega1 = (L2 * p1 - L1 * p2 * cos_delta) / (L1**2 * L2 * mass_term)
    omega2 = ((m1 + m2) * L1 * p2 - m2 * L2 * p1 * cos_delta) /(m2 * L1 * L2**2 * mass_term)
    return omega1, omega2


def angular_velocities_to_canonical_state(theta1, omega1,
theta2, omega2, m1, m2, L1, L2):
    """
    Convert angles and angular velocities to canonical coordinates.

    Returns:
    [theta1, p1, theta2, p2]
    """
    delta = theta1 - theta2
    cos_delta = np.cos(delta)

    p1 = (m1 + m2) * L1**2 * omega1 + m2 * L1 * L2 * omega2 * cos_delta
    p2 = m2 * L2**2 * omega2 + m2 * L1 * L2 * omega1 * cos_delta
    return np.array([theta1, p1, theta2, p2], dtype=float)


def double_pendulum_hamiltonian_rhs(state, m1, m2, L1, L2, g):
    """
    Return the canonical equations of motion.

    State convention:
    [theta1, p1, theta2, p2]
    """
    theta1, _, theta2, _ = state
    delta = theta1 - theta2

    omega1, omega2 = canonical_to_angular_velocities(state, m1, m2, L1, L2)
    coupling = m2 * L1 * L2 * omega1 * omega2 * np.sin(delta)

    dtheta1_dt = omega1
    dp1_dt = -(m1 + m2) * g * L1 * np.sin(theta1) - coupling
    dtheta2_dt = omega2
    dp2_dt = -m2 * g * L2 * np.sin(theta2) + coupling

    return np.array([dtheta1_dt, dp1_dt, dtheta2_dt, dp2_dt], dtype=float)


def verlet_step(f, y, dt, *args):
    """
    Advance one kick-drift-kick Verlet-style step in canonical coordinates.

    This is exact symplectic Verlet only for separable Hamiltonians H(q, p) = T(p) + V(q).
    For the double pendulum Hamiltonian, where T depends on q as well, it remains a useful
    explicit leapfrog-style integrator but is not guaranteed to be exactly symplectic.
    """
    rhs = f(y, *args)

    theta = y[[0, 2]]
    momentum = y[[1, 3]]
    momentum_half = momentum + 0.5 * dt * rhs[[1, 3]]

    state_half = np.array([theta[0], momentum_half[0], theta[1], momentum_half[1]], dtype=float)
    rhs_half = f(state_half, *args)
    theta_new = theta + dt * rhs_half[[0, 2]]

    state_new_half = np.array(
        [theta_new[0], momentum_half[0], theta_new[1], momentum_half[1]],
        dtype=float,
    )
    rhs_new_half = f(state_new_half, *args)
    momentum_new = momentum_half + 0.5 * dt * rhs_new_half[[1, 3]]

    return np.array([theta_new[0], momentum_new[0], theta_new[1], momentum_new[1]], dtype=float)


def canonical_trajectory_to_angular_velocities(traj, m1, m2, L1,
L2):
    """Convert a full canonical trajectory to omega1(t),
omega2(t)."""
    omegas = np.array(
        [canonical_to_angular_velocities(state, m1, m2, L1, L2) for state in traj],
        dtype=float,
    )
    return omegas[:, 0], omegas[:, 1]


def angles_to_cartesian(theta1, theta2, L1, L2):
    """Convert angular coordinates to Cartesian coordinates."""
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2


def compute_hamiltonian_energy(state, m1, m2, L1, L2, g):
    """
    Compute the total mechanical energy from a canonical state.

    [theta1, p1, theta2, p2]
    """
    theta1, _, theta2, _ = state
    omega1, omega2 = canonical_to_angular_velocities(state, m1, m2, L1, L2)
    delta = theta1 - theta2

    kinetic = (
        0.5 * (m1 + m2) * L1**2 * omega1**2
        + 0.5 * m2 * L2**2 * omega2**2
        + m2 * L1 * L2 * omega1 * omega2 * np.cos(delta)
    )

    potential = (
        -(m1 + m2) * g * L1 * np.cos(theta1)
        - m2 * g * L2 * np.cos(theta2)
    )

    return kinetic + potential
