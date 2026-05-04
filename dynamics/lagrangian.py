import numpy as np


def double_pendulum_lagrangian_rhs(state, m1, m2, L1, L2, g):
    """
    Return the time derivative of the Lagrangian state.

    State convention:
    [theta1, omega1, theta2, omega2]
    """
    theta1, omega1, theta2, omega2 = state

    delta = theta1 - theta2
    den = 2.0 * m1 + m2 - m2 * np.cos(2.0 * delta)

    dtheta1_dt = omega1
    dtheta2_dt = omega2

    domega1_dt = (
        -g * (2.0 * m1 + m2) * np.sin(theta1)
        - m2 * g * np.sin(theta1 - 2.0 * theta2)
        - 2.0
        * m2
        * np.sin(delta)
        * (L2 * omega2**2 + L1 * omega1**2 * np.cos(delta))
    ) / (L1 * den)

    domega2_dt = (
        2.0
        * np.sin(delta)
        * (
            L1 * omega1**2 * (m1 + m2)
            + g * (m1 + m2) * np.cos(theta1)
            + L2 * omega2**2 * m2 * np.cos(delta)
        )
    ) / (L2 * den)

    return np.array([dtheta1_dt, domega1_dt, dtheta2_dt,
domega2_dt], dtype=float)


def angles_to_cartesian(theta1, theta2, L1, L2):
    """Convert angular coordinates to Cartesian coordinates."""
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2


def compute_lagrangian_energy(state, m1, m2, L1, L2, g):
    """
    Compute total mechanical energy for a Lagrangian state.

    State convention:
    [theta1, omega1, theta2, omega2]
    """
    theta1, omega1, theta2, omega2 = state
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