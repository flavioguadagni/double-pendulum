import numpy as np


def rk4_step(f, y, dt, *args):
    """Advance one generic RK4 step."""
    k1 = f(y, *args)
    k2 = f(y + 0.5 * dt * k1, *args)
    k3 = f(y + 0.5 * dt * k2, *args)
    k4 = f(y + dt * k3, *args)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_trajectory(stepper, rhs, y0, t_max, dt, *args):
    """Integrate an ODE with a fixed-step method and return time and state arrays."""
    num_steps = int(np.floor(t_max / dt)) + 1
    t = np.linspace(0.0, dt * (num_steps - 1), num_steps)
    y = np.zeros((num_steps, len(y0)), dtype=float)
    y[0] = np.asarray(y0, dtype=float)

    for i in range(num_steps - 1):
        y[i + 1] = stepper(rhs, y[i], dt, *args)

    return t, y
