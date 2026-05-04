import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


def plot_solution(solution):
    """Plot the angles, optional canonical momenta, and the second-mass trajectory."""
    t = solution["t"]
    theta1 = solution["theta1"]
    theta2 = solution["theta2"]
    x2 = solution["x2"]
    y2 = solution["y2"]
    has_momenta = "p1" in solution and "p2" in solution

    if has_momenta:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(t, theta1, label=r"$\theta_1$")
    axes[0].plot(t, theta2, label=r"$\theta_2$")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel(r"Angle [$\mathrm{rad}$]")
    axes[0].set_title("Angular Motion")
    axes[0].legend()
    axes[0].grid(True)

    trajectory_axis = axes[1]

    if has_momenta:
        axes[1].plot(t, solution["p1"], label="p1")
        axes[1].plot(t, solution["p2"], label="p2")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Momentum")
        axes[1].set_title("Canonical Momenta")
        axes[1].legend()
        axes[1].grid(True)
        trajectory_axis = axes[2]

    trajectory_axis.plot(x2, y2, color="tab:red")
    trajectory_axis.set_xlabel("x [m]")
    trajectory_axis.set_ylabel("y [m]")
    trajectory_axis.set_title("Trajectory of Second Mass")
    trajectory_axis.axis("equal")
    trajectory_axis.grid(True)

    fig.tight_layout()
    plt.show()


def plot_comparison(solution_a, solution_b, label_a="Lagrangian", label_b="Hamiltonian"):
    """Overlay two solutions for direct comparison."""
    t_a = solution_a["t"]
    t_b = solution_b["t"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(t_a, solution_a["theta1"], label=rf"{label_a} $\theta_1$")
    axes[0].plot(t_a, solution_a["theta2"], label=rf"{label_a} $\theta_2$")
    axes[0].plot(t_b, solution_b["theta1"], "--", label=rf"{label_b} $\theta_1$")
    axes[0].plot(t_b, solution_b["theta2"], "--", label=rf"{label_b} $\theta_2$")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel(r"Angle [$\mathrm{rad}$]")
    axes[0].set_title("Angular Motion Comparison")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t_a, solution_a["omega1"], label=rf"{label_a} $\omega_1$")
    axes[1].plot(t_a, solution_a["omega2"], label=rf"{label_a} $\omega_2$")
    axes[1].plot(t_b, solution_b["omega1"], "--", label=rf"{label_b} $\omega_1$")
    axes[1].plot(t_b, solution_b["omega2"], "--", label=rf"{label_b} $\omega_2$")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel(r"Angular Velocity [$\mathrm{rad/s}$]")
    axes[1].set_title("Angular Velocity Comparison")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(solution_a["x2"], solution_a["y2"], label=label_a, color="tab:red")
    axes[2].plot(solution_b["x2"], solution_b["y2"], "--", label=label_b, color="tab:blue")
    axes[2].set_xlabel("x [m]")
    axes[2].set_ylabel("y [m]")
    axes[2].set_title("Second-Mass Trajectory Comparison")
    axes[2].axis("equal")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    plt.show()


def plot_comparison_error(solution_a, solution_b):
    """Plot absolute errors between two solutions over their common time grid."""
    t_a = solution_a["t"]
    t_b = solution_b["t"]

    if len(t_a) != len(t_b) or not np.allclose(t_a, t_b):
        raise ValueError("Comparison-error plots require matching time grids in both solution files.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    series = [
        (np.abs(solution_a["theta1"] - solution_b["theta1"]), r"$|\Delta \theta_1|$", "tab:blue"),
        (np.abs(solution_a["omega1"] - solution_b["omega1"]), r"$|\Delta \omega_1|$", "tab:green"),
        (np.abs(solution_a["theta2"] - solution_b["theta2"]), r"$|\Delta \theta_2|$", "tab:orange"),
        (np.abs(solution_a["omega2"] - solution_b["omega2"]), r"$|\Delta \omega_2|$", "tab:red"),
    ]

    for ax, (values, ylabel, color) in zip(axes, series):
        ax.plot(t_a, values, color=color)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True)

    axes[0].set_title(r"Absolute Error in $\theta_1$")
    axes[1].set_title(r"Absolute Error in $\omega_1$")
    axes[2].set_title(r"Absolute Error in $\theta_2$")
    axes[3].set_title(r"Absolute Error in $\omega_2$")

    fig.tight_layout()
    plt.show()


def plot_comparison_relative_error(solution_a, solution_b, eps=1e-12):
    """Plot relative errors between two solutions over their common time grid."""
    t_a = solution_a["t"]
    t_b = solution_b["t"]

    if len(t_a) != len(t_b) or not np.allclose(t_a, t_b):
        raise ValueError("Comparison-relative-error plots require matching time grids in both solution files.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    series = [
        (
            np.abs(solution_a["theta1"] - solution_b["theta1"]) / np.maximum(np.abs(solution_a["theta1"]), eps),
            r"$|\Delta \theta_1| / |\theta_1|$",
            "tab:blue",
        ),
        (
            np.abs(solution_a["omega1"] - solution_b["omega1"]) / np.maximum(np.abs(solution_a["omega1"]), eps),
            r"$|\Delta \omega_1| / |\omega_1|$",
            "tab:green",
        ),
        (
            np.abs(solution_a["theta2"] - solution_b["theta2"]) / np.maximum(np.abs(solution_a["theta2"]), eps),
            r"$|\Delta \theta_2| / |\theta_2|$",
            "tab:orange",
        ),
        (
            np.abs(solution_a["omega2"] - solution_b["omega2"]) / np.maximum(np.abs(solution_a["omega2"]), eps),
            r"$|\Delta \omega_2| / |\omega_2|$",
            "tab:red",
        ),
    ]

    for ax, (values, ylabel, color) in zip(axes, series):
        ax.plot(t_a, values, color=color)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True)

    axes[0].set_title(r"Relative Error in $\theta_1$")
    axes[1].set_title(r"Relative Error in $\omega_1$")
    axes[2].set_title(r"Relative Error in $\theta_2$")
    axes[3].set_title(r"Relative Error in $\omega_2$")

    fig.tight_layout()
    plt.show()


def plot_phase_space(solution, view="trajectory"):
    """Plot Hamiltonian trajectories either in phase space or versus time."""
    if "p1" not in solution or "p2" not in solution:
        raise ValueError("Phase-space plots require Hamiltonian data with p1 and p2.")

    if view == "trajectory":
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(solution["theta1"], solution["p1"], color="tab:blue")
        axes[0].set_xlabel(r"$\theta_1$ [$\mathrm{rad}$]")
        axes[0].set_ylabel("p1")
        axes[0].set_title(r"Phase Space: $\theta_1$ vs $p_1$")
        axes[0].grid(True)

        axes[1].plot(solution["theta2"], solution["p2"], color="tab:orange")
        axes[1].set_xlabel(r"$\theta_2$ [$\mathrm{rad}$]")
        axes[1].set_ylabel("p2")
        axes[1].set_title(r"Phase Space: $\theta_2$ vs $p_2$")
        axes[1].grid(True)
    elif view == "time-series":
        t = solution["t"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.ravel()
        series = [
            (r"$\theta_1$", r"$\theta_1$ [$\mathrm{rad}$]", "tab:blue", "theta1"),
            (r"$p_1$", r"$p_1$", "tab:cyan", "p1"),
            (r"$\theta_2$", r"$\theta_2$ [$\mathrm{rad}$]", "tab:orange", "theta2"),
            (r"$p_2$", r"$p_2$", "tab:red", "p2"),
        ]

        for ax, (name, ylabel, color, key) in zip(axes, series):
            ax.plot(t, solution[key], color=color)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{name} vs Time")
            ax.grid(True)
    else:
        raise ValueError(f"Unsupported phase-space view: {view}")

    fig.tight_layout()
    plt.show()


def _resolve_physical_parameters(solution, m1=None, m2=None, L1=None, L2=None, g=None):
    """Resolve physical parameters from explicit values or solution metadata."""
    defaults = {"m1": 1.0, "m2": 1.0, "L1": 1.0, "L2": 1.0, "g": 9.81}
    resolved = {}

    for key, value in {"m1": m1, "m2": m2, "L1": L1, "L2": L2, "g": g}.items():
        if value is not None:
            resolved[key] = value
        elif key in solution:
            resolved[key] = solution[key]
        else:
            resolved[key] = defaults[key]

    return resolved["m1"], resolved["m2"], resolved["L1"], resolved["L2"], resolved["g"]


def compute_total_energy(solution, m1=None, m2=None, L1=None, L2=None, g=None):
    """Compute the total mechanical energy from angular coordinates and velocities."""
    m1, m2, L1, L2, g = _resolve_physical_parameters(solution, m1=m1, m2=m2, L1=L1, L2=L2, g=g)
    theta1 = solution["theta1"]
    theta2 = solution["theta2"]
    omega1 = solution["omega1"]
    omega2 = solution["omega2"]
    delta = theta1 - theta2

    kinetic = (
        0.5 * (m1 + m2) * L1**2 * omega1**2
        + 0.5 * m2 * L2**2 * omega2**2
        + m2 * L1 * L2 * omega1 * omega2 * np.cos(delta)
    )
    potential = -(m1 + m2) * g * L1 * np.cos(theta1) - m2 * g * L2 * np.cos(theta2)
    return kinetic + potential


def plot_energy(
    solution,
    m1=None,
    m2=None,
    L1=None,
    L2=None,
    g=None,
):
    """Plot total mechanical energy versus time."""
    t = solution["t"]
    energy = compute_total_energy(solution, m1=m1, m2=m2, L1=L1, L2=L2, g=g)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, energy, color="tab:green", linewidth=1.5, label="Energy")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Energy [J]")
    ax.set_title("Total Mechanical Energy")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    plt.show()


def plot_energy_absolute(solution, m1=None, m2=None, L1=None, L2=None, g=None):
    """Plot total mechanical energy with full absolute values on the y-axis."""
    t = solution["t"]
    energy = compute_total_energy(solution, m1=m1, m2=m2, L1=L1, L2=L2, g=g)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, energy, color="tab:green", linewidth=1.5, label="Energy")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Energy [J]")
    ax.set_title("Total Mechanical Energy")
    ax.grid(True)
    ax.legend()

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    fig.tight_layout()
    plt.show()


def plot_relative_energy_error(solution, m1=None, m2=None, L1=None, L2=None, g=None):
    """Plot the relative energy deviation |E(t) - E(0)| / |E(0)|."""
    t = solution["t"]
    energy = compute_total_energy(solution, m1=m1, m2=m2, L1=L1, L2=L2, g=g)
    reference_energy = energy[0]

    if np.isclose(reference_energy, 0.0):
        raise ValueError("E(0) is too close to zero to compute a relative energy error.")

    relative_error = np.abs(energy - reference_energy) / np.abs(reference_energy)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, relative_error, color="tab:purple")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$|E(t)-E(0)| / |E(0)|$")
    ax.set_title("Relative Energy Error")
    ax.grid(True)

    fig.tight_layout()
    plt.show()
