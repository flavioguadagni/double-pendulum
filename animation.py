import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


_ANIMATION = None


def animate_solution(solution, trail=200, show=True):
    """Animate the pendulum motion and optionally return the animation for notebooks."""
    x1 = solution["x1"]
    y1 = solution["y1"]
    x2 = solution["x2"]
    y2 = solution["y2"]
    t = solution["t"]

    reach = max(np.max(np.abs(x2)), np.max(np.abs(y2)), 1.0)
    limit = 1.1 * reach

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_title("Double Pendulum Animation")
    ax.grid(True)

    line, = ax.plot([], [], "o-", lw=2, color="tab:blue")
    trace, = ax.plot([], [], "-", lw=1, color="tab:red", alpha=0.7)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        line.set_data([], [])
        trace.set_data([], [])
        time_text.set_text("")
        return line, trace, time_text

    def update(frame):
        line.set_data([0.0, x1[frame], x2[frame]], [0.0, y1[frame], y2[frame]])

        start = max(0, frame - trail)
        trace.set_data(x2[start:frame + 1], y2[start:frame + 1])
        time_text.set_text(f"t = {t[frame]:.2f} s")
        return line, trace, time_text

    dt = t[1] - t[0] if len(t) > 1 else 0.01
    interval_ms = max(1, int(1000 * dt))

    global _ANIMATION
    _ANIMATION = FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        interval=interval_ms,
        blit=True,
        repeat=True,
    )

    if show:
        plt.show()

    return _ANIMATION


def animate_phase_space(solution, trail=200, view="trajectory", show=True):
    """Animate Hamiltonian trajectories and optionally return the animation for notebooks."""
    if "p1" not in solution or "p2" not in solution:
        raise ValueError("Phase-space animation requires Hamiltonian data with p1 and p2.")

    theta1 = solution["theta1"]
    theta2 = solution["theta2"]
    p1 = solution["p1"]
    p2 = solution["p2"]
    t = solution["t"]

    if view == "trajectory":
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

        theta1_pad = 0.05 * max(np.ptp(theta1), 1.0)
        p1_pad = 0.05 * max(np.ptp(p1), 1.0)
        theta2_pad = 0.05 * max(np.ptp(theta2), 1.0)
        p2_pad = 0.05 * max(np.ptp(p2), 1.0)

        axes[0].set_xlim(np.min(theta1) - theta1_pad, np.max(theta1) + theta1_pad)
        axes[0].set_ylim(np.min(p1) - p1_pad, np.max(p1) + p1_pad)
        axes[0].set_xlabel(r"$\theta_1$ [$\mathrm{rad}$]")
        axes[0].set_ylabel("p1")
        axes[0].set_title(r"Phase Space: $\theta_1$ vs $p_1$")
        axes[0].grid(True)

        axes[1].set_xlim(np.min(theta2) - theta2_pad, np.max(theta2) + theta2_pad)
        axes[1].set_ylim(np.min(p2) - p2_pad, np.max(p2) + p2_pad)
        axes[1].set_xlabel(r"$\theta_2$ [$\mathrm{rad}$]")
        axes[1].set_ylabel("p2")
        axes[1].set_title(r"Phase Space: $\theta_2$ vs $p_2$")
        axes[1].grid(True)

        line1, = axes[0].plot([], [], "-", lw=1.5, color="tab:blue")
        point1, = axes[0].plot([], [], "o", color="tab:blue")
        line2, = axes[1].plot([], [], "-", lw=1.5, color="tab:orange")
        point2, = axes[1].plot([], [], "o", color="tab:orange")
        time_text = axes[0].text(0.02, 0.95, "", transform=axes[0].transAxes)

        def init():
            line1.set_data([], [])
            point1.set_data([], [])
            line2.set_data([], [])
            point2.set_data([], [])
            time_text.set_text("")
            return line1, point1, line2, point2, time_text

        def update(frame):
            start = max(0, frame - trail)
            line1.set_data(theta1[start:frame + 1], p1[start:frame + 1])
            point1.set_data([theta1[frame]], [p1[frame]])
            line2.set_data(theta2[start:frame + 1], p2[start:frame + 1])
            point2.set_data([theta2[frame]], [p2[frame]])
            time_text.set_text(f"t = {t[frame]:.2f} s")
            return line1, point1, line2, point2, time_text
    elif view == "time-series":
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        axes = axes.ravel()
        series = [
            (r"$\theta_1$", theta1, r"$\theta_1$ [$\mathrm{rad}$]", "tab:blue"),
            (r"$p_1$", p1, r"$p_1$", "tab:cyan"),
            (r"$\theta_2$", theta2, r"$\theta_2$ [$\mathrm{rad}$]", "tab:orange"),
            (r"$p_2$", p2, r"$p_2$", "tab:red"),
        ]
        artists = []

        for ax, (name, values, ylabel, color) in zip(axes, series):
            value_pad = 0.05 * max(np.ptp(values), 1.0)
            ax.set_xlim(t[0], t[-1])
            ax.set_ylim(np.min(values) - value_pad, np.max(values) + value_pad)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{name} vs Time")
            ax.grid(True)

            line, = ax.plot([], [], "-", lw=1.5, color=color)
            point, = ax.plot([], [], "o", color=color)
            artists.append((line, point, values))

        time_text = axes[0].text(0.02, 0.95, "", transform=axes[0].transAxes)

        def init():
            for line, point, _ in artists:
                line.set_data([], [])
                point.set_data([], [])
            time_text.set_text("")
            return tuple(artist for pair in artists for artist in pair[:2]) + (time_text,)

        def update(frame):
            start = max(0, frame - trail)
            for line, point, values in artists:
                line.set_data(t[start:frame + 1], values[start:frame + 1])
                point.set_data([t[frame]], [values[frame]])
            time_text.set_text(f"t = {t[frame]:.2f} s")
            return tuple(artist for pair in artists for artist in pair[:2]) + (time_text,)
    else:
        raise ValueError(f"Unsupported phase-space view: {view}")

    dt = t[1] - t[0] if len(t) > 1 else 0.01
    interval_ms = max(1, int(1000 * dt))

    global _ANIMATION
    _ANIMATION = FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    if show:
        plt.show()

    return _ANIMATION
