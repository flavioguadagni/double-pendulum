# Double Pendulum

Python project for simulating a double pendulum with both Lagrangian and Hamiltonian formulations, visualizing trajectories and energy behavior, and training neural surrogates for the canonical dynamics.

The repository currently includes:

- Deterministic numerical solvers for the double pendulum
- Plotting and animation utilities for trajectories, phase space, and energy
- Dataset generation from Hamiltonian trajectories
- Two neural models for learned dynamics:
  - a direct vector-field network
  - a Hamiltonian neural network (HNN)
- A Jupyter notebook that walks through the full workflow

## Repository Layout

```text
double-pendulum/
├── dynamics/
│   ├── config.py
│   ├── integrators.py
│   ├── lagrangian.py
│   ├── hamiltonian.py
│   └── solvers.py
├── neural_networks/
│   ├── dataset.py
│   ├── training.py
│   ├── rollout.py
│   ├── vector_field_nn.py
│   ├── hamiltonian_nn.py
│   ├── data/
│   └── checkpoints/
├── visualization/
├── simulation_input_card.json
└── test_notebook.ipynb
```

## Features

### Physics simulation

- Solve the double pendulum in Lagrangian coordinates with RK4
- Solve the double pendulum in canonical Hamiltonian coordinates with RK4 or Verlet
- Convert trajectories to Cartesian coordinates for plotting and animation

### Visualization

- Angular motion plots
- Canonical momenta plots
- Phase-space views
- Total energy and relative energy error plots
- Static comparisons between trajectories
- Local interactive animations

### Learning pipeline

- Sample random initial conditions
- Generate training data from Hamiltonian rollouts
- Standardize train/validation splits
- Train:
  - a direct canonical vector-field MLP
  - a Hamiltonian neural network that learns a scalar Hamiltonian
- Roll out learned models as continuous-time dynamical systems

## Requirements

The code is written for Python 3.11. Main dependencies are:

- `numpy`
- `matplotlib`
- `torch`
- `jupyter`
- `ipykernel`

## Setup

Create a virtual environment outside the project directory and install the dependencies:

```bash
cd ..
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy matplotlib torch jupyter ipykernel
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name double-pendulum --display-name "Python (.venv double-pendulum)"
```

Then launch Jupyter from the repository root:

```bash
cd double-pendulum
jupyter notebook
```

Open `test_notebook.ipynb` and select:

`Kernel` -> `Change kernel` -> `Python (.venv double-pendulum)`

## Quick Start

### 1. Run a simulation

```python
from dynamics import load_config, solve_lagrangian, solve_hamiltonian
from visualization import plot_solution, plot_phase_space, plot_energy

config = load_config("simulation_input_card.json")

solution_l = solve_lagrangian(config, method="rk4")
solution_h = solve_hamiltonian(config, method="rk4")

plot_solution(solution_l)
plot_solution(solution_h)
plot_phase_space(solution_h)
plot_energy(solution_h)
```

### 2. Generate a dataset

```python
from neural_networks.dataset import build_training_dataset, save_dataset

dataset = build_training_dataset(
    num_trajectories=200,
    t_max=10.0,
    dt=0.01,
    seed=42,
    val_fraction=0.2,
    m1=10.0,
    m2=5.0,
    L1=2.0,
    L2=1.0,
    g=9.81,
    omega_scale=2.0,
    as_torch=True,
)

save_dataset(dataset, "neural_networks/data/double_pendulum_dataset.pt")
```

### 3. Run animations locally

The project also provides Matplotlib-based animation helpers:

- `visualization.animate_solution(...)`
- `visualization.animate_phase_space(...)`

Example:

```python
from dynamics import load_config, solve_hamiltonian
from visualization import animate_solution, animate_phase_space

config = load_config("simulation_input_card.json")
solution = solve_hamiltonian(config, method="rk4")

animate_solution(solution, trail=200, show=True)
animate_phase_space(solution, trail=200, view="trajectory", show=True)
```

Important:

- These animations are intended for local interactive Python execution with a GUI-backed Matplotlib window.
- Treat them as a CLI/local-script feature.
- They do not work reliably inside Jupyter notebooks in the current project setup.
- Use the static plotting functions inside `test_notebook.ipynb` instead.

Typical way to run them from the command line:

```bash
python3.11
```

Then paste the example above into the Python session, or place it in a local script and run:

```bash
python3.11 your_script.py
```

### 4. Train a neural model

```python
from neural_networks.training import load_dataset, train_vector_field_model, train_hnn_model

dataset = load_dataset("neural_networks/data/double_pendulum_dataset.pt")

vector_model, vector_history = train_vector_field_model(
    dataset,
    epochs=200,
    checkpoint_path="neural_networks/checkpoints/vector_field.pt",
)

hnn_model, hnn_history = train_hnn_model(
    dataset,
    epochs=200,
    checkpoint_path="neural_networks/checkpoints/hamiltonian_nn.pt",
)
```

### 5. Roll out a trained checkpoint

```python
from dynamics import load_config
from neural_networks.rollout import rollout_vector_field_checkpoint, rollout_hnn_checkpoint
from visualization import plot_comparison

config = load_config("simulation_input_card.json")

vf_solution = rollout_vector_field_checkpoint(
    "neural_networks/checkpoints/vector_field.pt",
    config=config,
)

hnn_solution = rollout_hnn_checkpoint(
    "neural_networks/checkpoints/hamiltonian_nn.pt",
    config=config,
)

plot_comparison(vf_solution, hnn_solution, label_a="Vector Field", label_b="HNN")
```

## Configuration

Default simulation parameters live in `simulation_input_card.json`. The current example setup uses:

- `m1 = 10.0`: mass of the first pendulum bob
- `m2 = 5.0`: mass of the second pendulum bob
- `L1 = 2.0`: length of the first rod
- `L2 = 1.0`: length of the second rod
- `g = 9.81`: gravitational acceleration
- `theta1_0 = pi / 2`: initial angle of the first pendulum
- `theta2_0 = pi / 2 + 0.01`: initial angle of the second pendulum
- `omega1_0 = 0.0`: initial angular velocity of the first pendulum
- `omega2_0 = 0.0`: initial angular velocity of the second pendulum
- `t_max = 10.0`: total simulated physical time
- `dt = 0.01`: integration time step used by the numerical solver

You can either edit the JSON file or build a configuration in code with `dynamics.build_config(...)`.

## Notebook Workflow

The main walkthrough is `test_notebook.ipynb`. It covers:

1. Loading simulation parameters
2. Solving the system with the numerical physics models
3. Plotting the trajectories and phase space
4. Building and saving a training dataset
5. Training the vector-field network
6. Training the Hamiltonian neural network
7. Comparing learned rollouts against physics-based solutions

Animation is intentionally not part of the notebook workflow documented here. Use the animation helpers only from a local Python session or standalone script.

## Notes

- The neural models operate in canonical coordinates ordered as `[theta1, p1, theta2, p2]`.
- The direct vector-field model predicts `[dtheta1_dt, dp1_dt, dtheta2_dt, dp2_dt]`.
- The HNN predicts a scalar Hamiltonian and derives the dynamics through automatic differentiation.
- Checkpoints store both model weights and normalization statistics needed for rollout.

## Future Improvements

- Add a `requirements.txt` or `pyproject.toml`
- Add command-line entry points for dataset generation and training
- Add experiment tracking and result logging
