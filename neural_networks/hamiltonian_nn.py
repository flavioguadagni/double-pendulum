import torch
from torch import nn

from neural_networks.vector_field_nn import AngleEmbedding


class HamiltonianMLP(nn.Module):
    """
    Fully connected HNN that learns a scalar Hamiltonian.

    Input state:
      [theta1, p1, theta2, p2]
    Internal feature map:
      [sin(theta1), cos(theta1), p1, sin(theta2), cos(theta2), p2]
    Output:
      scalar H_hat
    """

    def __init__(self, hidden_dim=128, num_hidden_layers=3, activation_factory=nn.SiLU):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1.")

        self.embedding = AngleEmbedding()

        layers = []
        in_dim = 6
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_factory())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        features = self.embedding(x)
        return self.network(features)


def hamiltonian_vector_field(model, x):
    """
    Compute the canonical vector field induced by the learned Hamiltonian.

    The input x must be ordered as [theta1, p1, theta2, p2].
    The returned tensor is ordered as [dtheta1_dt, dp1_dt, dtheta2_dt, dp2_dt].
    """
    if not x.requires_grad:
        x = x.requires_grad_(True)

    hamiltonian = model(x).sum()
    grad = torch.autograd.grad(hamiltonian, x, create_graph=True)[0]

    dH_dtheta1 = grad[:, 0:1]
    dH_dp1 = grad[:, 1:2]
    dH_dtheta2 = grad[:, 2:3]
    dH_dp2 = grad[:, 3:4]

    return torch.cat(
        [
            dH_dp1,
            -dH_dtheta1,
            dH_dp2,
            -dH_dtheta2,
        ],
        dim=1,
    )


def build_hamiltonian_model(hidden_dim=128, num_hidden_layers=3, device="cpu", activation_factory=nn.SiLU):
    """Construct an HNN model and move it to the requested device."""
    return HamiltonianMLP(
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        activation_factory=activation_factory,
    ).to(device)


__all__ = [
    "HamiltonianMLP",
    "hamiltonian_vector_field",
    "build_hamiltonian_model",
]
