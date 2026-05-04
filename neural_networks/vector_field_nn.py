import torch
from torch import nn


class AngleEmbedding(nn.Module):
    """Map [theta1, p1, theta2, p2] to [sin(theta1), cos(theta1), p1, sin(theta2), cos(theta2), p2]."""

    def forward(self, x):
        theta1 = x[:, 0:1]
        p1 = x[:, 1:2]
        theta2 = x[:, 2:3]
        p2 = x[:, 3:4]
        return torch.cat(
            [
                torch.sin(theta1),
                torch.cos(theta1),
                p1,
                torch.sin(theta2),
                torch.cos(theta2),
                p2,
            ],
            dim=1,
        )


class VectorFieldMLP(nn.Module):
    """
    Fully connected baseline that learns the canonical vector field directly.

    Input state:
      [theta1, p1, theta2, p2]
    Internal feature map:
      [sin(theta1), cos(theta1), p1, sin(theta2), cos(theta2), p2]
    Output:
      [dtheta1_dt, dp1_dt, dtheta2_dt, dp2_dt]
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
        layers.append(nn.Linear(in_dim, 4))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        features = self.embedding(x)
        return self.network(features)


def build_vector_field_model(hidden_dim=128, num_hidden_layers=3, device="cpu", activation_factory=nn.SiLU):
    """Construct a vector-field model and move it to the requested device."""
    return VectorFieldMLP(
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        activation_factory=activation_factory,
    ).to(device)


__all__ = [
    "AngleEmbedding",
    "VectorFieldMLP",
    "build_vector_field_model",
]
