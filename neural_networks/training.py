from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from neural_networks.dataset import load_dataset as load_saved_dataset
from neural_networks.hamiltonian_nn import build_hamiltonian_model, hamiltonian_vector_field
from neural_networks.vector_field_nn import build_vector_field_model


def load_dataset(path, device="cpu"):
    """Load a serialized dataset dictionary and validate the required keys."""
    dataset = load_saved_dataset(path, device=device)
    required_keys = (
        "X_train",
        "y_train",
        "X_val",
        "y_val",
        "X_train_raw",
        "y_train_raw",
        "X_val_raw",
        "y_val_raw",
        "x_mean",
        "x_std",
        "y_mean",
        "y_std",
    )
    missing = [key for key in required_keys if key not in dataset]
    if missing:
        raise KeyError(f"Dataset is missing required keys: {missing}")
    return dataset


def create_vector_field_dataloaders(dataset, batch_size):
    """Create train/validation loaders for the direct vector-field model."""
    train_data = TensorDataset(dataset["X_train"].float(), dataset["y_train"].float())
    val_data = TensorDataset(dataset["X_val"].float(), dataset["y_val"].float())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def create_hnn_dataloaders(dataset, batch_size):
    """Create train/validation loaders for HNN training on raw canonical states."""
    train_data = TensorDataset(dataset["X_train_raw"].float(), dataset["y_train_raw"].float())
    val_data = TensorDataset(dataset["X_val_raw"].float(), dataset["y_val_raw"].float())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_optimizer(model, lr=1e-3, weight_decay=1e-5):
    """Construct the default optimizer used by both neural-network variants."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_loss_fn():
    """Construct the default supervised loss for the direct vector-field model."""
    return nn.MSELoss()


def run_vector_field_epoch(model, loader, optimizer, loss_fn, device):
    """Run one training epoch for the direct vector-field model."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(x_batch)
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = x_batch.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_vector_field(model, loader, loss_fn, device):
    """Evaluate the direct vector-field model on validation data."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        predictions = model(x_batch)
        loss = loss_fn(predictions, y_batch)

        batch_size = x_batch.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def hnn_loss(pred_raw, target_raw, y_mean, y_std):
    """Compute MSE on standardized derivatives for HNN training."""
    pred_std = (pred_raw - y_mean) / y_std
    target_std = (target_raw - y_mean) / y_std
    return nn.functional.mse_loss(pred_std, target_std)


def run_hnn_epoch(model, loader, optimizer, device, y_mean, y_std):
    """Run one training epoch for the Hamiltonian neural network."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device).requires_grad_(True)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = hamiltonian_vector_field(model, x_batch)
        loss = hnn_loss(predictions, y_batch, y_mean, y_std)
        loss.backward()
        optimizer.step()

        batch_size = x_batch.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def evaluate_hnn(model, loader, device, y_mean, y_std):
    """Evaluate HNN derivative loss on validation data."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device).requires_grad_(True)
        y_batch = y_batch.to(device)
        predictions = hamiltonian_vector_field(model, x_batch)
        loss = hnn_loss(predictions, y_batch, y_mean, y_std)

        batch_size = x_batch.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def save_checkpoint(path, model, dataset, hidden_dim, num_hidden_layers, best_val_loss, model_type):
    """Save model weights, normalization statistics, and training configuration."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "x_mean": dataset["x_mean"].float(),
        "x_std": dataset["x_std"].float(),
        "y_mean": dataset["y_mean"].float(),
        "y_std": dataset["y_std"].float(),
        "feature_order": dataset.get("feature_order"),
        "target_order": dataset.get("target_order"),
        "hidden_dim": hidden_dim,
        "num_hidden_layers": num_hidden_layers,
        "best_val_loss": best_val_loss,
        "model_type": model_type,
    }
    torch.save(checkpoint, path)


def train_vector_field_model(
    dataset,
    epochs=200,
    batch_size=512,
    lr=1e-3,
    weight_decay=1e-5,
    hidden_dim=128,
    num_hidden_layers=3,
    device="cpu",
    checkpoint_path=None,
    verbose=True,
):
    """Train the direct vector-field baseline and return the model plus history."""
    device = torch.device(device)
    train_loader, val_loader = create_vector_field_dataloaders(dataset, batch_size=batch_size)
    model = build_vector_field_model(
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        device=device,
    )
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    loss_fn = build_loss_fn()

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Training direct vector-field neural network.")
        print(f"Device: {device}")
        print(f"Model:  {num_hidden_layers} hidden layers, width {hidden_dim}")

    for epoch in range(1, epochs + 1):
        train_loss = run_vector_field_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate_vector_field(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    dataset,
                    hidden_dim=hidden_dim,
                    num_hidden_layers=num_hidden_layers,
                    best_val_loss=best_val_loss,
                    model_type="vector_field",
                )

        if verbose:
            print(
                f"Epoch {epoch:4d}/{epochs} "
                f"train_loss={train_loss:.6e} "
                f"val_loss={val_loss:.6e} "
                f"best_val_loss={best_val_loss:.6e}"
            )

    return model, history


def train_hnn_model(
    dataset,
    epochs=200,
    batch_size=512,
    lr=1e-3,
    weight_decay=1e-5,
    hidden_dim=128,
    num_hidden_layers=3,
    device="cpu",
    checkpoint_path=None,
    verbose=True,
):
    """Train the Hamiltonian neural network and return the model plus history."""
    device = torch.device(device)
    train_loader, val_loader = create_hnn_dataloaders(dataset, batch_size=batch_size)
    model = build_hamiltonian_model(
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        device=device,
    )
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    y_mean = dataset["y_mean"].float().to(device)
    y_std = dataset["y_std"].float().to(device)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Training Hamiltonian neural network.")
        print(f"Device: {device}")
        print(f"Model:  {num_hidden_layers} hidden layers, width {hidden_dim}")

    for epoch in range(1, epochs + 1):
        train_loss = run_hnn_epoch(model, train_loader, optimizer, device, y_mean, y_std)
        val_loss = evaluate_hnn(model, val_loader, device, y_mean, y_std)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    dataset,
                    hidden_dim=hidden_dim,
                    num_hidden_layers=num_hidden_layers,
                    best_val_loss=best_val_loss,
                    model_type="hnn",
                )

        if verbose:
            print(
                f"Epoch {epoch:4d}/{epochs} "
                f"train_loss={train_loss:.6e} "
                f"val_loss={val_loss:.6e} "
                f"best_val_loss={best_val_loss:.6e}"
            )

    return model, history


__all__ = [
    "build_loss_fn",
    "build_optimizer",
    "create_hnn_dataloaders",
    "create_vector_field_dataloaders",
    "evaluate_hnn",
    "evaluate_vector_field",
    "hnn_loss",
    "load_dataset",
    "run_hnn_epoch",
    "run_vector_field_epoch",
    "save_checkpoint",
    "train_hnn_model",
    "train_vector_field_model",
]
