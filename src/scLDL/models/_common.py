import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scLDL.device import resolve_device


def mixup_batch(x, y, alpha: float):
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[index]
    y_mix = lam * y + (1.0 - lam) * y[index]
    return x_mix, y_mix


def to_float_tensors(X, L, device):
    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    l = torch.as_tensor(L, dtype=torch.float32, device=device)
    return x, l


def make_loader(X, L, batch_size: int, shuffle: bool = True):
    dataset = TensorDataset(X, L)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class ResidualBlock(nn.Module):
    def __init__(self, n_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.bn2 = nn.BatchNorm1d(n_hidden)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out = self.relu(out + residual)
        return out


class CNNEncoder(nn.Module):
    def __init__(self, input_shape, n_hidden: int, n_out: int, out_activation=None):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_size = self.cnn(torch.zeros(1, *self.input_shape)).shape[1]
        layers = [nn.Linear(flat_size, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_out)]
        if out_activation is not None:
            layers.append(out_activation)
        self.head = nn.Sequential(*layers)

    def reshape(self, x):
        if x.dim() == 2:
            return x.view(-1, *self.input_shape)
        return x

    def forward(self, x):
        return self.head(self.cnn(self.reshape(x)))


def build_resnet_backbone(input_shape):
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    net = resnet18(weights=weights)
    if input_shape[0] == 1:
        original = net.conv1
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            net.conv1.weight.copy_(original.weight.sum(dim=1, keepdim=True))
    feature_dim = net.fc.in_features
    net.fc = nn.Identity()
    return net, feature_dim


class ModelBase(nn.Module):
    def __init__(self, device=None, lr=1e-3, epochs=100, batch_size=32, verbose=True):
        super().__init__()
        self.device = resolve_device(device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.history = {"loss": []}

    def _log_epoch(self, epoch, extra=""):
        if self.verbose and (epoch + 1) % 10 == 0:
            loss = self.history["loss"][-1] if self.history["loss"] else float("nan")
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}{extra}")
