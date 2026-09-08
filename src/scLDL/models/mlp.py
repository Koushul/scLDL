import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from scLDL.models._common import ModelBase, to_float_tensors


class MLPBaseline(ModelBase):
    """Standard MLP classifier used as an annotation baseline."""

    def __init__(
        self,
        n_features,
        n_outputs,
        n_hidden=256,
        dropout=0.2,
        lr=1e-3,
        epochs=50,
        batch_size=64,
        device=None,
        verbose=True,
    ):
        super().__init__(device=device, lr=lr, epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.n_outputs = n_outputs
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.BatchNorm1d(n_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden // 2, n_outputs),
        )
        self.to(self.device)

    def forward(self, x):
        return self.net(x)

    def fit(self, X, L):
        X_t, L_t = to_float_tensors(X, L, self.device)
        y = L_t.argmax(dim=1) if L_t.ndim == 2 else L_t.long()
        loader = DataLoader(TensorDataset(X_t, y.long()), batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        self.train()
        for epoch in range(self.epochs):
            total = 0.0
            n = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(self.forward(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                total += loss.item()
                n += 1
            self.history["loss"].append(total / max(n, 1))
            self._log_epoch(epoch)
        return self

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return F.softmax(self.forward(x), dim=1).cpu().numpy()
