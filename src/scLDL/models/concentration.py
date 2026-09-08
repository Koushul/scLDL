import torch
import torch.nn as nn
import torch.optim as optim

from scLDL.models._common import (
    CNNEncoder,
    ModelBase,
    build_resnet_backbone,
    make_loader,
    mixup_batch,
    to_float_tensors,
)


def evidential_mse(y, alpha):
    s = torch.sum(alpha, dim=1, keepdim=True)
    mean = alpha / s
    sq_err = torch.sum((y - mean) ** 2, dim=1, keepdim=True)
    var = torch.sum(alpha * (s - alpha) / (s * s * (s + 1)), dim=1, keepdim=True)
    return torch.mean(sq_err + var)


class ConcentrationLE(ModelBase):
    """Dirichlet / evidential label model. Predicts from X only."""

    def __init__(
        self,
        n_features,
        n_outputs,
        n_hidden=64,
        lr=1e-3,
        epochs=100,
        batch_size=32,
        device=None,
        encoder_type="mlp",
        input_shape=None,
        use_mixup=False,
        mixup_alpha=1.0,
        verbose=True,
    ):
        super().__init__(device=device, lr=lr, epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.n_outputs = n_outputs
        self.encoder_type = encoder_type
        self.input_shape = input_shape
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.is_resnet = encoder_type == "resnet" and input_shape is not None and len(input_shape) >= 2

        if self.is_resnet:
            self.feature_extractor, feature_dim = build_resnet_backbone(input_shape)
            self.encoder_head = nn.Sequential(
                nn.Linear(feature_dim, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_outputs),
                nn.Softplus(),
            )
        elif encoder_type == "cnn":
            if input_shape is None:
                raise ValueError("input_shape is required for CNN encoder")
            self.encoder_cnn = CNNEncoder(input_shape, n_hidden, n_outputs, out_activation=nn.Softplus())
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_features, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_outputs),
                nn.Softplus(),
            )
        self.to(self.device)

    def forward(self, x):
        if self.is_resnet:
            if x.dim() == 2:
                x = x.view(-1, *self.input_shape)
            evidence = self.encoder_head(self.feature_extractor(x))
        elif self.encoder_type == "cnn":
            evidence = self.encoder_cnn(x)
        else:
            evidence = self.encoder(x)
        return evidence, evidence + 1

    def fit(self, X, L):
        X_t, L_t = to_float_tensors(X, L, self.device)
        loader = make_loader(X_t, L_t, self.batch_size)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        for epoch in range(self.epochs):
            total = 0.0
            n = 0
            for batch_x, batch_l in loader:
                if self.use_mixup:
                    batch_x, batch_l = mixup_batch(batch_x, batch_l, self.mixup_alpha)
                optimizer.zero_grad()
                _, alpha = self.forward(batch_x)
                loss = evidential_mse(batch_l, alpha)
                loss.backward()
                optimizer.step()
                total += loss.item()
                n += 1
            self.history["loss"].append(total / max(n, 1))
            self._log_epoch(epoch)
        return self

    @torch.no_grad()
    def predict_evidence(self, X):
        self.eval()
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        evidence, alpha = self.forward(x)
        s = torch.sum(alpha, dim=1, keepdim=True)
        mean = (alpha / s).cpu().numpy()
        uncertainty = (self.n_outputs / s).cpu().numpy().ravel()
        return mean, uncertainty

    def predict(self, X):
        mean, _ = self.predict_evidence(X)
        return mean
