import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scLDL.models._common import ModelBase, make_loader, mixup_batch, to_float_tensors
from scLDL.models.concentration import evidential_mse


class HybridLEVI(ModelBase):
    """VAE backbone with an evidential label head. Predicts from X only."""

    def __init__(
        self,
        n_features,
        n_outputs,
        n_hidden=64,
        n_latent=64,
        alpha=1.0,
        gamma=1.0,
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
        self.alpha = alpha
        self.gamma = gamma
        self.encoder_type = encoder_type
        self.input_shape = input_shape
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        if encoder_type == "cnn":
            if input_shape is None:
                raise ValueError("input_shape is required for CNN encoder")
            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
            )
            with torch.no_grad():
                flat = self.encoder_cnn(torch.zeros(1, *input_shape)).shape[1]
            self.encoder_fc = nn.Sequential(nn.Linear(flat, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden))
        else:
            self.encoder_fc = nn.Sequential(
                nn.Linear(n_features, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden)
            )

        self.enc_mean = nn.Linear(n_hidden, n_latent)
        self.enc_logvar = nn.Linear(n_hidden, n_latent)
        self.decoder_X = nn.Sequential(nn.Linear(n_latent, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_features))
        self.decoder_evidence = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs),
            nn.Softplus(),
        )
        self.to(self.device)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def forward(self, x):
        if self.encoder_type == "cnn":
            if x.dim() == 2:
                x = x.view(-1, *self.input_shape)
            h = self.encoder_fc(self.encoder_cnn(x))
        else:
            h = self.encoder_fc(x)
        mean = self.enc_mean(h)
        logvar = self.enc_logvar(h)
        z = self.reparameterize(mean, logvar)
        return mean, logvar, z, self.decoder_X(z), self.decoder_evidence(z)

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
                mean, logvar, _, x_hat, evidence = self.forward(batch_x)
                kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
                rec = F.mse_loss(x_hat, batch_x.view(batch_x.size(0), -1))
                cdl = evidential_mse(batch_l, evidence + 1)
                loss = rec + self.gamma * cdl + self.alpha * kl
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
        _, _, _, _, evidence = self.forward(x)
        alpha = evidence + 1
        s = torch.sum(alpha, dim=1, keepdim=True)
        mean = (alpha / s).cpu().numpy()
        uncertainty = (self.n_outputs / s).cpu().numpy().ravel()
        return mean, uncertainty

    def predict(self, X, L=None):
        mean, _ = self.predict_evidence(X)
        return mean
