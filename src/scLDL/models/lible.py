import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scLDL.models._common import (
    CNNEncoder,
    ModelBase,
    make_loader,
    mixup_batch,
    to_float_tensors,
)


class LIBLE(ModelBase):
    """Label Information Bottleneck enhancer. Predicts from X only."""

    def __init__(
        self,
        n_features,
        n_outputs,
        n_hidden=64,
        n_latent=64,
        alpha=1e-3,
        beta=1e-3,
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
        self.beta = beta
        self.encoder_type = encoder_type
        self.input_shape = input_shape
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        if encoder_type == "cnn":
            if input_shape is None:
                raise ValueError("input_shape is required for CNN encoder")
            self.encoder_cnn = CNNEncoder(input_shape, n_hidden, n_hidden, out_activation=nn.Tanh())
            hidden_in = n_hidden
        else:
            self.encoder_hidden = nn.Sequential(nn.Linear(n_features, n_hidden), nn.Tanh())
            hidden_in = n_hidden

        self.encoder_mean = nn.Linear(hidden_in, n_latent)
        self.encoder_logvar = nn.Linear(hidden_in, n_latent)
        self.decoder_L = nn.Sequential(nn.Linear(n_latent, n_hidden), nn.Tanh(), nn.Linear(n_hidden, n_outputs))
        self.decoder_D = nn.Sequential(nn.Linear(n_latent, n_hidden), nn.Tanh(), nn.Linear(n_hidden, n_outputs))
        self.decoder_g = nn.Sequential(
            nn.Linear(n_latent, n_hidden), nn.Tanh(), nn.Linear(n_hidden, 1), nn.Sigmoid()
        )
        self.to(self.device)

    def encode(self, x):
        if self.encoder_type == "cnn":
            h = self.encoder_cnn(x)
        else:
            h = self.encoder_hidden(x)
        return self.encoder_mean(h), self.encoder_logvar(h)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def forward(self, x, transform=False):
        mean, logvar = self.encode(x)
        if transform:
            return self.decoder_D(mean)
        z = self.reparameterize(mean, logvar)
        return mean, logvar, self.decoder_L(z), self.decoder_D(z), self.decoder_g(z)

    def _loss(self, batch_x, batch_l):
        mean, logvar, l_hat, d_hat, g = self.forward(batch_x)
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
        rec_l = torch.sum((batch_l - l_hat) ** 2, dim=1).mean()
        g_sq = g.pow(2) + 1e-6
        rec_d = torch.sum((batch_l - d_hat) ** 2 / g_sq + torch.log(g_sq), dim=1).mean()
        return rec_l + self.alpha * kl + self.beta * rec_d

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
                loss = self._loss(batch_x, batch_l)
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
        return F.softmax(self.forward(x, transform=True), dim=1).cpu().numpy()
