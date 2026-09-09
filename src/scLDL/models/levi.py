import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scLDL.models._common import ModelBase, make_loader, mixup_batch, to_float_tensors


class LEVI(ModelBase):
    """Label Enhancement via Variational Inference.

    The encoder is q(z | x, l). ``predict`` therefore requires logical labels and
    is a label-enhancement step, not annotation of unlabeled cells.
    """

    def __init__(
        self,
        n_features,
        n_outputs,
        n_hidden=64,
        alpha=1.0,
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
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.alpha = alpha
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
            self.encoder_fc = nn.Sequential(
                nn.Linear(flat + n_outputs, n_hidden),
                nn.Softplus(),
                nn.Linear(n_hidden, n_outputs * 2),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_features + n_outputs, n_hidden),
                nn.Softplus(),
                nn.Linear(n_hidden, n_outputs * 2),
            )

        self.decoder_X = nn.Sequential(nn.Linear(n_outputs, n_hidden), nn.Softplus(), nn.Linear(n_hidden, n_features))
        self.decoder_L = nn.Sequential(nn.Linear(n_outputs, n_hidden), nn.Softplus(), nn.Linear(n_hidden, n_outputs))
        self.to(self.device)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def _encode(self, x, l):
        if self.encoder_type == "cnn":
            if x.dim() == 2:
                x = x.view(-1, *self.input_shape)
            h = torch.cat((self.encoder_cnn(x), l), dim=1)
            latent = self.encoder_fc(h)
        else:
            latent = self.encoder(torch.cat((x, l), dim=1))
        return latent[:, : self.n_outputs], latent[:, self.n_outputs :]

    def forward(self, x, l, transform=False):
        mean, logvar = self._encode(x, l)
        if transform:
            return mean
        z = self.reparameterize(mean, logvar)
        return mean, logvar, z, self.decoder_X(z), self.decoder_L(z)

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
                mean, logvar, z, x_hat, l_hat = self.forward(batch_x, batch_l)
                main = torch.sum((batch_l - z) ** 2)
                kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                rec_x = torch.sum((batch_x.view(batch_x.size(0), -1) - x_hat) ** 2)
                rec_l = F.binary_cross_entropy_with_logits(l_hat, batch_l, reduction="sum")
                loss = (main + self.alpha * (kl + rec_x + rec_l)) / batch_x.shape[0]
                loss.backward()
                optimizer.step()
                total += loss.item()
                n += 1
            self.history["loss"].append(total / max(n, 1))
            self._log_epoch(epoch)
        return self

    @torch.no_grad()
    def predict(self, X, L):
        self.eval()
        x, l = to_float_tensors(X, L, self.device)
        mean = self.forward(x, l, transform=True)
        return F.softmax(mean, dim=1).cpu().numpy()
