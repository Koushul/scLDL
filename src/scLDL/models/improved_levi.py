import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scLDL.models._common import (
    ResidualBlock,
    ModelBase,
    build_resnet_backbone,
    make_loader,
    mixup_batch,
    to_float_tensors,
)


class ImprovedLEVI(ModelBase):
    """Deeper LEVI variant. Encoder still conditions on labels; not for unlabeled annotation."""

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
        manifold_reg=0.0,
        verbose=True,
    ):
        super().__init__(device=device, lr=lr, epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.n_outputs = n_outputs
        self.n_latent = n_latent
        self.alpha = alpha
        self.gamma = gamma
        self.encoder_type = encoder_type
        self.input_shape = input_shape
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.manifold_reg = manifold_reg
        self.is_image = encoder_type == "resnet" and input_shape is not None and len(input_shape) >= 2

        if self.is_image:
            self.feature_extractor, self.feature_dim = build_resnet_backbone(input_shape)
        else:
            self.is_image = False
            self.feature_dim = 512
            self.feature_extractor = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                ResidualBlock(256),
                ResidualBlock(256),
                nn.Linear(256, self.feature_dim),
                nn.BatchNorm1d(self.feature_dim),
                nn.ReLU(),
            )

        self.encoder_fc = nn.Sequential(
            nn.Linear(self.feature_dim + n_outputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )
        self.enc_mean = nn.Linear(n_hidden, n_latent)
        self.enc_logvar = nn.Linear(n_hidden, n_latent)

        if self.is_image:
            self.decoder_fc = nn.Linear(n_latent, 64 * 7 * 7)
            self.decoder_cnn = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(),
            )
        else:
            self.decoder_x_1d = nn.Sequential(
                nn.Linear(n_latent, 256),
                nn.ReLU(),
                ResidualBlock(256),
                ResidualBlock(256),
                nn.Linear(256, n_features),
            )

        self.decoder_L = nn.Sequential(nn.Linear(n_latent, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_outputs))
        self.history["manifold_loss"] = []
        self.to(self.device)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def forward(self, x, l, transform=False):
        if self.is_image and x.dim() == 2:
            x_in = x.view(-1, *self.input_shape)
        else:
            x_in = x
        h_x = self.feature_extractor(x_in)
        h = self.encoder_fc(torch.cat((h_x, l), dim=1))
        mean = self.enc_mean(h)
        logvar = self.enc_logvar(h)
        if transform:
            return mean
        z = self.reparameterize(mean, logvar)
        if self.is_image:
            x_hat = self.decoder_cnn(self.decoder_fc(z).view(-1, 64, 7, 7)).view(x.size(0), -1)
        else:
            x_hat = self.decoder_x_1d(z)
        return mean, logvar, z, x_hat, self.decoder_L(z), h_x

    def _manifold_loss(self, features, pred_y):
        features_norm = F.normalize(features, p=2, dim=1)
        dist = 2 - 2 * torch.mm(features_norm, features_norm.t())
        s = torch.exp(-dist)
        p_norm = (pred_y ** 2).sum(1, keepdim=True)
        p_dist = p_norm + p_norm.t() - 2 * torch.mm(pred_y, pred_y.t())
        return torch.sum(s * p_dist) / (features.size(0) ** 2)

    def _step_loss(self, batch_x, batch_l):
        mean, logvar, _, x_hat, l_hat, h_x = self.forward(batch_x, batch_l)
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
        x_flat = batch_x.view(batch_x.size(0), -1)
        rec_x = F.mse_loss(x_hat, x_flat)
        rec_l = F.binary_cross_entropy_with_logits(l_hat, batch_l)
        loss = rec_x + self.gamma * rec_l + self.alpha * kl
        m_loss = torch.tensor(0.0, device=batch_x.device)
        if self.manifold_reg > 0:
            m_loss = self._manifold_loss(h_x.detach(), F.softmax(l_hat, dim=1))
            loss = loss + self.manifold_reg * m_loss
        return loss, m_loss

    def fit(self, X, L):
        X_t, L_t = to_float_tensors(X, L, self.device)
        loader = make_loader(X_t, L_t, self.batch_size)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        for epoch in range(self.epochs):
            total = 0.0
            total_m = 0.0
            n = 0
            for batch_x, batch_l in loader:
                if self.use_mixup:
                    batch_x, batch_l = mixup_batch(batch_x, batch_l, self.mixup_alpha)
                optimizer.zero_grad()
                loss, m_loss = self._step_loss(batch_x, batch_l)
                loss.backward()
                optimizer.step()
                total += loss.item()
                total_m += float(m_loss.item())
                n += 1
            self.history["loss"].append(total / max(n, 1))
            self.history["manifold_loss"].append(total_m / max(n, 1))
            extra = f", Manifold Loss: {self.history['manifold_loss'][-1]:.6f}"
            self._log_epoch(epoch, extra=extra)
        return self

    @torch.no_grad()
    def predict(self, X, L):
        self.eval()
        x, l = to_float_tensors(X, L, self.device)
        mean = self.forward(x, l, transform=True)
        return F.softmax(self.decoder_L(mean), dim=1).cpu().numpy()
