import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from scLDL.models._common import ModelBase, mixup_batch, to_float_tensors


class _Res(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, d))

    def forward(self, x):
        return x + self.fc(self.ln(x))


class StateConcentrationLE(ModelBase):
    """Evidential model for mixed cell states.

    Trains on soft targets (graph + markers + clusters) with a lineage Laplacian
    penalty, neighbor consistency, class-balanced sampling, and a weaker Dirichlet
    prior so rare states can accumulate evidence.
    """

    def __init__(
        self,
        n_features,
        n_outputs,
        n_hidden=256,
        lr=1e-3,
        epochs=80,
        batch_size=128,
        device=None,
        prior=0.2,
        lineage_weight=0.4,
        manifold_weight=0.15,
        kl_weight=0.25,
        vacuity_weight=0.1,
        mixup_alpha=0.4,
        class_balance=True,
        lineage_edges=None,
        verbose=True,
    ):
        super().__init__(device=device, lr=lr, epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.n_outputs = n_outputs
        self.prior = prior
        self.lineage_weight = lineage_weight
        self.manifold_weight = manifold_weight
        self.kl_weight = kl_weight
        self.vacuity_weight = vacuity_weight
        self.mixup_alpha = mixup_alpha
        self.class_balance = class_balance
        lap = np.eye(n_outputs, dtype=np.float32)
        if lineage_edges:
            from scLDL.state_targets import lineage_laplacian

            lap = lineage_laplacian(lineage_edges, n_outputs)
        self.register_buffer("lineage_L", torch.as_tensor(lap, dtype=torch.float32))

        self.trunk = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.GELU(),
            _Res(n_hidden),
            _Res(n_hidden),
            nn.Dropout(0.1),
        )
        self.evidence_head = nn.Sequential(nn.Linear(n_hidden, n_outputs), nn.Softplus())
        self.to(self.device)

    def forward(self, x):
        h = self.trunk(x)
        evidence = self.evidence_head(h)
        alpha = evidence + self.prior
        return h, evidence, alpha

    def _kl_target_pred(self, target, mean):
        t = torch.clamp(target, 1e-6, 1.0)
        m = torch.clamp(mean, 1e-6, 1.0)
        t = t / t.sum(dim=1, keepdim=True)
        m = m / m.sum(dim=1, keepdim=True)
        return torch.sum(t * (torch.log(t) - torch.log(m)), dim=1).mean()

    def _lineage_residual(self, mean, target):
        diff = mean - target
        return torch.mean(torch.sum(diff * (diff @ self.lineage_L), dim=1))

    def _manifold(self, h, mean, neighbor_p=None):
        p = mean
        pn = (p**2).sum(1, keepdim=True)
        pdist = torch.clamp(pn + pn.t() - 2 * (p @ p.t()), min=0)
        if neighbor_p is not None:
            mass = neighbor_p.sum()
            return (neighbor_p * pdist).sum() / torch.clamp(mass, min=1e-8)
        hn = F.normalize(h, dim=1)
        sim = torch.clamp(hn @ hn.t(), 0, 1)
        n = h.size(0)
        return (sim * pdist).sum() / (n * n)

    def _vacuity(self, alpha, target):
        s = torch.sum(alpha, dim=1)
        u = (self.prior * self.n_outputs) / s
        t = torch.clamp(target, 1e-8, 1.0)
        t = t / t.sum(dim=1, keepdim=True)
        ent = -torch.sum(t * torch.log(t), dim=1) / np.log(self.n_outputs)
        return F.mse_loss(u, ent.detach())

    def _sample_weights(self, L):
        mass = L.sum(dim=0)
        inv = 1.0 / torch.clamp(mass, min=1.0)
        w = L @ inv
        return w * (L.size(0) / torch.clamp(w.sum(), min=1e-6))

    def fit(self, X, L, neighbor_p=None):
        X_t, L_t = to_float_tensors(X, L, self.device)
        L_t = torch.clamp(L_t, min=0)
        L_t = L_t / torch.clamp(L_t.sum(dim=1, keepdim=True), min=1e-6)
        idx_t = torch.arange(len(X_t), device=self.device)
        P = None
        if neighbor_p is not None:
            P = torch.as_tensor(neighbor_p, dtype=torch.float32, device=self.device)
        if self.class_balance:
            y_hard = L_t.argmax(dim=1).cpu().numpy()
            freq = np.bincount(y_hard, minlength=self.n_outputs).astype(np.float64)
            freq[freq == 0] = 1.0
            sw = 1.0 / freq[y_hard]
            sw = sw / sw.mean()
            sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
            loader = DataLoader(
                TensorDataset(idx_t, X_t, L_t),
                batch_size=self.batch_size,
                sampler=sampler,
                drop_last=len(X_t) > self.batch_size,
            )
        else:
            loader = DataLoader(
                TensorDataset(idx_t, X_t, L_t), batch_size=self.batch_size, shuffle=True
            )

        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        self.train()
        for epoch in range(self.epochs):
            total = 0.0
            n = 0
            for batch_i, batch_x, batch_l in loader:
                optimizer.zero_grad()
                mixed = bool(self.mixup_alpha and self.mixup_alpha > 0)
                man_term = None
                if self.manifold_weight and P is not None and mixed:
                    h0, _, alpha0 = self.forward(batch_x)
                    mean0 = alpha0 / torch.sum(alpha0, dim=1, keepdim=True)
                    man_term = self._manifold(h0.detach(), mean0, P[batch_i][:, batch_i])
                if mixed:
                    batch_x, batch_l = mixup_batch(batch_x, batch_l, self.mixup_alpha)
                h, _, alpha = self.forward(batch_x)
                mean = alpha / torch.sum(alpha, dim=1, keepdim=True)
                weights = self._sample_weights(batch_l)
                s = torch.sum(alpha, dim=1, keepdim=True)
                m = alpha / s
                sq = torch.sum((batch_l - m) ** 2, dim=1)
                var = torch.sum(alpha * (s - alpha) / (s * s * (s + 1)), dim=1)
                cdl = torch.mean((sq + var) * weights)
                loss = cdl
                if self.kl_weight:
                    loss = loss + self.kl_weight * self._kl_target_pred(batch_l, mean)
                if self.lineage_weight:
                    loss = loss + self.lineage_weight * self._lineage_residual(mean, batch_l)
                if self.manifold_weight:
                    if man_term is None:
                        sub_p = P[batch_i][:, batch_i] if P is not None else None
                        man_term = self._manifold(h.detach(), mean, sub_p)
                    loss = loss + self.manifold_weight * man_term
                if self.vacuity_weight:
                    loss = loss + self.vacuity_weight * self._vacuity(alpha, batch_l)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()
                total += float(loss.item())
                n += 1
            self.history["loss"].append(total / max(n, 1))
            self._log_epoch(epoch)
        return self

    @torch.no_grad()
    def predict_evidence(self, X):
        self.eval()
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        chunks = []
        uncs = []
        bs = 1024
        for i in range(0, len(x), bs):
            _, evidence, alpha = self.forward(x[i : i + bs])
            s = torch.sum(alpha, dim=1, keepdim=True)
            chunks.append((alpha / s).cpu().numpy())
            uncs.append(((self.prior * self.n_outputs) / s).cpu().numpy().ravel())
        return np.concatenate(chunks, axis=0), np.concatenate(uncs, axis=0)

    def predict(self, X):
        mean, _ = self.predict_evidence(X)
        return mean
