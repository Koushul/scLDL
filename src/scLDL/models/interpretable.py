from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from scLDL.models._common import mixup_batch, to_float_tensors
from scLDL.models.concentration_state import StateConcentrationLE


class InterpretableLE(StateConcentrationLE):
    """State-aware evidential model with a marker concept bottleneck.

    Evidence is MLP(z) plus a linear map of class-aligned marker scores so
    P(state) can be read against programs. Optional penalties suppress
    lineage-illegal pairs and keep cell-type predictions peaked.
    """

    def __init__(
        self,
        n_features,
        n_outputs,
        n_concepts=0,
        illegal_weight=0.0,
        marker_kl_weight=0.0,
        peak_weight=0.0,
        **kwargs,
    ):
        super().__init__(n_features=n_features, n_outputs=n_outputs, **kwargs)
        self.n_concepts = int(n_concepts)
        self.illegal_weight = illegal_weight
        self.marker_kl_weight = marker_kl_weight
        self.peak_weight = peak_weight
        if self.n_concepts > 0:
            self.concept_map = nn.Linear(self.n_concepts, n_outputs, bias=False)
            nn.init.zeros_(self.concept_map.weight)
            k = min(self.n_concepts, n_outputs)
            with torch.no_grad():
                self.concept_map.weight[:k, :k] = 0.5 * torch.eye(k, device=self.concept_map.weight.device)
        else:
            self.concept_map = None
        adj = torch.clamp(-self.lineage_L, min=0)
        adj.fill_diagonal_(0)
        allowed = adj + torch.eye(n_outputs)
        self.register_buffer("illegal_mask", (1.0 - torch.clamp(allowed, 0, 1)))
        self.to(self.device)

    def forward(self, x, concepts=None):
        h = self.trunk(x)
        evidence = self.evidence_head(h)
        if self.concept_map is not None and concepts is not None:
            evidence = evidence + F.softplus(self.concept_map(concepts))
        alpha = evidence + self.prior
        return h, evidence, alpha

    def _illegal(self, mean):
        pair = mean.unsqueeze(2) * mean.unsqueeze(1)
        return torch.mean((pair * self.illegal_mask).sum(dim=(1, 2)))

    def _peak(self, mean):
        p = torch.clamp(mean, 1e-8, 1.0)
        return -torch.mean(torch.sum(p * torch.log(p), dim=1))

    def fit(self, X, L, neighbor_p=None, concepts=None):
        X_t, L_t = to_float_tensors(X, L, self.device)
        L_t = torch.clamp(L_t, min=0)
        L_t = L_t / torch.clamp(L_t.sum(dim=1, keepdim=True), min=1e-6)
        idx_t = torch.arange(len(X_t), device=self.device)
        if concepts is None:
            C_t = torch.zeros((len(X_t), max(self.n_concepts, 1)), device=self.device)
        else:
            C_t = torch.as_tensor(concepts, dtype=torch.float32, device=self.device)
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
                TensorDataset(idx_t, X_t, L_t, C_t),
                batch_size=self.batch_size,
                sampler=sampler,
                drop_last=len(X_t) > self.batch_size,
            )
        else:
            loader = DataLoader(
                TensorDataset(idx_t, X_t, L_t, C_t), batch_size=self.batch_size, shuffle=True
            )

        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        use_concepts = self.concept_map is not None and concepts is not None
        self.train()
        for epoch in range(self.epochs):
            total = 0.0
            n = 0
            for batch_i, batch_x, batch_l, batch_c in loader:
                optimizer.zero_grad()
                mixed = bool(self.mixup_alpha and self.mixup_alpha > 0)
                c_fwd = batch_c if use_concepts else None
                man_term = None
                if self.manifold_weight and P is not None and mixed:
                    h0, _, alpha0 = self.forward(batch_x, c_fwd)
                    mean0 = alpha0 / torch.sum(alpha0, dim=1, keepdim=True)
                    man_term = self._manifold(h0.detach(), mean0, P[batch_i][:, batch_i])
                if mixed:
                    batch_x, batch_l = mixup_batch(batch_x, batch_l, self.mixup_alpha)
                h, _, alpha = self.forward(batch_x, c_fwd)
                mean = alpha / torch.sum(alpha, dim=1, keepdim=True)
                weights = self._sample_weights(batch_l)
                s = torch.sum(alpha, dim=1, keepdim=True)
                m = alpha / s
                sq = torch.sum((batch_l - m) ** 2, dim=1)
                var = torch.sum(alpha * (s - alpha) / (s * s * (s + 1)), dim=1)
                loss = torch.mean((sq + var) * weights)
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
                if self.illegal_weight:
                    loss = loss + self.illegal_weight * self._illegal(mean)
                if self.peak_weight:
                    loss = loss + self.peak_weight * self._peak(mean)
                if self.marker_kl_weight and use_concepts and not mixed:
                    loss = loss + self.marker_kl_weight * self._kl_target_pred(batch_c, mean)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()
                total += float(loss.item())
                n += 1
            self.history["loss"].append(total / max(n, 1))
            self._log_epoch(epoch)
        return self

    @torch.no_grad()
    def predict_evidence(self, X, concepts=None):
        self.eval()
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        c = None
        if self.concept_map is not None and concepts is not None:
            c = torch.as_tensor(concepts, dtype=torch.float32, device=self.device)
        chunks = []
        uncs = []
        alphas = []
        bs = 1024
        for i in range(0, len(x), bs):
            cb = None if c is None else c[i : i + bs]
            _, evidence, alpha = self.forward(x[i : i + bs], cb)
            s = torch.sum(alpha, dim=1, keepdim=True)
            chunks.append((alpha / s).cpu().numpy())
            uncs.append(((self.prior * self.n_outputs) / s).cpu().numpy().ravel())
            alphas.append(alpha.cpu().numpy())
        return (
            np.concatenate(chunks, axis=0),
            np.concatenate(uncs, axis=0),
            np.concatenate(alphas, axis=0),
        )

    def predict(self, X, concepts=None):
        mean, _, _ = self.predict_evidence(X, concepts=concepts)
        return mean
