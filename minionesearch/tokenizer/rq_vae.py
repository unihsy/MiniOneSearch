"""PyTorch GPU/CPU RQ-VAE & Sinkhorn-Knopp Balanced RQ-VAE for Semantic ID Construction.

Supports clean controlled ablation of:
  1. `Vanilla RQ-VAE`: random uniform codebook init + standard nearest-neighbor STE
     (often suffers severe codebook collapse in deeper residual layers).
  2. `Sinkhorn RQ-VAE`: layer-wise K-Means++ initialization + Sinkhorn-Knopp optimal
     transport uniform assignment regularization + dead-code revival.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from minionesearch.utils import set_global_seed


@dataclass
class RQVAEConfig:
    codebook_sizes: Tuple[int, ...] = (256, 256, 256, 256, 256)
    latent_dim: int = 128
    hidden_dims: Tuple[int, ...] = (512, 256)
    commitment_weight: float = 0.25
    use_sinkhorn: bool = True
    sinkhorn_eps: float = 0.05
    sinkhorn_iters: int = 5
    kmeans_init: bool = True
    dead_code_threshold: int = 2
    lr: float = 1e-3
    batch_size: int = 4096
    epochs: int = 40
    seed: int = 42


def sinkhorn_knopp_probs(distances: torch.Tensor, eps: float = 0.05, n_iters: int = 5) -> torch.Tensor:
    """Compute doubly-stochastic Sinkhorn-Knopp assignment matrix `[B, K]` from squared distances."""
    # Stabilize scores before exp
    scores = -distances / max(eps, 1e-4)
    scores = scores - scores.max(dim=-1, keepdim=True).values
    q = torch.exp(scores).t()  # [K, B]
    q = q / q.sum().clamp(min=1e-12)
    k_cnt, b_cnt = q.shape
    for _ in range(n_iters):
        q = q / q.sum(dim=1, keepdim=True).clamp(min=1e-12) / float(k_cnt)
        q = q / q.sum(dim=0, keepdim=True).clamp(min=1e-12) / float(b_cnt)
    q = q * float(b_cnt)  # columns sum to 1
    return q.t()  # [B, K]


def _kmeans_plusplus_init_1d_layer(x: torch.Tensor, k: int, rng: torch.Generator) -> torch.Tensor:
    """Fast GPU/CPU K-Means++ centroid seeding on `[N, D]` tensor."""
    n, d = x.shape
    if n <= k:
        reps = (k + n - 1) // max(1, n)
        out = x.repeat(reps, 1)[:k].clone()
        out += 1e-4 * torch.randn(out.shape, device=x.device, dtype=x.dtype, generator=rng)
        return out

    centroids = torch.empty((k, d), device=x.device, dtype=x.dtype)
    first = int(torch.randint(0, n, (1,), generator=rng, device=x.device).item())
    centroids[0] = x[first]
    min_sq_dists = ((x - centroids[0].unsqueeze(0)) ** 2).sum(dim=-1)

    for c_idx in range(1, k):
        probs = min_sq_dists.clamp(min=1e-12)
        probs = probs / probs.sum()
        chosen = int(torch.multinomial(probs, 1, generator=rng).item())
        centroids[c_idx] = x[chosen]
        sq_d = ((x - centroids[c_idx].unsqueeze(0)) ** 2).sum(dim=-1)
        min_sq_dists = torch.minimum(min_sq_dists, sq_d)

    # 5 fast Lloyd refinement steps
    for _ in range(5):
        dists = torch.cdist(x, centroids, p=2.0)
        assign = dists.argmin(dim=-1)
        one_hot = F.one_hot(assign, num_classes=k).to(x.dtype)
        counts = one_hot.sum(dim=0).unsqueeze(-1)
        sums = one_hot.t() @ x
        centroids = torch.where(counts > 0, sums / counts.clamp(min=1.0), centroids)
    return centroids


class ResidualVQ(nn.Module):
    """Multi-layer Residual Vector Quantizer with optional K-Means++ init & Sinkhorn balancing."""

    def __init__(self, codebook_sizes: Sequence[int], dim: int, cfg: RQVAEConfig):
        super().__init__()
        self.codebook_sizes = tuple(codebook_sizes)
        self.dim = dim
        self.cfg = cfg
        self.codebooks = nn.ModuleList([nn.Embedding(k, dim) for k in self.codebook_sizes])
        for cb in self.codebooks:
            nn.init.uniform_(cb.weight, -1.0 / dim, 1.0 / dim)
        self.register_buffer("_initialized", torch.tensor(not cfg.kmeans_init, dtype=torch.bool))

    @torch.no_grad()
    def _init_from_batch(self, z: torch.Tensor) -> None:
        rng = torch.Generator(device=z.device)
        rng.manual_seed(self.cfg.seed)
        residual = z.detach().float()
        for cb in self.codebooks:
            k = cb.num_embeddings
            c_init = _kmeans_plusplus_init_1d_layer(residual, k, rng)
            cb.weight.data.copy_(c_init)
            dists = torch.cdist(residual, cb.weight.data, p=2.0)
            idx = dists.argmin(dim=-1)
            residual = residual - cb(idx)
        self._initialized.fill_(True)

    def forward(self, z: torch.Tensor, use_sinkhorn_assign: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        if self.training and not bool(self._initialized.item()):
            self._init_from_batch(z)

        residual = z
        z_q_sum = torch.zeros_like(z)
        commit_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        indices_per_layer: List[torch.Tensor] = []

        for cb in self.codebooks:
            weight = cb.weight  # [K, D]
            # Squared Euclidean distance `[B, K]`
            sq_dists = (
                (residual ** 2).sum(dim=-1, keepdim=True)
                + (weight ** 2).sum(dim=-1).unsqueeze(0)
                - 2.0 * (residual @ weight.t())
            ).clamp(min=0.0)

            if self.training and use_sinkhorn_assign and residual.size(0) >= cb.num_embeddings:
                with torch.no_grad():
                    q_probs = sinkhorn_knopp_probs(
                        sq_dists.detach(), eps=self.cfg.sinkhorn_eps, n_iters=self.cfg.sinkhorn_iters
                    )
                    idx = q_probs.argmax(dim=-1)
                    # Dead-code revival on unused centroids in large batches
                    if self.cfg.dead_code_threshold > 0:
                        counts = torch.bincount(idx, minlength=cb.num_embeddings)
                        dead_mask = counts == 0
                        n_dead = int(dead_mask.sum().item())
                        if n_dead > 0:
                            rand_src = torch.randperm(residual.size(0), device=z.device)[:n_dead]
                            cb.weight.data[dead_mask] = residual.detach()[rand_src] + 1e-4 * torch.randn_like(
                                residual.detach()[rand_src]
                            )
            else:
                idx = sq_dists.argmin(dim=-1)

            quant = cb(idx)
            # Codebook + Commitment loss
            layer_loss = F.mse_loss(quant, residual.detach()) + self.cfg.commitment_weight * F.mse_loss(
                residual, quant.detach()
            )
            commit_loss = commit_loss + layer_loss

            # Straight-through estimator
            quant_ste = residual + (quant - residual).detach()
            z_q_sum = z_q_sum + quant_ste
            residual = residual - quant.detach()
            indices_per_layer.append(idx)

        codes = torch.stack(indices_per_layer, dim=-1)  # [B, L]
        return z_q_sum, commit_loss / len(self.codebooks), codes


class RQVAEModel(nn.Module):
    def __init__(self, input_dim: int, cfg: RQVAEConfig):
        super().__init__()
        self.cfg = cfg
        enc_layers: List[nn.Module] = []
        prev_d = input_dim
        for hd in cfg.hidden_dims:
            enc_layers.extend([nn.Linear(prev_d, hd), nn.LayerNorm(hd), nn.GELU()])
            prev_d = hd
        enc_layers.append(nn.Linear(prev_d, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        self.rvq = ResidualVQ(cfg.codebook_sizes, cfg.latent_dim, cfg)

        dec_layers: List[nn.Module] = []
        prev_d = cfg.latent_dim
        for hd in reversed(cfg.hidden_dims):
            dec_layers.extend([nn.Linear(prev_d, hd), nn.LayerNorm(hd), nn.GELU()])
            prev_d = hd
        dec_layers.append(nn.Linear(prev_d, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        z = self.encoder(x)
        z_q, vq_loss, codes = self.rvq(z, use_sinkhorn_assign=self.cfg.use_sinkhorn)
        x_recon = F.normalize(self.decoder(z_q), p=2, dim=-1)
        recon_loss = F.mse_loss(x_recon, x) + 0.5 * (1.0 - (x_recon * x).sum(dim=-1).mean())
        return recon_loss + vq_loss, recon_loss, codes

    @torch.no_grad()
    def encode_codes(self, x: torch.Tensor) -> torch.LongTensor:
        self.eval()
        z = self.encoder(x)
        _, _, codes = self.rvq(z, use_sinkhorn_assign=False)
        return codes


def train_and_encode_rqvae(
    embeddings: np.ndarray,
    cfg: RQVAEConfig,
    device: torch.device | str | None = None,
) -> Tuple[np.ndarray, RQVAEModel]:
    """Train RQ-VAE (Vanilla or Sinkhorn) on L2-normalized item embeddings and return `[N, L]` codes."""
    set_global_seed(cfg.seed)
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_np = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(x_np, axis=-1, keepdims=True)
    x_np = x_np / np.maximum(norms, 1e-8)

    x_tensor = torch.from_numpy(x_np)
    ds = TensorDataset(x_tensor)
    eff_bs = min(cfg.batch_size, len(ds))
    loader = DataLoader(ds, batch_size=eff_bs, shuffle=True, drop_last=(len(ds) > eff_bs * 2))

    model = RQVAEModel(input_dim=x_np.shape[1], cfg=cfg).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))

    model.train()
    for _ in range(cfg.epochs):
        for (bx,) in loader:
            bx = bx.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            total_loss, _, _ = model(bx)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    all_codes: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_tensor), eff_bs):
            bx = x_tensor[start : start + eff_bs].to(dev)
            c = model.encode_codes(bx).cpu().numpy().astype(np.int32)
            all_codes.append(c)

    return np.vstack(all_codes), model
