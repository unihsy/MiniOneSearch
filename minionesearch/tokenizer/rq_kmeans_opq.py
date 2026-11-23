"""Search-Oriented Residual K-Means, Constrained RQ-KMeans, RQ-OPQ, and KHQE-RQ-OPQ.

All algorithms run seamlessly on both AMD ROCm MI300X, NVIDIA CUDA, and CPU without
requiring `faiss-gpu` C++ extensions:
  1. `RQ-KMeans`: Vectorized PyTorch GPU/CPU K-Means++ residual quantization.
  2. `Constrained-RQ-KMeans`: Sinkhorn-balanced residual K-Means + collision-free leaf suffix.
  3. `RQ-OPQ`: Coarse Residual K-Means + Orthogonal Product Quantization (SVD/FAISS OPQ) over residuals.
  4. `KHQE-RQ-OPQ` (Search Flagship):
       - Structured Attribute & Train Co-Clicked Query Enhancement (`KHQE`) for both
         Items and Queries, with item-content fallback for cold-start items.
       - Joint Query-Item Coarse RQ + Fine OPQ + deterministic leaf deduplication,
         placing Queries and Items into the *exact same* 5-level Semantic ID tree!
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from minionesearch.tokenizer.rq_vae import _kmeans_plusplus_init_1d_layer, sinkhorn_knopp_probs
from minionesearch.utils import set_global_seed


def l2_normalize_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


@torch.no_grad()
def fit_single_layer_kmeans_torch(
    x: torch.Tensor,
    k: int,
    n_iters: int = 25,
    balanced: bool = False,
    sinkhorn_eps: float = 0.04,
    chunk_size: int = 32768,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Fast GPU/CPU K-Means++ with optional Sinkhorn-balanced cluster assignment."""
    device = x.device
    n, d = x.shape
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # Initialize on random subsample if N is very large
    sub_n = min(n, max(k * 40, 32768))
    if sub_n < n:
        perm = torch.randperm(n, device=device, generator=rng)[:sub_n]
        x_init = x[perm]
    else:
        x_init = x
    centroids = _kmeans_plusplus_init_1d_layer(x_init, k, rng)  # [K, D]

    assignments = torch.zeros(n, dtype=torch.long, device=device)
    for it in range(n_iters):
        new_sums = torch.zeros((k, d), dtype=torch.float32, device=device)
        new_counts = torch.zeros(k, dtype=torch.float32, device=device)

        for st in range(0, n, chunk_size):
            bx = x[st : st + chunk_size]
            sq_dists = (
                (bx ** 2).sum(dim=-1, keepdim=True)
                + (centroids ** 2).sum(dim=-1).unsqueeze(0)
                - 2.0 * (bx @ centroids.t())
            ).clamp(min=0.0)

            if balanced and bx.size(0) >= k:
                # Normalize distance scale per batch so Sinkhorn temperature is scale-invariant
                med = sq_dists.median().clamp(min=1e-5)
                q = sinkhorn_knopp_probs(sq_dists / med, eps=sinkhorn_eps, n_iters=4)
                idx = q.argmax(dim=-1)
            else:
                idx = sq_dists.argmin(dim=-1)

            assignments[st : st + chunk_size] = idx
            one_hot = F.one_hot(idx, num_classes=k).float()
            new_counts += one_hot.sum(dim=0)
            new_sums += one_hot.t() @ bx.float()

        # Revive empty clusters if any
        empty_mask = new_counts == 0
        if bool(empty_mask.any()):
            n_empty = int(empty_mask.sum().item())
            ridx = torch.randint(0, n, (n_empty,), generator=rng, device=device)
            centroids[empty_mask] = x[ridx] + 1e-4 * torch.randn((n_empty, d), device=device, generator=rng)

        alive_mask = new_counts > 0
        centroids[alive_mask] = (new_sums[alive_mask] / new_counts[alive_mask].unsqueeze(-1)).to(x.dtype)

    # Final clean assignment pass
    for st in range(0, n, chunk_size):
        bx = x[st : st + chunk_size]
        sq_dists = (
            (bx ** 2).sum(dim=-1, keepdim=True)
            + (centroids ** 2).sum(dim=-1).unsqueeze(0)
            - 2.0 * (bx @ centroids.t())
        ).clamp(min=0.0)
        if balanced and bx.size(0) >= k:
            med = sq_dists.median().clamp(min=1e-5)
            q = sinkhorn_knopp_probs(sq_dists / med, eps=sinkhorn_eps, n_iters=5)
            assignments[st : st + chunk_size] = q.argmax(dim=-1)
        else:
            assignments[st : st + chunk_size] = sq_dists.argmin(dim=-1)

    return centroids, assignments


@torch.no_grad()
def assign_rq_centroids(
    x: np.ndarray,
    centroids_per_layer: Sequence[np.ndarray],
    device: Optional[torch.device] = None,
    chunk_size: int = 32768,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode arbitrary embeddings `[M, D]` through trained RQ centroids, returning `(codes [M, L], residuals [M, D])`."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(dev)
    n = res.size(0)
    codes_layers: List[np.ndarray] = []

    for c_np in centroids_per_layer:
        c_t = torch.from_numpy(np.asarray(c_np, dtype=np.float32)).to(dev)
        idx_all = torch.empty(n, dtype=torch.long, device=dev)
        for st in range(0, n, chunk_size):
            bx = res[st : st + chunk_size]
            dists = torch.cdist(bx, c_t, p=2.0)
            idx = dists.argmin(dim=-1)
            idx_all[st : st + chunk_size] = idx
            res[st : st + chunk_size] = bx - c_t[idx]
        codes_layers.append(idx_all.cpu().numpy().astype(np.int32))

    return np.stack(codes_layers, axis=-1), res.cpu().numpy().astype(np.float32)


def _deduplicate_final_layer_in_place(
    codes: np.ndarray,
    residuals: np.ndarray,
    last_k: int,
) -> np.ndarray:
    """Resolve leaf collisions deterministically while preserving non-colliding codes whenever possible!

    For prefixes `c[:-1]`:
      - Items whose original `c[-1]` is unique within `c[:-1]` keep their exact semantic/OPQ code!
      - Additional colliding items in the same prefix bucket are ordered by first residual projection
        and assigned unused codes in `[0, last_k)`.
    """
    out = codes.copy()
    # Project residual onto top principal direction for deterministic semantic ordering inside buckets
    proj = residuals @ residuals[0] if len(residuals) > 0 else np.zeros(len(codes))
    prefix_groups: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for row_i in range(len(out)):
        pref = tuple(int(x) for x in out[row_i, :-1])
        prefix_groups[pref].append(row_i)

    for pref, rows in prefix_groups.items():
        if len(rows) <= 1:
            continue
        rows_sorted = sorted(rows, key=lambda r: float(proj[r]))
        used_in_pref = set()
        overflow_rows: List[int] = []
        for r in rows_sorted:
            orig_c = int(out[r, -1])
            if 0 <= orig_c < last_k and orig_c not in used_in_pref:
                used_in_pref.add(orig_c)
            else:
                overflow_rows.append(r)

        free_slot = 0
        for r in overflow_rows:
            while free_slot < last_k and free_slot in used_in_pref:
                free_slot += 1
            if free_slot < last_k:
                out[r, -1] = free_slot
                used_in_pref.add(free_slot)
            else:
                # Only reachable if >last_k items collapsed into exact same L-1 prefix
                out[r, -1] = r % last_k
    return out


def fit_rq_kmeans(
    item_embs: np.ndarray,
    codebook_sizes: Sequence[int] = (256, 256, 256, 256, 256),
    n_iters: int = 25,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Standard multi-layer Residual K-Means (unconstrained)."""
    set_global_seed(seed)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    residual = torch.from_numpy(l2_normalize_np(item_embs)).to(dev)

    centroids_list: List[np.ndarray] = []
    codes_list: List[np.ndarray] = []
    for l_idx, k in enumerate(codebook_sizes):
        c_t, idx_t = fit_single_layer_kmeans_torch(
            residual, k=int(k), n_iters=n_iters, balanced=False, seed=seed + l_idx * 17
        )
        residual = residual - c_t[idx_t]
        centroids_list.append(c_t.cpu().numpy().astype(np.float32))
        codes_list.append(idx_t.cpu().numpy().astype(np.int32))

    return np.stack(codes_list, axis=-1), centroids_list


def fit_constrained_rq_kmeans(
    item_embs: np.ndarray,
    codebook_sizes: Sequence[int] = (256, 256, 256, 256, 256),
    n_iters: int = 25,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Sinkhorn-balanced Residual K-Means + deterministic collision-free final layer."""
    set_global_seed(seed)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    residual = torch.from_numpy(l2_normalize_np(item_embs)).to(dev)

    centroids_list: List[np.ndarray] = []
    codes_list: List[np.ndarray] = []
    for l_idx, k in enumerate(codebook_sizes):
        c_t, idx_t = fit_single_layer_kmeans_torch(
            residual, k=int(k), n_iters=n_iters, balanced=True, seed=seed + l_idx * 31
        )
        residual = residual - c_t[idx_t]
        centroids_list.append(c_t.cpu().numpy().astype(np.float32))
        codes_list.append(idx_t.cpu().numpy().astype(np.int32))

    raw_codes = np.stack(codes_list, axis=-1)
    dedup_codes = _deduplicate_final_layer_in_place(
        raw_codes, residual.cpu().numpy(), last_k=int(codebook_sizes[-1])
    )
    return dedup_codes, centroids_list


def _fit_orthogonal_pq_subspaces(
    train_residuals: np.ndarray,
    eval_residuals_list: Sequence[np.ndarray],
    subspace_ks: Sequence[int] = (256, 256),
    n_iters: int = 25,
    opq_alter_iters: int = 6,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray]]:
    """Ge et al. (CVPR 2013) Non-Parametric Optimized Product Quantization (Eigenvalue Allocation + Alternating Procrustes).

    Runs heavy matrix products and K-Means natively on PyTorch GPU and compact `[D, D]`
    eigendecomposition/SVD in float64, completely avoiding `faiss-cpu` OpenBLAS/libgomp
    symbol deadlocks on multi-core ROCm/CUDA servers while optimizing exact OPQ distortion
    `min_{R^T R = I, C^{(m)}} || X R - [C^{(1)}, ..., C^{(M)}] ||_F^2`.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_sub = len(subspace_ks)
    x_tr = np.asarray(train_residuals, dtype=np.float32)
    orig_d = x_tr.shape[1]
    pad_d = (m_sub - (orig_d % m_sub)) % m_sub
    if pad_d > 0:
        x_tr = np.pad(x_tr, ((0, 0), (0, pad_d)), mode="constant")
    d_eff = x_tr.shape[1]

    mean_vec = x_tr.mean(axis=0, keepdims=True)
    x_centered = x_tr - mean_vec

    rng = np.random.default_rng(seed)
    sub_n = min(len(x_centered), 40_000)
    sub_idx = (
        rng.choice(len(x_centered), size=sub_n, replace=False)
        if sub_n < len(x_centered)
        else np.arange(len(x_centered))
    )
    x_sub_t = torch.from_numpy(np.ascontiguousarray(x_centered[sub_idx], dtype=np.float32)).to(dev)

    # 1. Parametric Eigenvalue Allocation initialization on compact `[D_eff, D_eff]` covariance
    cov_np = ((x_sub_t.t() @ x_sub_t) / max(1, sub_n)).cpu().numpy().astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov_np)
    desc_order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, desc_order]  # [D_eff, D_eff], descending variance

    # Greedy variance-balanced bucket assignment (Ge et al. CVPR'13 Parametric OPQ init)
    sub_dim = d_eff // m_sub
    buckets: List[List[int]] = [[] for _ in range(m_sub)]
    bucket_log_var = np.zeros(m_sub, dtype=np.float64)
    desc_vars = np.maximum(eigvals[desc_order], 1e-12)
    for col_i in range(d_eff):
        eligible = [b for b in range(m_sub) if len(buckets[b]) < sub_dim]
        best_b = min(eligible, key=lambda b: bucket_log_var[b])
        buckets[best_b].append(col_i)
        bucket_log_var[best_b] += float(np.log(desc_vars[col_i]))

    perm_cols = [c for b in buckets for c in b]
    rot_t = torch.from_numpy(np.ascontiguousarray(eigvecs[:, perm_cols], dtype=np.float32)).to(dev)
    splits = [slice(m * sub_dim, (m + 1) * sub_dim) for m in range(m_sub)]

    # 2. Alternating Orthogonal Procrustes + Subspace K-Means (`min_{R^T R = I} || X R - Y_hat ||_F^2`)
    for alt_i in range(max(1, opq_alter_iters)):
        y_sub_t = x_sub_t @ rot_t
        y_hat_parts: List[torch.Tensor] = []
        for m_i, (sl, k_m) in enumerate(zip(splits, subspace_ks)):
            c_m, idx_m = fit_single_layer_kmeans_torch(
                y_sub_t[:, sl], k=int(k_m), n_iters=6, balanced=False, seed=seed + alt_i * 53 + m_i * 19
            )
            y_hat_parts.append(c_m[idx_m])
        y_hat_t = torch.cat(y_hat_parts, dim=-1)
        cross_np = (x_sub_t.t() @ y_hat_t).cpu().numpy().astype(np.float64)
        u_mat, _, vt_mat = np.linalg.svd(cross_np, full_matrices=False)
        rot_t = torch.from_numpy(np.ascontiguousarray(u_mat @ vt_mat, dtype=np.float32)).to(dev)

    # 3. Final Subspace Codebook Fit on Full Training Residuals
    rot_matrix = rot_t.cpu().numpy().astype(np.float32)
    x_full_t = torch.from_numpy(np.ascontiguousarray(x_centered, dtype=np.float32)).to(dev)
    y_full_t = x_full_t @ rot_t
    sub_centroids: List[np.ndarray] = []
    for m_i, (sl, k_m) in enumerate(zip(splits, subspace_ks)):
        c_t, _ = fit_single_layer_kmeans_torch(
            y_full_t[:, sl], k=int(k_m), n_iters=n_iters, balanced=True, seed=seed + 503 + m_i * 29
        )
        sub_centroids.append(c_t.cpu().numpy().astype(np.float32))

    # 4. Encode each requested evaluation residual array
    encoded_outputs: List[np.ndarray] = []
    for ev_res in eval_residuals_list:
        x_ev = np.asarray(ev_res, dtype=np.float32)
        if pad_d > 0:
            x_ev = np.pad(x_ev, ((0, 0), (0, pad_d)), mode="constant")
        ev_rot_t = torch.from_numpy(np.ascontiguousarray(x_ev - mean_vec, dtype=np.float32)).to(dev) @ rot_t
        ev_sub_codes: List[np.ndarray] = []
        for sl, c_np in zip(splits, sub_centroids):
            sub_ev_t = ev_rot_t[:, sl]
            c_t = torch.from_numpy(c_np).to(dev)
            idx_ev = torch.empty(len(x_ev), dtype=torch.long, device=dev)
            for st in range(0, len(x_ev), 32768):
                dists = torch.cdist(sub_ev_t[st : st + 32768], c_t, p=2.0)
                idx_ev[st : st + 32768] = dists.argmin(dim=-1)
            ev_sub_codes.append(idx_ev.cpu().numpy().astype(np.int32))
        encoded_outputs.append(np.stack(ev_sub_codes, axis=-1))

    return encoded_outputs, rot_matrix, sub_centroids


def _fit_and_encode_faiss_or_svd_opq(
    train_residuals: np.ndarray,
    eval_residuals_list: Sequence[np.ndarray],
    subspace_ks: Sequence[int] = (256, 256),
    n_iters: int = 25,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[List[np.ndarray], Any]:
    """Fit exact Ge et al. (CVPR'13) Alternating Procrustes OPQ natively on PyTorch GPU."""
    outs, rot_mat, opq_c = _fit_orthogonal_pq_subspaces(
        train_residuals=train_residuals,
        eval_residuals_list=eval_residuals_list,
        subspace_ks=subspace_ks,
        n_iters=n_iters,
        seed=seed,
        device=device,
    )
    return outs, {"backend": "procrustes_opq", "opq_rotation": rot_mat, "opq_centroids": opq_c}


def fit_rq_opq(
    item_embs: np.ndarray,
    coarse_ks: Sequence[int] = (256, 256, 256),
    opq_ks: Sequence[int] = (256, 256),
    l3_only_balanced: bool = True,
    deduplicate_items: bool = False,
    n_iters: int = 25,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, Dict]:
    """Item-only `RQ-OPQ`: Natural L1/L2 + L3-Balanced RQ + Native FAISS OPQ (`OneSearch §3.1.3`)."""
    set_global_seed(seed)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    residual = torch.from_numpy(l2_normalize_np(item_embs)).to(dev)

    rq_centroids: List[np.ndarray] = []
    rq_codes: List[np.ndarray] = []
    n_coarse = len(coarse_ks)
    for l_idx, k in enumerate(coarse_ks):
        # OneSearch §3.1.3: apply balanced K-Means ONLY on the 3rd RQ layer (`l_idx == n_coarse - 1`),
        # keeping L1 and L2 unconstrained so natural category granularity is preserved!
        use_bal = (l_idx == n_coarse - 1) if l3_only_balanced else True
        c_t, idx_t = fit_single_layer_kmeans_torch(
            residual, k=int(k), n_iters=n_iters, balanced=use_bal, seed=seed + l_idx * 23
        )
        residual = residual - c_t[idx_t]
        rq_centroids.append(c_t.cpu().numpy().astype(np.float32))
        rq_codes.append(idx_t.cpu().numpy().astype(np.int32))

    res_np = residual.cpu().numpy().astype(np.float32)
    (opq_codes,), opq_meta = _fit_and_encode_faiss_or_svd_opq(
        train_residuals=res_np,
        eval_residuals_list=[res_np],
        subspace_ks=opq_ks,
        n_iters=n_iters,
        seed=seed,
        device=dev,
    )
    full_codes = np.concatenate([np.stack(rq_codes, axis=-1), opq_codes], axis=-1)
    if deduplicate_items:
        full_codes = _deduplicate_final_layer_in_place(full_codes, res_np, last_k=int(opq_ks[-1]))

    artifacts = {"rq_centroids": rq_centroids, **opq_meta}
    return full_codes, artifacts


def compute_khqe_fused_embeddings(
    raw_item_embs: np.ndarray,
    attr_item_embs: np.ndarray,
    co_query_item_embs: Optional[np.ndarray] = None,
    has_co_query_mask: Optional[np.ndarray] = None,
    w_text: float = 0.55,
    w_attr: float = 0.25,
    w_co_query: float = 0.20,
) -> np.ndarray:
    """Fuse item semantic text, structured brand/category attributes, and train co-click query signals.

    Cold-start items (`has_co_query_mask == False`) redistribute weight to
    `text + structured_attributes` for item-content zero-shot placement.
    """
    e_txt = l2_normalize_np(raw_item_embs)
    e_att = l2_normalize_np(attr_item_embs)

    if co_query_item_embs is None or has_co_query_mask is None:
        s = w_text + w_attr
        return l2_normalize_np((w_text / s) * e_txt + (w_attr / s) * e_att)

    e_coq = l2_normalize_np(co_query_item_embs)
    mask = np.asarray(has_co_query_mask, dtype=bool)[:, None]

    warm_fused = w_text * e_txt + w_attr * e_att + w_co_query * e_coq
    s_cold = max(w_text + w_attr, 1e-6)
    cold_fused = (w_text / s_cold) * e_txt + (w_attr / s_cold) * e_att
    fused = np.where(mask, warm_fused, cold_fused)
    return l2_normalize_np(fused)


def fit_khqe_rq_opq(
    khqe_item_embs: np.ndarray,
    khqe_train_query_embs: Optional[np.ndarray] = None,
    khqe_infer_query_embs: Optional[np.ndarray] = None,
    coarse_ks: Sequence[int] = (256, 256, 256),
    opq_ks: Sequence[int] = (256, 256),
    l3_only_balanced: bool = True,
    deduplicate_items: bool = False,
    n_iters: int = 25,
    seed: int = 42,
    device: Optional[torch.device] = None,
    khqe_query_embs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """OneSearch-V1 §3.1 Flagship `KHQE-RQ-OPQ`:
      - Trains Natural-L1/L2 + L3-Balanced RQ + Native FAISS OPQ strictly on `[Items; TrainQueries]`.
      - Encodes unseen `InferQueries` (e.g. val/test queries) using ONLY frozen train centroids & OPQ!
    """
    set_global_seed(seed)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if khqe_train_query_embs is None and khqe_query_embs is not None:
        khqe_train_query_embs = khqe_query_embs

    norm_items = l2_normalize_np(khqe_item_embs)
    n_items = len(norm_items)
    if khqe_train_query_embs is not None and len(khqe_train_query_embs) > 0:
        norm_tr_q = l2_normalize_np(khqe_train_query_embs)
        joint_train_np = np.vstack([norm_items, norm_tr_q])
    else:
        norm_tr_q = None
        joint_train_np = norm_items

    residual = torch.from_numpy(joint_train_np).to(dev)
    rq_centroids: List[np.ndarray] = []
    joint_rq_codes: List[np.ndarray] = []
    n_coarse = len(coarse_ks)
    for l_idx, k in enumerate(coarse_ks):
        use_bal = (l_idx == n_coarse - 1) if l3_only_balanced else True
        c_t, idx_t = fit_single_layer_kmeans_torch(
            residual, k=int(k), n_iters=n_iters, balanced=use_bal, seed=seed + l_idx * 37
        )
        residual = residual - c_t[idx_t]
        rq_centroids.append(c_t.cpu().numpy().astype(np.float32))
        joint_rq_codes.append(idx_t.cpu().numpy().astype(np.int32))

    joint_res_np = residual.cpu().numpy().astype(np.float32)
    item_res_np = joint_res_np[:n_items]
    eval_res_list: List[np.ndarray] = [item_res_np]
    if norm_tr_q is not None:
        eval_res_list.append(joint_res_np[n_items:])

    infer_rq_codes: Optional[np.ndarray] = None
    if khqe_infer_query_embs is not None and len(khqe_infer_query_embs) > 0:
        norm_inf_q = l2_normalize_np(khqe_infer_query_embs)
        infer_rq_codes, infer_res_np = assign_rq_centroids(norm_inf_q, rq_centroids, device=dev)
        eval_res_list.append(infer_res_np)

    print(f"  [KHQE-RQ-OPQ] Coarse {n_coarse}-layer RQ finished (N={len(joint_train_np):,}); fitting OPQ...", flush=True)
    opq_outputs, opq_meta = _fit_and_encode_faiss_or_svd_opq(
        train_residuals=joint_res_np,
        eval_residuals_list=eval_res_list,
        subspace_ks=opq_ks,
        n_iters=n_iters,
        seed=seed,
        device=dev,
    )
    print(f"  [KHQE-RQ-OPQ] OPQ finished via backend={opq_meta.get('backend')} (dedup={deduplicate_items})", flush=True)

    joint_rq_arr = np.stack(joint_rq_codes, axis=-1)
    item_codes = np.concatenate([joint_rq_arr[:n_items], opq_outputs[0]], axis=-1)
    if deduplicate_items:
        item_codes = _deduplicate_final_layer_in_place(item_codes, item_res_np, last_k=int(opq_ks[-1]))

    train_q_codes: Optional[np.ndarray] = None
    ptr = 1
    if norm_tr_q is not None:
        train_q_codes = np.concatenate([joint_rq_arr[n_items:], opq_outputs[ptr]], axis=-1)
        ptr += 1

    infer_q_codes: Optional[np.ndarray] = None
    if infer_rq_codes is not None:
        infer_q_codes = np.concatenate([infer_rq_codes, opq_outputs[ptr]], axis=-1)

    artifacts = {"rq_centroids": rq_centroids, "infer_query_codes": infer_q_codes, **opq_meta}
    return item_codes, train_q_codes, artifacts
