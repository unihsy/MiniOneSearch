"""Core Post-Training Optimization Objectives for Generative Search (`Table 4` Suite).

Implements 4 cleanly comparable post-training algorithms under unified mathematical conventions:
  1. `Listwise DPO` (Plackett-Luce / Softmax DPO over 1 positive + K hard negatives)
  2. `Vanilla GRPO` (Sequence-level reward with group std-normalization broadcast to all tokens)
  3. `TPMA-GRPO` (OneSearch-V2 position-wise marginal advantage with per-position std-normalization)
  4. `SP-TPMA-GRPO` (Ours: Ancestor-Consistent & Scale-Preserving Prefix-Tree GRPO):
       - Enforces strict ancestor-consistent prefix matching `m_{i,t,l} = prod_{u<=l} I[c_{i,u} == y_{t,u}]`
         so residual codes under wrong coarse Voronoi cells never get spurious positive credit!
       - Separates ancestor prefix gating from item-level semantic relevance feedback so off-tree
         hard negatives still receive informative negative gradients across all levels!
       - Replaces noisy per-group/per-depth `1 / std` division with Scale-Preserving Mean-Centering,
         eliminating gradient explosion on near-unanimous coarse prefixes (`15/16` agreement) and
         preserving true reward magnitude across easy vs. ambiguous search queries!
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def default_hierarchy_weights(num_levels: int = 5) -> torch.Tensor:
    """Coarse-to-fine prefix importance weights (e.g. `[2.0, 2.0, 1.2, 1.0, 1.0]` for L=5)."""
    w = torch.ones(num_levels, dtype=torch.float32)
    if num_levels >= 1:
        w[0] = 2.0
    if num_levels >= 2:
        w[1] = 2.0
    if num_levels >= 3:
        w[2] = 1.2
    return w


def compute_vanilla_grpo_advantages(
    item_rewards: torch.Tensor,  # [B, G]
    num_levels: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard Sequence-Level GRPO: `(R - mean) / (std + eps)` broadcast to `[B, G, L]`."""
    r = item_rewards.float()
    mean = r.mean(dim=-1, keepdim=True)
    std = r.std(dim=-1, unbiased=False, keepdim=True)
    seq_adv = (r - mean) / (std + eps)
    token_adv = seq_adv.unsqueeze(-1).expand(-1, -1, num_levels).contiguous()
    gates = torch.ones_like(token_adv)
    return token_adv, gates


def compute_tpma_original_advantages(
    rollout_codes: torch.LongTensor,         # [B, G, L]
    gt_code_sets: Sequence[Sequence[Sequence[int]]],  # len B, each is list of positive code tuples
    item_rewards: torch.Tensor,              # [B, G]
    w_item: float = 0.5,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Faithful reproduction of OneSearch-V2 TPMA-GRPO (position-wise std-norm + prefix gate)."""
    bsz, g_size, num_levels = rollout_codes.shape
    device = rollout_codes.device
    w_pos = torch.ones(num_levels, dtype=torch.float32, device=device)
    for l in range(min(2, num_levels)):
        w_pos[l] = 2.0
    if num_levels >= 5:
        w_pos[-1] = 0.5

    all_adv = torch.zeros((bsz, g_size, num_levels), dtype=torch.float32, device=device)
    all_gates = torch.ones((bsz, g_size, num_levels), dtype=torch.float32, device=device)

    for b in range(bsz):
        gts = gt_code_sets[b]
        if gts:
            gt_t = torch.tensor(gts, dtype=torch.long, device=device)  # [T, L]
            comp = rollout_codes[b]  # [G, L]
            match = (comp.unsqueeze(1) == gt_t.unsqueeze(0)).float()  # [G, T, L]
            weighted_cum = (match * w_pos.view(1, 1, num_levels)).cumsum(dim=2)
            weighted_pref = weighted_cum.max(dim=1).values  # [G, L]
            unweighted_pref = match.cumsum(dim=2).max(dim=1).values  # [G, L]

            marginal = torch.zeros_like(weighted_pref)
            marginal[:, 0] = weighted_pref[:, 0]
            if num_levels > 1:
                marginal[:, 1:] = weighted_pref[:, 1:] - weighted_pref[:, :-1]

            for l in range(1, num_levels):
                all_gates[b, :, l] = (unweighted_pref[:, l - 1] / float(l)).clamp(0.0, 1.0)

            pos_adv = (marginal - marginal.mean(dim=0, keepdim=True)) / (
                marginal.std(dim=0, unbiased=False, keepdim=True) + eps
            )
        else:
            pos_adv = torch.zeros((g_size, num_levels), dtype=torch.float32, device=device)

        r_b = item_rewards[b].float()
        seq_adv = (r_b - r_b.mean()) / (r_b.std(unbiased=False) + eps)
        all_adv[b] = pos_adv + w_item * seq_adv.unsqueeze(-1)

    return all_adv, all_gates


def compute_sp_tpma_advantages(
    rollout_codes: torch.LongTensor,         # [B, G, L]
    gt_code_sets: Sequence[Sequence[Sequence[int]]],
    item_rewards: torch.Tensor,              # [B, G]
    prefix_scale: float = 2.0,
    item_scale: float = 1.5,
    w_item: float = 1.0,
    off_tree_gate_floor: float = 0.30,
    rms_floor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Compute Ancestor-Consistent & Scale-Preserving Prefix-Tree Advantages (`SP-TPMA-GRPO`).

    Key Mathematical Improvements over Original TPMA:
      1. Strict Ancestor Prefix Product (`cumprod`): `P_{i,l} = max_t prod_{u<=l} I[c_{i,u} == y_{t,u}]`.
         A sub-code `<b_5>` only receives positive prefix reward if `<a_*>` ALSO matched the same
         ground-truth item `t`!
      2. Monte-Carlo Subtree Value Backup + Exact Positive Indicator:
         For each depth `l`, computes both exact ground-truth prefix match `w_l * P_{i,l}` AND
         empirical mean `R_item` of rollouts sharing prefix `c_{i, <=l}`, propagating smooth
         4-Tier Relevance & CTR signal up the hierarchical SID tree even when exact positive wasn't hit!
      3. Scale-Preserving Mean-Centering (without noisy `1 / std` distortion):
         Preserves natural gradient magnitude across unanimous coarse prefixes (`std -> 0`) and
         high-variance ambiguous queries!
    """
    bsz, g_size, num_levels = rollout_codes.shape
    device = rollout_codes.device
    w_pos = default_hierarchy_weights(num_levels).to(device)

    final_adv = torch.zeros((bsz, g_size, num_levels), dtype=torch.float32, device=device)
    final_gates = torch.ones((bsz, g_size, num_levels), dtype=torch.float32, device=device)

    ancestor_match_rates: List[float] = []

    for b in range(bsz):
        comp = rollout_codes[b]          # [G, L]
        r_b = item_rewards[b].float()    # [G]
        gts = gt_code_sets[b]

        if gts:
            gt_t = torch.tensor(gts, dtype=torch.long, device=device)  # [T, L]
            eq_tok = (comp.unsqueeze(1) == gt_t.unsqueeze(0)).float()  # [G, T, L]
            # Strict ancestor-consistent prefix indicator!
            anc_match = torch.cumprod(eq_tok, dim=2)                   # [G, T, L]
            best_anc = anc_match.max(dim=1).values                     # [G, L] in {0.0, 1.0}
            ancestor_match_rates.append(float(best_anc[:, 0].mean().item()))
        else:
            best_anc = torch.zeros((g_size, num_levels), dtype=torch.float32, device=device)

        # Prefix Marginal Exact Reward `[G, L]`
        exact_marginal = best_anc * w_pos.unsqueeze(0)  # [G, L]
        centered_prefix = (exact_marginal - exact_marginal.mean(dim=0, keepdim=True)) / max(prefix_scale, 1e-4)

        # Monte-Carlo Prefix Subtree Expected Item Reward `[G, L]`:
        # Rollouts sharing the same prefix `comp[:, :l+1]` pool their item rewards at depth `l`,
        # giving coarse tokens dense relevance-aware credit assignment!
        subtree_adv = torch.zeros((g_size, num_levels), dtype=torch.float32, device=device)
        r_centered = (r_b - r_b.mean()) / max(item_scale, 1e-4)
        for l in range(num_levels):
            pref_l = comp[:, : l + 1]  # [G, l+1]
            # Pairwise prefix equality matrix `[G, G]`
            same_pref = (pref_l.unsqueeze(1) == pref_l.unsqueeze(0)).all(dim=-1).float()
            pref_mean_adv = (same_pref @ r_centered) / same_pref.sum(dim=-1).clamp(min=1.0)
            subtree_adv[:, l] = pref_mean_adv

        # Ancestor-consistent gate applies ONLY to exact ground-truth prefix marginals,
        # while Monte-Carlo subtree item rewards flow unattenuated across all depths!
        prefix_gates_b = torch.ones((g_size, num_levels), dtype=torch.float32, device=device)
        for l in range(1, num_levels):
            prefix_gates_b[:, l] = off_tree_gate_floor + (1.0 - off_tree_gate_floor) * best_anc[:, l - 1]

        final_adv[b] = prefix_gates_b * centered_prefix + w_item * subtree_adv
        final_gates[b] = torch.ones((g_size, num_levels), dtype=torch.float32, device=device)

    # Floor-Clamped Batch-Global RMS Calibration (Dr.GRPO / EchoSpec principle):
    #   - Never divides by per-prompt `std_g` (preventing 1/std explosion on 15/16 unanimous prefixes!)
    #   - Normalizes by `max(batch_rms, rms_floor)` so active multi-prompt batches match unit step budget!
    batch_rms = final_adv.pow(2).mean().sqrt().clamp(min=rms_floor)
    final_adv = final_adv / batch_rms

    diag = {
        "sp_tpma/depth0_exact_match_rate": float(sum(ancestor_match_rates) / max(1, len(ancestor_match_rates))),
        "sp_tpma/adv_abs_mean": float(final_adv.abs().mean().item()),
        "sp_tpma/gate_mean": float(final_gates.mean().item()),
    }
    return final_adv, final_gates, diag


def ppo_clipped_kl_surrogate_loss(
    cur_per_token_logps: torch.Tensor,   # [B*G, L_pad]
    old_per_token_logps: torch.Tensor,   # [B*G, L_pad]
    ref_per_token_logps: torch.Tensor,   # [B*G, L_pad]
    token_advantages: torch.Tensor,      # [B*G, L]
    token_gates: torch.Tensor,           # [B*G, L]
    completion_mask: torch.Tensor,       # [B*G, L_pad]
    clip_eps_low: float = 0.20,
    clip_eps_high: float = 0.28,
    kl_beta: float = 0.02,
    use_ppo_clip: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Asymmetric-Clipped (or Unclipped) Policy Gradient + Low-Variance K3 KL Regularizer over `L` SID tokens."""
    l_sid = token_advantages.size(1)
    cur_lp = cur_per_token_logps[:, :l_sid]
    old_lp = old_per_token_logps[:, :l_sid].detach()
    ref_lp = ref_per_token_logps[:, :l_sid].detach()
    mask = completion_mask[:, :l_sid].float()

    log_ratio = (cur_lp - old_lp).clamp(-10.0, 10.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high)

    adv = token_advantages.detach()
    gates = token_gates.detach()

    surr1 = ratio * adv
    if use_ppo_clip:
        surr2 = clipped_ratio * adv
        pg_loss_tok = -gates * torch.minimum(surr1, surr2)
    else:
        pg_loss_tok = -gates * surr1

    # Low-variance non-negative Schulman K3 KL estimator: exp(ref - cur) - (ref - cur) - 1
    log_ref_cur = (ref_lp - cur_lp).clamp(-10.0, 10.0)
    kl_tok = torch.exp(log_ref_cur) - log_ref_cur - 1.0

    total_tok = (pg_loss_tok + kl_beta * kl_tok) * mask
    denom = mask.sum().clamp(min=1.0)
    loss = total_tok.sum() / denom

    with torch.no_grad():
        was_clipped = ((ratio < (1.0 - clip_eps_low)) | (ratio > (1.0 + clip_eps_high))).float() * mask
        metrics = {
            "rl/pg_loss": float(((pg_loss_tok * mask).sum() / denom).item()),
            "rl/kl_div": float(((kl_tok * mask).sum() / denom).item()),
            "rl/ratio_mean": float(((ratio * mask).sum() / denom).item()),
            "rl/clip_frac": float((was_clipped.sum() / denom).item()) if use_ppo_clip else 0.0,
        }
    return loss, metrics


def compute_listwise_dpo_loss(
    policy_seq_logps: torch.Tensor,   # [B, 1 + K] (col 0 is positive, cols 1..K are negatives)
    ref_seq_logps: torch.Tensor,      # [B, 1 + K]
    beta: float = 0.10,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Listwise Softmax-DPO (S-DPO / Plackett-Luce DPO) over 1 positive and K hard negatives."""
    implicit_rewards = beta * (policy_seq_logps.float() - ref_seq_logps.detach().float())  # [B, 1+K]
    # Cross-entropy with target index 0 = `-log(exp(r_pos) / sum_j exp(r_j))`
    targets = torch.zeros(implicit_rewards.size(0), dtype=torch.long, device=implicit_rewards.device)
    loss = F.cross_entropy(implicit_rewards, targets)

    with torch.no_grad():
        pos_r = implicit_rewards[:, 0]
        neg_max = implicit_rewards[:, 1:].max(dim=-1).values
        win_rate = float((pos_r > neg_max).float().mean().item())
        margin = float((pos_r - implicit_rewards[:, 1:].mean(dim=-1)).mean().item())
    return loss, {"dpo/loss": float(loss.item()), "dpo/top1_win_rate": win_rate, "dpo/reward_margin": margin}
