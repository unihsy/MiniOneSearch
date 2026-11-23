"""Unit tests for Phase 2 Search-SID Tokenizer Zoo, SID-Bench, and Stateless GPU Prefix-Trie."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from minionesearch.decoding.trie import GPUPrefixTrie, PrefixTrieLogitsProcessor, SIDVocabulary
from minionesearch.tokenizer.rq_kmeans_opq import (
    compute_khqe_fused_embeddings,
    fit_constrained_rq_kmeans,
    fit_khqe_rq_opq,
    fit_rq_kmeans,
    fit_rq_opq,
)
from minionesearch.tokenizer.rq_vae import RQVAEConfig, train_and_encode_rqvae
from minionesearch.tokenizer.sid_bench import evaluate_sid_assignment, render_sid_bench_markdown


class DummyTokenizer:
    """Minimal HuggingFace-like tokenizer to test SID registration & `skip_special_tokens=True`."""

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {"<pad>": 0, "<eos>": 1, "历史:": 2, "推荐:": 3}
        self._id2tok: Dict[int, str] = {v: k for k, v in self._vocab.items()}
        self.eos_token_id = 1
        self.pad_token_id = 0
        self._special_ids = {0, 1}

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    def add_tokens(self, new_tokens: List[str], special_tokens: bool = False) -> int:
        added = 0
        for t in new_tokens:
            if t not in self._vocab:
                nid = len(self._vocab)
                self._vocab[t] = nid
                self._id2tok[nid] = t
                if special_tokens:
                    self._special_ids.add(nid)
                added += 1
        return added

    def batch_decode(self, batch_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        out: List[str] = []
        for row in batch_ids.tolist():
            toks = [self._id2tok[i] for i in row if not (skip_special_tokens and i in self._special_ids)]
            out.append("".join(toks))
        return out


def test_stateless_gpu_prefix_trie_100_percent_validity_and_save_load(tmp_path: Path) -> None:
    ks = (8, 8, 8, 8)
    sid_vocab = SIDVocabulary(ks)
    tok = DummyTokenizer()
    level_tids = sid_vocab.register_with_tokenizer(tok)

    # Build 25 distinct valid items in `[8,8,8,8]` space (out of 4,096 total possible code tuples!)
    rng = np.random.default_rng(123)
    valid_items = {}
    used_tuples = set()
    for iid in range(100, 125):
        while True:
            tup = tuple(int(x) for x in rng.integers(0, 8, size=4))
            if tup not in used_tuples:
                used_tuples.add(tup)
                valid_items[iid] = tup
                break

    trie = GPUPrefixTrie(sid_vocab, valid_items)
    trie_file = tmp_path / "trie.json"
    trie.save(trie_file)
    loaded_trie = GPUPrefixTrie.load(trie_file)

    vocab_size = len(tok.get_vocab())
    # Prompt intentionally contains historical SIDs BEFORE the separator token `推荐:` (id=3)
    # to ensure historical SIDs in prompt NEVER confuse our stateless PrefixTrieLogitsProcessor!
    sample_hist_ids = [
        int(level_tids[0][valid_items[100][0]]),
        int(level_tids[1][valid_items[100][1]]),
        int(level_tids[2][valid_items[100][2]]),
        int(level_tids[3][valid_items[100][3]]),
    ]
    prompt_row = [2] + sample_hist_ids + [3]  # len = 6
    batch_size = 64
    input_ids = torch.tensor([prompt_row] * batch_size, dtype=torch.long)

    proc = PrefixTrieLogitsProcessor(
        trie=loaded_trie,
        level_token_ids=level_tids,
        eos_token_id=tok.eos_token_id,
        vocab_size=vocab_size,
        prompt_len=None,  # test auto-latching prompt_len!
        renormalize_logprobs=True,
    )

    torch.manual_seed(99)
    for step in range(5):  # 4 SID levels + 1 EOS step
        random_logits = torch.randn((batch_size, vocab_size)) * 3.0
        # Simulate beam-search row permutation between steps!
        perm = torch.randperm(batch_size)
        input_ids = input_ids[perm]

        masked_logprobs = proc(input_ids, random_logits)
        # Every row's valid probabilities must sum to 1.0!
        probs = torch.exp(masked_logprobs)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-4)

        # Sample from constrained distribution
        next_tok = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_tok], dim=1)

    gen_sid_tokens = input_ids[:, 6:10]
    eos_tokens = input_ids[:, 10]
    assert torch.all(eos_tokens == tok.eos_token_id)

    decoded_sids = tok.batch_decode(gen_sid_tokens, skip_special_tokens=True)
    for s in decoded_sids:
        assert loaded_trie.is_valid_sid_str(s), f"Generated invalid SID {s}!"
        assert loaded_trie.decode_sid_str_to_item(s) in valid_items


def test_all_five_sid_tokenizers_and_sid_bench() -> None:
    rng = np.random.default_rng(7)
    # Create 4 semantic clusters of 30 items each = 120 items, dim=32
    centers = rng.normal(size=(4, 32)).astype(np.float32)
    embs = []
    corpus = []
    for i in range(120):
        c = i % 4
        vec = centers[c] + 0.15 * rng.normal(size=32).astype(np.float32)
        embs.append(vec)
        corpus.append(
            {
                "item_id": i + 1,
                "category_level1_id": (c // 2) + 1,
                "category_level2_id": c + 1,
                "category_level3_id": c + 1,
                "train_pos_count": 0 if (i % 6 == 0) else 3,
                "posterior_ctr": 0.08,
            }
        )
    raw_embs = np.stack(embs, axis=0)
    ks = (8, 8, 8, 8, 8)
    eval_sessions = [
        {"query": "q_cluster0", "clicked_item_ids": [1, 5, 9], "purchased_item_ids": [1]},
        {"query": "q_cluster1", "clicked_item_ids": [2, 6, 10], "purchased_item_ids": []},
    ]

    # 1. Vanilla & Sinkhorn RQ-VAE
    c_vae, _ = train_and_encode_rqvae(
        raw_embs, RQVAEConfig(codebook_sizes=ks, latent_dim=16, hidden_dims=(32,), epochs=6, batch_size=60, use_sinkhorn=False, kmeans_init=False)
    )
    c_sk_vae, _ = train_and_encode_rqvae(
        raw_embs, RQVAEConfig(codebook_sizes=ks, latent_dim=16, hidden_dims=(32,), epochs=6, batch_size=60, use_sinkhorn=True, kmeans_init=True)
    )
    assert c_vae.shape == (120, 5)
    assert c_sk_vae.shape == (120, 5)

    # 2. RQ-KMeans & Constrained RQ-KMeans
    c_rqkm, _ = fit_rq_kmeans(raw_embs, codebook_sizes=ks, n_iters=6)
    c_crqkm, _ = fit_constrained_rq_kmeans(raw_embs, codebook_sizes=ks, n_iters=6)
    m_crq = evaluate_sid_assignment(c_crqkm, ks, corpus, eval_sessions)
    assert m_crq["collision_rate"] == 0.0
    assert m_crq["unique_sids"] == 120

    # 3. RQ-OPQ & KHQE-RQ-OPQ
    c_rqopq, _ = fit_rq_opq(raw_embs, coarse_ks=(8, 8, 8), opq_ks=(8, 8), n_iters=6)
    attr_embs = np.stack([centers[i % 4] for i in range(120)], axis=0)
    coq_embs = np.stack([centers[i % 4] for i in range(120)], axis=0)
    has_coq = np.array([(i % 6 != 0) for i in range(120)], dtype=bool)
    fused_items = compute_khqe_fused_embeddings(raw_embs, attr_embs, coq_embs, has_coq)
    c_khqe, q_khqe, _ = fit_khqe_rq_opq(
        fused_items, khqe_query_embs=centers, coarse_ks=(8, 8, 8), opq_ks=(8, 8), deduplicate_items=False, n_iters=6
    )
    c_khqe_dedup, _, _ = fit_khqe_rq_opq(
        fused_items, khqe_query_embs=centers, coarse_ks=(8, 8, 8), opq_ks=(8, 8), deduplicate_items=True, n_iters=6
    )
    assert c_khqe.shape == (120, 5)
    assert q_khqe is not None and q_khqe.shape == (4, 5)

    m_khqe = evaluate_sid_assignment(c_khqe, ks, corpus, eval_sessions)
    m_khqe_dedup = evaluate_sid_assignment(c_khqe_dedup, ks, corpus, eval_sessions)
    assert m_khqe["cat_purity_l2_d2"] > 0.7
    assert m_khqe_dedup["collision_rate"] == 0.0
    assert m_khqe_dedup["unique_sids"] == 120

    md = render_sid_bench_markdown({"Constrained RQ-KMeans": m_crq, "KHQE-RQ-OPQ": m_khqe})
    assert "KHQE-RQ-OPQ" in md
