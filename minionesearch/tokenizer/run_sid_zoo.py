"""End-to-End Phase 2 Runner: OneSearch-Fidelity SID Tokenizer Zoo + SID-Bench + Prefix-Trie.

Compares 7 SID Tokenizers across identical 5-level codebook geometry:
  1. `Vanilla RQ-VAE (TIGER)`
  2. `Sinkhorn RQ-VAE (LC-Rec)`
  3. `RQ-KMeans` (Unconstrained global residual K-Means)
  4. `Constrained RQ-KMeans (Full-Bal)` (OneRec-V1 / MiniOneRec full-layer balanced)
  5. `RQ-OPQ (Full-Layer Balanced)` (Demonstrates OneSearch §3.1.3 early-balancing split issue)
  6. `RQ-OPQ (L3-Balanced, No KHQE)` (OneSearch §3.1.3 natural L1/L2 + L3-balanced + FAISS OPQ)
  7. `OneSearch KHQE-RQ-OPQ (Full)` (5-Loss Aligned Encoder + Eq.2 Aho-Corasick Keyword Enhancement
     + Natural L1/L2 & L3-Balanced Joint Q-I RQ + Native FAISS OPQ + Live Query-SID Quantizer!)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import jieba
import numpy as np
import torch

from minionesearch.baselines.dense_retriever import BiEncoderWrapper, encode_texts_single_or_sharded
from minionesearch.decoding.trie import GPUPrefixTrie, SIDVocabulary
from minionesearch.tokenizer.rq_kmeans_opq import (
    fit_constrained_rq_kmeans,
    fit_khqe_rq_opq,
    fit_rq_kmeans,
    fit_rq_opq,
    l2_normalize_np,
)
from minionesearch.tokenizer.rq_vae import RQVAEConfig, train_and_encode_rqvae
from minionesearch.tokenizer.sid_bench import evaluate_sid_assignment, render_sid_bench_markdown
from minionesearch.utils import iter_jsonl, set_global_seed


def mine_train_core_keyword_dictionary(
    corpus_recs: Sequence[Mapping[str, Any]],
    train_sessions: Sequence[Mapping[str, Any]],
    max_mined_modifiers: int = 3000,
) -> List[str]:
    """Mine OneSearch §3.1.2 Core Keywords strictly from catalog attributes + training query PV."""
    core_terms = set()
    for r in corpus_recs:
        for field in ("brand_name", "category_level1_name", "category_level2_name", "category_level3_name"):
            v = str(r.get(field) or "").strip()
            if 2 <= len(v) <= 16:
                core_terms.add(v.lower())

    pv_counter: Counter = Counter()
    for s in train_sessions:
        q = str(s.get("query") or "").strip().lower()
        for w in jieba.cut(q):
            w_s = w.strip()
            if 2 <= len(w_s) <= 10 and not w_s.isspace():
                pv_counter[w_s] += 1
    for term, freq in pv_counter.most_common(max_mined_modifiers):
        if freq >= 3:
            core_terms.add(term)

    return sorted(core_terms)


def _apply_eq2_keyword_enhancement(
    texts_to_match: Sequence[str],
    raw_embs: np.ndarray,
    kw_list: Sequence[str],
    kw_embs: np.ndarray,
) -> np.ndarray:
    """Exact OneSearch-V1 §3.1.2 Eq. (2): `e^o = Norm(0.5 * e_raw + 0.5 * mean(matched_kw_embs))`.

    Uses a pure-Python character prefix Trie so longest-span multi-keyword matching over
    100K-500K items runs in ~1 second on any OS/GPU container with zero C-compiler dependency!
    """
    kw_to_vec = {k: kw_embs[i] for i, k in enumerate(kw_list)}
    trie_root: Dict[str, Any] = {}
    END_KEY = "\x00"
    for kw in kw_list:
        node = trie_root
        for ch in kw:
            node = node.setdefault(ch, {})
        node[END_KEY] = kw

    def _match(text: str) -> List[str]:
        tl = text.lower()
        n_len = len(tl)
        found_set = set()
        for start_i in range(n_len):
            node = trie_root
            for cur_i in range(start_i, min(n_len, start_i + 16)):
                nxt = node.get(tl[cur_i])
                if nxt is None:
                    break
                node = nxt
                if END_KEY in node:
                    found_set.add(node[END_KEY])
        if not found_set:
            return []
        found = sorted(found_set, key=lambda s: -len(s))
        kept: List[str] = []
        for cand in found:
            if not any(cand in longer for longer in kept):
                kept.append(cand)
        return kept[:8]

    raw_norm = l2_normalize_np(raw_embs)
    enhanced = raw_norm.copy()
    for idx, txt in enumerate(texts_to_match):
        m_terms = _match(txt)
        if m_terms:
            vecs = [kw_to_vec[t] for t in m_terms if t in kw_to_vec]
            if vecs:
                mean_kw = np.mean(np.stack(vecs, axis=0), axis=0)
                mean_kw = mean_kw / max(float(np.linalg.norm(mean_kw)), 1e-8)
                enhanced[idx] = 0.5 * raw_norm[idx] + 0.5 * mean_kw
    return l2_normalize_np(enhanced)


def prepare_onesearch_khqe_embeddings(
    bench_dir: Path,
    aligned_encoder_ckpt: str | Path,
    cache_dir: Path,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, List[str], np.ndarray]:
    """Encode Items & Queries with Aligned Encoder and apply OneSearch Eq. (2) Keyword Enhancement."""
    from transformers import AutoModel, AutoTokenizer

    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_f = cache_dir / "khqe_v2_ready.json"
    if meta_f.is_file():
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        return (
            np.load(cache_dir / "aligned_raw_item_embs.npy"),
            np.load(cache_dir / "khqe_item_embs.npy"),
            meta["train_queries"],
            np.load(cache_dir / "khqe_train_q_embs.npy"),
            meta["infer_queries"],
            np.load(cache_dir / "khqe_infer_q_embs.npy"),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus_recs = list(iter_jsonl(bench_dir / "corpus.jsonl"))
    train_sessions = list(iter_jsonl(bench_dir / "train_sessions.jsonl"))
    val_sessions = list(iter_jsonl(bench_dir / "val_sessions.jsonl"))
    test_sessions = list(iter_jsonl(bench_dir / "test_sessions.jsonl"))

    tok = AutoTokenizer.from_pretrained(aligned_encoder_ckpt)
    backbone = AutoModel.from_pretrained(aligned_encoder_ckpt)
    enc = BiEncoderWrapper(backbone, pooling="cls").to(device)

    item_texts = [str(r["structured_text"]) for r in corpus_recs]
    raw_item_embs = encode_texts_single_or_sharded(
        enc, tok, item_texts, device=device, batch_size=batch_size, max_length=96, desc="Aligned Item Embs"
    )

    kw_list = mine_train_core_keyword_dictionary(corpus_recs, train_sessions)
    kw_embs = encode_texts_single_or_sharded(
        enc, tok, kw_list, device=device, batch_size=batch_size, max_length=32, desc="Core Keywords (18-Attr/PV)"
    )

    khqe_item_embs = _apply_eq2_keyword_enhancement(item_texts, raw_item_embs, kw_list, kw_embs)

    train_q_set = sorted({str(s["query"]).strip() for s in train_sessions if str(s.get("query", "")).strip()})
    all_obs_queries = set(train_q_set)
    for s in train_sessions + val_sessions + test_sessions:
        q_cur = str(s.get("query", "")).strip()
        if q_cur:
            all_obs_queries.add(q_cur)
        for hq in s.get("hist_queries") or []:
            if str(hq).strip():
                all_obs_queries.add(str(hq).strip())
    infer_only_qs = sorted(all_obs_queries - set(train_q_set))

    raw_tr_q = encode_texts_single_or_sharded(
        enc,
        tok,
        [f"搜索查询: {q}" for q in train_q_set],
        device=device,
        batch_size=batch_size,
        max_length=64,
        desc="Aligned Train Queries",
    )
    khqe_tr_q = _apply_eq2_keyword_enhancement(train_q_set, raw_tr_q, kw_list, kw_embs)

    if infer_only_qs:
        raw_inf_q = encode_texts_single_or_sharded(
            enc,
            tok,
            [f"搜索查询: {q}" for q in infer_only_qs],
            device=device,
            batch_size=batch_size,
            max_length=64,
            desc="Live Infer Queries (Val/Test)",
        )
        khqe_inf_q = _apply_eq2_keyword_enhancement(infer_only_qs, raw_inf_q, kw_list, kw_embs)
    else:
        khqe_inf_q = np.zeros((0, raw_item_embs.shape[1]), dtype=np.float32)

    np.save(cache_dir / "aligned_raw_item_embs.npy", raw_item_embs)
    np.save(cache_dir / "khqe_item_embs.npy", khqe_item_embs)
    np.save(cache_dir / "khqe_train_q_embs.npy", khqe_tr_q)
    np.save(cache_dir / "khqe_infer_q_embs.npy", khqe_inf_q)
    meta_f.write_text(
        json.dumps({"train_queries": train_q_set, "infer_queries": infer_only_qs}, ensure_ascii=False),
        encoding="utf-8",
    )
    return raw_item_embs, khqe_item_embs, train_q_set, khqe_tr_q, infer_only_qs, khqe_inf_q


def export_sid_bundle(
    out_subdir: Path,
    codebook_sizes: Sequence[int],
    corpus_records: Sequence[Mapping[str, Any]],
    item_codes: np.ndarray,
    uniq_queries: Optional[Sequence[str]] = None,
    query_codes: Optional[np.ndarray] = None,
    infer_queries: Optional[Sequence[str]] = None,
    infer_query_codes: Optional[np.ndarray] = None,
) -> GPUPrefixTrie:
    """Save `trie.json`, `item_sid_map.json`, `query_sid_map.json` (train), and `query_sid_all_live.json`."""
    out_subdir.mkdir(parents=True, exist_ok=True)
    vocab = SIDVocabulary(codebook_sizes)
    item_to_codes = {
        int(corpus_records[i]["item_id"]): tuple(int(x) for x in item_codes[i])
        for i in range(len(corpus_records))
    }
    item_priors = {int(r["item_id"]): float(r.get("posterior_ctr") or 0.0) for r in corpus_records}
    trie = GPUPrefixTrie(vocab, item_to_codes=item_to_codes, item_priors=item_priors)
    trie.save(out_subdir / "trie.json")

    item_sid_str_map = {str(k): vocab.codes_to_str(v) for k, v in item_to_codes.items()}
    (out_subdir / "item_sid_map.json").write_text(
        json.dumps(item_sid_str_map, ensure_ascii=False), encoding="utf-8"
    )

    if uniq_queries is not None and query_codes is not None:
        q_sid_map = {q: vocab.codes_to_str(query_codes[i]) for i, q in enumerate(uniq_queries)}
        (out_subdir / "query_sid_map.json").write_text(
            json.dumps(q_sid_map, ensure_ascii=False), encoding="utf-8"
        )
        q_all_map = dict(q_sid_map)
        if infer_queries is not None and infer_query_codes is not None:
            for i, iq in enumerate(infer_queries):
                q_all_map[iq] = vocab.codes_to_str(infer_query_codes[i])
        (out_subdir / "query_sid_all_live.json").write_text(
            json.dumps(q_all_map, ensure_ascii=False), encoding="utf-8"
        )
    return trie


def run_full_sid_zoo(
    bench_dir: str | Path,
    raw_emb_path: str | Path,
    encoder_ckpt: str | Path,
    out_dir: str | Path,
    codebook_sizes: Tuple[int, ...] = (512, 256, 256, 256, 256),
    rqvae_epochs: int = 45,
    kmeans_iters: int = 30,
    only_khqe: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    set_global_seed(seed)
    bpath = Path(bench_dir)
    opath = Path(out_dir)
    opath.mkdir(parents=True, exist_ok=True)

    corpus_recs = list(iter_jsonl(bpath / "corpus.jsonl"))
    test_sessions = list(iter_jsonl(bpath / "test_sessions.jsonl"))

    print("[SID-Zoo] Computing 5-Loss Aligned & Eq.(2) KHQE Representations...", flush=True)
    aligned_embs, khqe_items, tr_qs, khqe_tr_q, inf_qs, khqe_inf_q = prepare_onesearch_khqe_embeddings(
        bench_dir=bpath,
        aligned_encoder_ckpt=encoder_ckpt,
        cache_dir=opath / "khqe_v2_cache",
    )
    base_embs = np.load(raw_emb_path).astype(np.float32) if (raw_emb_path and Path(raw_emb_path).is_file()) else aligned_embs

    coarse_ks = codebook_sizes[:3]
    opq_ks = codebook_sizes[3:]
    bench_summary: Dict[str, Dict[str, Any]] = {}

    if not only_khqe:
        if raw_emb_path and Path(raw_emb_path).is_file():
            zs_embs = np.load(raw_emb_path).astype(np.float32)
            print("[SID-Zoo 0/7] Fitting RQ-OPQ on Zero-Shot BGE-M3 (isolates 5-Loss Aligned Encoder gain)...", flush=True)
            codes_zs_opq, _ = fit_rq_opq(
                zs_embs, coarse_ks=coarse_ks, opq_ks=opq_ks, l3_only_balanced=True, deduplicate_items=False, n_iters=kmeans_iters, seed=seed
            )
            bench_summary["RQ-OPQ (Zero-Shot BGE-M3, L3-Bal)"] = evaluate_sid_assignment(
                codes_zs_opq, codebook_sizes, corpus_recs, test_sessions
            )

        # All 7 core quantizers below share the same 5-Loss Aligned Encoder (`aligned_embs`) for controlled comparison.
        # 1. Vanilla RQ-VAE (TIGER)
        print("[SID-Zoo 1/7] Training Vanilla RQ-VAE (TIGER)...", flush=True)
        codes_vae, _ = train_and_encode_rqvae(
            aligned_embs,
            RQVAEConfig(codebook_sizes=codebook_sizes, use_sinkhorn=False, kmeans_init=False, dead_code_threshold=0, epochs=rqvae_epochs, seed=seed),
        )
        bench_summary["Vanilla RQ-VAE (TIGER)"] = evaluate_sid_assignment(codes_vae, codebook_sizes, corpus_recs, test_sessions)
        export_sid_bundle(opath / "vanilla_rqvae", codebook_sizes, corpus_recs, codes_vae)

        # 2. Sinkhorn RQ-VAE (LC-Rec)
        print("[SID-Zoo 2/7] Training Sinkhorn RQ-VAE (LC-Rec)...", flush=True)
        codes_sk_vae, _ = train_and_encode_rqvae(
            aligned_embs,
            RQVAEConfig(codebook_sizes=codebook_sizes, use_sinkhorn=True, kmeans_init=True, dead_code_threshold=2, epochs=rqvae_epochs, seed=seed),
        )
        bench_summary["Sinkhorn RQ-VAE (LC-Rec)"] = evaluate_sid_assignment(codes_sk_vae, codebook_sizes, corpus_recs, test_sessions)
        export_sid_bundle(opath / "sinkhorn_rqvae", codebook_sizes, corpus_recs, codes_sk_vae)

        # 3. Unconstrained RQ-KMeans
        print("[SID-Zoo 3/7] Fitting Unconstrained RQ-KMeans...", flush=True)
        codes_rqkm, _ = fit_rq_kmeans(aligned_embs, codebook_sizes=codebook_sizes, n_iters=kmeans_iters, seed=seed)
        bench_summary["RQ-KMeans"] = evaluate_sid_assignment(codes_rqkm, codebook_sizes, corpus_recs, test_sessions)
        export_sid_bundle(opath / "rq_kmeans", codebook_sizes, corpus_recs, codes_rqkm)

        # 4. Constrained Full-Layer Balanced RQ-KMeans (OneRec-V1 / MiniOneRec)
        print("[SID-Zoo 4/7] Fitting Constrained Full-Balanced RQ-KMeans (OneRec-V1 / MiniOneRec)...", flush=True)
        codes_crqkm, _ = fit_constrained_rq_kmeans(aligned_embs, codebook_sizes=codebook_sizes, n_iters=kmeans_iters, seed=seed)
        bench_summary["Constrained RQ-KMeans (Full-Bal)"] = evaluate_sid_assignment(codes_crqkm, codebook_sizes, corpus_recs, test_sessions)
        export_sid_bundle(opath / "constrained_rq_kmeans", codebook_sizes, corpus_recs, codes_crqkm)

        # 5. RQ-OPQ with Full-Layer Balancing (strictly single-variable test of OneSearch §3.1.3 L1/L2 balancing!)
        print("[SID-Zoo 5/7] Fitting RQ-OPQ (Full-Layer Balanced Ablation)...", flush=True)
        codes_opq_full, _ = fit_rq_opq(
            aligned_embs, coarse_ks=coarse_ks, opq_ks=opq_ks, l3_only_balanced=False, deduplicate_items=False, n_iters=kmeans_iters, seed=seed
        )
        bench_summary["RQ-OPQ (Full-Layer Balanced)"] = evaluate_sid_assignment(codes_opq_full, codebook_sizes, corpus_recs, test_sessions)

        # 6. RQ-OPQ with OneSearch L3-Only Balancing (Item-Only, No KHQE)
        print("[SID-Zoo 6/7] Fitting RQ-OPQ (Natural L1/L2 + L3-Balanced, No KHQE)...", flush=True)
        codes_rqopq, _ = fit_rq_opq(
            aligned_embs, coarse_ks=coarse_ks, opq_ks=opq_ks, l3_only_balanced=True, deduplicate_items=False, n_iters=kmeans_iters, seed=seed
        )
        bench_summary["RQ-OPQ (L3-Balanced, No KHQE)"] = evaluate_sid_assignment(codes_rqopq, codebook_sizes, corpus_recs, test_sessions)
        export_sid_bundle(opath / "rq_opq", codebook_sizes, corpus_recs, codes_rqopq)

    # 7. Flagship OneSearch KHQE-RQ-OPQ (Aligned 5-Loss + Eq.2 Keyword Enhancement + L3-Balanced Joint Q-I + Native FAISS OPQ)
    print("[SID-Zoo 7/7] Fitting Full OneSearch KHQE-RQ-OPQ (Joint Q-I + Live Query Quantizer)...", flush=True)
    codes_khqe, q_codes_khqe, khqe_art = fit_khqe_rq_opq(
        khqe_item_embs=khqe_items,
        khqe_train_query_embs=khqe_tr_q,
        khqe_infer_query_embs=khqe_inf_q,
        coarse_ks=coarse_ks,
        opq_ks=opq_ks,
        l3_only_balanced=True,
        deduplicate_items=False,
        n_iters=kmeans_iters,
        seed=seed,
    )
    bench_summary["OneSearch KHQE-RQ-OPQ (Full)"] = evaluate_sid_assignment(
        codes_khqe, codebook_sizes, corpus_recs, test_sessions
    )
    export_sid_bundle(
        opath / "khqe_rq_opq",
        codebook_sizes,
        corpus_recs,
        codes_khqe,
        uniq_queries=tr_qs,
        query_codes=q_codes_khqe,
        infer_queries=inf_qs,
        infer_query_codes=khqe_art.get("infer_query_codes"),
    )

    # 7b. Collision-Free Leaf-Disambiguated OneSearch KHQE-RQ-OPQ (`ICR = 100.00%, Max Bucket = 1`)
    codes_khqe_dedup, _, _ = fit_khqe_rq_opq(
        khqe_item_embs=khqe_items,
        khqe_train_query_embs=khqe_tr_q,
        khqe_infer_query_embs=khqe_inf_q,
        coarse_ks=coarse_ks,
        opq_ks=opq_ks,
        l3_only_balanced=True,
        deduplicate_items=True,
        n_iters=kmeans_iters,
        seed=seed,
    )
    bench_summary["OneSearch KHQE-RQ-OPQ (Dedup, Zero-Coll)"] = evaluate_sid_assignment(
        codes_khqe_dedup, codebook_sizes, corpus_recs, test_sessions
    )
    export_sid_bundle(
        opath / "khqe_rq_opq_dedup",
        codebook_sizes,
        corpus_recs,
        codes_khqe_dedup,
        uniq_queries=tr_qs,
        query_codes=q_codes_khqe,
        infer_queries=inf_qs,
        infer_query_codes=khqe_art.get("infer_query_codes"),
    )

    (opath / "sid_bench_results.json").write_text(
        json.dumps(bench_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = render_sid_bench_markdown(bench_summary)
    (opath / "sid_bench_table.md").write_text(md, encoding="utf-8")
    print("\n" + md, flush=True)
    return bench_summary


def main() -> None:
    p = argparse.ArgumentParser(description="Run OneSearch-Fidelity SID Tokenizer Zoo & SID-Bench")
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--raw_emb_path", default="")
    p.add_argument("--encoder_ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--codebook_sizes", default="512,256,256,256,256")
    p.add_argument("--rqvae_epochs", type=int, default=45)
    p.add_argument("--kmeans_iters", type=int, default=30)
    p.add_argument("--only_khqe", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    ks = tuple(int(x) for x in a.codebook_sizes.split(","))
    run_full_sid_zoo(
        bench_dir=a.bench_dir,
        raw_emb_path=a.raw_emb_path,
        encoder_ckpt=a.encoder_ckpt,
        out_dir=a.out_dir,
        codebook_sizes=ks,
        rqvae_epochs=a.rqvae_epochs,
        kmeans_iters=a.kmeans_iters,
        only_khqe=a.only_khqe,
        seed=a.seed,
    )


if __name__ == "__main__":
    main()
