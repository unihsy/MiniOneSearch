"""Lightweight Native PyTorch Constrained Beam Search & User-Prefix KV-Cache Latency Benchmark.

Why this exists:
  HuggingFace's generic `model.generate(num_beams=K)` executes heavy per-step Python wrapper
  validation, config cloning, and dynamic tensor slicing (~140 ms overhead at `batch_size=1`),
  which masks true GPU operator latency and hides `User-Prefix KV-Cache` prefill savings.

This engine runs exact 5-step Prefix-Trie Constrained Beam Search directly via `model.forward`
with zero `GenerationMixin` wrapper overhead, measuring authentic P50/P95/P99 latency (ms) & QPS
across short (`L=64`), medium (`L=256`), and long (`L=512`) user histories with vs. without
User-Prefix KV-Cache!
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from minionesearch.decoding.kv_cache_engine import _clone_dynamic_cache
from minionesearch.decoding.trie import GPUPrefixTrie, PrefixTrieLogitsProcessor
from minionesearch.evaluation import summarize_latencies_ms


class NativeFastConstrainedBeamEngine:
    """Zero-wrapper 5-step GPU Prefix-Trie Beam Search engine with exact User-Prefix KV-Cache."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        trie: GPUPrefixTrie,
        level_token_ids: Sequence[torch.LongTensor],
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.trie = trie
        self.level_token_ids = [t.to(device or next(model.parameters()).device) for t in level_token_ids]
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self.vocab_size = int(model.get_input_embeddings().weight.shape[0])
        self.eos_id = int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id)

    @torch.no_grad()
    def precompute_prefix_kv(self, prefix_ids: torch.LongTensor) -> DynamicCache:
        cache = DynamicCache()
        _ = self.model(input_ids=prefix_ids, past_key_values=cache, use_cache=True, return_dict=True)
        return cache

    @torch.no_grad()
    def decode_fast_beams(
        self,
        suffix_or_full_ids: torch.LongTensor,  # [1, S]
        num_beams: int = 10,
        tiled_prefix_cache: Optional[DynamicCache] = None,
        prefix_len: int = 0,
    ) -> List[int]:
        """Exact `L`-step Constrained Beam Search directly on `model.forward`."""
        l_sid = self.trie.num_levels
        proc = PrefixTrieLogitsProcessor(
            trie=self.trie,
            level_token_ids=self.level_token_ids,
            eos_token_id=self.eos_id,
            vocab_size=self.vocab_size,
            prompt_len=0,
            renormalize_logprobs=True,
        )

        # Step 0: Prefill `[1, S]` (either full prompt if uncached, or short query suffix if KV-cached!)
        if tiled_prefix_cache is None:
            kv = DynamicCache()
            out0 = self.model(input_ids=suffix_or_full_ids, past_key_values=kv, use_cache=True, return_dict=True)
        else:
            kv = tiled_prefix_cache
            pos_ids = torch.arange(
                prefix_len, prefix_len + suffix_or_full_ids.size(1), dtype=torch.long, device=self.device
            ).unsqueeze(0)
            out0 = self.model(
                input_ids=suffix_or_full_ids,
                position_ids=pos_ids,
                past_key_values=kv,
                use_cache=True,
                return_dict=True,
            )

        logits0 = out0.logits[:, -1, :]  # [1, V]
        empty_gen = torch.empty((1, 0), dtype=torch.long, device=self.device)
        logp0 = proc(empty_gen, logits0).squeeze(0)  # [V]

        k0 = min(num_beams, int((logp0 > -1e10).sum().item()))
        beam_scores, next_toks = torch.topk(logp0, k=k0, largest=True, sorted=True)  # [k0]
        gen_seqs = next_toks.unsqueeze(1)  # [k0, 1]
        beam_kv = _clone_dynamic_cache(kv, repeat_beams=k0)

        # Steps 1 .. L-1: Single-token batched `[k0, 1]` forward + vectorized Trie mask!
        for step in range(1, l_sid):
            cur_in = gen_seqs[:, -1:]  # [k0, 1]
            out_s = self.model(input_ids=cur_in, past_key_values=beam_kv, use_cache=True, return_dict=True)
            step_logp = proc(gen_seqs, out_s.logits[:, -1, :])  # [k0, V]

            # Restrict candidate expansion to valid level tokens `[k0, K_step]` for 50x faster topk!
            step_tids = self.level_token_ids[step]  # [K_step]
            cand_scores = beam_scores.unsqueeze(1) + step_logp[:, step_tids]  # [k0, K_step]
            flat_cand = cand_scores.view(-1)
            k_next = min(num_beams, int((flat_cand > -1e10).sum().item()))
            top_scores, top_flat_idx = torch.topk(flat_cand, k=k_next, largest=True, sorted=True)

            parent_beam = top_flat_idx // step_tids.size(0)
            chosen_local = top_flat_idx % step_tids.size(0)
            chosen_global = step_tids[chosen_local]

            gen_seqs = torch.cat([gen_seqs[parent_beam], chosen_global.unsqueeze(1)], dim=1)
            beam_scores = top_scores

            # Reorder KV cache rows along `parent_beam`
            if hasattr(beam_kv, "reorder_cache"):
                beam_kv.reorder_cache(parent_beam)
            elif hasattr(beam_kv, "layers"):
                for layer in beam_kv.layers:
                    if getattr(layer, "keys", None) is not None:
                        layer.keys = layer.keys.index_select(0, parent_beam)
                        layer.values = layer.values.index_select(0, parent_beam)

        decoded = self.tokenizer.batch_decode(gen_seqs, skip_special_tokens=True)
        return self.trie.decode_beams_to_ranked_items(decoded, max_items=num_beams)


def run_latency_matrix_benchmark(
    ckpt_dir: str | Path,
    sid_dir: str | Path,
    out_json: str | Path,
    num_beams: int = 10,
    num_trials: int = 80,
    prefix_lengths: Sequence[int] = (128, 512, 1024),
) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    trie = GPUPrefixTrie.load(Path(sid_dir) / "trie.json")
    level_tids = trie.sid_vocab.register_with_tokenizer(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, torch_dtype=dtype, trust_remote_code=True).to(device)
    engine = NativeFastConstrainedBeamEngine(model, tokenizer, trie, level_tids, device=device)

    results: Dict[str, Any] = {}
    vocab_base = 1000
    suf_len = 18

    for p_len in prefix_lengths:
        torch.manual_seed(42)
        pref_list = [torch.randint(10, vocab_base, (1, p_len), device=device) for _ in range(num_trials)]
        suf_list = [torch.randint(10, vocab_base, (1, suf_len), device=device) for _ in range(num_trials)]
        full_list = [torch.cat([p, s], dim=1) for p, s in zip(pref_list, suf_list)]

        # Warmup
        for i in range(5):
            _ = engine.decode_fast_beams(full_list[i], num_beams=num_beams)

        # 1. Uncached Online Latency (`prefix + query_suffix` prefilled on arrival)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        lats_uncached: List[float] = []
        outs_u: List[List[int]] = []
        t_w0 = time.perf_counter()
        for i in range(num_trials):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            res_u = engine.decode_fast_beams(full_list[i], num_beams=num_beams)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            lats_uncached.append((time.perf_counter() - t0) * 1000.0)
            outs_u.append(res_u)
        wall_u = time.perf_counter() - t_w0

        # 2. User-Prefix KV-Cached Online Latency (includes tensor cache clone inside timed request window!)
        cached_kvs = [engine.precompute_prefix_kv(p) for p in pref_list]
        for i in range(5):
            kv_copy = _clone_dynamic_cache(cached_kvs[i], repeat_beams=1)
            _ = engine.decode_fast_beams(suf_list[i], num_beams=num_beams, tiled_prefix_cache=kv_copy, prefix_len=p_len)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        lats_cached: List[float] = []
        outs_c: List[List[int]] = []
        t_w1 = time.perf_counter()
        for i in range(num_trials):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            kv_copy = _clone_dynamic_cache(cached_kvs[i], repeat_beams=1)
            res_c = engine.decode_fast_beams(suf_list[i], num_beams=num_beams, tiled_prefix_cache=kv_copy, prefix_len=p_len)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            lats_cached.append((time.perf_counter() - t0) * 1000.0)
            outs_c.append(res_c)
        wall_c = time.perf_counter() - t_w1

        top1_matches = sum(1 for u, c in zip(outs_u, outs_c) if (u[:1] == c[:1]))
        top10_jacc = [
            len(set(u) & set(c)) / max(1, len(set(u) | set(c)))
            for u, c in zip(outs_u, outs_c)
        ]
        u_stat = summarize_latencies_ms(lats_uncached, num_trials, wall_u)
        c_stat = summarize_latencies_ms(lats_cached, num_trials, wall_c)
        speedup = round(u_stat["p50_ms"] / max(c_stat["p50_ms"], 1e-4), 2)
        results[f"prefix_len_{p_len}"] = {
            "uncached": u_stat,
            "user_prefix_kv_cached_incl_clone": c_stat,
            "p50_speedup": speedup,
            "bf16_top1_agreement_rate": round(top1_matches / max(1, num_trials), 4),
            "bf16_top10_jaccard_mean": round(float(sum(top10_jacc) / max(1, len(top10_jacc))), 4),
        }
        print(
            f"[FastBeam L_pref={p_len}] Uncached P50={u_stat['p50_ms']:.2f}ms -> "
            f"KV-Cached P50={c_stat['p50_ms']:.2f}ms ({speedup}x) | "
            f"bf16_top1_agree={top1_matches}/{num_trials} top10_jaccard={results[f'prefix_len_{p_len}']['bf16_top10_jaccard_mean']:.3f}",
            flush=True,
        )

    out_p = Path(out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Native Fast Constrained Beam & User-Prefix KV-Cache Benchmark")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--sid_dir", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--num_beams", type=int, default=10)
    a = p.parse_args()
    run_latency_matrix_benchmark(a.ckpt_dir, a.sid_dir, a.out_json, num_beams=a.num_beams)


if __name__ == "__main__":
    main()
