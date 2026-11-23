"""User-Prefix KV-Cache Inference Engine for Low-Latency Personalized Generative Search.

In personalized search, a user's profile + historical search queries + historical clicked
item SIDs (`user_prefix`, often 150-400 tokens) can be pre-encoded and cached across a
user's active search session, leaving only the short real-time query suffix (`~12 tokens`)
plus `L=4..5` constrained SID tokens to decode online.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from transformers.cache_utils import DynamicCache
from transformers.generation import LogitsProcessorList

from minionesearch.decoding.trie import GPUPrefixTrie, PrefixTrieLogitsProcessor
from minionesearch.evaluation import summarize_latencies_ms


def _clone_dynamic_cache(cache: DynamicCache, repeat_beams: int = 1) -> DynamicCache:
    """Fast zero-deepcopy `DynamicCache` cloning/tiling across `repeat_beams` for Beam Search."""
    if hasattr(cache, "to_legacy_cache") and hasattr(DynamicCache, "from_legacy_cache"):
        legacy = cache.to_legacy_cache()
        tiled = tuple(
            (
                k.repeat_interleave(repeat_beams, dim=0) if repeat_beams > 1 else k.clone(),
                v.repeat_interleave(repeat_beams, dim=0) if repeat_beams > 1 else v.clone(),
            )
            for k, v in legacy
        )
        return DynamicCache.from_legacy_cache(tiled)

    cloned = DynamicCache()
    if hasattr(cache, "layers"):
        for layer_idx, layer in enumerate(cache.layers):
            if getattr(layer, "keys", None) is not None and layer.keys.numel() > 0:
                k_c = layer.keys.repeat_interleave(repeat_beams, dim=0) if repeat_beams > 1 else layer.keys.clone()
                v_c = layer.values.repeat_interleave(repeat_beams, dim=0) if repeat_beams > 1 else layer.values.clone()
                cloned.update(k_c, v_c, layer_idx)
        return cloned

    cloned = copy.deepcopy(cache)
    if repeat_beams > 1 and hasattr(cloned, "batch_repeat_interleave"):
        cloned.batch_repeat_interleave(repeat_beams)
    return cloned


class UserPrefixKVCacheEngine:
    """Constrained Beam Search engine supporting both standard and User-Prefix KV-Cached inference."""

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
        self.level_token_ids = list(level_token_ids)
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self.vocab_size = int(model.get_input_embeddings().weight.shape[0])
        self.eos_id = int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id)

    @torch.no_grad()
    def precompute_user_prefix_cache(self, user_prefix_text: str) -> Tuple[torch.LongTensor, DynamicCache]:
        """Pre-encode static user profile + historical sequence into reusable `DynamicCache`."""
        prefix_ids = self.tokenizer(
            user_prefix_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.device)
        cache = DynamicCache()
        if prefix_ids.shape[1] > 0:
            _ = self.model(
                input_ids=prefix_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        return prefix_ids, cache

    @torch.no_grad()
    def generate_constrained_beams_single(
        self,
        full_prompt_ids: torch.LongTensor,
        num_beams: int = 10,
        past_key_values: Optional[DynamicCache] = None,
    ) -> List[int]:
        """Run exact Constrained Beam Search returning up to `num_beams` ranked in-corpus item IDs."""
        prompt_len = int(full_prompt_ids.shape[1])
        proc = PrefixTrieLogitsProcessor(
            trie=self.trie,
            level_token_ids=self.level_token_ids,
            eos_token_id=self.eos_id,
            vocab_size=self.vocab_size,
            prompt_len=prompt_len,
            renormalize_logprobs=True,
        )

        gen_kwargs: Dict[str, Any] = {
            "input_ids": full_prompt_ids,
            "max_new_tokens": self.trie.num_levels + 1,
            "min_new_tokens": self.trie.num_levels,
            "num_beams": num_beams,
            "num_return_sequences": num_beams,
            "do_sample": False,
            "early_stopping": True,
            "pad_token_id": self.eos_id,
            "eos_token_id": self.eos_id,
            "logits_processor": LogitsProcessorList([proc]),
        }
        if past_key_values is not None:
            gen_kwargs["past_key_values"] = _clone_dynamic_cache(past_key_values, repeat_beams=num_beams)

        seq_out = self.model.generate(**gen_kwargs)
        gen_only = seq_out[:, prompt_len : prompt_len + self.trie.num_levels]
        decoded_strs = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        return self.trie.decode_beams_to_ranked_items(decoded_strs, max_items=num_beams)

    @torch.no_grad()
    def benchmark_online_latency(
        self,
        prefix_suffix_pairs: Sequence[Tuple[str, str]],
        num_beams: int = 10,
        use_prefix_cache: bool = True,
        warmup: int = 3,
    ) -> Tuple[List[List[int]], Dict[str, float]]:
        """Profile online single-request P50/P95/P99 latency with vs. without User-Prefix KV Cache."""
        # Pre-warm user prefix caches outside online query arrival timer when `use_prefix_cache=True`
        prepared: List[Tuple[torch.LongTensor, Optional[DynamicCache]]] = []
        for u_pref, q_suf in prefix_suffix_pairs:
            if use_prefix_cache and u_pref:
                pref_ids, kv = self.precompute_user_prefix_cache(u_pref)
                suf_ids = self.tokenizer(q_suf, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
                full_ids = torch.cat([pref_ids, suf_ids], dim=1)
                prepared.append((full_ids, kv))
            else:
                full_ids = self.tokenizer(
                    u_pref + q_suf, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(self.device)
                prepared.append((full_ids, None))

        for i in range(min(warmup, len(prepared))):
            full_ids, kv = prepared[i]
            _ = self.generate_constrained_beams_single(full_ids, num_beams=num_beams, past_key_values=kv)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        all_preds: List[List[int]] = []
        latencies_ms: List[float] = []
        t_wall0 = time.perf_counter()

        for full_ids, kv in prepared:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            preds = self.generate_constrained_beams_single(full_ids, num_beams=num_beams, past_key_values=kv)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            all_preds.append(preds)

        total_wall = time.perf_counter() - t_wall0
        return all_preds, summarize_latencies_ms(latencies_ms, len(prepared), total_wall)
