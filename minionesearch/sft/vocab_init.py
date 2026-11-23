"""Semantic Lexicon-Guided Initialization for Newly Added SID Tokens on Pretrained LLMs."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Sequence

import torch

from minionesearch.decoding.trie import SIDVocabulary


@torch.no_grad()
def initialize_sid_embeddings_semantic(
    model: Any,
    tokenizer: Any,
    sid_vocab: SIDVocabulary,
    corpus_records: Sequence[Mapping[str, Any]],
    item_to_codes: Mapping[int, Sequence[int]],
    noise_std: float = 1e-3,
) -> List[torch.LongTensor]:
    """Register SID tokens & warm-initialize their input/output embeddings from cluster category/title lexicons.

    Why this matters:
      Random Gaussian embeddings on 1,280 newly added tokens cause large initial logit spikes.
      Initializing each `<level_code>` token from the mean pretrained subword embeddings of the
      dominant category/brand words in that cluster (plus tiny symmetry-breaking noise) grounds
      SID tokens on the pretrained LM's semantic manifold from step 0!
    """
    orig_vocab_size = len(tokenizer)
    level_token_ids = sid_vocab.register_with_tokenizer(tokenizer)
    new_vocab_size = len(tokenizer)

    if new_vocab_size > model.get_input_embeddings().weight.shape[0]:
        try:
            model.resize_token_embeddings(new_vocab_size, mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(new_vocab_size)

    in_emb = model.get_input_embeddings().weight.data
    out_head = model.get_output_embeddings()
    out_emb = out_head.weight.data if (out_head is not None and out_head.weight.shape[0] == new_vocab_size) else None

    base_mean = in_emb[:orig_vocab_size].float().mean(dim=0)
    base_out_mean = out_emb[:orig_vocab_size].float().mean(dim=0) if out_emb is not None else None

    # Collect dominant category/brand strings for each `(level, code)` bucket
    bucket_phrases: Dict[tuple, Counter] = defaultdict(Counter)
    for rec in corpus_records:
        iid = int(rec["item_id"])
        codes = item_to_codes.get(iid)
        if codes is None:
            continue
        phrase_parts = [
            str(rec.get("category_level1_name") or ""),
            str(rec.get("category_level2_name") or ""),
            str(rec.get("category_level3_name") or ""),
            str(rec.get("brand_name") or ""),
        ]
        phrase = " ".join([p for p in phrase_parts if p]).strip()
        if not phrase:
            phrase = str(rec.get("item_title") or "")[:16]
        for lvl, c in enumerate(codes):
            bucket_phrases[(lvl, int(c))][phrase] += 1

    g = torch.Generator(device=in_emb.device)
    g.manual_seed(42)

    for lvl, tids in enumerate(level_token_ids):
        for code_idx, global_tid_t in enumerate(tids):
            gtid = int(global_tid_t.item())
            if gtid < orig_vocab_size:
                continue
            top_phrases = [p for p, _ in bucket_phrases.get((lvl, code_idx), Counter()).most_common(4)]
            sub_ids: List[int] = []
            for ph in top_phrases:
                ids = tokenizer(ph, add_special_tokens=False)["input_ids"]
                sub_ids.extend([i for i in ids if 0 <= i < orig_vocab_size])

            noise = noise_std * torch.randn(base_mean.shape, device=in_emb.device, dtype=torch.float32, generator=g)
            if sub_ids:
                idx_t = torch.tensor(sub_ids[:32], device=in_emb.device, dtype=torch.long)
                sem_vec = in_emb[idx_t].float().mean(dim=0)
                init_in = 0.7 * sem_vec + 0.3 * base_mean + noise
            else:
                init_in = base_mean + noise
            in_emb[gtid] = init_in.to(in_emb.dtype)

            if out_emb is not None and out_emb.data_ptr() != in_emb.data_ptr() and base_out_mean is not None:
                if sub_ids:
                    idx_t = torch.tensor(sub_ids[:32], device=out_emb.device, dtype=torch.long)
                    sem_out = out_emb[idx_t].float().mean(dim=0)
                    init_out = 0.7 * sem_out + 0.3 * base_out_mean + noise.to(out_emb.device)
                else:
                    init_out = base_out_mean + noise.to(out_emb.device)
                out_emb[gtid] = init_out.to(out_emb.dtype)

    return level_token_ids
