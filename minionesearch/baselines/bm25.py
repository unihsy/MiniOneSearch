"""Hybrid Word + Character-Bigram Vectorized Sparse BM25 Retriever.

Designed for Chinese & alphanumeric e-commerce search queries and titles:
  - Combines `jieba` word tokens with character bigrams & alphanumeric tokens
    so exact model numbers (`vivox200pro`, `45*70cm`) and brand terms match reliably.
  - Uses `scipy.sparse.csr_matrix` pre-weighted BM25 document-term matrix so
    batched retrieval over 100K-1M items runs in seconds with zero C++ build issues.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Dict, List, Mapping, Sequence, Tuple

import jieba
import numpy as np
from scipy.sparse import csr_matrix

from minionesearch.evaluation import summarize_latencies_ms

jieba.setLogLevel(20)

_ALNUM_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize_zh_ecommerce(text: str, use_char_bigrams: bool = True) -> List[str]:
    """Hybrid tokenizer tailored for Chinese e-commerce search."""
    if not text:
        return []
    lowered = text.lower().strip()
    tokens: List[str] = []

    # 1. Word-level jieba segmentation (filtering pure punctuation/whitespace)
    for tok in jieba.cut_for_search(lowered):
        t = tok.strip()
        if not t or t in {"|", ":", "/", "【", "】", "（", "）", "(", ")", ",", "，", " "}:
            continue
        tokens.append(f"w:{t}")

    # 2. Alphanumeric model/spec tokens
    for m in _ALNUM_RE.findall(lowered):
        tokens.append(f"a:{m}")

    # 3. Character bigrams (and single CJK chars for short 1-2 char queries)
    if use_char_bigrams:
        for seg in _CJK_RE.findall(lowered):
            if len(seg) == 1:
                tokens.append(f"c:{seg}")
            else:
                for idx in range(len(seg) - 1):
                    tokens.append(f"b:{seg[idx : idx + 2]}")
    return tokens


class BM25Retriever:
    """Fast CSR-matrix BM25 retriever with exact Top-K ranking and latency profiling."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, use_char_bigrams: bool = True):
        self.k1 = k1
        self.b = b
        self.use_char_bigrams = use_char_bigrams
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.empty(0, dtype=np.float32)
        self.doc_term_bm25: csr_matrix | None = None
        self.item_ids: np.ndarray = np.empty(0, dtype=np.int64)

    def index_corpus(self, corpus_records: Sequence[Mapping]) -> None:
        """Build CSR BM25 index over `corpus_records` (`item_id`, `structured_text` or `item_title`)."""
        n_docs = len(corpus_records)
        self.item_ids = np.array([int(r["item_id"]) for r in corpus_records], dtype=np.int64)

        tokenized_docs: List[Counter] = []
        doc_lens = np.zeros(n_docs, dtype=np.float32)
        df_counter: Counter = Counter()

        for i, rec in enumerate(corpus_records):
            text = str(rec.get("structured_text") or rec.get("item_title") or "")
            toks = tokenize_zh_ecommerce(text, use_char_bigrams=self.use_char_bigrams)
            tf = Counter(toks)
            tokenized_docs.append(tf)
            doc_lens[i] = float(len(toks))
            for term in tf.keys():
                df_counter[term] += 1

        avgdl = float(np.mean(doc_lens)) if n_docs > 0 and np.mean(doc_lens) > 0 else 1.0
        self.vocab = {term: idx for idx, term in enumerate(df_counter.keys())}
        n_terms = len(self.vocab)
        self.idf = np.zeros(n_terms, dtype=np.float32)
        for term, idx in self.vocab.items():
            df = df_counter[term]
            # Lucene-style non-negative Robertson-Sparck Jones IDF
            self.idf[idx] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        for doc_idx, tf in enumerate(tokenized_docs):
            dl_norm = self.k1 * (1.0 - self.b + self.b * (doc_lens[doc_idx] / avgdl))
            for term, freq in tf.items():
                tid = self.vocab[term]
                weight = self.idf[tid] * ((freq * (self.k1 + 1.0)) / (freq + dl_norm))
                rows.append(doc_idx)
                cols.append(tid)
                vals.append(weight)

        # Store transposed term -> doc matrix `[n_terms, n_docs]` in CSR for ultra-fast row slicing per query!
        doc_term = csr_matrix(
            (np.asarray(vals, dtype=np.float32), (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
            shape=(n_docs, n_terms),
        )
        self.doc_term_bm25 = doc_term.tocsc().T.tocsr()  # shape: [n_terms, n_docs]

    def retrieve_sessions(
        self,
        sessions: Sequence[Mapping],
        top_k: int = 50,
        batch_size: int = 128,
    ) -> Tuple[Dict[str, List[int]], Dict[str, float]]:
        """Retrieve Top-K item IDs for each session and profile single-query / batch latency."""
        if self.doc_term_bm25 is None:
            raise RuntimeError("BM25Retriever.index_corpus() must be called before retrieve_sessions().")

        n_docs = len(self.item_ids)
        eff_k = min(top_k, n_docs)
        n_terms = len(self.vocab)
        predictions: Dict[str, List[int]] = {}
        per_query_latencies_ms: List[float] = []

        t_total_start = time.perf_counter()
        for start in range(0, len(sessions), batch_size):
            batch = sessions[start : start + batch_size]
            t_b0 = time.perf_counter()

            q_rows: List[int] = []
            q_cols: List[int] = []
            q_vals: List[float] = []
            for b_idx, sess in enumerate(batch):
                q_toks = Counter(tokenize_zh_ecommerce(str(sess["query"]), use_char_bigrams=self.use_char_bigrams))
                for term, qtf in q_toks.items():
                    tid = self.vocab.get(term)
                    if tid is not None:
                        q_rows.append(b_idx)
                        q_cols.append(tid)
                        q_vals.append(float(qtf))

            if q_rows:
                q_mat = csr_matrix(
                    (np.asarray(q_vals, dtype=np.float32), (np.asarray(q_rows, dtype=np.int32), np.asarray(q_cols, dtype=np.int32))),
                    shape=(len(batch), n_terms),
                )
                scores_sparse = q_mat @ self.doc_term_bm25  # CSR [B, n_docs]
                scores_dense = scores_sparse.toarray()
            else:
                scores_dense = np.zeros((len(batch), n_docs), dtype=np.float32)

            for b_idx, sess in enumerate(batch):
                row_scores = scores_dense[b_idx]
                if eff_k < n_docs:
                    part_idx = np.argpartition(-row_scores, eff_k - 1)[:eff_k]
                    order = part_idx[np.argsort(-row_scores[part_idx], kind="stable")]
                else:
                    order = np.argsort(-row_scores, kind="stable")
                predictions[str(sess["session_id"])] = self.item_ids[order].tolist()

            dt_ms = (time.perf_counter() - t_b0) * 1000.0 / max(1, len(batch))
            per_query_latencies_ms.extend([dt_ms] * len(batch))

        total_wall = time.perf_counter() - t_total_start
        latency_stats = summarize_latencies_ms(per_query_latencies_ms, len(sessions), total_wall)
        return predictions, latency_stats
