# MiniOneSearch V3 Complete Benchmark & Statistical Results

All metrics, slice breakdowns, latency profiles, multi-seed aggregations, and 10,000-resample user-clustered paired bootstrap tests in this document are deterministically compiled from the synchronized JSON artifacts in [./summaries/](./summaries/).

---

## 0. Experimental Protocol & Benchmark Overview

* **Total Evaluated Entries (72 Total = 48 Generative Runs + 24 Traditional Baseline Evaluations)**:
  * **KuaiSearch-Core5 (11K)**: 6 traditional baselines + 23 generative runs (7 SFT ablations/3-seed bases + 12 3-seed 4-algo RL runs + 4 RL mechanism/tokenizer controls).
  * **KuaiSearch-Mini (100K)**: 6 traditional baselines + 22 generative runs (4 orthogonal $2\times 2$ &lt;SID<sub>q</sub>&gt; × Dedup cells + 5 clean single-variable SFT ablations + 1 clean CRQKM SDFT base + 3 additional 0.6B / 4B / 8B scaling backbones + 9 RL algorithm/control/tokenizer runs).
  * **ControlledPool-500K (Single-Variable Pool Crowding Control)**: 6 traditional baselines + 1 generative SDFT run (holding Mini's exact 21,079 train / 1,110 val / 3,930 test sessions and encoders constant while injecting 400,000 unseen distractors into corpus.jsonl).
  * **KuaiSearch-Scale (500K)**: 6 traditional baselines + 2 generative runs (Scale500K_E3_SDFT + Scale500K_E4_SPTPMAGRPO).
* **Chronological Session Splits (use_rank_prehistory = False)**:
  * Splits are built from timestamped recall_lite sessions (hist_query_times < cur_time, hist_clicked_times / hist_purchased_times < cur_time).
  * Training-split Bayesian smoothed item prior: posterior_ctr = (train_pos_cnt + 1.0) / (train_imp_cnt + 25.0).
* **Dataset Summary Across All Four Benchmark Tiers**:

| Benchmark&nbsp;Tier | Corpus&nbsp;Items | train_pos&nbsp;>=&nbsp;5&nbsp;Items | Zero&#8209;Pos&nbsp;Items | Selected&nbsp;Users&nbsp;(Train&nbsp;/&nbsp;Val&nbsp;/&nbsp;Test) | Sessions&nbsp;(Train&nbsp;/&nbsp;Val&nbsp;/&nbsp;Test) | Test&nbsp;Cold&#8209;Start&nbsp;Session&nbsp;Share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **KuaiSearch&#8209;Core5&nbsp;((1,5)&#8209;Core)** | 11,155 | **11,155&nbsp;(100.00%,&nbsp;min=5,&nbsp;med=6)** | **0&nbsp;(0.00%)** | 37,411&nbsp;(37,411&nbsp;/&nbsp;732&nbsp;/&nbsp;427) | 68,718&nbsp;/&nbsp;1,002&nbsp;/&nbsp;540 | 0.00%&nbsp;(0&nbsp;/&nbsp;540) |
| **KuaiSearch&#8209;Mini&nbsp;(100K)** | 100,000 | 195 | 54,692 | 2,604&nbsp;(2,564&nbsp;/&nbsp;478&nbsp;/&nbsp;2,166) | 21,079&nbsp;/&nbsp;1,110&nbsp;/&nbsp;3,930 | 92.72%&nbsp;(3,644&nbsp;/&nbsp;3,930) |
| **ControlledPool&#8209;500K&nbsp;(Mini&nbsp;+&nbsp;400K&nbsp;Distractors)** | 500,000 | 195 | 454,692 | 2,604&nbsp;(2,564&nbsp;/&nbsp;478&nbsp;/&nbsp;2,166) | 21,079&nbsp;/&nbsp;1,110&nbsp;/&nbsp;3,930 | 92.72%&nbsp;(3,644&nbsp;/&nbsp;3,930) |
| **KuaiSearch&#8209;Scale&nbsp;(500K)** | 500,000 | 8,543 | 163,343 | 53,405&nbsp;(51,134&nbsp;/&nbsp;5,613&nbsp;/&nbsp;3,371) | 214,463&nbsp;/&nbsp;11,288&nbsp;/&nbsp;6,444 | 67.77%&nbsp;(4,367&nbsp;/&nbsp;6,444) |

---

## 1. KuaiSearch-Core5 (11,155 Strict Final-Train 5-Core Items, 540 Test Sessions / 427 Warm Users)

### 1.1 Multi-Seed (42, 43, 44) Summary (Mean ± Std)

| Model&nbsp;(seeds&nbsp;=&nbsp;42,&nbsp;43,&nbsp;44) | Recall@1 | Recall@10 | NDCG@10 | MRR@10 | IndepStrict@10 | head_exact&nbsp;NDCG@10 | pers_ambig&nbsp;NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **OneSearch&#8209;V2&nbsp;Joint&nbsp;SDFT&nbsp;(Base)** | 19.41%&nbsp;±&nbsp;0.29% | 56.92%&nbsp;±&nbsp;0.35% | 38.16%&nbsp;±&nbsp;0.15% | 33.61%&nbsp;±&nbsp;0.17% | 78.90%&nbsp;±&nbsp;0.18% | 41.13%&nbsp;±&nbsp;0.26% | 35.73%&nbsp;±&nbsp;0.88% |
| **+&nbsp;Listwise&nbsp;DPO&nbsp;(val&nbsp;rollback&nbsp;to&nbsp;SDFT&nbsp;on&nbsp;all&nbsp;3&nbsp;seeds)** | 19.41%&nbsp;±&nbsp;0.29% | 56.92%&nbsp;±&nbsp;0.35% | 38.16%&nbsp;±&nbsp;0.15% | 33.61%&nbsp;±&nbsp;0.17% | 78.90%&nbsp;±&nbsp;0.18% | 41.13%&nbsp;±&nbsp;0.26% | 35.73%&nbsp;±&nbsp;0.88% |
| **+&nbsp;Vanilla&nbsp;GRPO&nbsp;(2&#8209;step&nbsp;PPO)** | 19.37%&nbsp;±&nbsp;0.21% | 57.11%&nbsp;±&nbsp;0.26% | 38.23%&nbsp;±&nbsp;0.18% | 33.67%&nbsp;±&nbsp;0.21% | 78.96%&nbsp;±&nbsp;0.29% | 41.34%&nbsp;±&nbsp;0.62% | 35.73%&nbsp;±&nbsp;0.87% |
| **+&nbsp;TPMA&#8209;GRPO&nbsp;(2&#8209;step&nbsp;PPO,&nbsp;w_item=0.50)** | 19.22%&nbsp;±&nbsp;0.10% | 57.41%&nbsp;±&nbsp;0.60% | 38.35%&nbsp;±&nbsp;0.25% | 33.83%&nbsp;±&nbsp;0.18% | 79.16%&nbsp;±&nbsp;0.09% | 41.30%&nbsp;±&nbsp;0.37% | 35.87%&nbsp;±&nbsp;0.59% |
| **+&nbsp;SP&#8209;TPMA&#8209;GRPO&nbsp;(2&#8209;step&nbsp;PPO,&nbsp;w_item=0.50)** | 19.28%&nbsp;±&nbsp;0.25% | 57.09%&nbsp;±&nbsp;0.33% | 38.17%&nbsp;±&nbsp;0.19% | 33.64%&nbsp;±&nbsp;0.29% | 79.12%&nbsp;±&nbsp;0.20% | 41.18%&nbsp;±&nbsp;0.56% | 35.74%&nbsp;±&nbsp;0.86% |

> **Note on Listwise DPO on Core5**: Across all three independent seeds (42, 43, 44), offline Listwise DPO did not improve validation NDCG@10 over the OneSearch-V2 SDFT checkpoint, so validation checkpoint selection (best_val_ckpt_tag = step0_init) retained the SDFT base. All val_history curves are recorded in core5_11k_summary.json.

### 1.2 Full Core5 Overall Table (6 Baselines + 23 Generative Runs)

<details>
<summary><b>📊 Expand Full 29-Entry Core5 Overall Table (Scroll ↔)</b></summary>
<br/>

| Model&nbsp;/&nbsp;Variant | Recall@1 | Recall@5 | Recall@10 | NDCG@10 | MRR@10 | Rel@10 | StrictRel@10 | IndepRel@10 | IndepStrict@10 | Valid@10 | P50&nbsp;ms&nbsp;(B=1&nbsp;/&nbsp;Batch) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **BM25** | 9.40% | 25.59% | 35.96% | 21.87% | 18.31% | 2.318 | 74.28% | 2.201 | 77.87% | 100.00% | —&nbsp;/&nbsp;0.08 |
| **BGE&#8209;M3&nbsp;(Zero&#8209;Shot)** | 10.85% | 28.98% | 41.27% | 24.75% | 20.36% | 2.367 | 76.80% | 2.229 | 78.46% | 100.00% | 11.73&nbsp;/&nbsp;0.26 |
| **FT&#8209;BGE&#8209;M3&nbsp;(Query&#8209;Only)** | 13.09% | 38.09% | 51.73% | 31.89% | 26.75% | 2.493 | 82.20% | 2.292 | 82.89% | 100.00% | 12.71&nbsp;/&nbsp;0.27 |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Query+Hist)** | 16.71% | 43.18% | 56.85% | 36.22% | 30.78% | 2.480 | 82.02% | 2.285 | 82.35% | 100.00% | 13.42&nbsp;/&nbsp;0.48 |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Query+Hist)&nbsp;+&nbsp;CrossEncoder&#8209;Rerank@100** | 10.79% | 31.79% | 44.27% | 26.60% | 21.88% | 2.617 | 88.37% | 2.328 | 84.59% | 100.00% | 40.67&nbsp;/&nbsp;40.67 |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;+&nbsp;CE&nbsp;Score&#8209;Fusion@100&nbsp;(val&nbsp;α=0.70)** | 16.28% | 43.89% | 56.48% | 35.96% | 30.47% | 2.533 | 84.13% | 2.307 | 83.61% | 100.00% | 40.67&nbsp;/&nbsp;40.67 |
| **C5_CRQKM_R4_SPTPMAGRPO_s42** | 18.98% | 44.52% | 58.66% | 38.58% | 33.50% | 2.381 | 77.85% | 2.214 | 78.98% | 100.00% | — |
| **C5_R1_ListwiseDPO_s42** | 19.31% | 44.51% | 57.31% | 38.32% | 33.76% | 2.362 | 77.17% | 2.203 | 78.72% | 100.00% | — |
| **C5_R1_ListwiseDPO_s43** | 19.74% | 44.77% | 56.64% | 38.13% | 33.63% | 2.364 | 77.54% | 2.204 | 79.07% | 100.00% | — |
| **C5_R1_ListwiseDPO_s44** | 19.20% | 45.20% | 56.82% | 38.03% | 33.43% | 2.367 | 77.56% | 2.202 | 78.89% | 100.00% | — |
| **C5_R2_VanillaGRPO_s42** | 19.31% | 44.51% | 57.31% | 38.32% | 33.76% | 2.362 | 77.17% | 2.203 | 78.72% | 100.00% | — |
| **C5_R2_VanillaGRPO_s43** | 19.60% | 44.98% | 57.21% | 38.34% | 33.82% | 2.368 | 77.56% | 2.208 | 79.28% | 100.00% | — |
| **C5_R2_VanillaGRPO_s44** | 19.20% | 45.20% | 56.82% | 38.03% | 33.43% | 2.367 | 77.56% | 2.202 | 78.89% | 100.00% | — |
| **C5_R3_TPMAGRPO_s42** | 19.34% | 45.49% | 57.41% | 38.34% | 33.90% | 2.372 | 77.59% | 2.210 | 79.13% | 100.00% | — |
| **C5_R3_TPMAGRPO_s43** | 19.14% | 44.54% | 56.81% | 38.11% | 33.62% | 2.372 | 77.89% | 2.209 | 79.26% | 100.00% | — |
| **C5_R3_TPMAGRPO_s44** | 19.20% | 45.91% | 58.01% | 38.60% | 33.97% | 2.367 | 77.67% | 2.206 | 79.09% | 100.00% | — |
| **C5_R4_SPTPMAGRPO_s42** | 19.09% | 45.15% | 57.45% | 38.10% | 33.54% | 2.372 | 77.69% | 2.211 | 79.26% | 100.00% | — |
| **C5_R4_SPTPMAGRPO_s43** | 19.57% | 45.11% | 56.99% | 38.39% | 33.97% | 2.370 | 77.70% | 2.208 | 79.20% | 100.00% | — |
| **C5_R4_SPTPMAGRPO_s44** | 19.20% | 45.20% | 56.82% | 38.03% | 33.43% | 2.367 | 77.56% | 2.202 | 78.89% | 100.00% | — |
| **C5_RLAbl1_SFTAnchorOnly_s42** | 19.31% | 44.51% | 57.31% | 38.32% | 33.76% | 2.362 | 77.17% | 2.203 | 78.72% | 100.00% | — |
| **C5_RLAbl2_PureOnPolicy_NoInj** | 19.37% | 45.31% | 57.89% | 38.45% | 33.80% | 2.368 | 77.50% | 2.209 | 79.13% | 100.00% | — |
| **C5_RLAbl3_Unclipped_1Step_TPMA** | 19.37% | 45.34% | 57.42% | 38.28% | 33.71% | 2.366 | 77.26% | 2.206 | 78.93% | 100.00% | — |
| **C5_S0_DirectCE_NoStage12** | 20.12% | 44.18% | 56.39% | 38.06% | 33.54% | 2.347 | 76.33% | 2.193 | 78.07% | 100.00% | 149.32&nbsp;/&nbsp;43.83 |
| **C5_S1_CRQKM_SDFT_s42** | 18.33% | 44.74% | 57.98% | 37.91% | 32.94% | 2.375 | 77.63% | 2.210 | 78.70% | 100.00% | 149.07&nbsp;/&nbsp;45.05 |
| **C5_S2_OneSearchV1_CE_s42** | 18.06% | 43.97% | 55.46% | 36.85% | 32.07% | 2.373 | 78.00% | 2.212 | 79.52% | 100.00% | 159.19&nbsp;/&nbsp;44.91 |
| **C5_S3_ExplicitCoT_NoTrunc** | 15.39% | 34.65% | 48.84% | 31.09% | 26.89% | 2.315 | 74.83% | 2.177 | 77.17% | 100.00% | 140.70&nbsp;/&nbsp;116.36 |
| **C5_S4_OneSearchV2_SDFT_s42** | 19.31% | 44.51% | 57.31% | 38.32% | 33.76% | 2.362 | 77.17% | 2.203 | 78.72% | 100.00% | 145.16&nbsp;/&nbsp;44.94 |
| **C5_S4_OneSearchV2_SDFT_s43** | 19.74% | 44.77% | 56.64% | 38.13% | 33.63% | 2.364 | 77.54% | 2.204 | 79.07% | 100.00% | 132.88&nbsp;/&nbsp;43.93 |
| **C5_S4_OneSearchV2_SDFT_s44** | 19.20% | 45.20% | 56.82% | 38.03% | 33.43% | 2.367 | 77.56% | 2.202 | 78.89% | 100.00% | 152.66&nbsp;/&nbsp;43.94 |

</details>

### 1.3 Full Core5 Diagnostic Slice Breakdown

<details>
<summary><b>📊 Expand Full 29-Entry Core5 Slice Breakdown Table (Scroll ↔)</b></summary>
<br/>

| Model&nbsp;/&nbsp;Variant | head_exact&nbsp;(N@10&nbsp;/&nbsp;R@10) | personalized_ambiguous&nbsp;(N@10&nbsp;/&nbsp;R@10) | long_tail_complex&nbsp;(N@10&nbsp;/&nbsp;R@10) | cold_start_item&nbsp;(N@10&nbsp;/&nbsp;R@10) |
| --- | --- | --- | --- | --- |
| **BM25** | 21.04%&nbsp;/&nbsp;38.01% | 10.17%&nbsp;/&nbsp;19.92% | 25.87%&nbsp;/&nbsp;39.40% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **BGE&#8209;M3&nbsp;(Zero&#8209;Shot)** | 27.28%&nbsp;/&nbsp;47.90% | 13.22%&nbsp;/&nbsp;26.02% | 29.46%&nbsp;/&nbsp;45.91% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **FT&#8209;BGE&#8209;M3&nbsp;(Query&#8209;Only)** | 32.83%&nbsp;/&nbsp;53.97% | 22.14%&nbsp;/&nbsp;39.13% | 36.90%&nbsp;/&nbsp;56.73% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Query+Hist)** | 36.78%&nbsp;/&nbsp;58.96% | 30.89%&nbsp;/&nbsp;47.46% | 38.86%&nbsp;/&nbsp;59.28% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Query+Hist)&nbsp;+&nbsp;CrossEncoder&#8209;Rerank@100** | 27.08%&nbsp;/&nbsp;46.81% | 16.05%&nbsp;/&nbsp;28.46% | 31.57%&nbsp;/&nbsp;51.43% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;+&nbsp;CE&nbsp;Score&#8209;Fusion@100&nbsp;(val&nbsp;α=0.70)** | 36.56%&nbsp;/&nbsp;57.55% | 31.19%&nbsp;/&nbsp;48.98% | 38.89%&nbsp;/&nbsp;59.28% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_CRQKM_R4_SPTPMAGRPO_s42** | 39.47%&nbsp;/&nbsp;62.38% | 36.09%&nbsp;/&nbsp;55.54% | 39.17%&nbsp;/&nbsp;59.43% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_R1_ListwiseDPO_s42** | 40.95%&nbsp;/&nbsp;60.28% | 36.42%&nbsp;/&nbsp;52.64% | 37.61%&nbsp;/&nbsp;58.26% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_R1_ListwiseDPO_s43** | 41.43%&nbsp;/&nbsp;62.07% | 36.02%&nbsp;/&nbsp;51.07% | 38.48%&nbsp;/&nbsp;57.13% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_R1_ListwiseDPO_s44** | 41.01%&nbsp;/&nbsp;62.07% | 34.75%&nbsp;/&nbsp;51.42% | 37.17%&nbsp;/&nbsp;56.65% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_R2_VanillaGRPO_s42** | 40.95%&nbsp;/&nbsp;60.28% | 36.42%&nbsp;/&nbsp;52.64% | 37.61%&nbsp;/&nbsp;58.26% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_R2_VanillaGRPO_s43** | 42.05%&nbsp;/&nbsp;63.55% | 36.01%&nbsp;/&nbsp;52.74% | 38.49%&nbsp;/&nbsp;57.93% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_R2_VanillaGRPO_s44** | 41.01%&nbsp;/&nbsp;62.07% | 34.75%&nbsp;/&nbsp;51.42% | 37.17%&nbsp;/&nbsp;56.65% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_R3_TPMAGRPO_s42** | 40.89%&nbsp;/&nbsp;60.28% | 36.45%&nbsp;/&nbsp;51.58% | 37.52%&nbsp;/&nbsp;58.81% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_R3_TPMAGRPO_s43** | 41.44%&nbsp;/&nbsp;63.08% | 35.27%&nbsp;/&nbsp;51.63% | 38.88%&nbsp;/&nbsp;58.22% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_R3_TPMAGRPO_s44** | 41.58%&nbsp;/&nbsp;63.08% | 35.90%&nbsp;/&nbsp;52.59% | 37.39%&nbsp;/&nbsp;57.53% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_R4_SPTPMAGRPO_s42** | 40.73%&nbsp;/&nbsp;59.81% | 36.27%&nbsp;/&nbsp;51.73% | 36.92%&nbsp;/&nbsp;58.81% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_R4_SPTPMAGRPO_s43** | 41.80%&nbsp;/&nbsp;62.85% | 36.21%&nbsp;/&nbsp;52.85% | 38.77%&nbsp;/&nbsp;57.57% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_R4_SPTPMAGRPO_s44** | 41.01%&nbsp;/&nbsp;62.07% | 34.75%&nbsp;/&nbsp;51.42% | 37.17%&nbsp;/&nbsp;56.65% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_RLAbl1_SFTAnchorOnly_s42** | 40.95%&nbsp;/&nbsp;60.28% | 36.42%&nbsp;/&nbsp;52.64% | 37.61%&nbsp;/&nbsp;58.26% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_RLAbl2_PureOnPolicy_NoInj** | 41.24%&nbsp;/&nbsp;60.51% | 37.58%&nbsp;/&nbsp;53.46% | 36.79%&nbsp;/&nbsp;58.81% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_RLAbl3_Unclipped_1Step_TPMA** | 41.07%&nbsp;/&nbsp;60.28% | 36.51%&nbsp;/&nbsp;51.93% | 37.27%&nbsp;/&nbsp;58.59% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_S0_DirectCE_NoStage12** | 41.06%&nbsp;/&nbsp;60.90% | 35.21%&nbsp;/&nbsp;52.95% | 38.32%&nbsp;/&nbsp;57.24% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_S1_CRQKM_SDFT_s42** | 38.95%&nbsp;/&nbsp;61.45% | 34.89%&nbsp;/&nbsp;52.59% | 39.10%&nbsp;/&nbsp;59.65% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_S2_OneSearchV1_CE_s42** | 39.00%&nbsp;/&nbsp;60.44% | 35.66%&nbsp;/&nbsp;53.76% | 37.96%&nbsp;/&nbsp;56.14% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_S3_ExplicitCoT_NoTrunc** | 33.92%&nbsp;/&nbsp;56.62% | 26.41%&nbsp;/&nbsp;41.31% | 33.06%&nbsp;/&nbsp;48.90% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_S4_OneSearchV2_SDFT_s42** | 40.95%&nbsp;/&nbsp;60.28% | 36.42%&nbsp;/&nbsp;52.64% | 37.61%&nbsp;/&nbsp;58.26% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_S4_OneSearchV2_SDFT_s43** | 41.43%&nbsp;/&nbsp;62.07% | 36.02%&nbsp;/&nbsp;51.07% | 38.48%&nbsp;/&nbsp;57.13% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |
| **C5_S4_OneSearchV2_SDFT_s44** | 41.01%&nbsp;/&nbsp;62.07% | 34.75%&nbsp;/&nbsp;51.42% | 37.17%&nbsp;/&nbsp;56.65% | N/A&nbsp;(0&nbsp;cold&nbsp;items) |

</details>

### 1.4 Core5 SID-Bench (512-256-256-256-256, Shared 5-Loss Aligned Encoder + Zero-Shot Control)

| Tokenizer | CUR&nbsp;↑ | Norm&nbsp;PPL&nbsp;↑ | D3&nbsp;Coll.&nbsp;↓ | L5&nbsp;ICR&nbsp;(Unique)&nbsp;↑ | L5&nbsp;Coll.&nbsp;↓ | Max&nbsp;Bucket&nbsp;↓ | CatPurity&nbsp;(D1/D2/D3)&nbsp;↑ | QueryCoClick&nbsp;(D1/D2)&nbsp;↑ | ColdStart&nbsp;D2&nbsp;Acc&nbsp;↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| **RQ&#8209;OPQ&nbsp;(Zero&#8209;Shot&nbsp;BGE&#8209;M3,&nbsp;L3&#8209;Bal)** | 100.00% | 75.52% | 12.58% | 95.41% | 4.59% | 11 | 86.1%&nbsp;/&nbsp;90.8%&nbsp;/&nbsp;98.3% | 50.5%&nbsp;/&nbsp;10.1% | N/A |
| **Vanilla&nbsp;RQ&#8209;VAE&nbsp;(TIGER)** | 4.96% | 4.49% | 96.52% | 11.73% | 88.27% | 302 | 42.8%&nbsp;/&nbsp;39.3%&nbsp;/&nbsp;38.5% | 81.2%&nbsp;/&nbsp;61.1% | N/A |
| **Sinkhorn&nbsp;RQ&#8209;VAE&nbsp;(LC&#8209;Rec)** | 97.81% | 65.84% | 21.43% | 94.11% | 5.89% | 23 | 83.6%&nbsp;/&nbsp;91.5%&nbsp;/&nbsp;96.0% | 60.1%&nbsp;/&nbsp;20.6% | N/A |
| **RQ&#8209;KMeans** | 100.00% | 26.40% | 19.18% | 93.53% | 6.47% | 12 | 88.2%&nbsp;/&nbsp;93.4%&nbsp;/&nbsp;96.8% | 57.4%&nbsp;/&nbsp;12.4% | N/A |
| **Constrained&nbsp;RQ&#8209;KMeans&nbsp;(Full&#8209;Bal)** | 100.00% | 91.56% | 14.34% | 100.00% | 0.00% | 1 | 87.5%&nbsp;/&nbsp;95.8%&nbsp;/&nbsp;98.2% | 48.9%&nbsp;/&nbsp;11.6% | N/A |
| **RQ&#8209;OPQ&nbsp;(Full&#8209;Layer&nbsp;Balanced)** | 100.00% | 93.72% | 13.94% | 93.82% | 6.18% | 8 | 87.5%&nbsp;/&nbsp;95.6%&nbsp;/&nbsp;98.3% | 48.9%&nbsp;/&nbsp;12.2% | N/A |
| **RQ&#8209;OPQ&nbsp;(L3&#8209;Balanced,&nbsp;No&nbsp;KHQE)** | 100.00% | 76.69% | 16.69% | 93.65% | 6.35% | 17 | 88.2%&nbsp;/&nbsp;93.3%&nbsp;/&nbsp;97.6% | 57.4%&nbsp;/&nbsp;15.1% | N/A |
| **OneSearch&nbsp;KHQE&#8209;RQ&#8209;OPQ&nbsp;(Full)** | 89.96% | 67.26% | 26.13% | 93.65% | 6.35% | 7 | 88.7%&nbsp;/&nbsp;91.2%&nbsp;/&nbsp;96.4% | 78.0%&nbsp;/&nbsp;25.9% | N/A |
| **OneSearch&nbsp;KHQE&#8209;RQ&#8209;OPQ&nbsp;(Dedup,&nbsp;Zero&#8209;Coll)** | 89.96% | 65.89% | 26.13% | 100.00% | 0.00% | 1 | 88.7%&nbsp;/&nbsp;91.2%&nbsp;/&nbsp;96.4% | 78.0%&nbsp;/&nbsp;25.9% | N/A |

---

## 2. KuaiSearch-Mini (100,000 Open-Catalog Items, 3,930 Test Sessions / 2,166 Users)

### 2.1 Full Mini-100K Overall Table (6 Baselines + 22 Generative Runs)

<details>
<summary><b>📊 Expand Full 28-Entry Mini-100K Overall Table (Scroll ↔)</b></summary>
<br/>

| Model&nbsp;/&nbsp;Variant | Recall@1 | Recall@5 | Recall@10 | NDCG@10 | MRR@10 | Rel@10 | StrictRel@10 | IndepRel@10 | IndepStrict@10 | Valid@10 | P50&nbsp;ms&nbsp;(B=1&nbsp;/&nbsp;Batch) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **BM25** | 11.80% | 29.00% | 37.37% | 27.43% | 27.05% | 2.116 | 65.78% | 2.097 | 70.65% | 100.00% | —&nbsp;/&nbsp;0.46 |
| **BGE&#8209;M3&nbsp;(Zero&#8209;Shot)** | 12.71% | 30.45% | 38.90% | 28.90% | 28.78% | 2.193 | 68.27% | 2.129 | 70.06% | 100.00% | 12.07&nbsp;/&nbsp;0.26 |
| **FT&#8209;BGE&#8209;M3&nbsp;(Query&#8209;Only)** | 14.20% | 34.33% | 43.93% | 32.56% | 31.65% | 2.332 | 74.68% | 2.204 | 74.86% | 100.00% | 12.79&nbsp;/&nbsp;0.26 |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Query+Hist)** | 14.51% | 35.64% | 45.82% | 33.76% | 32.67% | 2.324 | 74.53% | 2.206 | 74.97% | 100.00% | 13.45&nbsp;/&nbsp;0.53 |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Query+Hist)&nbsp;+&nbsp;CrossEncoder&#8209;Rerank@100** | 14.34% | 33.85% | 43.31% | 32.23% | 31.83% | 2.501 | 82.24% | 2.254 | 77.49% | 100.00% | 40.06&nbsp;/&nbsp;40.06 |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;+&nbsp;CE&nbsp;Score&#8209;Fusion@100&nbsp;(val&nbsp;α=0.20)** | 15.82% | 37.10% | 47.03% | 35.18% | 34.44% | 2.475 | 80.45% | 2.256 | 77.57% | 100.00% | 40.06&nbsp;/&nbsp;40.06 |
| **Mini_A1_Clean_DirectCE** | 4.01% | 10.02% | 14.53% | 10.00% | 11.33% | 1.862 | 53.68% | 1.870 | 58.22% | 100.00% | 165.34&nbsp;/&nbsp;50.10 |
| **Mini_A2_Clean_NonPers** | 4.07% | 10.97% | 16.12% | 10.90% | 11.87% | 1.928 | 56.75% | 1.923 | 61.10% | 100.00% | 145.71&nbsp;/&nbsp;15.74 |
| **Mini_A3_Clean_TigerRandUID** | 4.33% | 11.80% | 17.15% | 11.64% | 12.68% | 1.931 | 56.98% | 1.924 | 61.25% | 100.00% | 157.54&nbsp;/&nbsp;48.93 |
| **Mini_A4_Clean_OneSearchV1CE** | 4.45% | 11.87% | 17.01% | 11.63% | 12.65% | 1.929 | 56.86% | 1.924 | 61.03% | 100.00% | 159.87&nbsp;/&nbsp;50.24 |
| **Mini_A5_Clean_ExplicitCoT** | 3.87% | 11.59% | 17.44% | 11.45% | 12.19% | 1.903 | 55.53% | 1.906 | 60.16% | 100.00% | 152.33&nbsp;/&nbsp;131.03 |
| **Mini_CRQKM_RL1_ListwiseDPO** | 4.90% | 14.54% | 20.64% | 13.88% | 14.66% | 1.973 | 58.91% | 1.955 | 62.69% | 100.00% | — |
| **Mini_CRQKM_RL4_SPTPMAGRPO** | 5.28% | 15.29% | 21.66% | 14.54% | 15.26% | 1.981 | 59.38% | 1.959 | 62.77% | 100.00% | — |
| **Mini_CRQKM_q0_dedup_SDFT** | 4.93% | 14.53% | 20.59% | 13.88% | 14.69% | 1.973 | 58.91% | 1.955 | 62.68% | 100.00% | 157.57&nbsp;/&nbsp;50.23 |
| **Mini_F1_KHQE_q1_raw_SDFT** | 3.48% | 9.18% | 13.52% | 9.15% | 10.45% | 1.909 | 55.51% | 1.905 | 60.02% | 100.00% | 149.30&nbsp;/&nbsp;53.09 |
| **Mini_F2_KHQE_q1_dedup_SDFT** | 3.79% | 9.97% | 15.21% | 10.08% | 10.89% | 1.926 | 56.51% | 1.921 | 60.98% | 100.00% | 158.03&nbsp;/&nbsp;53.13 |
| **Mini_F3_KHQE_q0_raw_SDFT** | 4.33% | 11.47% | 16.71% | 11.36% | 12.63% | 1.921 | 56.23% | 1.914 | 60.63% | 100.00% | 149.30&nbsp;/&nbsp;50.25 |
| **Mini_F4_KHQE_q0_dedup_SDFT** | 4.33% | 11.98% | 18.08% | 11.93% | 12.88% | 1.932 | 56.98% | 1.924 | 61.13% | 100.00% | 153.27&nbsp;/&nbsp;50.21 |
| **Mini_RL1_KHQE_ListwiseDPO** | 4.52% | 12.86% | 18.69% | 12.56% | 13.68% | 1.938 | 57.35% | 1.925 | 61.19% | 100.00% | — |
| **Mini_RL2_KHQE_VanillaGRPO** | 4.46% | 12.87% | 18.46% | 12.40% | 13.41% | 1.945 | 57.50% | 1.931 | 61.40% | 100.00% | — |
| **Mini_RL3_KHQE_TPMAGRPO** | 4.50% | 12.64% | 18.50% | 12.36% | 13.33% | 1.944 | 57.43% | 1.931 | 61.36% | 100.00% | — |
| **Mini_RL4_KHQE_SPTPMAGRPO** | 4.57% | 12.87% | 18.58% | 12.44% | 13.38% | 1.945 | 57.49% | 1.931 | 61.41% | 100.00% | — |
| **Mini_RLAbl1_SFTAnchorOnly** | 4.31% | 12.09% | 18.05% | 11.93% | 12.85% | 1.934 | 57.08% | 1.925 | 61.19% | 100.00% | — |
| **Mini_RLAbl2_PureNoGTInj** | 4.62% | 13.02% | 18.84% | 12.63% | 13.58% | 1.948 | 57.65% | 1.933 | 61.38% | 100.00% | — |
| **Mini_RLAbl3_Unclipped1Step** | 4.46% | 12.36% | 18.19% | 12.15% | 13.16% | 1.941 | 57.30% | 1.928 | 61.24% | 100.00% | — |
| **Mini_Scale_0.6B_CleanSDFT** | 3.34% | 10.36% | 15.74% | 10.26% | 11.23% | 1.909 | 55.93% | 1.906 | 60.11% | 100.00% | 151.79&nbsp;/&nbsp;34.20 |
| **Mini_Scale_4B_CleanSDFT** | 4.24% | 11.94% | 17.87% | 11.85% | 12.69% | 1.955 | 57.91% | 1.943 | 62.00% | 100.00% | 215.34&nbsp;/&nbsp;104.91 |
| **Mini_Scale_8B_CleanSDFT** | 4.58% | 13.01% | 18.81% | 12.66% | 13.69% | 1.962 | 58.20% | 1.949 | 62.23% | 100.00% | 224.39&nbsp;/&nbsp;151.14 |

</details>

### 2.2 Full Mini-100K Diagnostic Slice Breakdown

<details>
<summary><b>📊 Expand Full 28-Entry Mini-100K Slice Breakdown Table (Scroll ↔)</b></summary>
<br/>

| Model&nbsp;/&nbsp;Variant | head_exact&nbsp;(N@10&nbsp;/&nbsp;R@10) | personalized_ambiguous&nbsp;(N@10&nbsp;/&nbsp;R@10) | long_tail_complex&nbsp;(N@10&nbsp;/&nbsp;R@10) | cold_start_item&nbsp;(N@10&nbsp;/&nbsp;R@10) |
| --- | --- | --- | --- | --- |
| **BM25** | 7.31%&nbsp;/&nbsp;12.79% | 6.15%&nbsp;/&nbsp;9.35% | 31.30%&nbsp;/&nbsp;42.19% | 28.49%&nbsp;/&nbsp;38.43% |
| **BGE&#8209;M3&nbsp;(Zero&#8209;Shot)** | 6.62%&nbsp;/&nbsp;11.43% | 6.38%&nbsp;/&nbsp;9.97% | 33.12%&nbsp;/&nbsp;44.18% | 30.03%&nbsp;/&nbsp;40.09% |
| **FT&#8209;BGE&#8209;M3&nbsp;(Query&#8209;Only)** | 10.63%&nbsp;/&nbsp;17.04% | 9.00%&nbsp;/&nbsp;13.44% | 37.04%&nbsp;/&nbsp;49.43% | 33.51%&nbsp;/&nbsp;44.87% |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Query+Hist)** | 13.16%&nbsp;/&nbsp;21.42% | 11.28%&nbsp;/&nbsp;17.29% | 37.89%&nbsp;/&nbsp;50.92% | 34.58%&nbsp;/&nbsp;46.60% |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Query+Hist)&nbsp;+&nbsp;CrossEncoder&#8209;Rerank@100** | 8.69%&nbsp;/&nbsp;16.64% | 9.02%&nbsp;/&nbsp;14.11% | 36.66%&nbsp;/&nbsp;48.66% | 33.22%&nbsp;/&nbsp;44.04% |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;+&nbsp;CE&nbsp;Score&#8209;Fusion@100&nbsp;(val&nbsp;α=0.20)** | 12.66%&nbsp;/&nbsp;19.36% | 11.57%&nbsp;/&nbsp;17.67% | 39.58%&nbsp;/&nbsp;52.49% | 36.11%&nbsp;/&nbsp;47.87% |
| **Mini_A1_Clean_DirectCE** | 12.70%&nbsp;/&nbsp;17.11% | 11.00%&nbsp;/&nbsp;14.94% | 9.72%&nbsp;/&nbsp;14.26% | 8.25%&nbsp;/&nbsp;12.21% |
| **Mini_A2_Clean_NonPers** | 12.09%&nbsp;/&nbsp;16.77% | 7.78%&nbsp;/&nbsp;11.81% | 11.21%&nbsp;/&nbsp;16.57% | 9.50%&nbsp;/&nbsp;14.09% |
| **Mini_A3_Clean_TigerRandUID** | 12.94%&nbsp;/&nbsp;18.34% | 8.76%&nbsp;/&nbsp;13.05% | 11.92%&nbsp;/&nbsp;17.68% | 10.07%&nbsp;/&nbsp;15.01% |
| **Mini_A4_Clean_OneSearchV1CE** | 13.07%&nbsp;/&nbsp;18.76% | 9.30%&nbsp;/&nbsp;13.91% | 11.83%&nbsp;/&nbsp;17.35% | 9.94%&nbsp;/&nbsp;14.74% |
| **Mini_A5_Clean_ExplicitCoT** | 12.25%&nbsp;/&nbsp;17.40% | 9.01%&nbsp;/&nbsp;13.23% | 11.73%&nbsp;/&nbsp;18.07% | 10.37%&nbsp;/&nbsp;16.12% |
| **Mini_CRQKM_RL1_ListwiseDPO** | 12.04%&nbsp;/&nbsp;17.60% | 8.68%&nbsp;/&nbsp;13.69% | 14.56%&nbsp;/&nbsp;21.57% | 12.34%&nbsp;/&nbsp;18.37% |
| **Mini_CRQKM_RL4_SPTPMAGRPO** | 12.27%&nbsp;/&nbsp;18.10% | 9.02%&nbsp;/&nbsp;14.46% | 15.31%&nbsp;/&nbsp;22.71% | 13.04%&nbsp;/&nbsp;19.43% |
| **Mini_CRQKM_q0_dedup_SDFT** | 12.04%&nbsp;/&nbsp;17.60% | 8.62%&nbsp;/&nbsp;13.49% | 14.58%&nbsp;/&nbsp;21.54% | 12.36%&nbsp;/&nbsp;18.34% |
| **Mini_F1_KHQE_q1_raw_SDFT** | 10.90%&nbsp;/&nbsp;14.13% | 6.29%&nbsp;/&nbsp;9.40% | 9.36%&nbsp;/&nbsp;13.89% | 7.57%&nbsp;/&nbsp;11.27% |
| **Mini_F2_KHQE_q1_dedup_SDFT** | 12.31%&nbsp;/&nbsp;17.11% | 7.81%&nbsp;/&nbsp;11.37% | 10.17%&nbsp;/&nbsp;15.42% | 8.32%&nbsp;/&nbsp;12.71% |
| **Mini_F3_KHQE_q0_raw_SDFT** | 12.40%&nbsp;/&nbsp;16.27% | 8.47%&nbsp;/&nbsp;12.16% | 11.62%&nbsp;/&nbsp;17.13% | 9.80%&nbsp;/&nbsp;14.66% |
| **Mini_F4_KHQE_q0_dedup_SDFT** | 13.38%&nbsp;/&nbsp;19.49% | 9.24%&nbsp;/&nbsp;13.18% | 12.17%&nbsp;/&nbsp;18.53% | 10.36%&nbsp;/&nbsp;15.81% |
| **Mini_RL1_KHQE_ListwiseDPO** | 12.60%&nbsp;/&nbsp;18.33% | 9.25%&nbsp;/&nbsp;13.38% | 12.94%&nbsp;/&nbsp;19.21% | 11.16%&nbsp;/&nbsp;16.70% |
| **Mini_RL2_KHQE_VanillaGRPO** | 13.16%&nbsp;/&nbsp;18.88% | 9.45%&nbsp;/&nbsp;13.71% | 12.70%&nbsp;/&nbsp;18.94% | 10.92%&nbsp;/&nbsp;16.29% |
| **Mini_RL3_KHQE_TPMAGRPO** | 13.31%&nbsp;/&nbsp;19.13% | 9.63%&nbsp;/&nbsp;13.95% | 12.62%&nbsp;/&nbsp;18.96% | 10.85%&nbsp;/&nbsp;16.34% |
| **Mini_RL4_KHQE_SPTPMAGRPO** | 13.19%&nbsp;/&nbsp;18.97% | 9.56%&nbsp;/&nbsp;13.76% | 12.73%&nbsp;/&nbsp;19.09% | 10.93%&nbsp;/&nbsp;16.42% |
| **Mini_RLAbl1_SFTAnchorOnly** | 13.22%&nbsp;/&nbsp;19.43% | 9.11%&nbsp;/&nbsp;12.92% | 12.21%&nbsp;/&nbsp;18.54% | 10.35%&nbsp;/&nbsp;15.79% |
| **Mini_RLAbl2_PureNoGTInj** | 13.12%&nbsp;/&nbsp;19.23% | 9.87%&nbsp;/&nbsp;14.01% | 12.90%&nbsp;/&nbsp;19.28% | 11.22%&nbsp;/&nbsp;16.79% |
| **Mini_RLAbl3_Unclipped1Step** | 13.15%&nbsp;/&nbsp;19.02% | 9.36%&nbsp;/&nbsp;13.69% | 12.44%&nbsp;/&nbsp;18.64% | 10.64%&nbsp;/&nbsp;15.99% |
| **Mini_Scale_0.6B_CleanSDFT** | 12.76%&nbsp;/&nbsp;18.82% | 9.15%&nbsp;/&nbsp;14.16% | 10.18%&nbsp;/&nbsp;15.66% | 8.59%&nbsp;/&nbsp;13.09% |
| **Mini_Scale_4B_CleanSDFT** | 11.19%&nbsp;/&nbsp;17.17% | 8.28%&nbsp;/&nbsp;13.39% | 12.32%&nbsp;/&nbsp;18.46% | 10.29%&nbsp;/&nbsp;15.40% |
| **Mini_Scale_8B_CleanSDFT** | 11.99%&nbsp;/&nbsp;16.20% | 8.72%&nbsp;/&nbsp;12.63% | 13.20%&nbsp;/&nbsp;19.72% | 11.26%&nbsp;/&nbsp;16.91% |

</details>

### 2.3 Mini-100K SID-Bench (512-256-256-256-256, Shared 5-Loss Aligned Encoder)

| Tokenizer | CUR&nbsp;↑ | Norm&nbsp;PPL&nbsp;↑ | D3&nbsp;Coll.&nbsp;↓ | L5&nbsp;ICR&nbsp;(Unique)&nbsp;↑ | L5&nbsp;Coll.&nbsp;↓ | Max&nbsp;Bucket&nbsp;↓ | CatPurity&nbsp;(D1/D2/D3)&nbsp;↑ | QueryCoClick&nbsp;(D1/D2)&nbsp;↑ | ColdStart&nbsp;D2&nbsp;Acc&nbsp;↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| **Vanilla&nbsp;RQ&#8209;VAE&nbsp;(TIGER)** | 13.32% | 9.71% | 95.81% | 51.48% | 48.52% | 185 | 50.9%&nbsp;/&nbsp;48.5%&nbsp;/&nbsp;53.1% | 86.6%&nbsp;/&nbsp;54.7% | 47.60% |
| **Sinkhorn&nbsp;RQ&#8209;VAE&nbsp;(LC&#8209;Rec)** | 100.00% | 93.77% | 19.53% | 93.88% | 6.12% | 114 | 83.1%&nbsp;/&nbsp;86.6%&nbsp;/&nbsp;96.2% | 55.6%&nbsp;/&nbsp;13.1% | 75.71% |
| **RQ&#8209;KMeans** | 100.00% | 46.29% | 32.29% | 90.52% | 9.48% | 139 | 85.6%&nbsp;/&nbsp;84.7%&nbsp;/&nbsp;92.9% | 67.1%&nbsp;/&nbsp;21.3% | 76.52% |
| **Constrained&nbsp;RQ&#8209;KMeans&nbsp;(Full&#8209;Bal)** | 100.00% | 94.00% | 26.00% | 100.00% | 0.00% | 1 | 83.9%&nbsp;/&nbsp;87.1%&nbsp;/&nbsp;94.6% | 59.2%&nbsp;/&nbsp;18.7% | 76.72% |
| **RQ&#8209;OPQ&nbsp;(Full&#8209;Layer&nbsp;Balanced)** | 100.00% | 96.61% | 25.84% | 91.71% | 8.29% | 139 | 83.9%&nbsp;/&nbsp;87.1%&nbsp;/&nbsp;94.6% | 59.2%&nbsp;/&nbsp;17.4% | 76.53% |
| **RQ&#8209;OPQ&nbsp;(L3&#8209;Balanced,&nbsp;No&nbsp;KHQE)** | 100.00% | 84.33% | 30.03% | 91.22% | 8.78% | 139 | 85.6%&nbsp;/&nbsp;85.2%&nbsp;/&nbsp;93.5% | 67.1%&nbsp;/&nbsp;20.0% | 76.89% |
| **OneSearch&nbsp;KHQE&#8209;RQ&#8209;OPQ&nbsp;(Full)** | 94.61% | 80.43% | 30.73% | 91.53% | 8.47% | 139 | 90.4%&nbsp;/&nbsp;88.0%&nbsp;/&nbsp;94.7% | 67.4%&nbsp;/&nbsp;19.6% | 81.76% |
| **OneSearch&nbsp;KHQE&#8209;RQ&#8209;OPQ&nbsp;(Dedup,&nbsp;Zero&#8209;Coll)** | 94.61% | 78.71% | 30.73% | 100.00% | 0.00% | 1 | 90.4%&nbsp;/&nbsp;88.0%&nbsp;/&nbsp;94.7% | 67.4%&nbsp;/&nbsp;19.6% | 81.76% |

---

## 3. Single-Variable Pool Crowding Control (ControlledPool-500K: Identical Mini Sessions + 400K Distractors)

### 3.1 ControlledPool-500K Overall Table (n = 3,930 Test Sessions / 2,166 Users)

| Model&nbsp;/&nbsp;Variant | Recall@1 | Recall@5 | Recall@10 | NDCG@10 | MRR@10 | Rel@10 | StrictRel@10 | IndepRel@10 | IndepStrict@10 | Valid@10 | P50&nbsp;ms&nbsp;(B=1&nbsp;/&nbsp;Batch) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **BM25** | 6.52% | 16.51% | 22.12% | 15.67% | 15.63% | 2.281 | 72.90% | 2.225 | 76.84% | 100.00% | —&nbsp;/&nbsp;2.57 |
| **BGE&#8209;M3&nbsp;(Zero&#8209;Shot)** | 6.71% | 17.37% | 23.47% | 16.45% | 16.56% | 2.357 | 75.96% | 2.248 | 75.82% | 100.00% | 11.84&nbsp;/&nbsp;0.26 |
| **FT&#8209;BGE&#8209;M3&nbsp;(Query&#8209;Only)** | 7.65% | 19.75% | 26.62% | 18.78% | 18.45% | 2.477 | 81.63% | 2.305 | 79.99% | 100.00% | 12.71&nbsp;/&nbsp;0.27 |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Query+Hist)** | 7.90% | 20.64% | 28.20% | 19.71% | 19.32% | 2.468 | 81.36% | 2.310 | 80.29% | 100.00% | 13.16&nbsp;/&nbsp;0.54 |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Query+Hist)&nbsp;+&nbsp;CrossEncoder&#8209;Rerank@100** | 7.87% | 20.01% | 26.90% | 19.03% | 18.81% | 2.635 | 88.83% | 2.356 | 82.83% | 100.00% | 40.08&nbsp;/&nbsp;40.08 |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;+&nbsp;CE&nbsp;Score&#8209;Fusion@100&nbsp;(val&nbsp;α=0.15)** | 8.50% | 21.91% | 29.66% | 20.95% | 20.64% | 2.613 | 87.59% | 2.353 | 82.77% | 100.00% | 40.08&nbsp;/&nbsp;40.08 |
| **ControlledPool500K_KHQE_SDFT** | 3.31% | 8.75% | 11.68% | 8.31% | 9.52% | 2.009 | 60.33% | 1.987 | 64.10% | 100.00% | 154.96&nbsp;/&nbsp;50.40 |

### 3.2 ControlledPool-500K Diagnostic Slice Breakdown

| Model&nbsp;/&nbsp;Variant | head_exact&nbsp;(N@10&nbsp;/&nbsp;R@10) | personalized_ambiguous&nbsp;(N@10&nbsp;/&nbsp;R@10) | long_tail_complex&nbsp;(N@10&nbsp;/&nbsp;R@10) | cold_start_item&nbsp;(N@10&nbsp;/&nbsp;R@10) |
| --- | --- | --- | --- | --- |
| **BM25** | 2.84%&nbsp;/&nbsp;5.02% | 3.19%&nbsp;/&nbsp;4.89% | 18.13%&nbsp;/&nbsp;25.44% | 16.26%&nbsp;/&nbsp;22.86% |
| **BGE&#8209;M3&nbsp;(Zero&#8209;Shot)** | 3.07%&nbsp;/&nbsp;5.26% | 3.47%&nbsp;/&nbsp;5.62% | 18.97%&nbsp;/&nbsp;26.88% | 17.03%&nbsp;/&nbsp;24.24% |
| **FT&#8209;BGE&#8209;M3&nbsp;(Query&#8209;Only)** | 6.16%&nbsp;/&nbsp;8.81% | 4.63%&nbsp;/&nbsp;7.42% | 21.48%&nbsp;/&nbsp;30.27% | 19.32%&nbsp;/&nbsp;27.21% |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Query+Hist)** | 7.13%&nbsp;/&nbsp;10.88% | 6.47%&nbsp;/&nbsp;10.72% | 22.20%&nbsp;/&nbsp;31.61% | 20.07%&nbsp;/&nbsp;28.62% |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Query+Hist)&nbsp;+&nbsp;CrossEncoder&#8209;Rerank@100** | 4.24%&nbsp;/&nbsp;7.81% | 4.90%&nbsp;/&nbsp;7.64% | 21.76%&nbsp;/&nbsp;30.41% | 19.63%&nbsp;/&nbsp;27.47% |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;+&nbsp;CE&nbsp;Score&#8209;Fusion@100&nbsp;(val&nbsp;α=0.15)** | 6.77%&nbsp;/&nbsp;10.28% | 7.02%&nbsp;/&nbsp;11.95% | 23.60%&nbsp;/&nbsp;33.13% | 21.44%&nbsp;/&nbsp;30.04% |
| **ControlledPool500K_KHQE_SDFT** | 10.54%&nbsp;/&nbsp;14.61% | 8.63%&nbsp;/&nbsp;11.63% | 8.17%&nbsp;/&nbsp;11.56% | 6.87%&nbsp;/&nbsp;9.79% |

---

## 4. KuaiSearch-Scale (500,000 Items, 214,463 Train Sessions, 6,444 Test Sessions / 3,371 Users)

### 4.1 Scale-500K Overall Table

| Model&nbsp;/&nbsp;Variant | Recall@1 | Recall@5 | Recall@10 | NDCG@10 | MRR@10 | Rel@10 | StrictRel@10 | IndepRel@10 | IndepStrict@10 | Valid@10 | P50&nbsp;ms&nbsp;(B=1&nbsp;/&nbsp;Batch) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **BM25** | 6.12% | 15.43% | 21.00% | 14.66% | 14.46% | 2.299 | 73.75% | 2.229 | 77.13% | 100.00% | —&nbsp;/&nbsp;2.25 |
| **BGE&#8209;M3&nbsp;(Zero&#8209;Shot)** | 6.29% | 16.71% | 22.70% | 15.63% | 15.52% | 2.380 | 76.87% | 2.262 | 76.74% | 100.00% | 11.66&nbsp;/&nbsp;0.26 |
| **FT&#8209;BGE&#8209;M3&nbsp;(Query&#8209;Only)** | 7.60% | 20.05% | 27.23% | 18.75% | 18.14% | 2.488 | 82.29% | 2.300 | 80.39% | 100.00% | 12.33&nbsp;/&nbsp;0.26 |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Query+Hist)** | 8.19% | 21.81% | 30.12% | 20.66% | 20.03% | 2.476 | 81.86% | 2.292 | 80.09% | 100.00% | 13.11&nbsp;/&nbsp;0.52 |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Query+Hist)&nbsp;+&nbsp;CrossEncoder&#8209;Rerank@100** | 7.40% | 19.15% | 26.82% | 18.42% | 17.87% | 2.642 | 89.13% | 2.355 | 83.13% | 100.00% | 40.31&nbsp;/&nbsp;40.31 |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;+&nbsp;CE&nbsp;Score&#8209;Fusion@100&nbsp;(val&nbsp;α=0.30)** | 8.79% | 22.72% | 30.98% | 21.55% | 20.99% | 2.594 | 86.94% | 2.337 | 82.61% | 100.00% | 40.31&nbsp;/&nbsp;40.31 |
| **Scale500K_E3_SDFT** | 5.45% | 13.48% | 18.47% | 12.67% | 13.45% | 2.118 | 65.00% | 2.061 | 67.82% | 100.00% | 150.14&nbsp;/&nbsp;48.27 |
| **Scale500K_E4_SPTPMAGRPO** | 5.62% | 13.82% | 18.79% | 12.94% | 13.82% | 2.131 | 65.57% | 2.071 | 68.31% | 100.00% | — |

### 4.2 Scale-500K Diagnostic Slice Breakdown

| Model&nbsp;/&nbsp;Variant | head_exact&nbsp;(N@10&nbsp;/&nbsp;R@10) | personalized_ambiguous&nbsp;(N@10&nbsp;/&nbsp;R@10) | long_tail_complex&nbsp;(N@10&nbsp;/&nbsp;R@10) | cold_start_item&nbsp;(N@10&nbsp;/&nbsp;R@10) |
| --- | --- | --- | --- | --- |
| **BM25** | 3.55%&nbsp;/&nbsp;6.35% | 3.49%&nbsp;/&nbsp;5.90% | 19.84%&nbsp;/&nbsp;27.73% | 18.65%&nbsp;/&nbsp;25.78% |
| **BGE&#8209;M3&nbsp;(Zero&#8209;Shot)** | 4.33%&nbsp;/&nbsp;7.53% | 3.85%&nbsp;/&nbsp;6.45% | 21.22%&nbsp;/&nbsp;30.25% | 19.39%&nbsp;/&nbsp;27.21% |
| **FT&#8209;BGE&#8209;M3&nbsp;(Query&#8209;Only)** | 7.12%&nbsp;/&nbsp;11.87% | 5.78%&nbsp;/&nbsp;9.37% | 24.92%&nbsp;/&nbsp;35.28% | 22.58%&nbsp;/&nbsp;31.74% |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Query+Hist)** | 8.81%&nbsp;/&nbsp;14.48% | 8.96%&nbsp;/&nbsp;14.46% | 26.52%&nbsp;/&nbsp;37.83% | 24.78%&nbsp;/&nbsp;34.75% |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Query+Hist)&nbsp;+&nbsp;CrossEncoder&#8209;Rerank@100** | 5.70%&nbsp;/&nbsp;10.59% | 5.11%&nbsp;/&nbsp;8.73% | 24.82%&nbsp;/&nbsp;35.28% | 22.59%&nbsp;/&nbsp;31.51% |
| **FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;+&nbsp;CE&nbsp;Score&#8209;Fusion@100&nbsp;(val&nbsp;α=0.30)** | 8.78%&nbsp;/&nbsp;14.30% | 9.03%&nbsp;/&nbsp;14.42% | 27.84%&nbsp;/&nbsp;39.11% | 25.99%&nbsp;/&nbsp;36.05% |
| **Scale500K_E3_SDFT** | 16.27%&nbsp;/&nbsp;23.07% | 12.53%&nbsp;/&nbsp;17.16% | 12.13%&nbsp;/&nbsp;17.93% | 10.08%&nbsp;/&nbsp;14.34% |
| **Scale500K_E4_SPTPMAGRPO** | 16.11%&nbsp;/&nbsp;22.65% | 12.48%&nbsp;/&nbsp;17.04% | 12.62%&nbsp;/&nbsp;18.60% | 10.55%&nbsp;/&nbsp;14.78% |

### 4.3 Scale-500K SID-Bench (512-512-256-256-256)

| Tokenizer | CUR&nbsp;↑ | Norm&nbsp;PPL&nbsp;↑ | D3&nbsp;Coll.&nbsp;↓ | L5&nbsp;ICR&nbsp;(Unique)&nbsp;↑ | L5&nbsp;Coll.&nbsp;↓ | Max&nbsp;Bucket&nbsp;↓ | CatPurity&nbsp;(D1/D2/D3)&nbsp;↑ | QueryCoClick&nbsp;(D1/D2)&nbsp;↑ | ColdStart&nbsp;D2&nbsp;Acc&nbsp;↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| **OneSearch&nbsp;KHQE&#8209;RQ&#8209;OPQ&nbsp;(Full)** | 98.32% | 83.25% | 45.02% | 88.51% | 11.49% | 155 | 87.2%&nbsp;/&nbsp;85.0%&nbsp;/&nbsp;91.0% | 70.9%&nbsp;/&nbsp;18.9% | 79.82% |
| **OneSearch&nbsp;KHQE&#8209;RQ&#8209;OPQ&nbsp;(Dedup,&nbsp;Zero&#8209;Coll)** | 98.32% | 80.32% | 45.02% | 99.99% | 0.01% | 3 | 87.2%&nbsp;/&nbsp;85.0%&nbsp;/&nbsp;91.0% | 70.9%&nbsp;/&nbsp;18.9% | 79.82% |

---

## 5. Serving Latency & User-Prefix KV-Cache Break-Even Analysis (0.6B – 8B)

All NativeFastConstrainedBeamEngine (Beam@10, B=1) measurements below run in uncompiled PyTorch Eager mode on AMD MI300X (bfloat16), pass explicit position_ids for RoPE alignment, and **include _clone_dynamic_cache tensor cloning inside the timed request window**:

| Backbone&nbsp;(Beam@10,&nbsp;B=1) | Prefix&nbsp;Len&nbsp;L_pref | Uncached&nbsp;P50&nbsp;(ms) | +User&#8209;Prefix&nbsp;KV&#8209;Cache&nbsp;P50&nbsp;(incl.&nbsp;clone,&nbsp;ms) | Net&nbsp;Speedup | bf16&nbsp;Top&#8209;1&nbsp;Agreement | bf16&nbsp;Top&#8209;10&nbsp;Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Qwen3&#8209;0.6B&#8209;Base** | 128 | 114.17 | 121.01 | **0.94x** | 96.25% | 0.912 |
| **Qwen3&#8209;0.6B&#8209;Base** | 512 | 116.92 | 122.26 | **0.96x** | 96.25% | 0.919 |
| **Qwen3&#8209;0.6B&#8209;Base** | 1024 | 117.24 | 122.92 | **0.95x** | 95.00% | 0.902 |
| **Qwen3&#8209;1.7B&#8209;Base** | 128 | 117.83 | 126.28 | **0.93x** | 83.75% | 0.808 |
| **Qwen3&#8209;1.7B&#8209;Base** | 512 | 125.10 | 133.54 | **0.94x** | 87.50% | 0.803 |
| **Qwen3&#8209;1.7B&#8209;Base** | 1024 | 128.70 | 132.13 | **0.97x** | 86.25% | 0.810 |
| **Qwen3&#8209;4B&#8209;Base** | 128 | 155.76 | 166.28 | **0.94x** | 95.00% | 0.929 |
| **Qwen3&#8209;4B&#8209;Base** | 512 | 161.59 | 170.84 | **0.95x** | 93.75% | 0.901 |
| **Qwen3&#8209;4B&#8209;Base** | 1024 | 170.46 | 170.78 | **1.00x** | 97.50% | 0.902 |
| **Qwen3&#8209;8B&#8209;Base** | 128 | 143.18 | 147.45 | **0.97x** | 86.25% | 0.847 |
| **Qwen3&#8209;8B&#8209;Base** | 512 | 152.14 | 156.31 | **0.97x** | 91.25% | 0.821 |
| **Qwen3&#8209;8B&#8209;Base** | 1024 | 172.46 | 157.60 | **1.09x** | 90.00% | 0.820 |

---

## 6. User-Clustered Paired Bootstrap Significance Tests (10,000 Resamples + Holm-Bonferroni FWER Adjustment)

### Family H1 — Traditional Cascade Objective Collision (CrossEncoder-Rerank@100 vs. FT-BGE-M3-Pers)

| Hypothesis&nbsp;Comparison&nbsp;(A&nbsp;&#8209;&nbsp;B) | Metric&nbsp;(Slice) | Mean&nbsp;A | Mean&nbsp;B | Δ&nbsp;(A&nbsp;&#8209;&nbsp;B) | 95%&nbsp;User&#8209;Cluster&nbsp;CI | Raw&nbsp;p | Holm&nbsp;p | Users&nbsp;/&nbsp;Sessions |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| **Core5:&nbsp;CE&#8209;Rerank@100&nbsp;vs&nbsp;FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Overall&nbsp;NDCG@10)** | NDCG@10&nbsp;(overall) | 26.60% | 36.22% | **&#8209;9.63pt** | [&#8209;12.85pt,&nbsp;&#8209;6.41pt] | 0.0001 | **0.0004** | 427&nbsp;/&nbsp;540 |
| **Core5:&nbsp;CE&#8209;Rerank@100&nbsp;vs&nbsp;FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(pers_ambig&nbsp;Recall@10)** | Recall@10&nbsp;(personalized_ambiguous) | 28.46% | 47.46% | **&#8209;19.00pt** | [&#8209;28.10pt,&nbsp;&#8209;9.67pt] | 0.0001 | **0.0004** | 137&nbsp;/&nbsp;164 |
| **Mini&#8209;100K:&nbsp;CE&#8209;Rerank@100&nbsp;vs&nbsp;FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Overall&nbsp;NDCG@10)** | NDCG@10&nbsp;(overall) | 32.23% | 33.76% | **&#8209;1.53pt** | [&#8209;2.41pt,&nbsp;&#8209;0.65pt] | 0.0002 | **0.0004** | 2166&nbsp;/&nbsp;3930 |
| **Scale&#8209;500K:&nbsp;CE&#8209;Rerank@100&nbsp;vs&nbsp;FT&#8209;BGE&#8209;M3&#8209;Pers&nbsp;(Overall&nbsp;NDCG@10)** | NDCG@10&nbsp;(overall) | 18.42% | 20.66% | **&#8209;2.24pt** | [&#8209;2.88pt,&nbsp;&#8209;1.61pt] | 0.0001 | **0.0004** | 3371&nbsp;/&nbsp;6444 |

### Family H2 — Core5 Generative Search vs. Validation-Tuned Score-Fusion@100 (427 User Clusters / 540 Sessions)

| Hypothesis&nbsp;Comparison&nbsp;(A&nbsp;&#8209;&nbsp;B) | Metric&nbsp;(Slice) | Mean&nbsp;A | Mean&nbsp;B | Δ&nbsp;(A&nbsp;&#8209;&nbsp;B) | 95%&nbsp;User&#8209;Cluster&nbsp;CI | Raw&nbsp;p | Holm&nbsp;p | Users&nbsp;/&nbsp;Sessions |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| **Core5:&nbsp;CRQKM+SP&#8209;TPMA&#8209;GRPO&nbsp;vs&nbsp;Score&#8209;Fusion@100&nbsp;(NDCG@10)** | NDCG@10&nbsp;(overall) | 38.58% | 35.96% | **+2.63pt** | [+0.02pt,&nbsp;+5.21pt] | 0.0478 | **0.2390** | 427&nbsp;/&nbsp;540 |
| **Core5:&nbsp;CRQKM+SP&#8209;TPMA&#8209;GRPO&nbsp;vs&nbsp;Score&#8209;Fusion@100&nbsp;(MRR@10)** | MRR@10&nbsp;(overall) | 33.50% | 30.47% | **+3.03pt** | [+0.22pt,&nbsp;+5.81pt] | 0.0363 | **0.2178** | 427&nbsp;/&nbsp;540 |
| **Core5:&nbsp;CRQKM+SP&#8209;TPMA&#8209;GRPO&nbsp;vs&nbsp;Score&#8209;Fusion@100&nbsp;(pers_ambig&nbsp;NDCG@10)** | NDCG@10&nbsp;(personalized_ambiguous) | 36.09% | 31.19% | **+4.90pt** | [+0.01pt,&nbsp;+9.89pt] | 0.0525 | **0.2390** | 137&nbsp;/&nbsp;164 |
| **Core5:&nbsp;3&#8209;Seed&nbsp;TPMA&#8209;GRPO&nbsp;Mean&nbsp;vs&nbsp;Score&#8209;Fusion@100&nbsp;(NDCG@10)** | NDCG@10&nbsp;(overall) | 38.35% | 35.96% | **+2.39pt** | [&#8209;0.26pt,&nbsp;+4.95pt] | 0.0727 | **0.2390** | 427&nbsp;/&nbsp;540 |
| **Core5:&nbsp;3&#8209;Seed&nbsp;TPMA&#8209;GRPO&nbsp;Mean&nbsp;vs&nbsp;Score&#8209;Fusion@100&nbsp;(MRR@10)** | MRR@10&nbsp;(overall) | 33.83% | 30.47% | **+3.36pt** | [+0.51pt,&nbsp;+6.16pt] | 0.0233 | **0.1864** | 427&nbsp;/&nbsp;540 |
| **Core5:&nbsp;3&#8209;Seed&nbsp;TPMA&#8209;GRPO&nbsp;Mean&nbsp;vs&nbsp;Score&#8209;Fusion@100&nbsp;(Recall@1)** | Recall@1&nbsp;(overall) | 19.22% | 16.28% | **+2.94pt** | [&#8209;0.06pt,&nbsp;+5.88pt] | 0.0553 | **0.2390** | 427&nbsp;/&nbsp;540 |
| **Core5:&nbsp;PureOnPolicy&nbsp;SP&#8209;TPMA&#8209;GRPO&nbsp;vs&nbsp;Score&#8209;Fusion@100&nbsp;(NDCG@10)** | NDCG@10&nbsp;(overall) | 38.45% | 35.96% | **+2.49pt** | [&#8209;0.41pt,&nbsp;+5.20pt] | 0.0813 | **0.2390** | 427&nbsp;/&nbsp;540 |
| **Core5:&nbsp;PureOnPolicy&nbsp;SP&#8209;TPMA&#8209;GRPO&nbsp;vs&nbsp;Score&#8209;Fusion@100&nbsp;(pers_ambig&nbsp;NDCG@10)** | NDCG@10&nbsp;(personalized_ambiguous) | 37.58% | 31.19% | **+6.39pt** | [+0.92pt,&nbsp;+12.04pt] | 0.0266 | **0.1864** | 137&nbsp;/&nbsp;164 |

### Family H3 — Orthogonal 2×2 (&lt;SID<sub>q</sub>&gt; × Dedup) Factorial & OneSearch-V2 SDFT vs. Un-Truncated Explicit CoT

| Hypothesis&nbsp;Comparison&nbsp;(A&nbsp;&#8209;&nbsp;B) | Metric&nbsp;(Slice) | Mean&nbsp;A | Mean&nbsp;B | Δ&nbsp;(A&nbsp;&#8209;&nbsp;B) | 95%&nbsp;User&#8209;Cluster&nbsp;CI | Raw&nbsp;p | Holm&nbsp;p | Users&nbsp;/&nbsp;Sessions |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| **Mini&nbsp;2x2:&nbsp;Remove&nbsp;<SID_q>&nbsp;at&nbsp;Raw&nbsp;(F3&nbsp;&#8209;&nbsp;F1,&nbsp;Recall@10)** | Recall@10&nbsp;(overall) | 16.71% | 13.52% | **+3.19pt** | [+2.40pt,&nbsp;+3.99pt] | 0.0001 | **0.0008** | 2166&nbsp;/&nbsp;3930 |
| **Mini&nbsp;2x2:&nbsp;Remove&nbsp;<SID_q>&nbsp;at&nbsp;Dedup&nbsp;(F4&nbsp;&#8209;&nbsp;F2,&nbsp;Recall@10)** | Recall@10&nbsp;(overall) | 18.08% | 15.21% | **+2.87pt** | [+1.97pt,&nbsp;+3.77pt] | 0.0001 | **0.0008** | 2166&nbsp;/&nbsp;3930 |
| **Mini&nbsp;2x2:&nbsp;Leaf&nbsp;Dedup&nbsp;at&nbsp;q=1&nbsp;(F2&nbsp;&#8209;&nbsp;F1,&nbsp;Recall@10)** | Recall@10&nbsp;(overall) | 15.21% | 13.52% | **+1.69pt** | [+0.95pt,&nbsp;+2.43pt] | 0.0001 | **0.0008** | 2166&nbsp;/&nbsp;3930 |
| **Mini&nbsp;2x2:&nbsp;Leaf&nbsp;Dedup&nbsp;at&nbsp;q=0&nbsp;(F4&nbsp;&#8209;&nbsp;F3,&nbsp;Recall@10)** | Recall@10&nbsp;(overall) | 18.08% | 16.71% | **+1.37pt** | [+0.56pt,&nbsp;+2.19pt] | 0.0011 | **0.0033** | 2166&nbsp;/&nbsp;3930 |
| **Core5:&nbsp;ExplicitCoT_NoTrunc&nbsp;vs&nbsp;OneSearchV1_CE&nbsp;(NDCG@10)** | NDCG@10&nbsp;(overall) | 31.09% | 36.85% | **&#8209;5.76pt** | [&#8209;8.16pt,&nbsp;&#8209;3.38pt] | 0.0001 | **0.0008** | 427&nbsp;/&nbsp;540 |
| **Core5:&nbsp;3&#8209;Seed&nbsp;OneSearchV2_SDFT&nbsp;vs&nbsp;OneSearchV1_CE&nbsp;(NDCG@10)** | NDCG@10&nbsp;(overall) | 38.16% | 36.85% | **+1.31pt** | [&#8209;0.12pt,&nbsp;+2.73pt] | 0.0711 | **0.0711** | 427&nbsp;/&nbsp;540 |
| **Mini:&nbsp;OneSearchV2_SDFT&nbsp;(F4)&nbsp;vs&nbsp;OneSearchV1_CE&nbsp;(A4)&nbsp;(Recall@10)** | Recall@10&nbsp;(overall) | 18.08% | 17.01% | **+1.07pt** | [+0.29pt,&nbsp;+1.86pt] | 0.0082 | **0.0164** | 2166&nbsp;/&nbsp;3930 |
| **Mini:&nbsp;CRQKM_SDFT&nbsp;vs&nbsp;KHQE_SDFT&nbsp;(cold_start&nbsp;Recall@10)** | Recall@10&nbsp;(cold_start_item) | 18.34% | 15.81% | **+2.54pt** | [+1.30pt,&nbsp;+3.75pt] | 0.0001 | **0.0008** | 2049&nbsp;/&nbsp;3644 |

### Family H4 — Multi-Step PPO Search-RL Gains & Mechanism Decomposition Controls

| Hypothesis&nbsp;Comparison&nbsp;(A&nbsp;&#8209;&nbsp;B) | Metric&nbsp;(Slice) | Mean&nbsp;A | Mean&nbsp;B | Δ&nbsp;(A&nbsp;&#8209;&nbsp;B) | 95%&nbsp;User&#8209;Cluster&nbsp;CI | Raw&nbsp;p | Holm&nbsp;p | Users&nbsp;/&nbsp;Sessions |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| **Mini&nbsp;CRQKM:&nbsp;SP&#8209;TPMA&#8209;GRPO&nbsp;vs&nbsp;CRQKM_SDFT&nbsp;(NDCG@10)** | NDCG@10&nbsp;(overall) | 14.54% | 13.88% | **+0.66pt** | [+0.46pt,&nbsp;+0.86pt] | 0.0001 | **0.0008** | 2166&nbsp;/&nbsp;3930 |
| **Mini&nbsp;CRQKM:&nbsp;SP&#8209;TPMA&#8209;GRPO&nbsp;vs&nbsp;Listwise&nbsp;DPO&nbsp;(NDCG@10)** | NDCG@10&nbsp;(overall) | 14.54% | 13.88% | **+0.66pt** | [+0.47pt,&nbsp;+0.86pt] | 0.0001 | **0.0008** | 2166&nbsp;/&nbsp;3930 |
| **Mini&nbsp;KHQE:&nbsp;PureOnPolicy&nbsp;SP&#8209;TPMA&nbsp;vs&nbsp;SDFT&nbsp;(NDCG@10)** | NDCG@10&nbsp;(overall) | 12.63% | 11.93% | **+0.70pt** | [+0.47pt,&nbsp;+0.95pt] | 0.0001 | **0.0008** | 2166&nbsp;/&nbsp;3930 |
| **Mini&nbsp;KHQE:&nbsp;Hybrid&nbsp;SP&#8209;TPMA&nbsp;vs&nbsp;SDFT&nbsp;(NDCG@10)** | NDCG@10&nbsp;(overall) | 12.44% | 11.93% | **+0.51pt** | [+0.32pt,&nbsp;+0.70pt] | 0.0001 | **0.0008** | 2166&nbsp;/&nbsp;3930 |
| **Mini&nbsp;KHQE:&nbsp;SFT&#8209;Anchor&#8209;Only&nbsp;vs&nbsp;SDFT&nbsp;(NDCG@10)** | NDCG@10&nbsp;(overall) | 11.93% | 11.93% | **&#8209;0.01pt** | [&#8209;0.09pt,&nbsp;+0.08pt] | 0.9066 | **0.9066** | 2166&nbsp;/&nbsp;3930 |
| **Mini&nbsp;KHQE:&nbsp;2&#8209;Step&nbsp;Clipped&nbsp;TPMA&nbsp;vs&nbsp;1&#8209;Step&nbsp;Unclipped&nbsp;(Recall@10)** | Recall@10&nbsp;(overall) | 18.50% | 18.19% | **+0.31pt** | [+0.09pt,&nbsp;+0.55pt] | 0.0080 | **0.0240** | 2166&nbsp;/&nbsp;3930 |
| **Scale&#8209;500K:&nbsp;SP&#8209;TPMA&#8209;GRPO&nbsp;vs&nbsp;SDFT&nbsp;(NDCG@10)** | NDCG@10&nbsp;(overall) | 12.94% | 12.67% | **+0.27pt** | [+0.14pt,&nbsp;+0.41pt] | 0.0004 | **0.0016** | 3371&nbsp;/&nbsp;6444 |
| **Core5&nbsp;CRQKM:&nbsp;SP&#8209;TPMA&#8209;GRPO&nbsp;vs&nbsp;CRQKM_SDFT&nbsp;(NDCG@10)** | NDCG@10&nbsp;(overall) | 38.58% | 37.91% | **+0.67pt** | [&#8209;0.02pt,&nbsp;+1.41pt] | 0.0629 | **0.1258** | 427&nbsp;/&nbsp;540 |

### Family H5 — 500K Candidate-Pool Crowding & Collaborative Slice Crossover

| Hypothesis&nbsp;Comparison&nbsp;(A&nbsp;&#8209;&nbsp;B) | Metric&nbsp;(Slice) | Mean&nbsp;A | Mean&nbsp;B | Δ&nbsp;(A&nbsp;&#8209;&nbsp;B) | 95%&nbsp;User&#8209;Cluster&nbsp;CI | Raw&nbsp;p | Holm&nbsp;p | Users&nbsp;/&nbsp;Sessions |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| **ControlledPool&#8209;500K:&nbsp;Generative&nbsp;SDFT&nbsp;vs&nbsp;Score&#8209;Fusion@100&nbsp;(head_exact&nbsp;NDCG@10)** | NDCG@10&nbsp;(head_exact) | 10.54% | 6.77% | **+3.78pt** | [+1.05pt,&nbsp;+6.63pt] | 0.0080 | **0.0160** | 245&nbsp;/&nbsp;314 |
| **ControlledPool&#8209;500K:&nbsp;Generative&nbsp;SDFT&nbsp;vs&nbsp;Score&#8209;Fusion@100&nbsp;(pers_ambig&nbsp;NDCG@10)** | NDCG@10&nbsp;(personalized_ambiguous) | 8.63% | 7.02% | **+1.61pt** | [&#8209;0.46pt,&nbsp;+3.70pt] | 0.1239 | **0.1239** | 384&nbsp;/&nbsp;485 |
| **Scale&#8209;500K:&nbsp;SP&#8209;TPMA&#8209;GRPO&nbsp;vs&nbsp;Score&#8209;Fusion@100&nbsp;(head_exact&nbsp;NDCG@10)** | NDCG@10&nbsp;(head_exact) | 16.11% | 8.78% | **+7.33pt** | [+5.50pt,&nbsp;+9.21pt] | 0.0001 | **0.0005** | 896&nbsp;/&nbsp;1179 |
| **Scale&#8209;500K:&nbsp;SP&#8209;TPMA&#8209;GRPO&nbsp;vs&nbsp;Score&#8209;Fusion@100&nbsp;(pers_ambig&nbsp;NDCG@10)** | NDCG@10&nbsp;(personalized_ambiguous) | 12.48% | 9.03% | **+3.45pt** | [+2.01pt,&nbsp;+4.90pt] | 0.0001 | **0.0005** | 972&nbsp;/&nbsp;1362 |
| **Scale&#8209;500K:&nbsp;SP&#8209;TPMA&#8209;GRPO&nbsp;vs&nbsp;Score&#8209;Fusion@100&nbsp;(cold_start&nbsp;NDCG@10)** | NDCG@10&nbsp;(cold_start_item) | 10.55% | 25.99% | **&#8209;15.44pt** | [&#8209;16.64pt,&nbsp;&#8209;14.27pt] | 0.0001 | **0.0005** | 2368&nbsp;/&nbsp;4367 |
