"""Unified Generative Search SFT, Progressive Multi-Task Alignment & Joint Self-Distillation (SDFT)."""

from .dataset import (
    PromptBuilder,
    SFTMultiStageDataset,
    build_stage1_alignment_records,
    build_stage2_cooccurrence_records,
    build_stage3_personalized_records,
)
from .sft_trainer import JointSDFTConfig, compute_completion_kl_loss, train_generative_search_sft
from .vocab_init import initialize_sid_embeddings_semantic

__all__ = [
    "JointSDFTConfig",
    "PromptBuilder",
    "SFTMultiStageDataset",
    "build_stage1_alignment_records",
    "build_stage2_cooccurrence_records",
    "build_stage3_personalized_records",
    "compute_completion_kl_loss",
    "initialize_sid_embeddings_semantic",
    "train_generative_search_sft",
]
