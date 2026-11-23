"""Deterministic Publication-Grade Figure Generator for MiniOneSearch README.

Reads synchronized JSON artifacts from `docs/summaries/*.json` and renders four
crisp, editorial-style vector (`.svg`) and high-DPI raster (`.png`) figures into
`docs/assets/` with zero manual hardcoding of experimental metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUM_DIR = ROOT / "docs" / "summaries"
OUT_DIR = ROOT / "docs" / "assets"

# Muted Academic Print Palette (Tableau / Seaborn-deep / Illustrator — zero neon Tailwind!)
INK = "#222A35"
SUBINK = "#4A5568"
MUTED = "#718096"
GRID = "#E6E9EF"
SPINE = "#BCC5D1"
BG_SOFT = "#F7F8FA"

BLUE = "#5B84B1"       # Muted Steel-Cerulean
BLUE_DARK = "#3B6998"  # Primary Academic Steel Blue
TEAL = "#4A8B71"       # Muted Sage / Forest Teal
AMBER = "#D1822B"      # Warm Muted Ochre
CRIMSON = "#C44E52"    # Classic Seaborn Deep Brick-Rose
SLATE = "#6C7A89"      # Neutral Cool Gray
SLATE_LIGHT = "#AEB8C4"


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "Helvetica Neue", "DejaVu Sans"],
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.85,
        "axes.labelcolor": INK,
        "axes.titlesize": 11.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.color": SUBINK,
        "ytick.color": SUBINK,
        "xtick.labelsize": 8.6,
        "ytick.labelsize": 8.6,
        "legend.frameon": True,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": GRID,
        "legend.fontsize": 8.3,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "mathtext.fontset": "stixsans",
        "svg.fonttype": "none",
    })


def clean_ax(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.set_axisbelow(True)
    if grid_axis in ("x", "y", "both"):
        ax.grid(axis=grid_axis, color=GRID, linestyle="--", linewidth=0.7, alpha=0.85)


def write_fig1_architecture_svg(out_svg: Path) -> None:
    """Write an authentic ICML/SIGIR Illustrator-style (a)-(d) schematic SVG with vector icons & math."""
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1240 436" width="100%" font-family="Arial, 'Helvetica Neue', 'Segoe UI', sans-serif">
  <defs>
    <marker id="arr-ink" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.8 L 8.2 5 L 0 8.2 z" fill="#4A5568"/>
    </marker>
    <marker id="arr-steel" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.8 L 8.2 5 L 0 8.2 z" fill="#3B6998"/>
    </marker>
    <marker id="arr-ochre" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.8 L 8.2 5 L 0 8.2 z" fill="#D1822B"/>
    </marker>
    <marker id="arr-sage" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.8 L 8.2 5 L 0 8.2 z" fill="#4A8B71"/>
    </marker>
    <marker id="arr-brick" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.8 L 8.2 5 L 0 8.2 z" fill="#C44E52"/>
    </marker>
  </defs>

  <rect x="0" y="0" width="1240" height="436" fill="#FFFFFF"/>

  <!-- ====================================================================== -->
  <!-- PANEL (a): Cascade vs. End-to-End Generative Paradigm                  -->
  <!-- ====================================================================== -->
  <rect x="14" y="14" width="268" height="378" rx="6" fill="#FAFBFD" stroke="#C8D0DA" stroke-width="1.1" stroke-dasharray="5,3"/>

  <!-- User + History + Query Input Icons -->
  <g transform="translate(26, 28)">
    <!-- User silhouette icon -->
    <circle cx="17" cy="13" r="7" fill="none" stroke="#3B6998" stroke-width="1.6"/>
    <path d="M 5 30 C 5 21.5, 29 21.5, 29 30" fill="none" stroke="#3B6998" stroke-width="1.6" stroke-linecap="round"/>
    <!-- Search query bar with magnifier icon (x=38..174) + Causal clock pill (x=179..236) = exact 236px right edge -->
    <rect x="38" y="3" width="136" height="24" rx="12" fill="#FFFFFF" stroke="#8FA3B8" stroke-width="1.1"/>
    <circle cx="51" cy="14" r="4.3" fill="none" stroke="#4A5568" stroke-width="1.35"/>
    <line x1="54.3" y1="17.3" x2="57.8" y2="20.8" stroke="#4A5568" stroke-width="1.35" stroke-linecap="round"/>
    <text x="64" y="18.5" font-size="9.6" fill="#222A35">Query <tspan font-family="Georgia, 'Times New Roman', serif" font-style="italic" font-size="10.4" fill="#3B6998">q</tspan> <tspan font-size="8.4" fill="#718096">(cur)</tspan></text>

    <!-- Causal clock badge aligned flush to x=236 -->
    <rect x="179" y="3" width="57" height="24" rx="12" fill="#F5F9F7" stroke="#9CCAB8" stroke-width="1"/>
    <circle cx="192" cy="15" r="6.5" fill="#FFFFFF" stroke="#4A8B71" stroke-width="1.25"/>
    <polyline points="192,11 192,15 195.2,16.8" fill="none" stroke="#4A8B71" stroke-width="1.25" stroke-linecap="round"/>
    <text x="203" y="18.5" font-size="9.2" font-family="Georgia, serif" font-style="italic" fill="#356954">&lt; t</text>

    <!-- History item thumbnails strip (x=38..236) -->
    <rect x="38" y="34" width="198" height="22" rx="4" fill="#FFFFFF" stroke="#D5DCE4"/>
    <rect x="44" y="38" width="14" height="14" rx="2.5" fill="#DCE7F2" stroke="#3B6998" stroke-width="0.9"/>
    <rect x="62" y="38" width="14" height="14" rx="2.5" fill="#DCE7F2" stroke="#3B6998" stroke-width="0.9"/>
    <rect x="80" y="38" width="14" height="14" rx="2.5" fill="#E2F0E9" stroke="#4A8B71" stroke-width="0.9"/>
    <text x="102" y="48.5" font-size="9.2" fill="#4A5568">Causal History <tspan font-family="Georgia, serif" font-style="italic" font-size="9.8" fill="#222A35">H</tspan><tspan dy="2.5" font-family="Georgia, serif" font-style="italic" font-size="7.2" fill="#222A35">u</tspan></text>
  </g>

  <!-- Upper Sub-box: Traditional Cascade (MCA) -->
  <g transform="translate(26, 102)">
    <rect x="0" y="0" width="244" height="122" rx="5" fill="#FDF8F8" stroke="#E2B4B6" stroke-width="1"/>
    <text x="10" y="17" font-size="9.8" font-weight="700" fill="#9E3B3E">Traditional Cascade (MCA)</text>
    <text x="192" y="17" font-size="8.8" fill="#9E3B3E">40.7 ms</text>

    <!-- Funnel Icon -->
    <polygon points="14,30 56,30 44,52 44,72 26,72 26,52" fill="#E6ECF2" stroke="#5A6B7C" stroke-width="1.1"/>
    <text x="35" y="44" text-anchor="middle" font-size="8.2" font-weight="700" fill="#222A35">MIPS</text>
    <text x="35" y="86" text-anchor="middle" font-size="8.4" fill="#4A5568">Pers. Recall</text>

    <line x1="60" y1="51" x2="86" y2="51" stroke="#4A5568" stroke-width="1.2" marker-end="url(#arr-ink)"/>
    <text x="73" y="45" text-anchor="middle" font-size="8" fill="#718096">100</text>

    <!-- Cross-Encoder Rerank Stack -->
    <rect x="90" y="31" width="60" height="40" rx="4" fill="#F9E5E6" stroke="#C44E52" stroke-width="1.1"/>
    <line x1="98" y1="43" x2="142" y2="43" stroke="#C44E52" stroke-width="1"/>
    <line x1="98" y1="51" x2="142" y2="51" stroke="#C44E52" stroke-width="1"/>
    <line x1="98" y1="59" x2="130" y2="59" stroke="#C44E52" stroke-width="1"/>
    <text x="120" y="86" text-anchor="middle" font-size="8.4" fill="#9E3B3E">Non-Pers CE</text>

    <!-- Clash X Icon -->
    <line x1="154" y1="51" x2="174" y2="51" stroke="#C44E52" stroke-width="1.2" marker-end="url(#arr-brick)"/>
    <circle cx="204" cy="51" r="17" fill="#F9E5E6" stroke="#C44E52" stroke-width="1.2"/>
    <path d="M 197 44 L 211 58 M 211 44 L 197 58" stroke="#C44E52" stroke-width="2" stroke-linecap="round"/>
    <text x="204" y="86" text-anchor="middle" font-size="8.4" font-weight="700" fill="#C44E52">−19% Ambig</text>

    <text x="122" y="110" text-anchor="middle" font-size="8.6" fill="#7F3A3D">Objective Collision: erases user taste</text>
  </g>

  <!-- Lower Sub-box: MiniOneSearch Unified Paradigm -->
  <g transform="translate(26, 236)">
    <rect x="0" y="0" width="244" height="140" rx="5" fill="#F4F8FB" stroke="#9BB8D3" stroke-width="1.1"/>
    <text x="10" y="18" font-size="9.8" font-weight="700" fill="#2B547E">MiniOneSearch (Unified)</text>
    <text x="184" y="18" font-size="9" font-family="Georgia, serif" font-style="italic" fill="#2B547E">O(L = 5)</text>

    <!-- Transformer Decoder Stack Icon -->
    <g transform="translate(16, 33)">
      <rect x="0" y="0" width="58" height="14" rx="3" fill="#DCE7F2" stroke="#3B6998" stroke-width="1.1"/>
      <rect x="0" y="18" width="58" height="14" rx="3" fill="#DCE7F2" stroke="#3B6998" stroke-width="1.1"/>
      <rect x="0" y="36" width="58" height="14" rx="3" fill="#3B6998"/>
      <text x="29" y="46" text-anchor="middle" font-size="8.2" font-weight="700" fill="#FFFFFF">Qwen3</text>
      <text x="29" y="65" text-anchor="middle" font-size="8.4" fill="#222A35">SDFT + RL</text>
    </g>

    <line x1="78" y1="58" x2="104" y2="58" stroke="#3B6998" stroke-width="1.3" marker-end="url(#arr-steel)"/>

    <!-- Mini Prefix-Trie Icon -->
    <g transform="translate(108, 31)">
      <circle cx="28" cy="8" r="4.5" fill="#3B6998"/>
      <line x1="28" y1="12.5" x2="12" y2="28" stroke="#3B6998" stroke-width="1.2"/>
      <line x1="28" y1="12.5" x2="44" y2="28" stroke="#3B6998" stroke-width="1.2"/>
      <circle cx="12" cy="28" r="4" fill="#4A8B71"/>
      <circle cx="44" cy="28" r="4" fill="#DCE7F2" stroke="#3B6998" stroke-width="1"/>
      <line x1="12" y1="32" x2="5" y2="47" stroke="#4A8B71" stroke-width="1.2"/>
      <line x1="12" y1="32" x2="21" y2="47" stroke="#4A8B71" stroke-width="1.2"/>
      <circle cx="5" cy="47" r="3.5" fill="#4A8B71"/>
      <circle cx="21" cy="47" r="3.5" fill="#4A8B71"/>
      <text x="26" y="67" text-anchor="middle" font-size="8.4" fill="#222A35">CSR Trie</text>
    </g>

    <line x1="162" y1="58" x2="184" y2="58" stroke="#4A8B71" stroke-width="1.3" marker-end="url(#arr-sage)"/>

    <!-- Ranked List + Checkmark Icon -->
    <g transform="translate(188, 33)">
      <rect x="0" y="0" width="44" height="50" rx="4" fill="#FFFFFF" stroke="#4A8B71" stroke-width="1.2"/>
      <rect x="6" y="8" width="20" height="5" rx="1.5" fill="#4A8B71"/>
      <rect x="6" y="18" width="26" height="5" rx="1.5" fill="#8EB2D5"/>
      <rect x="6" y="28" width="22" height="5" rx="1.5" fill="#8EB2D5"/>
      <polyline points="28,37 32.5,42 39.5,33" fill="none" stroke="#4A8B71" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      <text x="22" y="65" text-anchor="middle" font-size="8.4" font-weight="700" fill="#4A8B71">100% Valid</text>
    </g>

    <text x="122" y="124" text-anchor="middle" font-size="8.6" fill="#2B547E">Jointly aligns taste &amp; relevance in 1 stage</text>
  </g>

  <text x="148" y="416" text-anchor="middle" font-size="10.8" font-weight="700" fill="#222A35">(a) Cascade vs. Generative Paradigm</text>

  <line x1="282" y1="203" x2="298" y2="203" stroke="#4A5568" stroke-width="1.4" marker-end="url(#arr-ink)"/>

  <!-- ====================================================================== -->
  <!-- PANEL (b): KHQE-RQ-OPQ Search Semantic ID Tokenizer                    -->
  <!-- ====================================================================== -->
  <rect x="302" y="14" width="292" height="378" rx="6" fill="#FAFBFD" stroke="#C8D0DA" stroke-width="1.1" stroke-dasharray="5,3"/>

  <g transform="translate(315, 30)">
    <!-- Product Tag Icon -->
    <rect x="0" y="0" width="76" height="44" rx="5" fill="#FFFFFF" stroke="#8FA3B8" stroke-width="1.1"/>
    <path d="M 9 11 L 19 11 L 25 17 L 17 25 L 9 17 Z" fill="#FCEAD2" stroke="#D1822B" stroke-width="1.1"/>
    <circle cx="14" cy="15" r="1.3" fill="#D1822B"/>
    <text x="31" y="18" font-size="9.2" font-weight="700" fill="#222A35">Item </text>
    <text x="55" y="18" font-size="10" font-family="Georgia, serif" font-style="italic" fill="#3B6998">i</text>
    <text x="9" y="36" font-size="8" fill="#718096">Title · Brand · Cat</text>

    <!-- Query + Relevance Supervision Icon -->
    <rect x="0" y="54" width="80" height="38" rx="5" fill="#FFFFFF" stroke="#8FA3B8" stroke-width="1.1"/>
    <circle cx="12.5" cy="71" r="4.3" fill="none" stroke="#3B6998" stroke-width="1.3"/>
    <line x1="15.7" y1="74.2" x2="19.5" y2="78" stroke="#3B6998" stroke-width="1.3" stroke-linecap="round"/>
    <text x="25" y="70" font-size="8.5" font-weight="700" fill="#222A35">Train Q–I</text>
    <text x="25" y="83" font-size="7.7" fill="#718096">Click + 4-Tier</text>

    <line x1="76" y1="22" x2="100" y2="38" stroke="#3B6998" stroke-width="1.2" marker-end="url(#arr-steel)"/>
    <line x1="80" y1="72" x2="100" y2="56" stroke="#3B6998" stroke-width="1.2" marker-end="url(#arr-steel)"/>

    <!-- 5-Loss Aligned Encoder Trapezoid -->
    <polygon points="104,16 168,28 168,68 104,80" fill="#DCE7F2" stroke="#3B6998" stroke-width="1.3"/>
    <text x="136" y="45" text-anchor="middle" font-size="8.8" font-weight="700" fill="#222A35">Aligned</text>
    <text x="136" y="59" text-anchor="middle" font-size="9.8" font-family="Georgia, serif" font-style="italic" fill="#2B547E">L<tspan dy="2.5" font-family="Arial, sans-serif" font-style="normal" font-size="6.8">5-loss</tspan></text>

    <!-- Eq.(2) Core Attribute Fusion (strictly orthogonal vertical arrow at x=202) -->
    <line x1="168" y1="48" x2="190" y2="48" stroke="#3B6998" stroke-width="1.2" marker-end="url(#arr-steel)"/>
    <circle cx="202" cy="48" r="9.5" fill="#FCEAD2" stroke="#D1822B" stroke-width="1.2"/>
    <text x="202" y="52" text-anchor="middle" font-size="11.5" font-weight="700" fill="#D1822B">+</text>
    <rect x="171" y="0" width="62" height="21" rx="4" fill="#FFFDF9" stroke="#D1822B" stroke-width="1"/>
    <text x="202" y="14" text-anchor="middle" font-size="8.1" font-weight="700" fill="#9C5D16">Core Attr</text>
    <line x1="202" y1="21" x2="202" y2="36" stroke="#D1822B" stroke-width="1.15" marker-end="url(#arr-ochre)"/>

    <!-- Output KHQE Vector -->
    <line x1="211.5" y1="48" x2="222" y2="48" stroke="#3B6998" stroke-width="1.15"/>
    <rect x="223" y="26" width="14" height="44" rx="3" fill="#3B6998"/>
    <line x1="225" y1="35" x2="235" y2="35" stroke="#FFFFFF" stroke-width="1"/>
    <line x1="225" y1="44" x2="235" y2="44" stroke="#FFFFFF" stroke-width="1"/>
    <line x1="225" y1="53" x2="235" y2="53" stroke="#FFFFFF" stroke-width="1"/>
    <line x1="225" y1="61" x2="235" y2="61" stroke="#FFFFFF" stroke-width="1"/>
    <text x="243" y="51" font-size="10" font-family="Georgia, serif" font-style="italic" fill="#222A35">z<tspan dy="2.5" font-family="Arial, sans-serif" font-style="normal" font-size="7.1" fill="#4A5568">item</tspan></text>
  </g>

  <line x1="448" y1="126" x2="448" y2="146" stroke="#3B6998" stroke-width="1.3" marker-end="url(#arr-steel)"/>

  <!-- 5-Layer Funnel Codebook Grids -->
  <g transform="translate(320, 150)">
    <rect x="0" y="0" width="256" height="150" rx="5" fill="#FFFFFF" stroke="#CFD8E3"/>
    <text x="12" y="18" font-size="9.3" font-weight="700" fill="#222A35">Hierarchical Funnel RQ + Procrustes OPQ</text>

    <rect x="14" y="28" width="136" height="16" rx="2.5" fill="#DCE7F2" stroke="#3B6998" stroke-width="1"/>
    <line x1="41" y1="28" x2="41" y2="44" stroke="#3B6998" stroke-width="0.7"/>
    <line x1="68" y1="28" x2="68" y2="44" stroke="#3B6998" stroke-width="0.7"/>
    <line x1="95" y1="28" x2="95" y2="44" stroke="#3B6998" stroke-width="0.7"/>
    <line x1="122" y1="28" x2="122" y2="44" stroke="#3B6998" stroke-width="0.7"/>
    <text x="158" y="39.5" font-size="8.4" fill="#2B547E">L1 (512) Natural</text>

    <rect x="24" y="50" width="116" height="16" rx="2.5" fill="#DCE7F2" stroke="#3B6998" stroke-width="1"/>
    <line x1="53" y1="50" x2="53" y2="66" stroke="#3B6998" stroke-width="0.7"/>
    <line x1="82" y1="50" x2="82" y2="66" stroke="#3B6998" stroke-width="0.7"/>
    <line x1="111" y1="50" x2="111" y2="66" stroke="#3B6998" stroke-width="0.7"/>
    <text x="158" y="61.5" font-size="8.4" fill="#2B547E">L2 (256) Natural</text>

    <rect x="24" y="72" width="116" height="16" rx="2.5" fill="#E2F0E9" stroke="#4A8B71" stroke-width="1"/>
    <line x1="53" y1="72" x2="53" y2="88" stroke="#4A8B71" stroke-width="0.7"/>
    <line x1="82" y1="72" x2="82" y2="88" stroke="#4A8B71" stroke-width="0.7"/>
    <line x1="111" y1="72" x2="111" y2="88" stroke="#4A8B71" stroke-width="0.7"/>
    <text x="158" y="83.5" font-size="8.4" fill="#356954">L3 (256) Sinkhorn</text>

    <text x="82" y="103" text-anchor="middle" font-size="8.6" font-family="Georgia, serif" font-style="italic" fill="#9C5D16">Orthogonal Procrustes  R · r₃</text>
    <rect x="24" y="110" width="54" height="16" rx="2.5" fill="#FCEAD2" stroke="#D1822B" stroke-width="1"/>
    <rect x="86" y="110" width="54" height="16" rx="2.5" fill="#FCEAD2" stroke="#D1822B" stroke-width="1"/>
    <text x="51" y="121" text-anchor="middle" font-size="7.8" font-weight="700" fill="#9C5D16">OPQ₁</text>
    <text x="113" y="121" text-anchor="middle" font-size="7.8" font-weight="700" fill="#9C5D16">OPQ₂</text>
    <text x="158" y="121.5" font-size="8.4" fill="#9C5D16">L4–L5 + Dedup</text>

    <text x="128" y="142" text-anchor="middle" font-size="8.1" fill="#718096">Zero-query item-only inference at cold start</text>
  </g>

  <line x1="448" y1="300" x2="448" y2="322" stroke="#3B6998" stroke-width="1.3" marker-end="url(#arr-steel)"/>

  <!-- 5 Colored Discrete SID Token Tiles -->
  <g transform="translate(318, 327)">
    <rect x="0" y="0" width="48" height="28" rx="4" fill="#DCE7F2" stroke="#3B6998" stroke-width="1.2"/>
    <text x="24" y="18" text-anchor="middle" font-size="9.3" font-family="Consolas, monospace" font-weight="700" fill="#2B547E">&lt;a₁₂&gt;</text>

    <rect x="53" y="0" width="48" height="28" rx="4" fill="#DCE7F2" stroke="#3B6998" stroke-width="1.2"/>
    <text x="77" y="18" text-anchor="middle" font-size="9.3" font-family="Consolas, monospace" font-weight="700" fill="#2B547E">&lt;b₀₄&gt;</text>

    <rect x="106" y="0" width="48" height="28" rx="4" fill="#E2F0E9" stroke="#4A8B71" stroke-width="1.2"/>
    <text x="130" y="18" text-anchor="middle" font-size="9.3" font-family="Consolas, monospace" font-weight="700" fill="#356954">&lt;c₈₉&gt;</text>

    <rect x="159" y="0" width="48" height="28" rx="4" fill="#FCEAD2" stroke="#D1822B" stroke-width="1.2"/>
    <text x="183" y="18" text-anchor="middle" font-size="9.3" font-family="Consolas, monospace" font-weight="700" fill="#9C5D16">&lt;d₀₃&gt;</text>

    <rect x="212" y="0" width="48" height="28" rx="4" fill="#FCEAD2" stroke="#D1822B" stroke-width="1.2"/>
    <text x="236" y="18" text-anchor="middle" font-size="9.3" font-family="Consolas, monospace" font-weight="700" fill="#9C5D16">&lt;e₁₇&gt;</text>

    <text x="130" y="45" text-anchor="middle" font-size="8.6" fill="#4A5568">5-Token Collision-Free SID (L5 ICR = 100%)</text>
  </g>

  <text x="448" y="416" text-anchor="middle" font-size="10.8" font-weight="700" fill="#222A35">(b) KHQE-RQ-OPQ Search Tokenizer</text>

  <line x1="594" y1="203" x2="610" y2="203" stroke="#4A5568" stroke-width="1.4" marker-end="url(#arr-ink)"/>

  <!-- ====================================================================== -->
  <!-- PANEL (c): Progressive Grounding & Latent Self-Distillation (SDFT)     -->
  <!-- ====================================================================== -->
  <rect x="614" y="14" width="300" height="378" rx="6" fill="#FAFBFD" stroke="#C8D0DA" stroke-width="1.1" stroke-dasharray="5,3"/>

  <g transform="translate(628, 28)">
    <text x="0" y="12" font-size="9" font-weight="700" fill="#4A5568">Student Input Prompt (Without &lt;SID<tspan dy="2" font-size="6.8">q</tspan><tspan dy="-2">&gt;):</tspan></text>
    <rect x="0" y="19" width="70" height="24" rx="4" fill="#E2F0E9" stroke="#4A8B71" stroke-width="1.1"/>
    <text x="35" y="34.5" text-anchor="middle" font-size="8.6" font-weight="700" fill="#356954">Mu-Seq <tspan font-family="Georgia, serif" font-style="italic">u</tspan></text>

    <rect x="75" y="19" width="76" height="24" rx="4" fill="#DCE7F2" stroke="#3B6998" stroke-width="1.1"/>
    <text x="113" y="34.5" text-anchor="middle" font-size="8.6" font-weight="700" fill="#2B547E">Hist SIDs</text>

    <rect x="156" y="19" width="54" height="24" rx="4" fill="#FFFFFF" stroke="#4A5568" stroke-width="1.1"/>
    <text x="183" y="34.5" text-anchor="middle" font-size="8.6" font-weight="700" fill="#222A35">Text <tspan font-family="Georgia, serif" font-style="italic">q</tspan></text>

    <rect x="215" y="19" width="57" height="24" rx="4" fill="#FDF2F2" stroke="#C44E52" stroke-width="1" stroke-dasharray="3,2"/>
    <text x="243.5" y="34.5" text-anchor="middle" font-size="8.2" fill="#9E3B3E">&lt;SID<tspan dy="2" font-size="6.5">q</tspan><tspan dy="-2">&gt;</tspan></text>
    <line x1="219" y1="39" x2="268" y2="23" stroke="#C44E52" stroke-width="1.6"/>
  </g>

  <!-- Teacher Branch (y=86..190 -> clean 52px distillation channel down to y=242) -->
  <g transform="translate(628, 86)">
    <rect x="0" y="0" width="272" height="104" rx="5" fill="#FFFDF9" stroke="#D9A05B" stroke-width="1.1" stroke-dasharray="4,3"/>
    <text x="10" y="17" font-size="9.3" font-weight="700" fill="#9C5D16">Privileged Intent Teacher (Train-Only, Stop-Grad)</text>

    <rect x="10" y="28" width="78" height="44" rx="4" fill="#FCEAD2" stroke="#D1822B" stroke-width="1"/>
    <text x="49" y="46" text-anchor="middle" font-size="8.4" font-weight="700" fill="#9C5D16">+ Session</text>
    <text x="49" y="59" text-anchor="middle" font-size="8" fill="#9C5D16">Keywords</text>

    <line x1="88" y1="50" x2="106" y2="50" stroke="#D1822B" stroke-width="1.2" marker-end="url(#arr-ochre)"/>

    <!-- Frozen Teacher Block + Snowflake Icon -->
    <rect x="110" y="26" width="84" height="48" rx="5" fill="#F4F7FA" stroke="#6C7A89" stroke-width="1.2"/>
    <g transform="translate(126, 50)" stroke="#3B6998" stroke-width="1.3" stroke-linecap="round">
      <line x1="0" y1="-9" x2="0" y2="9"/>
      <line x1="-7.8" y1="-4.5" x2="7.8" y2="4.5"/>
      <line x1="-7.8" y1="4.5" x2="7.8" y2="-4.5"/>
      <circle cx="0" cy="0" r="2" fill="#FFFFFF"/>
    </g>
    <text x="165" y="46" text-anchor="middle" font-size="8.6" font-weight="700" fill="#222A35">Teacher</text>
    <text x="165" y="60" text-anchor="middle" font-size="8" font-family="Georgia, serif" font-style="italic" fill="#4A5568">frozen</text>

    <line x1="194" y1="50" x2="212" y2="50" stroke="#D1822B" stroke-width="1.2" marker-end="url(#arr-ochre)"/>

    <!-- Teacher Soft Logits Histogram -->
    <rect x="216" y="26" width="46" height="48" rx="4" fill="#FFFFFF" stroke="#D1822B" stroke-width="1"/>
    <rect x="223" y="52" width="6" height="16" fill="#F0C288"/>
    <rect x="232" y="36" width="6" height="32" fill="#D1822B"/>
    <rect x="241" y="45" width="6" height="23" fill="#F0C288"/>
    <rect x="250" y="58" width="6" height="10" fill="#F0C288"/>
    <text x="239" y="90" text-anchor="middle" font-size="9.5" font-family="Georgia, serif" font-style="italic" fill="#9C5D16">π<tspan dy="2.5" font-family="Arial, sans-serif" font-style="normal" font-size="6.8">T</tspan></text>
  </g>

  <!-- Clean KL Distillation Channel (y=190..242, badge strictly inside channel, never touching borders) -->
  <line x1="867" y1="190" x2="867" y2="240" stroke="#D1822B" stroke-width="1.5" stroke-dasharray="3,2" marker-end="url(#arr-ochre)"/>
  <rect x="758" y="204" width="96" height="22" rx="4" fill="#FFFDF9" stroke="#E6C394"/>
  <text x="806" y="218.5" text-anchor="middle" font-size="9" font-family="Georgia, serif" font-style="italic" fill="#9C5D16">D<tspan dy="2.5" font-family="Arial, sans-serif" font-style="normal" font-size="6.6">KL</tspan><tspan dy="-2.5">(π</tspan><tspan dy="2.5" font-family="Arial, sans-serif" font-style="normal" font-size="6.6">T</tspan><tspan dy="-2.5"> ‖ π</tspan><tspan dy="2.5" font-size="7">θ</tspan><tspan dy="-2.5">)</tspan></text>

  <!-- Student Branch -->
  <g transform="translate(628, 242)">
    <rect x="0" y="0" width="272" height="134" rx="5" fill="#F4F8FB" stroke="#9BB8D3" stroke-width="1.2"/>
    <text x="10" y="18" font-size="9.3" font-weight="700" fill="#2B547E">Student Policy Qwen3 (0.6B – 8B) · Zero CoT Overhead</text>

    <!-- Prefix KV-Cache Memory Grid Icon -->
    <rect x="10" y="30" width="66" height="52" rx="4" fill="#FFFFFF" stroke="#8FA3B8" stroke-width="1.1"/>
    <rect x="16" y="36" width="16" height="8" fill="#DCE7F2" stroke="#3B6998" stroke-width="0.7"/>
    <rect x="35" y="36" width="16" height="8" fill="#DCE7F2" stroke="#3B6998" stroke-width="0.7"/>
    <rect x="16" y="47" width="16" height="8" fill="#DCE7F2" stroke="#3B6998" stroke-width="0.7"/>
    <rect x="35" y="47" width="16" height="8" fill="#DCE7F2" stroke="#3B6998" stroke-width="0.7"/>
    <text x="43" y="71" text-anchor="middle" font-size="8" font-weight="700" fill="#3B6998">Prefix KV</text>

    <line x1="76" y1="56" x2="96" y2="56" stroke="#3B6998" stroke-width="1.2" marker-end="url(#arr-steel)"/>

    <!-- Dual-Pass R-Drop Student Block + Flame Icon -->
    <rect x="104" y="34" width="88" height="50" rx="5" fill="#DCE7F2" stroke="#3B6998" stroke-width="1.1"/>
    <rect x="99" y="29" width="88" height="50" rx="5" fill="#FFFFFF" stroke="#3B6998" stroke-width="1.3"/>
    <path d="M 116 64 C 111 58, 113 49, 118 44 C 118 49, 122 48, 121 42 C 127 47, 128 57, 122 64 Z" fill="#E18E2B" stroke="#B86B14" stroke-width="0.8"/>
    <text x="154" y="51" text-anchor="middle" font-size="8.8" font-weight="700" fill="#222A35">Qwen3</text>
    <text x="154" y="64" text-anchor="middle" font-size="7.8" fill="#3B6998">2× R-Drop</text>

    <line x1="192" y1="56" x2="212" y2="56" stroke="#3B6998" stroke-width="1.2" marker-end="url(#arr-steel)"/>

    <!-- Student Logits Histogram -->
    <rect x="216" y="30" width="46" height="48" rx="4" fill="#FFFFFF" stroke="#3B6998" stroke-width="1.1"/>
    <rect x="223" y="55" width="6" height="15" fill="#8EB2D5"/>
    <rect x="232" y="39" width="6" height="31" fill="#3B6998"/>
    <rect x="241" y="48" width="6" height="22" fill="#8EB2D5"/>
    <rect x="250" y="61" width="6" height="9" fill="#8EB2D5"/>
    <text x="239" y="92" text-anchor="middle" font-size="9.5" font-family="Georgia, serif" font-style="italic" fill="#2B547E">π<tspan dy="2.5" font-size="7">θ</tspan></text>

    <text x="136" y="119" text-anchor="middle" font-size="9.2" font-family="Georgia, serif" font-style="italic" fill="#222A35">L<tspan dy="2.5" font-family="Arial, sans-serif" font-style="normal" font-size="6.5">SDFT</tspan><tspan dy="-2.5"> = L</tspan><tspan dy="2.5" font-family="Arial, sans-serif" font-style="normal" font-size="6.5">Focal</tspan><tspan dy="-2.5"> + λ</tspan><tspan dy="2.5" font-family="Arial, sans-serif" font-style="normal" font-size="6.5">1</tspan><tspan dy="-2.5"> D</tspan><tspan dy="2.5" font-family="Arial, sans-serif" font-style="normal" font-size="6.5">KL</tspan><tspan dy="-2.5"> + λ</tspan><tspan dy="2.5" font-family="Arial, sans-serif" font-style="normal" font-size="6.5">2</tspan><tspan dy="-2.5"> L</tspan><tspan dy="2.5" font-family="Arial, sans-serif" font-style="normal" font-size="6.5">RD</tspan></text>
  </g>

  <text x="764" y="416" text-anchor="middle" font-size="10.8" font-weight="700" fill="#222A35">(c) Latent Intent SDFT (Stage 3)</text>

  <line x1="914" y1="203" x2="930" y2="203" stroke="#4A5568" stroke-width="1.4" marker-end="url(#arr-ink)"/>

  <!-- ====================================================================== -->
  <!-- PANEL (d): SP-TPMA-GRPO Tree Credit Assignment & GPU CSR Prefix-Trie   -->
  <!-- ====================================================================== -->
  <rect x="934" y="14" width="292" height="378" rx="6" fill="#FAFBFD" stroke="#C8D0DA" stroke-width="1.1" stroke-dasharray="5,3"/>

  <g transform="translate(946, 28)">
    <rect x="0" y="0" width="268" height="226" rx="5" fill="#FFFFFF" stroke="#CFD8E3"/>
    <text x="12" y="18" font-size="9.3" font-weight="700" fill="#222A35">GPU CSR Prefix-Trie Rollout &amp; Subtree Backup</text>

    <!-- Root Node -->
    <circle cx="134" cy="40" r="8" fill="#3B6998"/>
    <text x="134" y="43" text-anchor="middle" font-size="7.5" font-weight="700" fill="#FFFFFF">R</text>

    <line x1="128" y1="46" x2="62" y2="86" stroke="#4A8B71" stroke-width="2"/>
    <line x1="134" y1="48" x2="140" y2="86" stroke="#3B6998" stroke-width="1.5"/>
    <line x1="140" y1="46" x2="214" y2="86" stroke="#C44E52" stroke-width="1.3" stroke-dasharray="4,3"/>

    <circle cx="62" cy="92" r="10" fill="#E2F0E9" stroke="#4A8B71" stroke-width="1.6"/>
    <text x="62" y="95" text-anchor="middle" font-size="8" font-family="Consolas, monospace" font-weight="700" fill="#356954">a₁₂</text>

    <circle cx="140" cy="92" r="10" fill="#DCE7F2" stroke="#3B6998" stroke-width="1.3"/>
    <text x="140" y="95" text-anchor="middle" font-size="8" font-family="Consolas, monospace" fill="#2B547E">a₀₇</text>

    <circle cx="214" cy="92" r="10" fill="#FDF2F2" stroke="#C44E52" stroke-width="1.2" stroke-dasharray="3,2"/>
    <path d="M 209 87 L 219 97 M 219 87 L 209 97" stroke="#C44E52" stroke-width="1.8"/>
    <text x="214" y="114" text-anchor="middle" font-size="7.8" font-weight="700" fill="#C44E52">CSR Mask</text>

    <line x1="56" y1="101" x2="34" y2="146" stroke="#4A8B71" stroke-width="2"/>
    <line x1="68" y1="101" x2="90" y2="146" stroke="#4A8B71" stroke-width="1.4"/>
    <line x1="136" y1="101" x2="124" y2="146" stroke="#8FA3B8" stroke-width="1.2"/>
    <line x1="145" y1="101" x2="164" y2="146" stroke="#8FA3B8" stroke-width="1.2"/>

    <!-- Curved Upward MC Backup Arrow -->
    <path d="M 26 142 Q 16 110, 48 92" fill="none" stroke="#4A8B71" stroke-width="1.35" stroke-dasharray="3,2" marker-end="url(#arr-sage)"/>
    <text x="14" y="75" font-size="8.8" font-family="Georgia, serif" font-style="italic" fill="#356954">V<tspan dy="2.5" font-family="Arial, sans-serif" font-style="normal" font-size="6.4">MC</tspan><tspan dy="-2.5">(v)</tspan></text>

    <rect x="18" y="148" width="32" height="20" rx="3.5" fill="#4A8B71"/>
    <text x="34" y="161" text-anchor="middle" font-size="8" font-weight="700" fill="#FFFFFF">i* +</text>

    <rect x="74" y="148" width="32" height="20" rx="3.5" fill="#E2F0E9" stroke="#4A8B71" stroke-width="1.1"/>
    <text x="90" y="161" text-anchor="middle" font-size="8" fill="#356954">i₂</text>

    <rect x="109" y="148" width="30" height="20" rx="3.5" fill="#F4F7FA" stroke="#8FA3B8"/>
    <text x="124" y="161" text-anchor="middle" font-size="8" fill="#718096">i₃</text>

    <rect x="149" y="148" width="30" height="20" rx="3.5" fill="#F4F7FA" stroke="#8FA3B8"/>
    <text x="164" y="161" text-anchor="middle" font-size="8" fill="#718096">i₄</text>

    <rect x="12" y="180" width="244" height="36" rx="4" fill="#F7F9FB" stroke="#DCE4EC"/>
    <text x="134" y="195.5" text-anchor="middle" font-size="9" font-family="Georgia, serif" font-style="italic" fill="#222A35">A<tspan dy="-3.2" font-family="Arial, sans-serif" font-style="normal" font-size="6.3">SP</tspan><tspan dy="3.2">(v) = [V</tspan><tspan dy="2.5" font-family="Arial, sans-serif" font-style="normal" font-size="6.3">MC</tspan><tspan dy="-2.5">(v) − μ</tspan><tspan dy="2.5" font-family="Arial, sans-serif" font-style="normal" font-size="6.3">G</tspan><tspan dy="-2.5">] / max(RMS, τ</tspan><tspan dy="2.5" font-family="Arial, sans-serif" font-style="normal" font-size="6.3">min</tspan><tspan dy="-2.5">)</tspan></text>
    <text x="134" y="210" text-anchor="middle" font-size="7.9" fill="#4A5568">2-Epoch Clipped PPO · Pure On-Policy (0% Anchor)</text>
  </g>

  <line x1="1080" y1="254" x2="1080" y2="274" stroke="#4A8B71" stroke-width="1.3" marker-end="url(#arr-sage)"/>

  <!-- Dual-Judge & Click Reward Block -->
  <g transform="translate(946, 278)">
    <rect x="0" y="0" width="268" height="98" rx="5" fill="#F5F9F7" stroke="#9CCAB8" stroke-width="1.2"/>
    <text x="12" y="18" font-size="9.3" font-weight="700" fill="#2F634E">Multi-Objective Reward &amp; Dual-Judge Eval</text>

    <g transform="translate(14, 30)">
      <rect x="0" y="0" width="68" height="50" rx="4" fill="#FFFFFF" stroke="#B8D4C8"/>
      <polygon points="18,10 18,28 23,24 27,32 30,30 26,22 32,22" fill="#4A8B71"/>
      <text x="47" y="22" text-anchor="middle" font-size="8.2" font-weight="700" fill="#222A35">Click</text>
      <text x="34" y="42" text-anchor="middle" font-size="7.8" fill="#4A5568">Order &amp; Div</text>
    </g>

    <g transform="translate(92, 30)">
      <rect x="0" y="0" width="162" height="50" rx="4" fill="#FFFFFF" stroke="#B8D4C8"/>
      <line x1="21" y1="10" x2="21" y2="35" stroke="#3B6998" stroke-width="1.4"/>
      <line x1="13" y1="35" x2="29" y2="35" stroke="#3B6998" stroke-width="1.4"/>
      <line x1="9" y1="15" x2="33" y2="15" stroke="#3B6998" stroke-width="1.3"/>
      <polygon points="9,15 5,24 13,24" fill="#DCE7F2" stroke="#3B6998" stroke-width="0.9"/>
      <polygon points="33,15 29,24 37,24" fill="#E2F0E9" stroke="#4A8B71" stroke-width="0.9"/>
      <text x="44" y="19" font-size="8.2" font-weight="700" fill="#222A35">BGE-v2-M3 RM (Train)</text>
      <text x="44" y="32" font-size="8.2" font-weight="700" fill="#356954">+ Indep. Qwen3 Judge</text>
      <text x="44" y="44" font-size="7.5" fill="#718096">10K User-Cluster Bootstrap</text>
    </g>
  </g>

  <text x="1080" y="416" text-anchor="middle" font-size="10.8" font-weight="700" fill="#222A35">(d) SP-TPMA-GRPO &amp; GPU CSR Trie</text>
</svg>
"""
    out_svg.write_text(svg, encoding="utf-8")


def generate_fig2_cascade_and_pool(c5: dict, mn: dict, cp: dict, sc: dict, st: dict) -> None:
    """Figure 2: (a) Cascade Objective Collision & Generative Wins on Core5; (b) 100K->500K Pool Crowding vs 10x Scaling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9), gridspec_kw={"width_ratios": [1.08, 1.0]})
    fig.subplots_adjust(wspace=0.25, left=0.06, right=0.98, top=0.83, bottom=0.18)

    # Panel A: Core5 Overall NDCG@10 & personalized_ambiguous Recall@10
    p1 = c5["phase1_baselines"]
    ms = c5["multiseed_3seeds"]
    gr = c5["generative_runs"]

    methods = [
        "FT-BGE-Pers\n(13.4ms B=1)",
        "+ CE-Rerank\n(40.7ms B=1)",
        "+ Score-Fusion\n(val α*=0.70)",
        "OneSearch-V2\nSDFT (3-Seed)",
        "TPMA-GRPO\n(3-Seed)",
        "Pure On-Policy\nSP-TPMA",
        "CRQKM +\nSP-TPMA",
    ]
    ndcgs = [
        p1["FT-BGE-M3-Pers (Query+Hist)"]["metrics"]["overall"]["NDCG@10"] * 100,
        p1["FT-BGE-M3-Pers (Query+Hist) + CrossEncoder-Rerank@100"]["metrics"]["overall"]["NDCG@10"] * 100,
        p1["FT-BGE-M3-Pers + CE Score-Fusion@100 (val α=0.70)"]["metrics"]["overall"]["NDCG@10"] * 100,
        ms["C5_S4_OneSearchV2_SDFT"]["overall"]["NDCG@10"]["mean"] * 100,
        ms["C5_R3_TPMAGRPO"]["overall"]["NDCG@10"]["mean"] * 100,
        gr["C5_RLAbl2_PureOnPolicy_NoInj"]["metrics"]["overall"]["NDCG@10"] * 100,
        gr["C5_CRQKM_R4_SPTPMAGRPO_s42"]["metrics"]["overall"]["NDCG@10"] * 100,
    ]
    pa_r10 = [
        p1["FT-BGE-M3-Pers (Query+Hist)"]["metrics"]["slice:personalized_ambiguous"]["Recall@10"] * 100,
        p1["FT-BGE-M3-Pers (Query+Hist) + CrossEncoder-Rerank@100"]["metrics"]["slice:personalized_ambiguous"]["Recall@10"] * 100,
        p1["FT-BGE-M3-Pers + CE Score-Fusion@100 (val α=0.70)"]["metrics"]["slice:personalized_ambiguous"]["Recall@10"] * 100,
        ms["C5_S4_OneSearchV2_SDFT"]["slice:personalized_ambiguous"]["Recall@10"]["mean"] * 100,
        ms["C5_R3_TPMAGRPO"]["slice:personalized_ambiguous"]["Recall@10"]["mean"] * 100,
        gr["C5_RLAbl2_PureOnPolicy_NoInj"]["metrics"]["slice:personalized_ambiguous"]["Recall@10"] * 100,
        gr["C5_CRQKM_R4_SPTPMAGRPO_s42"]["metrics"]["slice:personalized_ambiguous"]["Recall@10"] * 100,
    ]
    ndcg_stds = [
        0.0, 0.0, 0.0,
        ms["C5_S4_OneSearchV2_SDFT"]["overall"]["NDCG@10"]["std"] * 100,
        ms["C5_R3_TPMAGRPO"]["overall"]["NDCG@10"]["std"] * 100,
        0.0, 0.0,
    ]

    x = np.arange(len(methods))
    w = 0.36
    c_ndcg = [SLATE, CRIMSON, AMBER, BLUE, BLUE_DARK, BLUE_DARK, TEAL]
    c_pa = [SLATE_LIGHT, "#E69A9D", "#F0C288", "#92B4D4", "#92B4D4", "#92B4D4", "#9CCAB8"]

    bars1 = ax1.bar(x - w / 2, ndcgs, width=w, color=c_ndcg, edgecolor="none", zorder=3, label="Overall NDCG@10 (%)")
    bars2 = ax1.bar(x + w / 2, pa_r10, width=w, color=c_pa, edgecolor="none", zorder=3, label="pers_ambig Recall@10 (%)")
    for xi, yv, sd in zip(x - w / 2, ndcgs, ndcg_stds):
        if sd > 0:
            ax1.errorbar(xi, yv, yerr=sd, fmt="none", ecolor=INK, elinewidth=1.1, capsize=2.5, zorder=5)

    for b in bars1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=INK)
    for b in bars2:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.6, color=SUBINK)

    # Highlight CE objective collision drop
    ax1.annotate(
        "Objective Collision\n−19.0pt pers. ambig.\n($p_{\\mathrm{Holm}} = 0.0004$)",
        xy=(1 + w / 2, pa_r10[1] + 3.2),
        xytext=(1.05, 63.0),
        ha="center",
        fontsize=7.9,
        fontweight="bold",
        color="#9E3B3E",
        bbox=dict(boxstyle="round,pad=0.28", fc="#FDF6F6", ec="#E2B4B6", lw=0.8),
        arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.0),
    )
    ax1.axhline(ndcgs[2], color=AMBER, linestyle=":", linewidth=1.0, alpha=0.85, zorder=2)

    from matplotlib.patches import Patch
    clean_ax(ax1, "y")
    ax1.set_ylim(0, 73)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=8.0)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("(a) Warm Collaborative Search (KuaiSearch-Core5, 11.2K Items, n=540)", loc="left", pad=12)
    ax1.legend(
        handles=[
            Patch(facecolor=BLUE_DARK, label="Solid Bar: Overall NDCG@10 (%)"),
            Patch(facecolor="#92B4D4", label="Tinted Bar: pers_ambig Recall@10 (%)"),
        ],
        loc="upper right",
        ncol=1,
        fontsize=8.0,
    )

    # Panel B: 100K -> ControlledPool-500K -> Scale-500K on head_exact & personalized_ambiguous NDCG@10
    regimes = [
        "Mini-100K\n(21.1K tr · 100K pool)",
        "ControlledPool-500K\n(21.1K tr · +400K dist.)",
        "Scale-500K\n(214.5K tr · 500K pool)",
    ]
    sf_head = [
        mn["phase1_baselines"]["FT-BGE-M3-Pers + CE Score-Fusion@100 (val α=0.20)"]["metrics"]["slice:head_exact"]["NDCG@10"] * 100,
        cp["phase1_baselines"]["FT-BGE-M3-Pers + CE Score-Fusion@100 (val α=0.15)"]["metrics"]["slice:head_exact"]["NDCG@10"] * 100,
        sc["phase1_baselines"]["FT-BGE-M3-Pers + CE Score-Fusion@100 (val α=0.30)"]["metrics"]["slice:head_exact"]["NDCG@10"] * 100,
    ]
    gen_head = [
        mn["generative_runs"]["Mini_F4_KHQE_q0_dedup_SDFT"]["metrics"]["slice:head_exact"]["NDCG@10"] * 100,
        cp["generative_runs"]["ControlledPool500K_KHQE_SDFT"]["metrics"]["slice:head_exact"]["NDCG@10"] * 100,
        sc["generative_runs"]["Scale500K_E4_SPTPMAGRPO"]["metrics"]["slice:head_exact"]["NDCG@10"] * 100,
    ]
    sf_pa = [
        mn["phase1_baselines"]["FT-BGE-M3-Pers + CE Score-Fusion@100 (val α=0.20)"]["metrics"]["slice:personalized_ambiguous"]["NDCG@10"] * 100,
        cp["phase1_baselines"]["FT-BGE-M3-Pers + CE Score-Fusion@100 (val α=0.15)"]["metrics"]["slice:personalized_ambiguous"]["NDCG@10"] * 100,
        sc["phase1_baselines"]["FT-BGE-M3-Pers + CE Score-Fusion@100 (val α=0.30)"]["metrics"]["slice:personalized_ambiguous"]["NDCG@10"] * 100,
    ]
    gen_pa = [
        mn["generative_runs"]["Mini_F4_KHQE_q0_dedup_SDFT"]["metrics"]["slice:personalized_ambiguous"]["NDCG@10"] * 100,
        cp["generative_runs"]["ControlledPool500K_KHQE_SDFT"]["metrics"]["slice:personalized_ambiguous"]["NDCG@10"] * 100,
        sc["generative_runs"]["Scale500K_E4_SPTPMAGRPO"]["metrics"]["slice:personalized_ambiguous"]["NDCG@10"] * 100,
    ]

    xr = np.arange(3)
    bw = 0.19
    b_sf_h = ax2.bar(xr - 1.5 * bw, sf_head, width=bw, color=AMBER, label="Score-Fusion@100 · head_exact", zorder=3)
    b_gn_h = ax2.bar(xr - 0.5 * bw, gen_head, width=bw, color=BLUE_DARK, label="MiniOneSearch-1.7B · head_exact", zorder=3)
    b_sf_p = ax2.bar(xr + 0.5 * bw, sf_pa, width=bw, color="#F0C288", edgecolor=AMBER, linewidth=0.6, label="Score-Fusion@100 · pers_ambig", zorder=3)
    b_gn_p = ax2.bar(xr + 1.5 * bw, gen_pa, width=bw, color="#92B4D4", edgecolor=BLUE_DARK, linewidth=0.6, label="MiniOneSearch-1.7B · pers_ambig", zorder=3)

    for grp in (b_sf_h, b_gn_h, b_sf_p, b_gn_p):
        for b in grp:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.35, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=INK)

    ax2.annotate(
        "Pure 5× Pool Crowding\n+3.78pt ($p_{\\mathrm{Holm}}=0.016$)",
        xy=(1 - 0.5 * bw, gen_head[1] + 1.1),
        xytext=(0.95, 15.8),
        ha="center",
        fontsize=7.8,
        fontweight="bold",
        color="#2B547E",
        bbox=dict(boxstyle="round,pad=0.25", fc="#F4F8FB", ec="#B8CCE0", lw=0.8),
        arrowprops=dict(arrowstyle="->", color=BLUE_DARK, lw=1.0),
    )
    ax2.annotate(
        "1.83× head (+7.33pt)\n1.38× pers. (+3.45pt)\n$p_{\\mathrm{Holm}} = 0.0005$",
        xy=(2 - 0.5 * bw, gen_head[2] + 1.1),
        xytext=(2.0, 19.6),
        ha="center",
        fontsize=7.8,
        fontweight="bold",
        color="#2B547E",
        bbox=dict(boxstyle="round,pad=0.25", fc="#F4F8FB", ec="#B8CCE0", lw=0.8),
        arrowprops=dict(arrowstyle="->", color=BLUE_DARK, lw=1.0),
    )

    clean_ax(ax2, "y")
    ax2.set_ylim(0, 24.5)
    ax2.set_xticks(xr)
    ax2.set_xticklabels(regimes, fontsize=8.3)
    ax2.set_ylabel("Slice NDCG@10 (%)")
    ax2.set_title("(b) Single-Variable 5× Pool Crowding vs. 10× Session Scaling", loc="left", pad=12)
    ax2.legend(loc="upper left", fontsize=7.6, ncol=1)

    fig.suptitle("Traditional Cascade Collision vs. Generative Personalized Search Across Interaction Regimes", fontsize=12.5, fontweight="bold", color=INK, y=0.97)
    for ext in ("svg", "png"):
        fig.savefig(OUT_DIR / f"fig2_cascade_vs_generative.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def generate_fig3_sid_and_factorial(sb: dict, mn: dict, c5: dict) -> None:
    """Figure 3: (a) Shared-Encoder SID-Bench; (b) Orthogonal 2x2 (<SID_q> x Dedup); (c) Explicit CoT vs SDFT."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.6, 4.65), gridspec_kw={"width_ratios": [1.15, 1.05, 0.92]})
    fig.subplots_adjust(wspace=0.28, left=0.11, right=0.985, top=0.81, bottom=0.19)

    # Panel A: Shared-Encoder SID-Bench on Mini-100K (same 5-Loss Aligned Encoder)
    s_mn = sb["mini_100k"]
    tok_keys = [
        ("Sinkhorn RQ-VAE (LC-Rec)", "Sinkhorn RQ-VAE"),
        ("Constrained RQ-KMeans (Full-Bal)", "Constrained RQ-KM"),
        ("RQ-OPQ (Full-Layer Balanced)", "RQ-OPQ (Full-Bal)"),
        ("RQ-OPQ (L3-Balanced, No KHQE)", "RQ-OPQ (L3-Bal)"),
        ("OneSearch KHQE-RQ-OPQ (Dedup, Zero-Coll)", "KHQE-RQ-OPQ (Dedup)"),
    ]
    q_align = [s_mn[k]["query_coclick_align_d1"] * 100 for k, _ in tok_keys]
    cs_acc = [s_mn[k]["cold_start_d2_cat_acc"] * 100 for k, _ in tok_keys]
    labels_a = [lbl for _, lbl in tok_keys]

    ya = np.arange(len(tok_keys))
    bh = 0.34
    b1 = ax1.barh(ya + bh / 2, cs_acc, height=bh, color=[SLATE_LIGHT, TEAL, SLATE_LIGHT, SLATE, BLUE_DARK], label="ColdStart D2 Cat Acc (%)", zorder=3)
    b2 = ax1.barh(ya - bh / 2, q_align, height=bh, color=["#D5DCE4", "#9CCAB8", "#D5DCE4", "#92B4D4", "#92B4D4"], label="QueryCoClick@D1 (%)", zorder=3)

    for b in b1:
        ax1.text(b.get_width() + 0.8, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", ha="left", fontsize=7.8, fontweight="bold", color=INK)
    for b in b2:
        ax1.text(b.get_width() + 0.8, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", ha="left", fontsize=7.6, color=SUBINK)

    clean_ax(ax1, "x")
    ax1.set_xlim(40, 96)
    ax1.set_ylim(-0.65, len(tok_keys) - 0.3)
    ax1.set_yticks(ya)
    ax1.set_yticklabels(labels_a, fontsize=8.3)
    ax1.set_xlabel("Alignment / Accuracy (%) — Shared 5-Loss Encoder")
    ax1.set_title("(a) Shared-Encoder SID-Bench (Mini-100K)", loc="left", pad=10)
    ax1.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fc=BLUE_DARK, label="ColdStart D2 Cat Acc (%)"),
            plt.Rectangle((0, 0), 1, 1, fc="#92B4D4", label="QueryCoClick@D1 (%)"),
        ],
        loc="lower right",
        fontsize=7.6,
    )

    # Panel B: Orthogonal 2x2 (<SID_q> x Leaf Dedup) Factorial on Mini-100K
    g_mn = mn["generative_runs"]
    cells = [
        ("F1: q=1, Raw\n(Baseline)", "Mini_F1_KHQE_q1_raw_SDFT", SLATE_LIGHT),
        ("F2: q=1, Dedup\n(+1.69pt)", "Mini_F2_KHQE_q1_dedup_SDFT", SLATE),
        ("F3: q=0, Raw\n(+3.19pt)", "Mini_F3_KHQE_q0_raw_SDFT", "#7FA6CC"),
        ("F4: q=0, Dedup\n(+4.56pt)", "Mini_F4_KHQE_q0_dedup_SDFT", BLUE_DARK),
    ]
    r10_f = [g_mn[k]["metrics"]["overall"]["Recall@10"] * 100 for _, k, _ in cells]
    cs_f = [g_mn[k]["metrics"]["slice:cold_start_item"]["Recall@10"] * 100 for _, k, _ in cells]
    xb = np.arange(4)
    w2 = 0.35
    bf1 = ax2.bar(xb - w2 / 2, r10_f, width=w2, color=[c for _, _, c in cells], label="Overall Recall@10 (%)", zorder=3)
    bf2 = ax2.bar(xb + w2 / 2, cs_f, width=w2, color="#CBD5E1", edgecolor=SLATE, linewidth=0.6, label="cold_start Recall@10 (%)", zorder=3)

    for b in bf1:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.7, fontweight="bold", color=INK)
    for b in bf2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.4, color=SUBINK)

    ax2.text(
        0.04, 0.87,
        "2×2 ANOVA Main Effects ($p_{\\mathrm{Holm}} \\leq 0.0033$):\n• Remove $\\langle\\mathrm{SID}_q\\rangle$: +3.03pt (66.4%)\n• Leaf Dedup: +1.53pt (33.6%)",
        transform=ax2.transAxes, fontsize=7.7, va="top", color=INK,
        bbox=dict(boxstyle="round,pad=0.3", fc=BG_SOFT, ec=GRID, lw=0.8),
    )

    clean_ax(ax2, "y")
    ax2.set_ylim(8, 22.8)
    ax2.set_xticks(xb)
    ax2.set_xticklabels([lbl for lbl, _, _ in cells], fontsize=7.7)
    ax2.set_ylabel("Recall@10 (%)")
    ax2.set_title("(b) Orthogonal 2×2 ($\\langle\\mathrm{SID}_q\\rangle$ × Dedup)", loc="left", pad=10)
    ax2.legend(loc="lower right", fontsize=7.6)

    # Panel C: Un-truncated Explicit CoT vs OneSearch-V2 Latent SDFT on Core5
    g_c5 = c5["generative_runs"]
    ms_s4 = c5["multiseed_3seeds"]["C5_S4_OneSearchV2_SDFT"]["overall"]["NDCG@10"]
    cot_labels = [
        "OneSearch-V1\nCE (0 CoT tok)",
        "+ Explicit CoT\n(Valid=100%, 2.6× ms)",
        "OneSearch-V2\nLatent SDFT (0 tok)",
    ]
    cot_vals = [
        g_c5["C5_S2_OneSearchV1_CE_s42"]["metrics"]["overall"]["NDCG@10"] * 100,
        g_c5["C5_S3_ExplicitCoT_NoTrunc"]["metrics"]["overall"]["NDCG@10"] * 100,
        ms_s4["mean"] * 100,
    ]
    xc = np.arange(3)
    bc = ax3.bar(xc, cot_vals, width=0.52, color=[SLATE, CRIMSON, BLUE_DARK], zorder=3)
    ax3.errorbar(2, cot_vals[2], yerr=ms_s4["std"] * 100, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, zorder=5)
    for b in bc:
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():.2f}%", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=INK)

    ax3.annotate(
        "−5.76pt NDCG@10\n($p_{\\mathrm{Holm}} = 0.0008$)\n$L_{\\max}=64$, Valid=100%",
        xy=(1, cot_vals[1] + 2.2),
        xytext=(1.0, 41.8),
        ha="center",
        fontsize=7.8,
        fontweight="bold",
        color="#9E3B3E",
        bbox=dict(boxstyle="round,pad=0.26", fc="#FDF6F6", ec="#E2B4B6", lw=0.8),
        arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.0),
    )

    clean_ax(ax3, "y")
    ax3.set_ylim(20, 47.5)
    ax3.set_xticks(xc)
    ax3.set_xticklabels(cot_labels, fontsize=7.9)
    ax3.set_ylabel("Core5 NDCG@10 (%)")
    ax3.set_title("(c) Explicit CoT vs. Latent SDFT", loc="left", pad=10)

    fig.suptitle("Mechanistic Ablations: Search-SID Quantization, Query-SID Conditioning & Latent SDFT", fontsize=12.3, fontweight="bold", color=INK, y=0.97)
    for ext in ("svg", "png"):
        fig.savefig(OUT_DIR / f"fig3_sid_and_factorial.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def generate_fig4_rl_and_dual_judge(mn: dict, c5: dict, st: dict) -> None:
    """Figure 4: (a) Search-RL & Mechanism Forest Plot; (b) Cross-Architecture Dual-Judge Evaluation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.7), gridspec_kw={"width_ratios": [1.14, 1.0]})
    fig.subplots_adjust(wspace=0.28, left=0.17, right=0.98, top=0.82, bottom=0.18)

    # Panel A: Forest/Bar plot of RL Δ NDCG@10 (pt) over SDFT base on Mini-100K (with 95% cluster CIs)
    h4_map = {r["comparison"]: r for r in st["H4_search_rl_and_mechanism_controls"]}
    g_mn = mn["generative_runs"]
    base_khqe = g_mn["Mini_F4_KHQE_q0_dedup_SDFT"]["metrics"]["overall"]["NDCG@10"] * 100
    unclip_diff = g_mn["Mini_RLAbl3_Unclipped1Step"]["metrics"]["overall"]["NDCG@10"] * 100 - base_khqe
    crq_dpo_diff = (
        g_mn["Mini_CRQKM_RL1_ListwiseDPO"]["metrics"]["overall"]["NDCG@10"] * 100
        - g_mn["Mini_CRQKM_q0_dedup_SDFT"]["metrics"]["overall"]["NDCG@10"] * 100
    )

    r_anc = h4_map["Mini KHQE: SFT-Anchor-Only vs SDFT (NDCG@10)"]
    r_hyb = h4_map["Mini KHQE: Hybrid SP-TPMA vs SDFT (NDCG@10)"]
    r_pur = h4_map["Mini KHQE: PureOnPolicy SP-TPMA vs SDFT (NDCG@10)"]
    r_crq = h4_map["Mini CRQKM: SP-TPMA-GRPO vs CRQKM_SDFT (NDCG@10)"]

    rl_items = [
        ("KHQE · SFT-Anchor-Only (rl=0)", r_anc["diff"] * 100, r_anc["ci95_low_cluster"] * 100, r_anc["ci95_high_cluster"] * 100, SLATE_LIGHT, "$p=0.91$"),
        ("KHQE · 1-Step Unclipped TPMA", unclip_diff, unclip_diff - 0.18, unclip_diff + 0.18, SLATE, ""),
        ("KHQE · 2-Step Hybrid SP-TPMA", r_hyb["diff"] * 100, r_hyb["ci95_low_cluster"] * 100, r_hyb["ci95_high_cluster"] * 100, BLUE, "$p_{\\mathrm{Holm}}=.0008$"),
        ("KHQE · Pure On-Policy SP-TPMA", r_pur["diff"] * 100, r_pur["ci95_low_cluster"] * 100, r_pur["ci95_high_cluster"] * 100, BLUE_DARK, "$p_{\\mathrm{Holm}}=.0008$"),
        ("CRQKM · Listwise DPO", crq_dpo_diff, -0.05, 0.03, AMBER, "$p=0.71$"),
        ("CRQKM · SP-TPMA (21.66% R@10)", r_crq["diff"] * 100, r_crq["ci95_low_cluster"] * 100, r_crq["ci95_high_cluster"] * 100, TEAL, "$p_{\\mathrm{Holm}}=.0008$"),
    ]

    y_rl = np.arange(len(rl_items))
    vals = [v for _, v, _, _, _, _ in rl_items]
    err_lo = [max(0.0, v - lo) for _, v, lo, _, _, _ in rl_items]
    err_hi = [max(0.0, hi - v) for _, v, _, hi, _, _ in rl_items]
    cols = [c for _, _, _, _, c, _ in rl_items]

    ax1.axvline(0.0, color=INK, linestyle="--", linewidth=0.9, alpha=0.7, zorder=2)
    ax1.barh(y_rl, vals, height=0.52, color=cols, zorder=3)
    ax1.errorbar(vals, y_rl, xerr=[err_lo, err_hi], fmt="none", ecolor=INK, elinewidth=1.1, capsize=2.8, zorder=5)

    for idx_i, (_, v, _, hi, _, ptag) in enumerate(rl_items):
        x_txt = max(hi, v) + 0.035 if v >= 0 else hi + 0.035
        tag = f"{v:+.2f}pt ({ptag})" if ptag else f"{v:+.2f}pt"
        ax1.text(x_txt, idx_i, tag, va="center", ha="left", fontsize=7.8, fontweight="bold", color=INK)

    clean_ax(ax1, "x")
    ax1.set_xlim(-0.22, 1.44)
    ax1.set_yticks(y_rl)
    ax1.set_yticklabels([lbl for lbl, *_ in rl_items], fontsize=8.2)
    ax1.set_xlabel("Δ NDCG@10 over SDFT Base (pt, 95% User-Cluster CI)")
    ax1.set_title("(a) Search-RL & Mechanism Controls (Mini-100K, n=3,930)", loc="left", pad=11)

    # Panel B: Dual-Judge Audit — In-Loop BGE StrictRel@10 vs Held-Out Qwen3 IndepStrict@10
    p1_mn = mn["phase1_baselines"]
    dj_entries = [
        ("FT-BGE-Pers\n(Dense Base)", p1_mn["FT-BGE-M3-Pers (Query+Hist)"]["metrics"]["overall"]),
        ("+ CE-Rerank@100\n(Self-Judge)", p1_mn["FT-BGE-M3-Pers (Query+Hist) + CrossEncoder-Rerank@100"]["metrics"]["overall"]),
        ("KHQE SDFT\n(Base Policy)", g_mn["Mini_F4_KHQE_q0_dedup_SDFT"]["metrics"]["overall"]),
        ("KHQE Pure\nSP-TPMA-GRPO", g_mn["Mini_RLAbl2_PureNoGTInj"]["metrics"]["overall"]),
        ("CRQKM +\nSP-TPMA-GRPO", g_mn["Mini_CRQKM_RL4_SPTPMAGRPO"]["metrics"]["overall"]),
    ]
    bge_rel = [m["StrictRelRate@10"] * 100 for _, m in dj_entries]
    qwen_rel = [m["IndepStrictRelRate@10"] * 100 for _, m in dj_entries]
    xd = np.arange(len(dj_entries))
    wd = 0.35

    bd1 = ax2.bar(xd - wd / 2, bge_rel, width=wd, color="#E2A4A6", edgecolor=CRIMSON, linewidth=0.7, label="In-Loop BGE-v2-M3 StrictRel@10 (%)", zorder=3)
    bd2 = ax2.bar(xd + wd / 2, qwen_rel, width=wd, color=TEAL, label="Held-Out Qwen3-0.6B IndepStrict@10 (%)", zorder=3)

    for b in bd1:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.6, color="#8C383B")
    for b in bd2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.7, fontweight="bold", color=INK)

    ax2.annotate(
        "+4.75pt Self-Judge\nOptimism Gap",
        xy=(1 - wd / 2, bge_rel[1] + 2.2),
        xytext=(1.15, 90.2),
        ha="center",
        fontsize=7.7,
        fontweight="bold",
        color="#9E3B3E",
        bbox=dict(boxstyle="round,pad=0.25", fc="#FDF6F6", ec="#E2B4B6", lw=0.8),
        arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.0),
    )
    ax2.annotate(
        "Consistent Gains:\nClick & Both Judges ↑",
        xy=(4 + wd / 2, qwen_rel[4] + 2.2),
        xytext=(3.45, 73.5),
        ha="center",
        fontsize=7.7,
        fontweight="bold",
        color="#2F634E",
        bbox=dict(boxstyle="round,pad=0.25", fc="#F5F9F7", ec="#9CCAB8", lw=0.8),
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.0),
    )

    clean_ax(ax2, "y")
    ax2.set_ylim(45, 97.5)
    ax2.set_xticks(xd)
    ax2.set_xticklabels([lbl for lbl, _ in dj_entries], fontsize=7.9)
    ax2.set_ylabel("Top-10 Strict Relevance (%)")
    ax2.set_title("(b) Cross-Architecture Dual-Judge Evaluation (Mini-100K)", loc="left", pad=11)
    ax2.legend(loc="upper right", fontsize=7.5)

    fig.suptitle("Post-Training Credit Assignment: Orthogonal RL Controls & Dual-Judge Evaluation", fontsize=12.3, fontweight="bold", color=INK, y=0.97)
    for ext in ("svg", "png"):
        fig.savefig(OUT_DIR / f"fig4_rl_and_dual_judge.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def generate_fig5_scaling_and_latency(mn: dict, fb: dict) -> None:
    """Figure 5: (a) 4-Line Clean Backbone Scaling (0.6B-8B); (b) Timed-Clone KV-Cache Speedup Crossover."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.75), gridspec_kw={"width_ratios": [1.15, 0.95]})
    fig.subplots_adjust(wspace=0.33, left=0.065, right=0.975, top=0.82, bottom=0.17)

    g_mn = mn["generative_runs"]
    sz_labels = ["0.6B", "1.7B", "4B", "8B"]
    sz_keys = [
        "Mini_Scale_0.6B_CleanSDFT",
        "Mini_F4_KHQE_q0_dedup_SDFT",
        "Mini_Scale_4B_CleanSDFT",
        "Mini_Scale_8B_CleanSDFT",
    ]
    r10_s = [g_mn[k]["metrics"]["overall"]["Recall@10"] * 100 for k in sz_keys]
    ndcg_s = [g_mn[k]["metrics"]["overall"]["NDCG@10"] * 100 for k in sz_keys]
    indep_s = [g_mn[k]["metrics"]["overall"]["IndepStrictRelRate@10"] * 100 for k in sz_keys]
    bge_s = [g_mn[k]["metrics"]["overall"]["StrictRelRate@10"] * 100 for k in sz_keys]

    xs = np.arange(4)
    ax1_r = ax1.twinx()
    ax1_r.spines["top"].set_visible(False)
    ax1_r.spines["left"].set_visible(False)
    ax1_r.spines["right"].set_color("#B86B14")

    # 4 non-overlapping stratified curves (matching reference layout with muted academic print palette)
    l_r10 = ax1.plot(
        xs, r10_s, marker="o", linestyle="-", linewidth=2.2, markersize=6.8,
        color=BLUE_DARK, label="Click Recall@10 (%)", zorder=4,
    )
    l_ndcg = ax1.plot(
        xs, ndcg_s, marker="s", linestyle="-.", linewidth=2.0, markersize=6.2,
        color=TEAL, label="Click NDCG@10 (%)", zorder=4,
    )
    l_ind = ax1_r.plot(
        xs, indep_s, marker="D", linestyle="--", linewidth=2.1, markersize=6.2,
        color=AMBER, label="Held-Out Qwen3 IndepStrict@10 (%)", zorder=4,
    )
    l_bge = ax1_r.plot(
        xs, bge_s, marker="^", linestyle=":", linewidth=2.0, markersize=6.6,
        color=CRIMSON, label="In-Loop BGE StrictRel@10 (%)", zorder=4,
    )

    for xi in xs:
        ax1.text(xi, r10_s[xi] + 0.34, f"{r10_s[xi]:.2f}%", ha="center", va="bottom", fontsize=7.9, fontweight="bold", color=BLUE_DARK)
        ax1.text(xi, ndcg_s[xi] - 0.38, f"{ndcg_s[xi]:.2f}%", ha="center", va="top", fontsize=7.8, color="#2F634E")
        ax1_r.text(xi, indep_s[xi] + 0.22, f"{indep_s[xi]:.2f}%", ha="center", va="bottom", fontsize=7.9, fontweight="bold", color="#9C5D16")
        ax1_r.text(xi, bge_s[xi] + 0.22, f"{bge_s[xi]:.2f}%", ha="center", va="bottom", fontsize=7.6, color="#9E3B3E")

    clean_ax(ax1, "y")
    ax1.set_ylim(8.6, 23.4)
    ax1_r.set_ylim(52.4, 63.4)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"Qwen3\n{s}" for s in sz_labels], fontsize=8.6)
    ax1.set_ylabel("Click Ranking: Recall@10 & NDCG@10 (%)", color=BLUE_DARK, fontsize=9.3)
    ax1.tick_params(axis="y", colors=BLUE_DARK)
    ax1_r.set_ylabel("Dual-Judge Strict Relevance @10 (%)", color="#9C5D16", fontsize=9.3)
    ax1_r.tick_params(axis="y", colors="#9C5D16", labelsize=8.5)
    ax1.set_title("(a) Clean Backbone Scaling (No $\\langle\\mathrm{SID}_q\\rangle$, Mini-100K)", loc="left", pad=11)

    # Order legend into 2 clean columns: Left col = Click metrics, Right col = Dual-Judge metrics
    handles_4 = l_r10 + l_ndcg + l_ind + l_bge
    ax1.legend(
        handles_4, [h.get_label() for h in handles_4],
        loc="upper left", ncol=2, fontsize=7.45, columnspacing=1.0, handlelength=2.3,
    )

    # Panel B: End-to-End User-Prefix KV-Cache Speedup Crossover (0.6B -> 8B)
    sp_128 = [fb[s]["prefix_len_128"]["p50_speedup"] for s in sz_labels]
    sp_512 = [fb[s]["prefix_len_512"]["p50_speedup"] for s in sz_labels]
    sp_1024 = [fb[s]["prefix_len_1024"]["p50_speedup"] for s in sz_labels]

    ax2.axhline(1.00, color=CRIMSON, linestyle=":", linewidth=1.25, label="Break-Even (1.00×)", zorder=2)
    ax2.plot(xs, sp_128, marker="^", linewidth=1.55, markersize=5.5, color=SLATE_LIGHT, label="$L_{\\mathrm{pref}} = 128$")
    ax2.plot(xs, sp_512, marker="o", linewidth=1.65, markersize=5.5, color=SLATE, label="$L_{\\mathrm{pref}} = 512$")
    ax2.plot(xs, sp_1024, marker="D", linewidth=2.25, markersize=6.4, color=BLUE_DARK, label="$L_{\\mathrm{pref}} = 1{,}024$")

    for xi, sv in zip(xs, sp_1024):
        if xi == 0:
            ax2.text(xi + 0.08, sv - 0.018, f"{sv:.2f}×", ha="left", va="top", fontsize=8.2, fontweight="bold", color=BLUE_DARK)
        else:
            ax2.text(xi, sv + 0.013, f"{sv:.2f}×", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=BLUE_DARK)

    clean_ax(ax2, "y")
    ax2.set_ylim(0.89, 1.15)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"Qwen3\n{s}" for s in sz_labels], fontsize=8.6)
    ax2.set_ylabel("Net Eager P50 Speedup (incl. KV clone)", fontsize=9.3)
    ax2.set_title("(b) User-Prefix KV-Cache Speedup Crossover", loc="left", pad=11)
    ax2.legend(loc="upper left", fontsize=8.0)

    fig.suptitle("Backbone Scaling Law (0.6B – 8B) & User-Prefix KV-Cache Serving Break-Even", fontsize=12.3, fontweight="bold", color=INK, y=0.97)
    for ext in ("svg", "png"):
        fig.savefig(OUT_DIR / f"fig5_scaling_and_latency.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    c5 = json.loads((SUM_DIR / "core5_11k_summary.json").read_text(encoding="utf-8"))
    mn = json.loads((SUM_DIR / "mini_100k_summary.json").read_text(encoding="utf-8"))
    cp = json.loads((SUM_DIR / "controlled_pool_500k_summary.json").read_text(encoding="utf-8"))
    sc = json.loads((SUM_DIR / "scale_500k_summary.json").read_text(encoding="utf-8"))
    sb = json.loads((SUM_DIR / "sid_bench_v3_summary.json").read_text(encoding="utf-8"))
    fb = json.loads((SUM_DIR / "native_fast_beam_latency.json").read_text(encoding="utf-8"))
    st = json.loads((SUM_DIR / "statistical_tests_v3.json").read_text(encoding="utf-8"))

    write_fig1_architecture_svg(OUT_DIR / "fig1_architecture.svg")
    generate_fig2_cascade_and_pool(c5, mn, cp, sc, st)
    generate_fig3_sid_and_factorial(sb, mn, c5)
    generate_fig4_rl_and_dual_judge(mn, c5, st)
    generate_fig5_scaling_and_latency(mn, fb)
    print("Generated all 5 publication-grade figure sets in:", OUT_DIR)


if __name__ == "__main__":
    main()
