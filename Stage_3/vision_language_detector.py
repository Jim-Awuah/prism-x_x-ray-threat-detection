# Stage_3/vision_language_detector.py

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

#  Optional BERT import
try:
    from transformers import AutoModel, AutoTokenizer
    _HF_OK = True
except ImportError:
    _HF_OK = False

#  SwinV2 backbone registry (mirrors Stage 1) 
from torchvision.models import (
    swin_v2_t, Swin_V2_T_Weights,
    swin_v2_s, Swin_V2_S_Weights,
    swin_v2_b, Swin_V2_B_Weights,
)

_SWIN = {
    "swin_v2_t": (swin_v2_t, Swin_V2_T_Weights.DEFAULT, 768),
    "swin_v2_s": (swin_v2_s, Swin_V2_S_Weights.DEFAULT, 768),
    "swin_v2_b": (swin_v2_b, Swin_V2_B_Weights.DEFAULT, 1024),
}


# ── 1. Frozen Vision Encoder ──────────────────────────────────────────────────

class FrozenVisionEncoder(nn.Module):
    """
    SwinV2 backbone loaded from a Stage 1 BYOL checkpoint.

    All backbone parameters are frozen (requires_grad = False).
    A trainable linear projection maps backbone features → fusion_dim.

    Args:
        variant    : "swin_v2_t" | "swin_v2_s" | "swin_v2_b"
        fusion_dim : output dimension (same as text projection)
        ckpt_path  : optional path to Stage 1 BYOL student checkpoint
    """

    def __init__(
        self,
        variant:    str = "swin_v2_t",
        fusion_dim: int = 256,
        ckpt_path:  Optional[str] = None,
    ) -> None:
        super().__init__()
        factory, weights, feat_dim = _SWIN[variant]
        swin = factory(weights=weights)

        self.features = swin.features
        self.norm     = swin.norm
        self.avgpool  = swin.avgpool
        self.feat_dim = feat_dim

        # Load Stage 1 weights if provided
        if ckpt_path:
            self._load_stage1_weights(ckpt_path)

        # Freeze all backbone parameters
        for p in self.parameters():
            p.requires_grad = False

        # Trainable projection: feat_dim → fusion_dim
        self.proj = nn.Linear(feat_dim, fusion_dim)

    def _load_stage1_weights(self, ckpt_path: str) -> None:
        """Load SwinV2 weights from BYOL student checkpoint saved in Stage 1."""
        ckpt   = torch.load(ckpt_path, map_location="cpu")
        # Stage 1 saves the full BYOL model; extract student backbone weights
        state  = ckpt.get("model_state_dict", ckpt)
        prefix = "student."
        backbone_state = {
            k[len(prefix):]: v
            for k, v in state.items()
            if k.startswith(prefix) and not k.startswith(f"{prefix}head")
        }
        missing, unexpected = self.load_state_dict(backbone_state, strict=False)
        if missing:
            print(f"[FrozenVisionEncoder] Missing keys: {missing[:5]} ...")
        print(f"[FrozenVisionEncoder] Loaded Stage 1 backbone from {ckpt_path}")

    @torch.no_grad()
    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features, no grad (backbone is frozen)."""
        x = self.norm(self.features(x))                              # (B,H',W',C)
        x = torch.flatten(self.avgpool(x.permute(0, 3, 1, 2)), 1)   # (B, feat_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, H, W)
        Returns:
            (B, fusion_dim) — projected visual features
        """
        feats = self._extract(x)          # frozen
        return self.proj(feats)           # trainable projection


# ── 2. Text Encoder ───────────────────────────────────────────────────────────

class TextEncoder(nn.Module):
    """
    Lightweight BERT-based text encoder f_t.

    Embeds category-name prompts (e.g. "gun", "knife") into the same
    fusion_dim space as visual features.

    Only the projection layer is trained; BERT weights are frozen.

    Args:
        model_name : HuggingFace model id (default: "bert-base-uncased")
        text_dim   : hidden size of the chosen BERT model
        fusion_dim : output projection dimension
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        text_dim:   int = 768,
        fusion_dim: int = 256,
    ) -> None:
        super().__init__()
        if not _HF_OK:
            raise ImportError(
                "transformers is required for the text encoder.\n"
                "Install: pip install transformers"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert      = AutoModel.from_pretrained(model_name)

        # Freeze BERT
        for p in self.bert.parameters():
            p.requires_grad = False

        # Trainable projection: text_dim → fusion_dim
        self.proj = nn.Linear(text_dim, fusion_dim)

    def forward(self, class_names: list[str], device: torch.device) -> torch.Tensor:
        """
        Encode a list of class-name strings into fixed-size vectors.

        Args:
            class_names : list of N strings, e.g. ["gun", "knife", ...]
            device      : target device for output tensor

        Returns:
            (N, fusion_dim) text embeddings
        """
        tokens = self.tokenizer(
            class_names,
            padding     = True,
            truncation  = True,
            return_tensors = "pt",
        ).to(device)

        with torch.no_grad():
            out = self.bert(**tokens)               # frozen

        # CLS token as sentence representation
        cls_feats = out.last_hidden_state[:, 0, :]  # (N, text_dim)
        return self.proj(cls_feats)                 # (N, fusion_dim)


# ── 3. Spatial-Visual Fusion MLP ─────────────────────────────────────────────

class SpatialFusionMLP(nn.Module):
    """
    Projects fused [visual_feat ; bbox_coords] into fusion_dim.

    Implements:  ṽ_i = MLP([v_i ; b_i])   (paper §4.4)

    where b_i ∈ ℝ⁴ are normalised bounding box coordinates [x1,y1,x2,y2].

    Args:
        visual_dim : dimension of v_i  (= fusion_dim from vision encoder)
        fusion_dim : output dimension
    """

    def __init__(self, visual_dim: int = 256, fusion_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim + 4, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, fusion_dim),
        )

    def forward(
        self,
        visual_feats: torch.Tensor,   # (B, K, visual_dim)
        bbox_coords:  torch.Tensor,   # (B, K, 4)  normalised [0,1]
    ) -> torch.Tensor:
        """Returns (B, K, fusion_dim)."""
        combined = torch.cat([visual_feats, bbox_coords], dim=-1)  # (B, K, D+4)
        return self.mlp(combined)                                   # (B, K, fusion_dim)


# ── 4. Two-Stream Transformer Decoder Layer ───────────────────────────────────

class TwoStreamDecoderLayer(nn.Module):
    """
    One layer of the two-stream transformer decoder (paper §4.4).

    Step 1 — Self-Attention (SA):
        Visual tokens attend to each other.
        Text   tokens attend to each other.

    Step 2 — Bidirectional Cross-Attention (CA):
        Image → Text:  visual queries attend to text keys/values.
        Text  → Image: text queries attend to visual keys/values.

    Args:
        d_model    : embedding dimension (= fusion_dim)
        num_heads  : attention heads
        ffn_dim    : feed-forward inner dimension
        dropout    : dropout probability
    """

    def __init__(
        self,
        d_model:   int = 256,
        num_heads: int = 8,
        ffn_dim:   int = 1024,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()

        # Self-attention for each stream
        self.sa_vis  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.sa_txt  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Bidirectional cross-attention
        self.ca_v2t  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ca_t2v  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Feed-forward networks
        self.ffn_vis = self._make_ffn(d_model, ffn_dim, dropout)
        self.ffn_txt = self._make_ffn(d_model, ffn_dim, dropout)

        # Layer norms
        self.norm_vis_sa  = nn.LayerNorm(d_model)
        self.norm_txt_sa  = nn.LayerNorm(d_model)
        self.norm_vis_ca  = nn.LayerNorm(d_model)
        self.norm_txt_ca  = nn.LayerNorm(d_model)
        self.norm_vis_ffn = nn.LayerNorm(d_model)
        self.norm_txt_ffn = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _make_ffn(d_model: int, ffn_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        vis: torch.Tensor,   # (B, K, D) visual token sequence
        txt: torch.Tensor,   # (B, C, D) text  token sequence
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vis : (B, K, D) fused visual region tokens
            txt : (B, C, D) class text embedding tokens

        Returns:
            vis_out : (B, K, D) updated visual tokens
            txt_out : (B, C, D) updated text   tokens
        """
        # ── Step 1: Self-Attention ────────────────────────────────────────────
        vis_sa, _ = self.sa_vis(vis, vis, vis)
        vis = self.norm_vis_sa(vis + self.dropout(vis_sa))

        txt_sa, _ = self.sa_txt(txt, txt, txt)
        txt = self.norm_txt_sa(txt + self.dropout(txt_sa))

        # ── Step 2: Bidirectional Cross-Attention ─────────────────────────────
        # Image → Text:  visual queries, text keys & values  (Eq. 2)
        vis_ca, _ = self.ca_v2t(query=vis, key=txt, value=txt)
        vis = self.norm_vis_ca(vis + self.dropout(vis_ca))

        # Text → Image:  text queries, visual keys & values  (reversed roles)
        txt_ca, _ = self.ca_t2v(query=txt, key=vis, value=vis)
        txt = self.norm_txt_ca(txt + self.dropout(txt_ca))

        # ── Step 3: Feed-Forward ──────────────────────────────────────────────
        vis = self.norm_vis_ffn(vis + self.ffn_vis(vis))
        txt = self.norm_txt_ffn(txt + self.ffn_txt(txt))

        return vis, txt


# ── 5. Detection Head (FCN) ───────────────────────────────────────────────────

class DetectionHead(nn.Module):
    """
    Two-branch FCN detection head.

    Branch 1 (class head) : fusion_dim → num_classes + 1  (includes background)
    Branch 2 (box  head)  : fusion_dim → 4  (normalised [x1,y1,x2,y2])

    Internally the box branch predicts (cx, cy, w, h) — each passed through
    a sigmoid so all four lie in [0,1] — and is converted to [x1,y1,x2,y2]
    before being returned. This guarantees x2 > x1 and y2 > y1 by
    construction (since w, h >= 0), which a raw 4-way independent sigmoid
    over [x1,y1,x2,y2] does NOT guarantee — that earlier parameterisation
    allowed degenerate/inverted boxes (x2 < x1) that blow up GIoU into
    huge, meaningless values (including large negative "losses").

    Args:
        fusion_dim  : input dimension
        num_classes : number of threat categories (background added internally)
        hidden_dim  : intermediate FCN width
    """

    def __init__(
        self,
        fusion_dim:  int = 256,
        num_classes: int = 6,
        hidden_dim:  int = 256,
    ) -> None:
        super().__init__()

        self.class_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes + 1),    # +1 for background
        )
        # Predicts (cx, cy, w, h), each in [0,1] via sigmoid. Converted to
        # [x1,y1,x2,y2] in forward() so x2>x1 and y2>y1 hold by construction.
        self.box_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),                               # normalise to [0,1]
        )

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert (cx, cy, w, h) ∈ [0,1] to (x1, y1, x2, y2).

        Because cx, cy, w, h all come from a sigmoid (so w, h >= 0), the
        resulting x2 >= x1 and y2 >= y1 always hold — no degenerate boxes.
        Final clamp to [0,1] guards against boxes that spill slightly
        outside the image when a box is centred near an edge.
        """
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)

    def forward(
        self, queries: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries : (B, Q, fusion_dim)

        Returns:
            class_logits : (B, Q, num_classes + 1)
            boxes        : (B, Q, 4)  normalised [x1,y1,x2,y2], guaranteed
                           valid (x2 > x1, y2 > y1)
        """
        cxcywh = self.box_head(queries)               # (B, Q, 4) in [0,1]
        boxes  = self._cxcywh_to_xyxy(cxcywh)          # (B, Q, 4) valid xyxy
        return self.class_head(queries), boxes


# ── 6. VisionLanguageDetector (full Stage 3 model) ───────────────────────────

class VisionLanguageDetector(nn.Module):
    """
    Full Stage 3 model: vision-language detection and alignment.

    Combines FrozenVisionEncoder, TextEncoder, SpatialFusionMLP,
    TwoStreamDecoder, and DetectionHead into a single forward pass.

    Trainable components (only):
        - vision encoder projection layer
        - text   encoder projection layer
        - SpatialFusionMLP
        - TwoStreamDecoder (all layers)
        - DetectionHead

    Frozen:
        - SwinV2 backbone (loaded from Stage 1)
        - BERT            (pretrained, language prior only)

    Args:
        num_classes       : number of threat categories
        backbone_variant  : "swin_v2_t" | "swin_v2_s" | "swin_v2_b"
        text_encoder_name : HuggingFace BERT model id
        text_feat_dim     : BERT hidden size
        fusion_dim        : shared projection dimension
        num_heads         : transformer attention heads
        num_decoder_layers: number of two-stream decoder layers
        ffn_dim           : feed-forward inner dimension
        dropout           : dropout probability
        num_queries       : max detections per image
        stage1_ckpt       : path to Stage 1 BYOL checkpoint (optional)
        class_names       : list of category name strings (for text prompts)
    """

    def __init__(
        self,
        num_classes:        int,
        backbone_variant:   str   = "swin_v2_t",
        text_encoder_name:  str   = "bert-base-uncased",
        text_feat_dim:      int   = 768,
        fusion_dim:         int   = 256,
        num_heads:          int   = 8,
        num_decoder_layers: int   = 3,
        ffn_dim:            int   = 1024,
        dropout:            float = 0.1,
        num_queries:        int   = 100,
        stage1_ckpt:        Optional[str] = None,
        class_names:        Optional[list[str]] = None,
    ) -> None:
        super().__init__()

        self.num_classes  = num_classes
        self.num_queries  = num_queries
        self.fusion_dim   = fusion_dim

        # Store class names for text prompt generation
        self.class_names  = class_names or [f"class_{i}" for i in range(num_classes)]

        # ── Sub-modules ───────────────────────────────────────────────────────

        # f_v : frozen SwinV2 + trainable projection
        self.vision_encoder = FrozenVisionEncoder(
            variant    = backbone_variant,
            fusion_dim = fusion_dim,
            ckpt_path  = stage1_ckpt,
        )

        # f_t : frozen BERT + trainable projection
        self.text_encoder = TextEncoder(
            model_name = text_encoder_name,
            text_dim   = text_feat_dim,
            fusion_dim = fusion_dim,
        )

        # ṽ_i = MLP([v_i ; b_i])
        self.spatial_mlp = SpatialFusionMLP(
            visual_dim = fusion_dim,
            fusion_dim = fusion_dim,
        )

        # Two-stream transformer decoder stack
        self.decoder = nn.ModuleList([
            TwoStreamDecoderLayer(fusion_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Language-guided learnable query embeddings: (num_queries, fusion_dim)
        # Initialised by text embeddings at forward time; these are the
        # trainable offsets added to the text-initialised queries.
        self.query_embed = nn.Embedding(num_queries, fusion_dim)

        # Detection head: queries → class logits + box coords
        self.det_head = DetectionHead(fusion_dim, num_classes)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        images:       torch.Tensor,          # (B, 3, H, W)
        bbox_proposals: torch.Tensor,         # (B, K, 4) normalised [0,1]
        pseudo_labels:  torch.Tensor,         # (B, K)    class indices (int)
        region_images:  Optional[torch.Tensor] = None,  # (B*K, 3, h, w) pre-cropped
    ) -> dict[str, torch.Tensor]:
        """
        Full Stage 3 forward pass.

        Args:
            images         : batch of full X-ray images
            bbox_proposals : K region proposals per image (normalised coords)
            pseudo_labels  : predicted class index per proposal (from Stage 2)
            region_images  : optionally pre-cropped region images.  If None,
                             the full image features are reused (faster but
                             coarser; cropping is recommended for best results).

        Returns dict containing:
            class_logits   : (B, Q, num_classes+1) — classification scores
            pred_boxes     : (B, Q, 4)             — predicted boxes
            visual_embeds  : (B, K, fusion_dim)    — fused ṽ (for VLC loss)
            text_embeds    : (B, K, fusion_dim)    — text t  (for VLC loss)
        """
        B, K, _ = bbox_proposals.shape
        device  = images.device

        # ── 1. Visual features ────────────────────────────────────────────────
        if region_images is not None:
            # (B*K, 3, h, w) → (B*K, fusion_dim) → (B, K, fusion_dim)
            v_flat  = self.vision_encoder(region_images)
            v_i     = v_flat.view(B, K, self.fusion_dim)
        else:
            # Reuse whole-image features for all K proposals (fast fallback)
            v_whole = self.vision_encoder(images)                   # (B, fusion_dim)
            v_i     = v_whole.unsqueeze(1).expand(-1, K, -1)        # (B, K, fusion_dim)

        # ── 2. Spatial fusion: ṽ_i = MLP([v_i ; b_i]) ────────────────────────
        vis_fused = self.spatial_mlp(v_i, bbox_proposals)           # (B, K, fusion_dim)

        # ── 3. Text embeddings ────────────────────────────────────────────────
        # Map pseudo-label indices → class name strings, then encode with BERT
        # Result: (B, K, fusion_dim)
        txt_tokens = self._get_text_embeddings(pseudo_labels, device)  # (B, K, D)

        # ── 4. Two-stream transformer decoder ─────────────────────────────────
        vis_out = vis_fused    # (B, K, fusion_dim)
        txt_out = txt_tokens   # (B, K, fusion_dim)
        for layer in self.decoder:
            vis_out, txt_out = layer(vis_out, txt_out)

        # ── 5. Language-guided query initialisation ───────────────────────────
        # Learnable query offsets added to text-derived tokens,
        # clamped to num_queries to handle variable K.
        q_idx     = torch.arange(min(K, self.num_queries), device=device)
        q_offsets = self.query_embed(q_idx)                         # (q, fusion_dim)

        # Pad/crop vis_out to exactly num_queries
        q_len   = min(K, self.num_queries)
        queries = vis_out[:, :q_len, :] + q_offsets.unsqueeze(0)    # (B, q, D)

        # ── 6. Detection head ─────────────────────────────────────────────────
        class_logits, pred_boxes = self.det_head(queries)           # (B,q,C+1), (B,q,4)

        return {
            "class_logits":  class_logits,                          # (B, q, C+1)
            "pred_boxes":    pred_boxes,                            # (B, q, 4)
            "visual_embeds": vis_fused[:, :q_len, :],               # (B, q, D) — for VLC
            "text_embeds":   txt_tokens[:, :q_len, :],              # (B, q, D) — for VLC
        }

    # ── Text embedding helper ─────────────────────────────────────────────────

    def _get_text_embeddings(
        self,
        label_indices: torch.Tensor,    # (B, K) int
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert a (B, K) tensor of class indices into (B, K, fusion_dim)
        text embeddings using the frozen BERT encoder.

        Each unique class name is encoded once, then gathered to build
        the full (B, K, D) matrix efficiently.
        """
        B, K = label_indices.shape

        # Encode every class name once  → (num_classes, fusion_dim)
        all_text_embeds = self.text_encoder(self.class_names, device)   # (C, D)

        # Clamp indices to valid range (guard against background index C)
        idx_clamped = label_indices.clamp(0, self.num_classes - 1)      # (B, K)

        # Gather:  (B, K, D)
        idx_exp = idx_clamped.unsqueeze(-1).expand(-1, -1, self.fusion_dim)
        text_embeds = all_text_embeds.unsqueeze(0).expand(B, -1, -1)    # (B, C, D)
        return text_embeds.gather(1, idx_exp)                           # (B, K, D)

    # ── Inference only ────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        images:         torch.Tensor,
        bbox_proposals: torch.Tensor,
        pseudo_labels:  torch.Tensor,
        score_threshold: float = 0.5,
        region_images:   Optional[torch.Tensor] = None,
    ) -> list[dict]:
        """
        Run inference and return filtered detections per image.

        Args:
            images          : (B, 3, H, W)
            bbox_proposals  : (B, K, 4)
            pseudo_labels   : (B, K) int class indices
            score_threshold : minimum softmax confidence to keep
            region_images   : optional (B*K, 3, h, w)

        Returns:
            List of B dicts, each with:
                boxes  : (N, 4)  kept boxes
                scores : (N,)    confidence scores
                labels : (N,)    predicted class indices
        """
        out = self.forward(images, bbox_proposals, pseudo_labels, region_images)

        class_logits = out["class_logits"]   # (B, Q, C+1)
        pred_boxes   = out["pred_boxes"]     # (B, Q, 4)

        probs        = F.softmax(class_logits, dim=-1)      # (B, Q, C+1)
        scores, lbls = probs[..., :-1].max(dim=-1)          # (B, Q) — exclude background

        results = []
        for b in range(images.shape[0]):
            keep  = scores[b] >= score_threshold
            results.append({
                "boxes":  pred_boxes[b][keep],
                "scores": scores[b][keep],
                "labels": lbls[b][keep],
            })
        return results