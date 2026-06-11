# Stage_3/stage3_config.py
#
# All hyper-parameters for Stage 3: Vision-Language Detection & Alignment.
# Values follow the paper §4.4 and §5.1 unless otherwise noted.

STAGE3_CONFIG = {
    #  Visual encoder (frozen SwinV2) 
    "backbone_variant":    "swin_v2_t",   # must match Stage 1 checkpoint
    "backbone_pretrained": True,

    # Text encoder 
    # Lightweight BERT used as the language encoder f_t.
    # "bert-base-uncased" is the default; "prajjwal1/bert-tiny" is faster.
    "text_encoder_name":  "bert-base-uncased",
    "text_feat_dim":      768,            # hidden size of the chosen BERT

    # Cross-modal fusion transformer
    "fusion_dim":         256,            # projection dim for both modalities
    "num_heads":          8,              # attention heads in SA / CA layers
    "num_decoder_layers": 3,             # stack depth for the two-stream decoder
    "ffn_dim":            1024,           # feed-forward inner dim
    "dropout":            0.1,

    # Detection head
    "num_queries":        100,            # max detections per image

    # Contrastive loss (Eq. 3) 
    "temperature":        0.07,           # τ — temperature scaling factor

    # Training
    "epochs":             50,             # paper §5.1
    "batch_size":         32,             # paper §5.1
    "lr":                 1e-4,           # AdamW initial lr  (paper §5.1)
    "lr_min":             1e-6,           # cosine schedule floor
    "weight_decay":       1e-4,
    "grad_clip":          1.0,
    "num_workers":        4,

    # ── Pseudo-label refresh cadence ─────────────────────────────────────
    "pseudo_refresh_every": 10,           # re-generate every N epochs (paper §4.3)

    # Loss weights 
    "loss_weight_cls":   1.0,             # classification CE
    "loss_weight_bbox":  5.0,             # L1 box regression
    "loss_weight_giou":  2.0,             # GIoU box regression
    "loss_weight_vlc":   1.0,             # vision-language contrastive

    # Shared
    "num_classes":        6,              # updated by main.py per dataset
    "img_size":           224,            # must match Stage 1 / Stage 2

    #  I/O 
    "conf_threshold":     0.65,           # kept consistent with Stage 2
    "iou_threshold":      0.45,
}
