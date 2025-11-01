# transformer_block.py - Add verbose logging
import torch.nn as nn
from .multi_head_attention import MultiHeadAttentionV1, MultiHeadAttentionV2
from .gpt_layers import LayerNorm, FeedForward

class TransformerBlockV1(nn.Module):
    def __init__(self, cfg, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.verbose = cfg.get("verbose", False)
        self.att = MultiHeadAttentionV1(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        self.attn_weights = None

    def forward(self, x):
        if self.verbose:
            print(f"\n--- Layer {self.layer_idx} ---")
            print(f"Input shape: {x.shape}")
            print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
        
        shortcut = x
        x = self.norm1(x)
        
        if self.verbose:
            print(f"After norm1 - mean: {x.mean():.4f}, std: {x.std():.4f}")
        
        x = self.att(x)
        
        if self.verbose:
            print(f"After attention - mean: {x.mean():.4f}, std: {x.std():.4f}")
        
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        if self.verbose:
            print(f"After residual 1 - mean: {x.mean():.4f}, std: {x.std():.4f}")
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        
        if self.verbose:
            print(f"After feedforward - mean: {x.mean():.4f}, std: {x.std():.4f}")
        
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        if self.verbose:
            print(f"After residual 2 - mean: {x.mean():.4f}, std: {x.std():.4f}")
        
        return x

class TransformerBlockV2(nn.Module):
    def __init__(self, cfg, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.verbose = cfg.get("verbose", False)
        self.att = MultiHeadAttentionV2(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            window_size=cfg.get("kv_window_size", cfg["context_length"])
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        if self.verbose:
            print(f"\n--- Layer {self.layer_idx} (KV Cache: {use_cache}) ---")
            print(f"Input shape: {x.shape}")
            print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
        
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_cache=use_cache)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x