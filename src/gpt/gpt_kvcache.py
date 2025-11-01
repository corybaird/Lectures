# gpt_kvcache.py
# Description: Defines the V2 GPT model with KV caching and sliding window support.

import torch
import torch.nn as nn
from .gpt_layers import LayerNorm, FeedForward
from .transformer_block import TransformerBlockV2

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList([TransformerBlockV2(cfg) for _ in range(cfg["n_layers"])])
        self.ptr_current_pos = 0
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        if use_cache:
            pos_ids = torch.arange(self.ptr_current_pos, self.ptr_current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.ptr_current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.ptr_current_pos = 0