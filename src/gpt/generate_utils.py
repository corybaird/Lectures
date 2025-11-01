# generate_utils.py
# Description: Utility functions for generating text, with and without cache.

import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """V1 generation function (no cache)."""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def generate_text_simple_cached(model, idx, max_new_tokens, context_size=None, use_cache=True):
    """V2 generation function (with cache support)."""
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            model.reset_kv_cache()
            # Process the initial prompt (context)
            logits = model(idx[:, -ctx_len:], use_cache=True)
            
            # Generate new tokens one by one
            for _ in range(max_new_tokens):
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)
                logits = model(next_idx, use_cache=True)
        else:
            # Fallback to V1 behavior if cache is off
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)
    return idx