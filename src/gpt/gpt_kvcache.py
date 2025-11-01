# gpt_kvcache.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .gpt_layers import LayerNorm, FeedForward
from .transformer_block import TransformerBlockV2

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList([
            TransformerBlockV2(cfg, layer_idx=i) for i in range(cfg["n_layers"])
        ])
        self.ptr_current_pos = 0
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
        self.visualize = False
        self.layer_outputs = []
        self.layer_stats = []
        self.cache_usage = []
        self.verbose = cfg.get("verbose", False)

    def enable_visualization(self):
        self.visualize = True
        self.layer_outputs = []
        self.layer_stats = []
        self.cache_usage = []

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Forward Pass - use_cache={use_cache}, ptr_pos={self.ptr_current_pos}")
            print(f"Input shape: {in_idx.shape}, batch_size={batch_size}, seq_len={seq_len}")
        
        tok_embeds = self.tok_emb(in_idx)
        
        if use_cache:
            pos_ids = torch.arange(
                self.ptr_current_pos, 
                self.ptr_current_pos + seq_len, 
                device=in_idx.device, 
                dtype=torch.long
            )
            if self.verbose:
                print(f"Position IDs (cached): {pos_ids.tolist()}")
            self.ptr_current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
            if self.verbose:
                print(f"Position IDs (no cache): {pos_ids.tolist()}")
        
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        if self.verbose:
            print(f"After embeddings - mean: {x.mean():.4f}, std: {x.std():.4f}")
        
        if self.visualize:
            if not use_cache:
                self.layer_outputs = []
                self.layer_stats = []
                self.cache_usage = []
        
        for i, blk in enumerate(self.trf_blocks):
            x = blk(x, use_cache=use_cache)
            
            if self.visualize:
                self.layer_outputs.append(x.detach().cpu())
                self.layer_stats.append({
                    'mean': x.mean().item(),
                    'std': x.std().item(),
                    'max': x.max().item(),
                    'min': x.min().item(),
                    'layer': i,
                    'cache_used': use_cache
                })
                
                if use_cache and hasattr(blk.att, 'ptr_cur'):
                    self.cache_usage.append({
                        'layer': i,
                        'cache_pos': blk.att.ptr_cur,
                        'window_size': blk.att.window_size
                    })
        
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        if self.verbose:
            print(f"Output logits shape: {logits.shape}")
            print(f"Logits mean: {logits.mean():.4f}, std: {logits.std():.4f}")
            print(f"{'='*60}\n")
        
        return logits

    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.ptr_current_pos = 0
        if self.verbose:
            print("KV Cache reset")

    def plot_layer_stats(self, save_dir):
        if not self.layer_stats:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        layers = [s['layer'] for s in self.layer_stats]
        
        axes[0, 0].plot(layers, [s['mean'] for s in self.layer_stats], 'o-', linewidth=2)
        axes[0, 0].set_title('Layer Output Mean', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Mean Activation')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(layers, [s['std'] for s in self.layer_stats], 'o-', color='orange', linewidth=2)
        axes[0, 1].set_title('Layer Output Standard Deviation', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('Std Deviation')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(layers, [s['max'] for s in self.layer_stats], 'o-', color='green', label='Max', linewidth=2)
        axes[1, 0].plot(layers, [s['min'] for s in self.layer_stats], 'o-', color='red', label='Min', linewidth=2)
        axes[1, 0].set_title('Layer Output Range', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_ylabel('Activation Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        if self.layer_outputs:
            layer_output = self.layer_outputs[-1][0, -1, :].numpy()
            axes[1, 1].hist(layer_output, bins=50, edgecolor='black', alpha=0.7)
            axes[1, 1].set_title('Final Layer Activation Distribution', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Activation Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'layer_statistics.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_cache_usage(self, save_dir):
        if not self.cache_usage:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        layers = [c['layer'] for c in self.cache_usage]
        cache_pos = [c['cache_pos'] for c in self.cache_usage]
        window_size = self.cache_usage[0]['window_size']
        
        ax1.bar(layers, cache_pos, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axhline(y=window_size, color='red', linestyle='--', linewidth=2, label=f'Window Size ({window_size})')
        ax1.set_title('KV Cache Position per Layer', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Cache Position')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        cache_fill = [(pos / window_size) * 100 for pos in cache_pos]
        colors = ['green' if f < 75 else 'orange' if f < 90 else 'red' for f in cache_fill]
        ax2.bar(layers, cache_fill, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('KV Cache Usage (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Cache Fill %')
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'cache_usage.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_attention_maps(self, save_dir):
        pass

    def plot_all_visualizations(self, save_dir):
        self.plot_layer_stats(save_dir)
        self.plot_cache_usage(save_dir)
        print(f"All visualizations saved to {save_dir}/")