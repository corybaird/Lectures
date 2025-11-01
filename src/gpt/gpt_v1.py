# gpt_v1.py - Add visualization tracking
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .gpt_layers import LayerNorm, FeedForward
from .transformer_block import TransformerBlockV1

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList([
            TransformerBlockV1(cfg, layer_idx=i) for i in range(cfg["n_layers"])
        ])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
        self.visualize = False
        self.layer_outputs = []
        self.layer_stats = []

    def enable_visualization(self):
        self.visualize = True
        self.layer_outputs = []
        self.layer_stats = []

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        if self.visualize:
            self.layer_outputs = []
            self.layer_stats = []
        
        for i, block in enumerate(self.trf_blocks):
            x = block(x)
            if self.visualize:
                self.layer_outputs.append(x.detach().cpu())
                self.layer_stats.append({
                    'mean': x.mean().item(),
                    'std': x.std().item(),
                    'max': x.max().item(),
                    'min': x.min().item(),
                    'layer': i
                })
        
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def plot_layer_stats(self, save_dir):
        if not self.layer_stats:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        layers = [s['layer'] for s in self.layer_stats]
        
        axes[0, 0].plot(layers, [s['mean'] for s in self.layer_stats], 'o-')
        axes[0, 0].set_title('Layer Output Mean')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Mean')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(layers, [s['std'] for s in self.layer_stats], 'o-', color='orange')
        axes[0, 1].set_title('Layer Output Std')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Std')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(layers, [s['max'] for s in self.layer_stats], 'o-', color='green', label='Max')
        axes[1, 0].plot(layers, [s['min'] for s in self.layer_stats], 'o-', color='red', label='Min')
        axes[1, 0].set_title('Layer Output Range')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        layer_output = self.layer_outputs[-1][0, -1, :].numpy()
        axes[1, 1].hist(layer_output, bins=50, edgecolor='black')
        axes[1, 1].set_title('Final Layer Output Distribution')
        axes[1, 1].set_xlabel('Activation Value')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'layer_statistics.png', dpi=150)
        plt.close()

    def plot_attention_maps(self, save_dir):
        pass