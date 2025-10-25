# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-4.
# This file can be run as a standalone script.

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """
    Implements Layer Normalization.

    This layer normalizes the activations across the features (embedding dimension)
    for each token independently. It helps stabilize training by maintaining
    a consistent distribution of activations.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # A small value added to the variance for numerical stability
        # A learnable gain parameter (gamma) initialized to 1
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # A learnable bias parameter (beta) initialized to 0
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # Calculate mean across the last dimension (embedding dimension)
        mean = x.mean(dim=-1, keepdim=True)
        # Calculate variance across the last dimension (embedding dimension)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize the input
        # (x - mean) / sqrt(var + epsilon)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply the learnable scale and shift parameters
        # gamma * norm_x + beta
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Implements the Gaussian Error Linear Unit (GELU) activation function.

    This is a smooth, non-linear activation function used in transformers (like GPT)
    as an alternative to ReLU. It uses an approximation of the GELU formula.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # This is the approximate GELU formula used in the original GPT-2 paper:
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    Implements the position-wise Feed-Forward Network (FFN).

    This block consists of two linear transformations with a GELU activation
    in between. It is applied independently to each token's embedding.
    The dimensionality is first expanded (usually by 4x) and then
    projected back to the original embedding dimension.
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            # Expand the embedding dimension by a factor of 4
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            # Apply the GELU non-linear activation
            GELU(),
            # Project back to the original embedding dimension
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    Implements a single Transformer block (decoder-style).

    This block contains two main sub-layers:
    1. Multi-Head Attention (self-attention)
    2. Feed-Forward Network (position-wise)

    Each sub-layer uses pre-normalization (LayerNorm before the operation)
    and a residual connection (adding the input back to the output).
    Dropout is also applied after each sub-layer.
    """
    def __init__(self, cfg):
        super().__init__()
        # First sub-layer: Multi-Head Attention
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        
        # Second sub-layer: Feed-Forward Network
        self.ff = FeedForward(cfg)
        
        # Normalization layer for the attention sub-layer
        self.norm1 = LayerNorm(cfg["emb_dim"])
        # Normalization layer for the feed-forward sub-layer
        self.norm2 = LayerNorm(cfg["emb_dim"])
        
        # Dropout for the residual connections
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # --- First Sub-layer: Multi-Head Attention ---
        
        # Save the original input for the residual connection
        shortcut = x
        
        # 1. Pre-normalization: Normalize the input before attention
        x = self.norm1(x)
        
        # 2. Multi-Head Attention: Compute attention
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        
        # 3. Dropout: Apply dropout to the attention output
        x = self.drop_shortcut(x)
        
        # 4. Residual Connection: Add the original input back
        x = x + shortcut

        # --- Second Sub-layer: Feed-Forward Network ---
        
        # Save the input to this sub-layer for the residual connection
        shortcut = x
        
        # 1. Pre-normalization: Normalize the input before the FFN
        x = self.norm2(x)
        
        # 2. Feed-Forward Network: Apply the FFN
        x = self.ff(x)
        
        # 3. Dropout: Apply dropout to the FFN output
        x = self.drop_shortcut(x)
        
        # 4. Residual Connection: Add the input back
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    args = parser.parse_args()
    
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": False
    }
    
    print(f"\n{50*'='}\n{20*' '}CONFIG\n{50*'='}")
    print(f"Embedding dimension: {args.emb_dim}")
    print(f"Number of heads: {args.n_heads}")
    print(f"Number of layers: {args.n_layers}")
    print(f"Context length: {args.context_length}")
    print(f"Dropout rate: {args.drop_rate}")
    print(f"Max new tokens: {args.max_new_tokens}")
    
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    
    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)
    
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=args.max_new_tokens,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    
    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-4.
# This file can be run as a standalone script.

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """
    Implements Layer Normalization.

    This layer normalizes the activations across the features (embedding dimension)
    for each token independently. It helps stabilize training by maintaining
    a consistent distribution of activations.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # A small value added to the variance for numerical stability
        # A learnable gain parameter (gamma) initialized to 1
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # A learnable bias parameter (beta) initialized to 0
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # Calculate mean across the last dimension (embedding dimension)
        mean = x.mean(dim=-1, keepdim=True)
        # Calculate variance across the last dimension (embedding dimension)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize the input
        # (x - mean) / sqrt(var + epsilon)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply the learnable scale and shift parameters
        # gamma * norm_x + beta
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Implements the Gaussian Error Linear Unit (GELU) activation function.

    This is a smooth, non-linear activation function used in transformers (like GPT)
    as an alternative to ReLU. It uses an approximation of the GELU formula.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # This is the approximate GELU formula used in the original GPT-2 paper:
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    Implements the position-wise Feed-Forward Network (FFN).

    This block consists of two linear transformations with a GELU activation
    in between. It is applied independently to each token's embedding.
    The dimensionality is first expanded (usually by 4x) and then
    projected back to the original embedding dimension.
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            # Expand the embedding dimension by a factor of 4
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            # Apply the GELU non-linear activation
            GELU(),
            # Project back to the original embedding dimension
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    Implements a single Transformer block (decoder-style).

    This block contains two main sub-layers:
    1. Multi-Head Attention (self-attention)
    2. Feed-Forward Network (position-wise)

    Each sub-layer uses pre-normalization (LayerNorm before the operation)
    and a residual connection (adding the input back to the output).
    Dropout is also applied after each sub-layer.
    """
    def __init__(self, cfg):
        super().__init__()
        # First sub-layer: Multi-Head Attention
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        
        # Second sub-layer: Feed-Forward Network
        self.ff = FeedForward(cfg)
        
        # Normalization layer for the attention sub-layer
        self.norm1 = LayerNorm(cfg["emb_dim"])
        # Normalization layer for the feed-forward sub-layer
        self.norm2 = LayerNorm(cfg["emb_dim"])
        
        # Dropout for the residual connections
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # --- First Sub-layer: Multi-Head Attention ---
        
        # Save the original input for the residual connection
        shortcut = x
        
        # 1. Pre-normalization: Normalize the input before attention
        x = self.norm1(x)
        
        # 2. Multi-Head Attention: Compute attention
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        
        # 3. Dropout: Apply dropout to the attention output
        x = self.drop_shortcut(x)
        
        # 4. Residual Connection: Add the original input back
        x = x + shortcut

        # --- Second Sub-layer: Feed-Forward Network ---
        
        # Save the input to this sub-layer for the residual connection
        shortcut = x
        
        # 1. Pre-normalization: Normalize the input before the FFN
        x = self.norm2(x)
        
        # 2. Feed-Forward Network: Apply the FFN
        x = self.ff(x)
        
        # 3. Dropout: Apply dropout to the FFN output
        x = self.drop_shortcut(x)
        
        # 4. Residual Connection: Add the input back
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    args = parser.parse_args()
    
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": False
    }
    
    print(f"\n{50*'='}\n{20*' '}CONFIG\n{50*'='}")
    print(f"Embedding dimension: {args.emb_dim}")
    print(f"Number of heads: {args.n_heads}")
    print(f"Number of layers: {args.n_layers}")
    print(f"Context length: {args.context_length}")
    print(f"Dropout rate: {args.drop_rate}")
    print(f"Max new tokens: {args.max_new_tokens}")
    
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    
    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)
    
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=args.max_new_tokens,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    
    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)