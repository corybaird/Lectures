# main.py - Add verbose argument and tracking
import time
import tiktoken
import torch
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .gpt_v1 import GPTModel as GPTModelV1
from .gpt_kvcache import GPTModel as GPTModelV2
from .generate_utils import generate_text_simple, generate_text_simple_cached

def main(model_version, verbose=False, visualize=False):
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "verbose": verbose
    }

    if model_version == "v2":
        GPT_CONFIG_124M["kv_window_size"] = 1024
        VERSION_NAME = "V2 (KV Cache)"
        print("Importing V2 (KV Cache) Model...")
        model = GPTModelV2(GPT_CONFIG_124M)
        generate_fn = generate_text_simple_cached
    else:
        VERSION_NAME = "V1 (No Cache)"
        print("Importing V1 (No Cache) Model...")
        model = GPTModelV1(GPT_CONFIG_124M)
        generate_fn = generate_text_simple

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if visualize:
        model.enable_visualization()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    print(f"\n{50*'='}\n IN ({VERSION_NAME})\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    if model_version == "v2":
        token_ids = generate_fn(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=200
        )
    else:
        token_ids = generate_fn(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=200,
            context_size=GPT_CONFIG_124M["context_length"]
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start
    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n OUT ({VERSION_NAME})\n{50*'='}")
    print("\nOutput text:", decoded_text)
    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")

# main.py - Updated visualization section
    if visualize:
        print("\nGenerating visualizations...")
        save_dir = Path("visualizations")
        save_dir.mkdir(exist_ok=True)
        
        if model_version == "v2":
            model.plot_all_visualizations(save_dir)
        else:
            model.plot_layer_stats(save_dir)
            model.plot_attention_maps(save_dir)
        
        print(f"Visualizations saved to {save_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT model (V1 or V2).")
    parser.add_argument("--version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--verbose", action="store_true", help="Print layer-by-layer updates")
    parser.add_argument("--visualize", action="store_true", help="Generate attention and layer visualizations")
    args = parser.parse_args()
    main(args.version, args.verbose, args.visualize)