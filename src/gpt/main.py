import time
import tiktoken
import torch
import argparse
from .gpt_v1 import GPTModel as GPTModelV1
from .gpt_kvcache import GPTModel as GPTModelV2
from .generate_utils import generate_text_simple, generate_text_simple_cached

def main(model_version):
    # Base config
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    # Add KV cache config only if needed
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

    # Run the selected generation function
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT model (V1 or V2).")
    parser.add_argument("--version", type=str, default="v1", choices=["v1", "v2"],
                        help="Select model version: 'v1' (no cache) or 'v2' (KV cache).")
    args = parser.parse_args()
    main(args.version)


# # main.py
# # Description: Main execution script to run either V1 (no cache)
# #              or V2 (KV cache) model implementations.

# import time
# import tiktoken
# import torch
# import sys

# # --- Configuration Flag ---
# # Set to False to run the V1 model
# # Set to True to run the V2 (KV cache) model
# USE_KV_CACHE = True
# # --------------------------


# # Dynamically import the correct model and generation function
# if USE_KV_CACHE:
#     print("Importing V2 (KV Cache) Model...")
#     from .gpt_kvcache import GPTModel
#     from .modules.generate_utils import generate_text_simple_cached as generate_fn
#     VERSION_NAME = "V2 (KV Cache)"
# else:
#     print("Importing V1 (No Cache) Model...")
#     from .gpt_v1 import GPTModel
#     from .modules.generate_utils import generate_text_simple as generate_fn
#     VERSION_NAME = "V1 (No Cache)"


# def main():
#     # Base config
#     GPT_CONFIG_124M = {
#         "vocab_size": 50257,
#         "context_length": 1024,
#         "emb_dim": 768,
#         "n_heads": 12,
#         "n_layers": 12,
#         "drop_rate": 0.1,
#         "qkv_bias": False
#     }
    
#     # Add KV cache config only if needed
#     if USE_KV_CACHE:
#         GPT_CONFIG_124M["kv_window_size"] = 1024

#     torch.manual_seed(123)
#     model = GPTModel(GPT_CONFIG_124M)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()

#     start_context = "Hello, I am"
#     tokenizer = tiktoken.get_encoding("gpt2")
#     encoded = tokenizer.encode(start_context)
#     encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

#     print(f"\n{50*'='}\n IN ({VERSION_NAME})\n{50*'='}")
#     print("\nInput text:", start_context)
#     print("Encoded input text:", encoded)

#     if torch.cuda.is_available(): torch.cuda.synchronize()
#     start = time.time()

#     # --- Run the selected generation function ---
#     if USE_KV_CACHE:
#         token_ids = generate_fn(
#             model=model,
#             idx=encoded_tensor,
#             max_new_tokens=200
#         )
#     else:
#         token_ids = generate_fn(
#             model=model,
#             idx=encoded_tensor,
#             max_new_tokens=200,
#             context_size=GPT_CONFIG_124M["context_length"]
#         )
#     # ---------------------------------------------
    
#     if torch.cuda.is_available(): torch.cuda.synchronize()
#     total_time = time.time() - start

#     decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

#     print(f"\n\n{50*'='}\n OUT ({VERSION_NAME})\n{50*'='}")
#     print("\nOutput text:", decoded_text)
#     print(f"\nTime: {total_time:.2f} sec")
#     print(f"{int(len(token_ids[0])/total_time)} tokens/sec")

# if __name__ == "__main__":
#     main()