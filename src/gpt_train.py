import matplotlib.pyplot as plt
import os
import requests
import torch
import tiktoken
import argparse

from .gpt_dummy import GPTModel, create_dataloader_v1, generate_text_simple


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


class GPTTrainer:
    def __init__(self, model, config, settings, device):
        self.model = model.to(device)
        self.config = config
        self.settings = settings
        self.device = device
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=settings["learning_rate"], 
            weight_decay=settings["weight_decay"]
        )
        
        self.train_losses = []
        self.val_losses = []
        self.tokens_seen = []
        self.train_loader = None
        self.val_loader = None
    
    def prepare_data(self, text_data, train_ratio=0.90):
        split_idx = int(train_ratio * len(text_data))
        
        self.train_loader = create_dataloader_v1(
            text_data[:split_idx],
            batch_size=self.settings["batch_size"],
            max_length=self.config["context_length"],
            stride=self.config["context_length"],
            drop_last=True,
            shuffle=True,
            num_workers=0
        )
        
        self.val_loader = create_dataloader_v1(
            text_data[split_idx:],
            batch_size=self.settings["batch_size"],
            max_length=self.config["context_length"],
            stride=self.config["context_length"],
            drop_last=False,
            shuffle=False,
            num_workers=0
        )
    
    def train_step(self, input_batch, target_batch):
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        
        self.optimizer.zero_grad()
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), 
            target_batch.flatten()
        )
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, eval_iter=1):
        self.model.eval()
        with torch.no_grad():
            train_loss = self._calc_loader_loss(self.train_loader, eval_iter)
            val_loss = self._calc_loader_loss(self.val_loader, eval_iter)
        self.model.train()
        return train_loss, val_loss
    
    def _calc_loader_loss(self, loader, num_batches):
        total_loss = 0.
        if len(loader) == 0:
            return float("nan")
        
        num_batches = min(num_batches or len(loader), len(loader))
        
        for i, (input_batch, target_batch) in enumerate(loader):
            if i >= num_batches:
                break
            input_batch = input_batch.to(self.device)
            target_batch = target_batch.to(self.device)
            logits = self.model(input_batch)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), 
                target_batch.flatten()
            )
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(self, num_epochs, eval_freq=5, eval_iter=1, start_context="Every effort moves you"):
        tokens_seen = 0
        global_step = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            
            for input_batch, target_batch in self.train_loader:
                loss = self.train_step(input_batch, target_batch)
                tokens_seen += input_batch.numel()
                
                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate(eval_iter=eval_iter)
                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)
                    self.tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                
                global_step += 1
            
            self.generate_sample(start_context)
    
    def generate_sample(self, start_context, max_tokens=50):
        self.model.eval()
        context_size = self.model.pos_emb.weight.shape[0]
        
        encoded = text_to_token_ids(start_context, self.tokenizer).to(self.device)
        
        with torch.no_grad():
            token_ids = generate_text_simple(
                self.model, encoded, max_tokens, context_size
            )
            decoded = token_ids_to_text(token_ids, self.tokenizer)
            print(decoded.replace("\n", " "))
        
        self.model.train()
    
    def test(self, test_prompts=None, max_tokens=100):
        if test_prompts is None:
            test_prompts = [
                "The ecomomy was very",
                "Demand is",
                "Supply is too constrained"
            ]
        
        self.model.eval()
        context_size = self.model.pos_emb.weight.shape[0]
        
        print("\n" + "="*50)
        print("Testing Model")
        print("="*50)
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            encoded = text_to_token_ids(prompt, self.tokenizer).to(self.device)
            
            with torch.no_grad():
                token_ids = generate_text_simple(
                    model=self.model, 
                    idx=encoded,
                    max_new_tokens=max_tokens, 
                    context_size=context_size
                )
                decoded_text = token_ids_to_text(token_ids, self.tokenizer)
                print(f"Generated: {decoded_text}\n")
    
    def plot_losses(self):
        if not self.train_losses:
            print("No training data to plot")
            return
        
        epochs_tensor = torch.linspace(0, self.settings["num_epochs"], len(self.train_losses))
        
        fig, ax1 = plt.subplots()
        ax1.plot(epochs_tensor, self.train_losses, label="Training loss")
        ax1.plot(epochs_tensor, self.val_losses, linestyle="-.", label="Validation loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        
        ax2 = ax1.twiny()
        ax2.plot(self.tokens_seen, self.train_losses, alpha=0)
        ax2.set_xlabel("Tokens seen")
        
        fig.tight_layout()
        os.makedirs("reports", exist_ok = True)
        plt.savefig("reports/loss.pdf")
        print("Loss plot saved to reports/loss.pdf")
    
    def save(self, path="models/econ_gpt.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load(self, path="models/econ_gpt.pth"):
        print(os.listdir("src/"))
        self.model.load_state_dict(torch.load("models/econ_gpt.pth", weights_only=True))
        print(f"Model loaded from {path}")


def download_data(file_path="data/llm/economic_training_corpus.txt"):
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    
    return text_data


def main(gpt_config, settings, model_path=None, test_prompts=None):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GPTModel(gpt_config)
    trainer = GPTTrainer(model, gpt_config, settings, device)
    
    if model_path:
        trainer.load(model_path)
        trainer.test(test_prompts)
        return trainer
    
    text_data = download_data()
    trainer.prepare_data(text_data)
    trainer.train(
        num_epochs=settings["num_epochs"],
        eval_freq=5,
        eval_iter=1,
        start_context="The economy is"
    )
    
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test GPT model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model for testing (skips training)')
    parser.add_argument('--prompts', type=str, nargs='+', default=None,
                        help='Custom test prompts')
    args = parser.parse_args()

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 50,
        "batch_size": 2,
        "weight_decay": 0.1
    }

    trainer = main(
        GPT_CONFIG_124M, 
        OTHER_SETTINGS, 
        model_path=args.model_path,
        test_prompts=args.prompts
    )

    if args.model_path is None:
        trainer.plot_losses()
        trainer.save("models/econ_gpt.pth")