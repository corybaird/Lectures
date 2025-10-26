import os
import argparse
from dotenv import load_dotenv
import ollama
import re

load_dotenv()

class LLMInferenceOllama:
    def __init__(self, model_name="llama3.2:latest", host="http://localhost:11434", **kwargs):
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=self.host)
        self.num_predict = kwargs.get('max_tokens')
        self.temperature = kwargs.get('temperature')
        self.top_p = kwargs.get('top_p')
        self.top_k = kwargs.get('top_k')

    def list_models(self):
        try:
            models = self.client.list()
            return models
        except Exception as e:
            print(f"Error listing models: {e}")
            return None

    def load_prompt(self, prompt_dict, verbose=False):
        if verbose:
            print(f"Instruction Length: {len(prompt_dict['instruction'])}")
        self.system_message = prompt_dict['instruction']
        self.user_message_content = prompt_dict['output_format']

    def extract_possible_thought(self, response):
        """
        Automatically detects and extracts train of thought (CoT) if present.
        Returns a tuple of (thought, final_answer) or (None, response) if no CoT found.
        """
        thought_patterns = [
            r"(thought[s]?|reasoning|let's think|step\s+\d+|first[,:\s])",
            r"(because|so that|therefore|thus|in order to)"
        ]

        if any(re.search(pat, response.lower()) for pat in thought_patterns):
            split_match = re.split(r"(In (summary|conclusion)|Therefore|Final answer[:\s])", response, flags=re.IGNORECASE)
            if len(split_match) >= 2:
                thought = split_match[0].strip()
                rest = ''.join(split_match[1:]).strip()
                return thought, rest
            else:
                return response.strip(), None
        return None, response.strip()

    def infer(self, max_tokens_response=None, extract_thought=False):
        try:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": self.user_message_content}
            ]

            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "stream": False
            }

            options = {}
            num_predict_val = max_tokens_response or self.num_predict
            if num_predict_val:
                options["num_predict"] = num_predict_val
            if self.temperature is not None:
                options["temperature"] = self.temperature
            if self.top_p is not None:
                options["top_p"] = self.top_p
            if self.top_k is not None:
                options["top_k"] = self.top_k
            if options:
                kwargs["options"] = options

            response = self.client.chat(**kwargs)
            output = response["message"]["content"]

            if extract_thought:
                thought, final_answer = self.extract_possible_thought(output)
                return {
                    "full": output,
                    "thought": thought,
                    "answer": final_answer
                }

            return output

        except Exception as e:
            print(f"Error during inference: {e}")


import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM inference via Ollama")
    parser.add_argument("--model", type=str, default="magistral:latest", help="Ollama model name")
    args = parser.parse_args()

    llm = LLMInferenceOllama(model_name=args.model)

    models = llm.list_models()
    if models:
        print("Installed Ollama models:")
        for model in models['models']:
            print(f"{model['model']}")
    else:
        print("No models found or error occurred")

    prompt_dict = {
        "instruction": "You are an economist.",
        "output_format": "Tell me about capital controls. Provide me an answer in bullet points"
    }

    llm.load_prompt(prompt_dict, verbose=True)

    print(f"\nRunning inference on model: {args.model} ...")
    start_time = time.time()

    result = llm.infer(max_tokens_response=200, extract_thought=True)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Inference completed in {duration:.2f} seconds.")

    if result:
        if isinstance(result, dict):
            print("\n--- Train of Thought ---")
            print(result.get("thought", "None"))
            print("\n--- Final Answer ---")
            print(result.get("answer", "None"))
        else:
            print("Response:")
            print(result)
    else:
        print("Failed to get response from Ollama")


