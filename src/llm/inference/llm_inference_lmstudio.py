import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class LLMInferenceLMStudio:
    def __init__(self, model_name="liquid/lfm2-1.2b", base_url="http://localhost:1234/v1", **kwargs):
        self.model_name = model_name
        self.base_url = base_url
        self.client = OpenAI(
            api_key="lm-studio",
            base_url=self.base_url
        )
        self.max_tokens = kwargs.get('max_tokens')
        self.temperature = kwargs.get('temperature')
        self.top_p = kwargs.get('top_p')
        self.frequency_penalty = kwargs.get('frequency_penalty')
        self.presence_penalty = kwargs.get('presence_penalty')

    def get_loaded_model(self):
        try:
            models = self.client.models.list()
            loaded_models = [model.id for model in models.data]
            return loaded_models
        except Exception as e:
            print(f"Error getting loaded models: {e}")
            return []

    def check_model_loaded(self, model_name=None):
        target_model = model_name or self.model_name
        loaded_models = self.get_loaded_model()
        is_loaded = target_model in loaded_models
        
        print(f"Checking model: {target_model}")
        print(f"Loaded models: {loaded_models}")
        print(f"Model loaded in memory: {is_loaded}")
        
        return is_loaded

    def load_prompt(self, prompt, system_message=None, verbose=False):
        if verbose:
            print(f"Prompt Length: {len(prompt)}")
        self.system_message = system_message or "You are a helpful assistant."
        self.user_message_content = prompt

    def infer(self, max_tokens_response=None):
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
            
            max_tokens_val = max_tokens_response or self.max_tokens
            if max_tokens_val:
                kwargs["max_tokens"] = max_tokens_val
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            if self.frequency_penalty is not None:
                kwargs["frequency_penalty"] = self.frequency_penalty
            if self.presence_penalty is not None:
                kwargs["presence_penalty"] = self.presence_penalty
            
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during inference: {e}")
            return None

if __name__ == "__main__":
    import time
    import argparse
    parser = argparse.ArgumentParser(description="Run LLM inference via LM Studio")
    parser.add_argument("--model", type=str, default="liquid/lfm2-1.2b", help="LM Studio model name")
    args = parser.parse_args()
    
    llm = LLMInferenceLMStudio(model_name=args.model)

    llm.check_model_loaded()

    prompt = "Tell me about capital controls. Provide me an answer in bullet points"
    system_message = "You are an economist."

    llm.load_prompt(prompt, system_message=system_message, verbose=True)
    
    print(f"\nRunning inference on model: {args.model} ...")
    start_time = time.time()

    result = llm.infer(max_tokens_response=50)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Inference completed in {duration:.2f} seconds.\n")
    if result:
        print(f"Response: {result}")
    else:
        print("Failed to get response from LM Studio")