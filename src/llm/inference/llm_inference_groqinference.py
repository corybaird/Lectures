from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

class LLMInferenceGroqInference:
    def __init__(self, model_name='openai/gpt-oss-20b', **kwargs):
        self.groq_model = model_name
        self.groq_api_key = os.getenv("API_GROQ")
        self.client = Groq(api_key=self.groq_api_key)
        self.max_tokens = kwargs.get('max_tokens')
        self.temperature = kwargs.get('temperature')
        self.top_p = kwargs.get('top_p')

    def load_prompt(self, prompt, system_message=None, verbose=False):
        if verbose:
            print(f"Prompt Length: {len(prompt)}")
        self.system_message = system_message or "You are a helpful assistant."
        self.user_message_content = prompt

    def infer(self, max_tokens_response=None):
        try:
            kwargs = {
                "model": self.groq_model,
                "messages": [
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": self.user_message_content}
                ]
            }
            
            max_tokens_val = max_tokens_response or self.max_tokens or 3000
            kwargs["max_tokens"] = max_tokens_val
            
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            
            completion = self.client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error during inference: {e}")
            return None


if __name__ == "__main__":
    import time
    
    llm = LLMInferenceGroqInference(model_name='llama-3.3-70b-versatile')
    
    prompt = "Tell me about capital controls. Provide me an answer in bullet points"
    system_message = "You are an economist."
    
    llm.load_prompt(prompt, system_message=system_message, verbose=True)
    
    print("\nRunning inference...")
    start_time = time.time()
    
    result = llm.infer(max_tokens_response=50)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Inference completed in {duration:.2f} seconds.\n")
    
    if result:
        print(f"Response: {result}")
    else:
        print("Failed to get response from Groq")