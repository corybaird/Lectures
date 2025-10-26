import os
import re
import ast
import time
import json
from tqdm import tqdm

class LLMInferenceRunner:
    def __init__(self, llm_provider, model_name, save_path=None, **kwargs):
        self.llm_provider = llm_provider.lower()
        self.model_name = model_name
        self.save_path = save_path
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        self.prompt = ""
        self.system_message = None
        
        if self.llm_provider == 'groqinference':
            from src.llm.inference.llm_inference_groqinference import LLMInferenceGroqInference
            self.llm = LLMInferenceGroqInference(model_name=model_name, **kwargs)
        elif self.llm_provider == 'lmstudio':
            from src.llm.inference.llm_inference_lmstudio import LLMInferenceLMStudio
            self.llm = LLMInferenceLMStudio(model_name=model_name, **kwargs)
        elif self.llm_provider == 'ollama':
            from src.llm.inference.llm_inference_ollama import LLMInferenceOllama
            self.llm = LLMInferenceOllama(model_name=model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {llm_provider}")
    
    def set_prompt(self, prompt, system_message=None, verbose=False):
        self.prompt = prompt
        self.system_message = system_message
        self.llm.load_prompt(prompt, system_message=system_message, verbose=verbose)
    
    def infer(self, input_list, seconds_delay=0, save_intermediate_n=None):
        results = []
        
        for i, item in enumerate(tqdm(input_list)):
            if isinstance(self.prompt, str) and '{text}' in self.prompt:
                modified_prompt = self.prompt.format(text=item)
                self.llm.load_prompt(modified_prompt, system_message=self.system_message, verbose=False)
                try:
                    if seconds_delay > 0:
                        time.sleep(seconds_delay)
                    output = self.llm.infer()
                    results.append(output)
                except Exception as e:
                    results.append(None)
                    print(f"INFERENCE FAILED for item: {item}. Error: {e}")
            
            if save_intermediate_n and self.save_path and (i + 1) % save_intermediate_n == 0:
                self._save_progress_json(results, input_list[:i+1], f"{self.save_path}/results_until_n{i+1}.json")
                print(f"Progress saved: {i + 1}/{len(input_list)} items processed")
        
        if self.save_path:
            self._save_progress_json(results, input_list, f"{self.save_path}/results_full.json")
        
        return results
    
    def _save_progress_json(self, results, input_list, save_file):
        data = []
        for i, (inp, out) in enumerate(zip(input_list, results)):
            if out is None:
                data.append({"input": inp, "output": None, "error": "Inference failed"})
            else:
                data.append({"input": inp, "output": out})
        
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    runner = LLMInferenceRunner(
        llm_provider='lmstudio',
        model_name='liquid/lfm2-1.2b',
        save_path='./test_results'
    )
    
    prompt = "Classify this text according to topic: {text}"
    system_message = "You are a helpful assistant"
    
    runner.set_prompt(prompt, system_message=system_message, verbose=True)
    
    test_inputs = [
        "The quick brown fox jumps over the lazy dog. This is a common pangram used in typography.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "Climate change is affecting weather patterns around the world, causing more extreme weather events."
    ]
    
    print("\nRunning batch inference...")
    results = runner.infer(test_inputs, seconds_delay=1, save_intermediate_n=2)
    
    print("\n=== Results ===")
    for i, (inp, out) in enumerate(zip(test_inputs, results)):
        print(f"\nInput {i+1}: {inp[:50]}...")
        print(f"Output {i+1}: {out}")