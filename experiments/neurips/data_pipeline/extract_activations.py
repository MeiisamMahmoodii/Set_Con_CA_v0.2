import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

class ActivationExtractor:
    def __init__(self, model_name, layer, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use bfloat16 to avoid OOM on 8B+ models
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            torch_dtype=dtype,
            device_map="auto" # use auto to split across GPUs if available
        )
        self.layer = layer

    def extract(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Hidden states: 0 is embedding, 1 to L are layers.
        # Handle layer indexing based on the model's actual number of layers
        total_layers = len(outputs.hidden_states)
        target_layer = self.layer if self.layer < total_layers else -1

        hidden_states = outputs.hidden_states[target_layer]
        # Exact extraction rule: last token of prompt
        vector = hidden_states[:, -1, :]  

        # Normalization rule: L2 normalize
        vec_np = vector.squeeze().cpu().float().numpy()
        vec_np = vec_np / (np.linalg.norm(vec_np) + 1e-8)
        
        return vec_np

if __name__ == "__main__":
    extractor = ActivationExtractor("google/gemma-2-2b", layer=20)
    print("Testing extraction...")
    v = extractor.extract("Explain why the sky is blue.")
    print("Extracted shape:", v.shape)
    print("Normalized norm:", np.linalg.norm(v))
