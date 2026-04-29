from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ActivationExtractor:
    def __init__(self, model_name, layer=None, relative_depth=0.6, device=None, max_length=256):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.relative_depth = relative_depth
        print(f"Loading '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use bfloat16 to avoid OOM on 8B+ models
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        offload_dir = Path("scratch") / "hf_offload" / model_name.replace("/", "__")
        offload_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto",
            offload_folder=str(offload_dir),
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.layer = layer

    def _pick_input_device(self):
        for p in self.model.parameters():
            if p.device.type != "meta":
                return p.device
        return torch.device("cpu")

    def _resolve_target_layer(self, outputs):
        total_layers = len(outputs.hidden_states)
        if self.layer is not None:
            return self.layer if self.layer < total_layers else -1
        hidden_layer_count = total_layers - 1
        idx = int(round(hidden_layer_count * self.relative_depth))
        return max(1, min(hidden_layer_count, idx))

    def extract_batch(self, prompts):
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_device = self._pick_input_device()
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)

        target_layer = self._resolve_target_layer(outputs)
        hidden_states = outputs.hidden_states[target_layer]
        last_idx = inputs["attention_mask"].sum(dim=1) - 1
        row_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        vector = hidden_states[row_idx, last_idx, :]

        vec_np = vector.detach().cpu().float().numpy()
        norms = np.linalg.norm(vec_np, axis=1, keepdims=True) + 1e-8
        return vec_np / norms

    def extract(self, prompt):
        return self.extract_batch([prompt])[0]

if __name__ == "__main__":
    extractor = ActivationExtractor("google/gemma-2-2b", layer=20)
    print("Testing extraction...")
    v = extractor.extract("Explain why the sky is blue.")
    print("Extracted shape:", v.shape)
    print("Normalized norm:", np.linalg.norm(v))
