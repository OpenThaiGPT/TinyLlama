import torch
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer  # Ensure to import your custom model class

# Specify path and config
model_path = "out/tinyllama_1b_mixed/iter-200000-ckpt.bin"
config = LlamaConfig(**{
#   "_name_or_path": "meta-llama/Llama-2-7b-hf",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 5632,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 22,
  "num_key_value_heads": 4,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": None,
  "tie_word_embeddings": False,
  "torch_dtype": "float32",
#   "transformers_version": "4.31.0.dev0",
  "use_cache": True,
  # "vocab_size": 32000
})

# Loading a model (make sure your custom class is correctly imported)
model = LlamaForCausalLM(config)  # Ensure to implement the config properly if not using transformers.Config
state_dict = torch.load(model_path, map_location="cpu")
print(state_dict.keys(), 'keys')
model.load_state_dict(state_dict, strict=False)

tokenizer = LlamaTokenizer.from_pretrained('/project/lt200056-opgpth/tokenizer_spm_v5')

print(model, 'model')

# hf_path_name = '/project/lt200056-opgpth/new/TinyLlama/huggingface/tinyllama_1b_mixed-200000'
hf_path_name = 'new5558/tinyllama_1b-200000'

model.push_to_hub(hf_path_name)
tokenizer.push_to_hub(hf_path_name)
config.push_to_hub(hf_path_name)