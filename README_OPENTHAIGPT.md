# Training OpenThaiGPT Dataset



## Training Steps
1. Install all requirements from [PRETRAIN.md](PRETRAIN.md)
2. Run [submit_data_hf.sh](submit_data_hf.sh) to convert Huggingface datasets to JSONL format  
2.1 Convert The Pile HF to The Pile JSONL  
2.2 Convert OpenThaiGPT HF to OpenThaiGPT JSONL
3. Run [submit_data_openthai.sh](submit_data_openthai.sh) to tokenize and save OpenThaiGPT JSONL data to TinyLlama format
3. Run [submit_data_thepile.sh](submit_data_thepile.sh) to tokenize and save ThePile JSONL (Convert from Huggingface) data to TinyLlama format
4. Run [submit_train.sh](submit_train.sh) to train model


## Upload Model to Huggingface hub

1. Run command to convert TinyLlama checkpoint to Huggingface Pytorch checkpoint
```bash
python scripts/convert_lit_checkpoint.py \
--checkpoint_name iter-200000-ckpt.pth \
--out_dir out/tinyllama_1b \
--model_name tiny_LLaMA_1b
```
2. Run command to push to hub
```bash
python scripts/push_to_hub.py
```