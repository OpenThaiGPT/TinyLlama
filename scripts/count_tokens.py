from transformers import LlamaTokenizer
from datasets import load_from_disk
from tqdm.auto import tqdm

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained("/project/lt200056-opgpth/tokenizer_spm_v5")

print(tokenizer.tokenize('ทดสอบภาษาไทยหน่อยนะครับ'))

# Load your dataset
# Note: You'll need to replace this with code that loads your actual dataset.
# Here's how you might load a dataset from a file.
dataset = load_from_disk("/scratch/lt200056-opgpth/HF_V6_Colassal_deduplicated_128_09_decontaminated_128_03_blinded")
# dataset = load_from_disk("/scratch/lt200056-opgpth/HF_V5_555_Dataset_deduplicated_128_09_decontaminated_128_03_blinded")
print(dataset)

# Sample 0.1% of the dataset
# Note: Replace `your_dataset` with the actual variable containing your dataset.
sample_size = int(len(dataset['train']) * 0.002)
sampled_data = dataset['train'][:sample_size]


# sampled_data = random.sample(your_dataset, sample_size)

# Tokenize the sampled data and count tokens
token_count = 0
i = 0
for data_point in tqdm(sampled_data['text']):
    if i % 2000 == 0:
        print(data_point, 'data_point\n')
    tokens = tokenizer.tokenize(data_point)
    token_count += len(tokens)
    i += 1

# Scale up to estimate the total number of tokens in the full dataset
estimated_total_tokens = (token_count / sample_size) * len(dataset['train'])

# Report the result in billions
estimated_total_tokens_in_billions = estimated_total_tokens / 1e9

print(f"Estimated total tokens in the dataset: {estimated_total_tokens_in_billions:.2f} billion")
