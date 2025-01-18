import json
from transformers import PreTrainedTokenizerFast
from src.training.config import TrainingConfig  # Adjust the import path as needed

def preprocess_dataset(input_path, output_path, tokenizer, max_length):
    """
    Preprocess and tokenize a dataset, then save to a new file.

    Args:
        input_path (str): Path to the raw JSONL dataset.
        output_path (str): Path to save the pre-tokenized dataset.
        tokenizer (PreTrainedTokenizerFast): Tokenizer instance.
        max_length (int): Maximum sequence length for tokenization.
    """
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            sample = json.loads(line)
            prompt = sample["prompt"]
            completion = sample["completion"]

            # Combine and tokenize
            combined = f"<s>[INST] {prompt} [/INST] {completion}</s>"
            encoded = tokenizer(
                combined,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="np",
            )

            # Save tokenized data
            tokenized_sample = {
                "input_ids": encoded["input_ids"].tolist(),
                "attention_mask": encoded["attention_mask"].tolist(),
            }
            outfile.write(json.dumps(tokenized_sample) + "\n")


if __name__ == "__main__":
    # Load configuration
    config = TrainingConfig()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        str(config.BASE_MODEL), trust_remote_code=True
    )

    # Set the pad_token if it's not already defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
        
    preprocess_dataset(
        input_path="data/processed/training_data.jsonl",  # Replace with actual input
        output_path="data/processed/tokenized_training_data.jsonl",  # Replace with desired output
        tokenizer=tokenizer,
        max_length=config.MAX_SEQ_LEN,  # Get max sequence length from config
    )
