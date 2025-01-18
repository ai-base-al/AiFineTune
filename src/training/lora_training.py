from pathlib import Path
import json
import time
import warnings
from typing import List
import argparse
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam
import numpy as np
from transformers import PreTrainedTokenizerFast
from .config import TrainingConfig
from .flame_model import FlameModel
from .metal_config import init_metal, configure_metal, monitor_memory, clear_memory


class LoRATrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        mx.random.seed(self.config.METAL_CONFIG["seed"])

        # Initialize Metal
        self.device, is_metal = init_metal()
        if is_metal:
            configure_metal(self.device)

        TrainingConfig.log_step("Loading model configuration...")
        config_path = Path(self.config.BASE_MODEL) / "config.json"
        with open(config_path, "r") as f:
            self.config.model_config = json.load(f)

        TrainingConfig.log_step("Initializing tokenizer...")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(self.config.BASE_MODEL), trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Model initialization
        self.model = FlameModel(self.config)
        self.model.load_weights(self.config.BASE_MODEL)
        self.freeze_base_params()

        TrainingConfig.log_step("Initializing optimizer...")
        self.optimizer = Adam(learning_rate=self.config.LEARNING_RATE)

        self.start_time = time.time()

    def freeze_base_params(self):
        """Freeze parameters of the base model to avoid updating them."""
        for param in self.model.parameters():
            if hasattr(param, "requires_grad"):
                param.requires_grad = False

    def prepare_batch(self, batch_samples):
        """Prepare a batch from pre-tokenized samples."""
        input_ids = [np.squeeze(sample["input_ids"]) for sample in batch_samples]
        attention_masks = [np.squeeze(sample["attention_mask"]) for sample in batch_samples]

        # Ensure consistent dimensions
        assert len(input_ids) == len(attention_masks), "Mismatched batch sizes!"
        assert all(len(x) == len(input_ids[0]) for x in input_ids), "Inconsistent input lengths!"

        input_ids_tensor = np.array(input_ids, dtype=np.int32)
        attention_mask_tensor = np.array(attention_masks, dtype=np.int32)

        return {
            "input_ids": mx.array(input_ids_tensor, dtype=mx.int32),
            "attention_mask": mx.array(attention_mask_tensor, dtype=mx.int32),
        }


    def train_step(self, batch):
        """Perform a single training step on a batch."""
        clear_memory()  # Ensure GPU memory is cleared before starting

        def loss_fn(model_params):
            """Compute the loss for the given model parameters."""
            self.model.update(model_params)
            logits = self.model(batch["input_ids"], batch["attention_mask"])  # Forward pass
            targets = batch["input_ids"][:, 1:]  # Shift targets by 1 for prediction
            logits = logits[:, :-1]  # Align logits to targets
            mask = batch["attention_mask"][:, 1:]  # Align mask with shifted targets

            vocab_size = self.config.model_config["vocab_size"]
            logits_flat = logits.reshape(-1, vocab_size)  # Flatten logits for loss calculation
            targets_flat = targets.reshape(-1)  # Flatten targets
            mask_flat = mask.reshape(-1)  # Flatten mask

            # Compute cross-entropy loss with masking
            loss = nn.losses.cross_entropy(
                logits_flat, targets_flat, reduction="none"
            )
            # Apply mask and normalize
            return (loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)

        # Initialize gradient accumulation variables
        accumulated_grads = None

        for _ in range(self.config.GRADIENT_ACCUMULATION_STEPS):
            loss, grads = mx.value_and_grad(loss_fn)(self.model.parameters())
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                for g, acc_g in zip(grads, accumulated_grads):
                    acc_g += g  # Accumulate gradients

        # Update model parameters with accumulated gradients
        self.optimizer.update(self.model.parameters(), accumulated_grads)

        # Evaluate updated parameters
        mx.eval(self.model.parameters(), self.optimizer.state)

        return float(loss)  # Convert loss to a float for easy logging


    def train(self, batch_size=None, epochs=None):
        """Train the model using the configured dataset."""
        batch_size = batch_size or self.config.BATCH_SIZE
        epochs = epochs or self.config.NUM_EPOCHS

        training_data = self._load_training_data()
        num_samples = len(training_data)
        steps_per_epoch = (num_samples + batch_size - 1) // batch_size

        self._print_training_config(num_samples, batch_size, steps_per_epoch)

        for epoch in range(epochs):
            TrainingConfig.log_step(f"Starting LoRA Epoch {epoch + 1}/{epochs}")
            total_loss = 0.0

            for step in range(steps_per_epoch):
                batch_start = step * batch_size
                batch_end = batch_start + batch_size
                batch_samples = training_data[batch_start:batch_end]

                if len(batch_samples) < batch_size:
                    # Break early if the batch is incomplete
                    break

                try:
                    batch_data = self.prepare_batch(batch_samples)
                    loss = self.train_step(batch_data)
                    total_loss += loss
                    self.config.update_stats(loss)
                    if step % self.config.SAVE_STEPS == 0:
                        self.save_checkpoint(epoch, step)
                    if step % self.config.LOG_STEPS == 0:
                        self._log_progress(epoch, step, total_loss, steps_per_epoch)

                except Exception as e:
                    warnings.warn(
                        f"Error processing batch at step {self.config.TRAINING_STATS['completed_steps']}: {str(e)}"
                    )
                    clear_memory()
                    continue

            mem_stats = monitor_memory("Training")
            print(f"GPU Memory: {mem_stats}")
            self._log_epoch_summary(epoch, total_loss, len(training_data))

    def save_checkpoint(self, epoch, step):
        """Save the model checkpoint."""
        checkpoint_dir = Path(self.config.OUTPUT_DIR) / f"checkpoint-epoch-{epoch + 1}-step-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        model_path = checkpoint_dir / "model.bin"
        self.model.save_weights(model_path)

        # Save tokenizer
        tokenizer_path = checkpoint_dir / "tokenizer"
        self.tokenizer.save_pretrained(tokenizer_path)

        # Save configuration
        config_path = checkpoint_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.model_config, f, indent=2)

        TrainingConfig.log_step(f"Checkpoint saved at {checkpoint_dir}")


    def _load_training_data(self):
        """Load pre-tokenized training data from the configured file."""
        TrainingConfig.log_step("Loading pre-tokenized training data...")

        # Ensure the path is a string before opening
        training_data_path = str(self.config.TRAINING_DATA)
        training_data = []

        with open(training_data_path, "r") as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    training_data.append({
                        "input_ids": sample["input_ids"],
                        "attention_mask": sample["attention_mask"],
                    })
                except (json.JSONDecodeError, KeyError):
                    continue

        if not training_data:
            raise ValueError("No valid training samples found!")
        return training_data


    def _print_training_config(self, num_samples, batch_size, steps_per_epoch):
        TrainingConfig.log_step("\n=== LoRA Training Started ===")
        TrainingConfig.log_step(f"Samples: {num_samples:,}")
        TrainingConfig.log_step(f"Batch Size: {batch_size}")
        TrainingConfig.log_step(f"Steps per Epoch: {steps_per_epoch:,}")

    def _log_progress(self, epoch, step, total_loss, steps_per_epoch):
        avg_loss = total_loss / (step + 1)
        progress_bar = self.config.create_progress_bar(step + 1, steps_per_epoch)
        elapsed = time.time() - self.start_time
        TrainingConfig.log_step(
            f"Epoch {epoch + 1}, Step {step}/{steps_per_epoch} {progress_bar}, "
            f"Loss: {avg_loss:.4f}, Time Elapsed: {elapsed:.2f}s"
        )

    def _log_epoch_summary(self, epoch, total_loss, num_samples):
        avg_loss = total_loss / num_samples
        TrainingConfig.log_step(
            f"Epoch {epoch + 1} Summary: Average Loss: {avg_loss:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the LoRA model.")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to the tokenized training data file."
    )
    args = parser.parse_args()

    config = TrainingConfig()
    config.TRAINING_DATA = args.data

    trainer = LoRATrainer(config)
    trainer.train()
