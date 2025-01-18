# File: src/training/mlx_trainer.py
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.simplefilter('ignore', NotOpenSSLWarning)
import json
from pathlib import Path
import time
from typing import List

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam
import numpy as np
from transformers import PreTrainedTokenizerFast

from .config import TrainingConfig
from .flame_model import FlameModel

class MLXTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        mx.random.seed(self.config.METAL_CONFIG["seed"])
        
        TrainingConfig.log_step("Loading model configuration...")
        config_path = Path(self.config.BASE_MODEL) / "config.json"
        with open(config_path, "r") as f:
            self.config.model_config = json.load(f)

        TrainingConfig.log_step("Initializing tokenizer...")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(self.config.BASE_MODEL),
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = FlameModel(self.config)
        self.model.load_weights(self.config.BASE_MODEL)

        TrainingConfig.log_step("Initializing optimizer...")
        self.optimizer = Adam(learning_rate=self.config.LEARNING_RATE)
        
        self.start_time = time.time()
        self.batch_times = []
        
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
            "input_ids": mx.array(input_ids_tensor, dtype=mx.float16),
            "attention_mask": mx.array(attention_mask_tensor, dtype=mx.float16),
        }


    def train_step(self, batch):
        start_time = time.time()
        def loss_fn(params):
            self.model.update(params)
            logits = self.model(batch["input_ids"], batch["attention_mask"])
            targets = batch["input_ids"][:, 1:]
            logits = logits[:, :-1]
            mask = batch["attention_mask"][:, 1:]
            vocab_size = self.config.model_config["vocab_size"]
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)
            mask_flat = mask.reshape(-1)
            loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='none')
            loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
            return loss

        loss, grads = mx.value_and_grad(loss_fn)(self.model.parameters())
        self.optimizer.update(self.model.parameters(), grads)
        mx.eval(loss, self.model.parameters(), self.optimizer.state)

        batch_time = time.time() - start_time
        self.batch_times.append(batch_time)
        return loss.item()

    def train(self, batch_size=None, epochs=None):
        batch_size = batch_size or self.config.BATCH_SIZE
        epochs = epochs or self.config.NUM_EPOCHS
        
        TrainingConfig.log_step("Loading training data...")
        training_data = []
        with open(self.config.TRAINING_DATA, 'r') as f:
            for line in f:
                ex = json.loads(line)
                prompt = ex["prompt"].strip()
                completion = ex["completion"].strip()
                training_data.append(f"<s>[INST] {prompt} [/INST] {completion}</s>")
        
        num_samples = len(training_data)
        steps_per_epoch = num_samples // batch_size
        total_steps = steps_per_epoch * epochs
        
        TrainingConfig.log_step(f"\n=== Training Started ===")
        TrainingConfig.log_step(f"Samples: {num_samples:,}")
        TrainingConfig.log_step(f"Batch Size: {batch_size}")
        TrainingConfig.log_step(f"Steps per Epoch: {steps_per_epoch:,}")
        TrainingConfig.log_step(f"Total Steps: {total_steps:,}\n")

        for epoch in range(epochs):
            TrainingConfig.log_step(f"Starting Epoch {epoch+1}/{epochs}")
            total_loss = 0.0
            num_batches = 0
            epoch_start = time.time()

            for i in range(0, num_samples, batch_size):
                texts = training_data[i:i+batch_size]
                if len(texts) < batch_size:
                    batch_size = len(texts)
                try:
                    batch_data = self.prepare_batch(texts)
                    loss_val = self.train_step(batch_data)
                    total_loss += loss_val
                    num_batches += 1
                    self.config.update_stats(loss_val)
                except Exception as e:
                    warnings.warn(f"Error during training at batch {num_batches}: {str(e)}")
                    continue

                elapsed = time.time() - self.start_time
                speed = (num_batches*batch_size) / elapsed
                if num_batches % self.config.LOG_STEPS == 0:
                    avg_loss = total_loss / num_batches
                    bar = self.config.create_progress_bar(num_batches, steps_per_epoch)
                    TrainingConfig.log_step(
                        f"Batch {num_batches}/{steps_per_epoch} {bar}\n"
                        f"Loss: {avg_loss:.4f}, Speed: {speed:.2f} samples/sec"
                    )

                if num_batches % self.config.SAVE_STEPS == 0:
                    self.save_checkpoint(epoch, num_batches, loss_val)

            epoch_time = time.time() - epoch_start
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                TrainingConfig.log_step(
                    f"\nEpoch {epoch+1} Summary:\n"
                    f"Average Loss: {avg_loss:.4f}\n"
                    f"Time: {epoch_time:.1f}s\n"
                    f"Speed: {(num_batches*batch_size)/epoch_time:.2f} samples/sec"
                )
                self.save_checkpoint(epoch, num_batches, avg_loss)
            else:
                TrainingConfig.log_step(f"⚠ No batches processed in epoch {epoch+1}")

        self.config.print_training_summary()

    def save_checkpoint(self, epoch, step, loss):
        ckpt_dir = self.config.OUTPUT_DIR / f"checkpoint-{epoch+1}-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        TrainingConfig.log_step("Saving checkpoint...")
        mx.save_safetensors(ckpt_dir / "model.safetensors", self.model.parameters())
        self.tokenizer.save_pretrained(ckpt_dir)
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump(self.config.model_config, f, indent=2)
        TrainingConfig.log_step(f"✓ Checkpoint saved to {ckpt_dir}")

def main():
    config = TrainingConfig()
    trainer = MLXTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
