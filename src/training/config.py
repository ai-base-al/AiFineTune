import time
from datetime import datetime, timedelta
from pathlib import Path
from .metal_config import init_metal, configure_metal

class TrainingConfig:
    START_TIME = time.time()

    @classmethod
    def log_step(cls, message: str, end="\n"):
        elapsed = time.time() - cls.START_TIME
        timestamp = datetime.now().strftime("%H:%M:%S")
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        print(f"[{timestamp}] ({elapsed_str}) {message}", end=end)

    def __init__(self):
        self.log_step("Initializing Metal device...")
        self.DEVICE, metal_available = init_metal()

        if metal_available:
            metal_configured = configure_metal(self.DEVICE)
            if metal_configured:
                self.log_step("✓ Metal GPU acceleration enabled and configured")
            else:
                self.log_step("⚠ Metal GPU available but configuration failed")
        else:
            self.log_step("⚠ Metal GPU not available, falling back to CPU")

        # Base directories
        self.BASE_DIR = Path.home() / "AiFineTune"
        self.DATA_DIR = self.BASE_DIR / "data"
        self.MODELS_DIR = self.BASE_DIR / "models"

        self.TRAINING_DATA = self.DATA_DIR / "processed" / "test_data.jsonl"
        self.BASE_MODEL = Path.home() / ".lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit"
        self.OUTPUT_DIR = self.MODELS_DIR / "flame-finetuned"

        # Optimized training parameters
        self.BATCH_SIZE = 8
        self.GRADIENT_ACCUMULATION_STEPS = 4
        self.NUM_EPOCHS = 1
        self.LEARNING_RATE = 1e-5
        self.MAX_SEQ_LEN = 256  # Correctly define here
        self.LOG_STEPS = 1
        self.SAVE_STEPS = 10

        # Metal configuration
        self.METAL_CONFIG = {
            "seed": 42,
            "max_mem_mb": 16000,
            "memory_limit_ratio": 0.7,
        }

        self.TRAINING_STATS = {
            "start_time": time.time(),
            "total_steps": 0,
            "completed_steps": 0,
            "best_loss": float("inf"),
            "losses": [],  # Initialize as an empty list
            "checkpoints": [],
            "memory_stats": [],
        }

        self._print_configuration()

    def _validate_vocab_size(self):
        """Validate vocabulary size matches the tokenizer"""
        if hasattr(self, 'tokenizer'):
            actual_vocab_size = len(self.tokenizer.get_vocab())
            if actual_vocab_size != self.model_config["vocab_size"]:
                print(f"Warning: Vocab size mismatch. Config: {self.model_config['vocab_size']}, "
                      f"Tokenizer: {actual_vocab_size}")
                self.model_config["vocab_size"] = actual_vocab_size

    def _print_configuration(self):
        self.log_step("\n=== Training Configuration ===")
        self.log_step(f"• Base Directory: {self.BASE_DIR}")
        self.log_step(f"• Model Path: {self.BASE_MODEL}")
        self.log_step(f"• Output Directory: {self.OUTPUT_DIR}")
        self.log_step(f"• Batch Size: {self.BATCH_SIZE}")
        self.log_step(f"• Learning Rate: {self.LEARNING_RATE}")
        self.log_step(f"• Max Sequence Length: {self.MAX_SEQ_LEN}")
        device_str = "Metal GPU" if self.DEVICE else "CPU"
        self.log_step(f"• Device: {device_str}")
        self.log_step("=" * 30 + "\n")

    def update_stats(self, loss: float, checkpoint_saved: bool = False):
        """Update training statistics."""
        if not isinstance(loss, float):
            raise ValueError(f"Expected loss to be a float, got {type(loss)}")
        
        self.TRAINING_STATS["completed_steps"] += 1
        self.TRAINING_STATS["losses"].append(loss)  # Append the new loss
        
        if loss < self.TRAINING_STATS["best_loss"]:
            self.TRAINING_STATS["best_loss"] = loss

        if checkpoint_saved:
            self.TRAINING_STATS["checkpoints"].append({
                "step": self.TRAINING_STATS["completed_steps"],
                "loss": loss,
                "time": time.time(),
            })


    def print_training_summary(self):
        duration = time.time() - self.TRAINING_STATS["start_time"]
        hours = duration // 3600
        minutes = (duration % 3600) // 60

        self.log_step("\n=== Training Summary ===")
        self.log_step(f"Duration: {int(hours)}h {int(minutes)}m")
        self.log_step(f"Steps Completed: {self.TRAINING_STATS['completed_steps']}")
        self.log_step(f"Best Loss: {self.TRAINING_STATS['best_loss']:.4f}")
        self.log_step(f"Checkpoints Saved: {len(self.TRAINING_STATS['checkpoints'])}")

        if self.TRAINING_STATS["losses"]:
            avg_loss = sum(self.TRAINING_STATS["losses"]) / len(self.TRAINING_STATS["losses"])
            self.log_step(f"Average Loss: {avg_loss:.4f}")

        self.log_step("=" * 30 + "\n")

    def create_progress_bar(self, current, total, width=50):
        progress = int(width * current / total)
        return f"[{'=' * progress}{' ' * (width - progress)}] {current}/{total} ({100 * current / total:.1f}%)"
