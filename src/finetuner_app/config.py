"""Configuration models for the finetuner app."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

try:  # pragma: no cover - runtime import guard
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w


@dataclass
class TrainingConfig:
    """Hyperparameters for LoRA fine-tuning."""

    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    output_dir: Path = Path("outputs")
    resolution: int = 512
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    max_train_steps: int = 1000
    checkpointing_steps: int = 250
    validation_prompt: str = "A DSLR photo of a custom object"
    validation_steps: int = 250
    seed: int = 42
    mixed_precision: str = "bf16"
    use_8bit_adam: bool = True
    enable_xformers_memory_efficient_attention: bool = True
    gradient_checkpointing: bool = True
    prior_loss_weight: float = 1.0
    num_class_images: int = 200
    caption_extension: str = ".txt"
    center_crop: bool = False
    color_jitter: bool = False
    train_text_encoder: bool = False
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: ["finetuner-app"])

    def dump(self, path: Path) -> None:
        """Serialize the configuration to a TOML file."""

        path.write_text(tomli_w.dumps(asdict(self)), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        """Load a configuration from a TOML file."""

        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


DEFAULT_CONFIG = TrainingConfig()
