"""Training orchestration for LoRA fine-tuning."""
from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import ContextManager, Iterable, Optional

from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from .config import TrainingConfig
from .dataset import CaptionRecord


class CaptionDataset(torch.utils.data.Dataset):
    """A torch dataset wrapping caption records."""

    def __init__(self, records: Iterable[CaptionRecord], resolution: int, center_crop: bool) -> None:
        self.records = list(records)
        interpolation = transforms.InterpolationMode.BILINEAR
        transform_list = [transforms.Resize(resolution, interpolation=interpolation)]
        if center_crop:
            transform_list.append(transforms.CenterCrop(resolution))
        transform_list.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        with Image.open(record.image_path) as img:
            image = self.transform(img.convert("RGB"))
        caption = record.caption or ""
        return {"pixel_values": image, "caption": caption}


def _to_torch_dataset(records: Iterable[CaptionRecord], config: TrainingConfig) -> DataLoader:
    dataset = CaptionDataset(records=records, resolution=config.resolution, center_crop=config.center_crop)
    return DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        pin_memory=True,
    )


def _init_pipeline(config: TrainingConfig) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
        safety_checker=None,
    )
    if config.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()
    return pipe


def _setup_lora(pipe: StableDiffusionPipeline) -> AttnProcsLayers:
    lora_attn_procs = AttnProcsLayers(pipe.unet.attn_processors)
    for param in pipe.unet.parameters():
        param.requires_grad_(False)
    return lora_attn_procs


def train_lora(
    records: Iterable[CaptionRecord],
    config: TrainingConfig,
    *,
    output_dir: Optional[Path] = None,
) -> Path:
    """Train a LoRA adapter and return the output directory."""

    output_dir = output_dir or config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloader = _to_torch_dataset(records, config)
    pipe = _init_pipeline(config)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    pipe.to(device)

    lora_layers = _setup_lora(pipe)
    optimizer = AdamW(lora_layers.parameters(), lr=config.learning_rate)
    ema_model = EMAModel(lora_layers.parameters())

    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=config.max_train_steps,
    )

    global_step = 0
    progress_bar = tqdm(range(config.max_train_steps))

    pipe.unet.train()
    if config.train_text_encoder:
        pipe.text_encoder.train()

    def autocast_context() -> ContextManager[None]:
        if device == "cpu":
            return nullcontext()
        return torch.autocast(device_type=device, dtype=pipe.unet.dtype)

    optimizer.zero_grad()

    while global_step < config.max_train_steps:
        for step, batch in enumerate(dataloader, start=1):
            with autocast_context():
                latents = pipe.vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

                text_inputs = pipe.tokenizer(batch["caption"], return_tensors="pt", padding="max_length", truncation=True)
                text_inputs = text_inputs.input_ids.to(device)
                encoder_hidden_states = pipe.text_encoder(text_inputs)[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                target = noise
                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = loss / config.gradient_accumulation_steps

            loss.backward()

            if step % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(lora_layers.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                ema_model.step(lora_layers.parameters())

                if global_step % config.checkpointing_steps == 0 and global_step > 0:
                    save_dir = output_dir / f"checkpoint-{global_step}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    pipe.save_pretrained(save_dir)

                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item() * config.gradient_accumulation_steps})
                global_step += 1

                if global_step >= config.max_train_steps:
                    break

        if global_step >= config.max_train_steps:
            break

    ema_model.copy_to(lora_layers.parameters())

    weights_path = output_dir / "lora_weights.safetensors"
    pipe.save_lora_weights(weights_path)

    (output_dir / "training_args.json").write_text(json.dumps(config.__dict__, indent=2), encoding="utf-8")

    if config.validation_prompt:
        generator = torch.manual_seed(config.seed)
        with autocast_context():
            images = pipe(config.validation_prompt, num_inference_steps=30, generator=generator).images
        samples_dir = output_dir / "sample_images"
        samples_dir.mkdir(exist_ok=True)
        for idx, image in enumerate(images):
            image.save(samples_dir / f"sample_{idx:03d}.png")

    return output_dir
