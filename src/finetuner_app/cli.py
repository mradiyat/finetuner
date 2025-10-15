"""Typer-powered CLI entrypoint."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import DEFAULT_CONFIG, TrainingConfig
from .dataset import iter_dataset_files, validate_images
from .trainer import train_lora

app = typer.Typer(help="Fine-tune image generation models with LoRA adapters.")


def _resolve_dataset(path: Path) -> Path:
    if not path.exists():
        raise typer.BadParameter(f"Dataset path does not exist: {path}")
    return path


@app.command()
def init_config(dataset_dir: Path = typer.Argument(..., help="Path to your dataset directory")) -> None:
    """Create a config template next to the dataset."""

    dataset_dir = _resolve_dataset(dataset_dir)
    config_path = dataset_dir / "config.toml"
    if config_path.exists():
        typer.echo(f"Config already exists at {config_path}")
        raise typer.Exit(code=1)
    DEFAULT_CONFIG.dump(config_path)
    typer.echo(f"Wrote config template to {config_path}")


@app.command()
def validate(dataset_dir: Path = typer.Argument(..., help="Dataset directory to validate")) -> None:
    """Validate that all images and captions can be read."""

    dataset_dir = _resolve_dataset(dataset_dir)
    warnings = validate_images(iter_dataset_files(dataset_dir))
    if not warnings:
        typer.secho("Dataset looks good!", fg=typer.colors.GREEN)
        raise typer.Exit(code=0)

    typer.secho("Found potential issues:", fg=typer.colors.YELLOW)
    for warning in warnings:
        typer.echo(f"- {warning}")
    raise typer.Exit(code=1)


@app.command()
def train(
    dataset_dir: Path = typer.Argument(..., help="Directory containing your dataset"),
    config: Optional[Path] = typer.Option(None, help="Path to a config.toml file"),
    output_dir: Optional[Path] = typer.Option(None, help="Override the default output directory"),
) -> None:
    """Launch LoRA fine-tuning using the provided dataset."""

    dataset_dir = _resolve_dataset(dataset_dir)
    records = list(iter_dataset_files(dataset_dir))
    if not records:
        raise typer.BadParameter("No images found in dataset. Ensure it follows the documented layout.")

    train_config = TrainingConfig.load(config) if config else DEFAULT_CONFIG
    if output_dir is not None:
        train_config.output_dir = output_dir

    typer.echo("Starting training...")
    result_dir = train_lora(records, train_config)
    typer.secho(f"Training complete! Artifacts saved to {result_dir}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
