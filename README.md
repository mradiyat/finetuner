# Finetuner App

Finetuner App is a command-line application you can package on macOS to fine-tune Stable Diffusion style image generation models with lightweight LoRA adapters. The tool wraps Hugging Face Diffusers utilities into a single workflow that:

1. Prepares and validates an image/text dataset stored locally.
2. Launches LoRA fine-tuning with sensible defaults optimized for consumer GPUs.
3. Exports trained adapters and metadata for use in compatible image generation UIs.

The project ships as a Python package that you can install into a virtual environment or bundle into a standalone application using tools like [pyinstaller](https://pyinstaller.org/) or [briefcase](https://beeware.org/project/projects/tools/briefcase/).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

> **Tip:** To produce a downloadable `.app` for macOS, build a standalone binary with `pyinstaller`:
>
> ```bash
> pip install pyinstaller
> pyinstaller --onefile --name FinetunerApp src/finetuner_app/cli.py
> ```
>
> The resulting binary in `dist/FinetunerApp` can be codesigned and wrapped into a `.dmg`.

## Usage

The CLI exposes three core commands:

```bash
# 1. Generate a config template to customize your run
finetuner-app init-config ./my-dataset

# 2. Validate your dataset before launching training
finetuner-app validate ./my-dataset

# 3. Launch LoRA fine-tuning
finetuner-app train ./my-dataset --config config.toml
```

### Dataset layout

Datasets should contain images and optional text captions. The app supports:

```
my-dataset/
├── metadata.jsonl        # optional: JSON lines with {"file_name": ..., "text": ...}
├── class_images/         # optional: prior preservation images
└── data/
    ├── image_0001.png
    ├── image_0001.txt    # optional: caption alongside the image
    └── ...
```

### Training outputs

After a successful run, the tool creates:

- `lora_weights.safetensors`: the trained LoRA adapter.
- `training_args.json`: hyperparameters used during training.
- `sample_images/`: periodically generated samples for monitoring progress.

## macOS considerations

- Use Python 3.10 or newer (available via [Homebrew](https://brew.sh/) `brew install python@3.11`).
- Install PyTorch with Metal Performance Shaders support for Apple Silicon:

  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

- Provide at least 30 GB of free disk space for checkpoints and caches.

## Development

```bash
pip install -e .[develop]
ruff check src
pytest
```

## License

MIT
