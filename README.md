# PACT: Proportion Assessment of Collaborative Texts between Human and AI

English | [中文](README.zh-CN.md)

PACT is an open dataset and construction pipeline for controlled human-AI mixed text. It starts from human-written documents, selects sentences according to target AI proportions, rewrites those sentences with an LLM, and outputs JSONL records with both sentence-level labels and document-level continuous labels.

This repository currently focuses on:

- documenting and releasing the PACT dataset format;
- providing a reproducible batch dataset construction pipeline;
- supporting LLM rewriting experiments across multiple providers and models.

Single-text interactive testing has been temporarily removed from the public entrypoint and will be added back later.

## Dataset Task

Each sample contains an original human document, `original_text`, and a mixed document, `mixed_text`. The mixed text is produced by rewriting only part of the sentences while keeping the remaining sentences unchanged from the human source.

Default construction settings:

- AI sentence ratios: `0.0`, `0.2`, `0.4`, `0.6`, `0.8`, `1.0`
- mixing modes: `block_replace`, `random_scatter`
- sentence-level labels: `0` for human sentences, `1` for AI-rewritten sentences
- document-level continuous labels: `lir`, `jaccard_distance`, `sentence_jaccard`, `cosine_distance`

## Human Text Sources

Human seed texts are sampled from four English sources:

- ArXiv abstracts
- OpenWebText
- XSum news documents
- DAIGT-v2 human essays

The current default input file is:

```text
data/human_texts_1k.cleaned.jsonl
```

Each input JSONL record must contain at least:

```json
{"id": "arxiv_0902.3253", "text": "...", "sentence_count": 12}
```

## Output Format

Batch construction writes JSONL files under `output/`. The default filename pattern is:

```text
output/mixed_dataset_<model-name>.jsonl
```

Example record:

```json
{
  "id": "arxiv_1409.3719_r40_block",
  "source_dataset": "arxiv",
  "source_domain": "academic",
  "original_text": "...",
  "mixed_text": "...",
  "n_sentences": 6,
  "target_ai_ratio": 0.4,
  "mixing_mode": "block_replace",
  "rewrite_model": "qwen3.5-plus",
  "sentence_labels": [1, 0, 1, 0, 0, 0],
  "lir": 0.3822,
  "jaccard_distance": 0.2451,
  "sentence_jaccard": 0.2113,
  "cosine_distance": 0.1893,
  "extra": {}
}
```

## Pipeline Overview

The batch construction pipeline:

1. Loads cleaned human seed texts.
2. Splits each document into sentences.
3. Assigns one mixing mode to each source document.
4. Generates variants for each target AI ratio.
5. Calls the selected LLM to rewrite only the selected sentence indices.
6. Validates the LLM response and skips missing, incomplete, or malformed samples.
7. Fills rewritten sentences back into the document to produce `mixed_text`.
8. Computes sentence-level labels and document-level continuous labels.
9. Appends valid samples to JSONL and maintains checkpoints.

For resumable runs, the final JSONL output is treated as the source of truth. Checkpoints only store runtime statistics such as processed source documents and API token usage.

## Installation

Using Conda:

```bash
conda env create -f environment.yml
conda activate ob_dataset
```

Or using pip:

```bash
pip install -r requirements.txt
```

## API Keys

Copy the environment template and fill in the provider keys you need:

```bash
cp .env.example .env
```

The current code supports OpenAI, Anthropic, Gemini, MiniMax, DashScope, DeepSeek, Doubao/Ark, OpenRouter, and Ollama-compatible local endpoints. You only need to configure the provider used by your selected model.

## Usage

List supported rewrite models:

```bash
python run.py list-models
```

Run a dry run without API calls to validate the pipeline:

```bash
python run.py batch --dry-run --max-docs 10
```

Run batch construction with the default model:

```bash
python run.py batch
```

Run batch construction with a specific model:

```bash
python run.py batch --model MiniMax-M2.7
python run.py batch --model qwen3.5-flash
python run.py batch --model gemini-3.1-flash-lite-preview
python run.py batch --model claude-haiku-4.5
python run.py batch --model gpt-5.4
```

Override generation and rate-limit settings at runtime:

```bash
python run.py batch \
  --model MiniMax-M2.7 \
  --temperature 0.2 \
  --rpm 30 \
  --max-output-tokens 1024
```

## Configuration

Core configuration lives in `src/config.py`:

- `source_path`: default input JSONL path
- `output_dir`: output directory
- `concurrent_requests`: async concurrency
- `random_seed`: random seed
- `ai_ratios`: target AI sentence ratios
- `mixing_modes`: mixing strategies
- `SUPPORTED_MODELS`: model names, providers, real model IDs, and model-level overrides

Models share default values for `temperature`, `max_output_tokens`, and `requests_per_minute`. Individual models only override fields that differ from the shared defaults.

## Current Scope

Currently supported:

- batch dataset construction;
- dry-run pipeline validation;
- multi-provider model registry;
- append-only JSONL writing with resumable checkpoints;
- sentence-level and document-level continuous label computation.

Temporarily not exposed:

- single-text interactive testing;
- direct CLI overrides for input path, output directory, random seed, or ratio list through `run.py`.

For implementation details, see `docs/dataset_plan.md`.
