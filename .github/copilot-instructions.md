# PACT Project Guidelines

## Build And Test
- Preferred environment: `conda env create -f environment.yml` then `conda activate ob_dataset`.
- Alternative install: `pip install -r requirements.txt`.
- Verify available models before running jobs: `python run.py list-models`.
- Always validate with a small dry run first: `python run.py batch --dry-run --max-docs 10`.
- Only run full batch jobs after dry-run succeeds.

## Architecture
- Main CLI entrypoint: `run.py`.
- Core pipeline modules are under `src/` (config, pipeline, sentence processing, rewriting, label calculation, writers).
- Operational scripts are under `scripts/` (sampling, cleaning, auditing, analysis).
- Data flow is generally: `data/` -> pipeline -> `output/` -> audits/statistics in `analysis/`.

## Conventions
- Primary structured data format is JSONL.
- Dataset outputs should preserve the expected fields (`id`, source metadata, `original_text`, `mixed_text`, `n_sentences`, `target_ai_ratio`, `mixing_mode`, `rewrite_model`, `sentence_labels`, distance metrics, `extra`).
- Keep sentence-level integrity: `n_sentences` must align with sentence label length after processing.
- Sentence splitting policy is sensitive. Follow project governance guidance before changing segmentation behavior.
- Prefer adding new analysis/audit artifacts under `analysis/` and keeping generated datasets/logs under `output/`.

## Operational Safety
- Do not modify or commit secrets from `.env`.
- Treat `data/` and `output/` as large/generated assets; avoid unnecessary rewrites and destructive operations.
- Avoid expensive full-dataset or multi-model runs unless explicitly requested.
- When changing pipeline logic, run at least one small dry-run and one relevant audit script.

## Documentation Map
- Project overview and commands: `README.md`.
- End-to-end data construction plan: `docs/dataset_plan.md`.
- Data governance and sentence-cleaning policy: `数据治理.md`.
