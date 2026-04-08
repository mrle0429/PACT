# AI人类混合文本数据集构建工具

## 快速开始

1. 安装依赖

```bash
# conda
conda env create -f environment.yml

# 或 pip
pip install -r requirements.txt
```

2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env
```

3. 查看支持模型

```bash
python run.py list-models
```

## 目录结构

```text
ob/
├── run.py              # 主入口：批量构建、模型列表
├── src/                # 核心实现
├── scripts/            # 采样、清洗、单条调试、分析脚本
├── docs/               # 流程文档
├── data/               # 原始与清洗后的数据
├── output/             # 数据集输出、checkpoint、API 日志
├── analysis/           # 图表等分析产物
└── logs/               # 本地运行日志
```

## 运行方式

### 批量构建（batch）

```bash
# dry-run（不调用 API）
python run.py batch --dry-run --max-docs 10

# 正式运行（默认模型：MiniMax-M2.7）
python run.py batch

# 指定模型
python run.py batch --model llama4-fast:latest
python run.py batch --model gemma4
python run.py batch --model MiniMax-M2.7
python run.py batch --model claude-haiku-4.5
```

常用参数：

- `--source`：数据源路径（`.json` 或 `.jsonl`）
- `--ratios`：AI 比例列表，如 `0.0,0.2,0.4,0.6,0.8,1.0`
- `--modes`：混合模式列表，如 `block_replace,random_scatter`
- `--domains`：领域过滤（逗号分隔）
- `--max-docs`：最多处理文档数
- `--concurrent`：并发请求数
- `--seed`：随机种子

### 单文本测试

```bash
# 直接修改脚本顶部参数后运行
python scripts/run_single.py
```

更多数据准备与分析脚本：

- 采样：`python scripts/sample_human_texts.py`
- 清洗：`python scripts/clean_human_texts.py --input data/human_texts_10k.jsonl --output data/human_texts_10k.cleaned.jsonl`
- 质量检查：`python scripts/check_data_quality.py`
- 比例分析图：`python scripts/plot_ratio_analysis.py`

## 输出文件

- 数据集：`output/mixed_dataset.jsonl`
- checkpoint：`output/checkpoints/run_{model}_{dataset}_{fingerprint}.json`
- API 日志：`output/api_logs/{model}_{dataset}.jsonl`

### JSONL 记录结构（当前实现）

```json
{
  "id": "raid_dev__books-e52f03d8__r40_block",
  "source_dataset": "raid_dev",
  "source_domain": "books",
  "original_text": "...",
  "mixed_text": "...",
  "n_sentences": 6,
  "target_ai_ratio": 0.4,
  "mixing_mode": "block_replace",
  "rewrite_model": "qwen3.5-plus",
  "sentence_labels": [1, 0, 1, 0, 0, 0],
  "lir": 0.3822,
  "jaccard_distance": 0.2451,
  "cosine_distance": 0.1893,
  "extra": {}
}
```

## 关键行为说明

- `target_ratio=0.0` 不调用 API，直接保留原文。
- API 改写失败或覆盖不完整（返回句子数小于请求句子数）时，该样本会被跳过，不写入数据集。

## Ollama 接入

- 默认模型：`MiniMax-M2.7`
- 默认 Base URL：`http://127.0.0.1:11434/api`
- 调用接口：`POST /api/chat`
- 默认 `keep_alive`：`5m`
- 默认关闭环境代理影响：`trust_env=False`
- `gemma4` 通过 `think=false` 显式关闭思考输出

如需覆盖本地默认值，可在 `.env` 中设置：

```bash
OLLAMA_BASE_URL=http://127.0.0.1:11434/api
OLLAMA_KEEP_ALIVE=5m
```

## OpenRouter / Qwen 3.6 接入

- 模型 key：`qwen3.6-plus-preview-free`
- 实际 OpenRouter model id：`qwen/qwen3.6-plus-preview:free`
- 接口基址：`https://openrouter.ai/api/v1`
- 当前实现按 OpenRouter 要求开启 reasoning，但通过 `exclude=true` 不返回 reasoning 内容；同时优先请求 `json_object` 结构化输出

在 `.env` 中配置：

```bash
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

同一个 OpenRouter 配置也适用于：

- `claude-haiku-4.5` -> `anthropic/claude-haiku-4.5`

## 当前支持模型

- `llama4-fast:latest`（Ollama / 原生 HTTP API）
- `gemma4`（Ollama / 原生 HTTP API，thinking disabled）
- `qwen3.6-plus-preview-free`（OpenRouter / OpenAI-compatible）
- `claude-haiku-4.5`（OpenRouter / OpenAI-compatible）
- `qwen3.5-plus`（DashScope / OpenAI-compatible）
- `MiniMax-M2.7`（Anthropic-compatible，thinking disabled，利用被动 prompt cache）
- `gemini-3.1-flash-lite-preview`（google-genai）

模型列表由 `src/config.py` 的 `SUPPORTED_MODELS` 决定。

详细流程说明见 [`docs/dataset_plan.md`](/Volumes/Mac/Project/ob/docs/dataset_plan.md)。
