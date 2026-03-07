# AI 混合文本数据集构建工具

按 `dataset_plan.md` 的流程，生成带连续标签的人机混合文本数据集。

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

## 运行方式

### 批量构建（batch）

```bash
# dry-run（不调用 API）
python run.py batch --dry-run --max-docs 10

# 正式运行（默认模型：qwen3.5-plus）
python run.py batch

# 指定模型
python run.py batch --model gemini-3.1-flash-lite-preview
```

常用参数：

- `--source`：数据源路径（`.json` 或 `.jsonl`）
- `--ratios`：AI 比例列表，如 `0.0,0.2,0.4,0.6,0.8,1.0`
- `--modes`：混合模式列表，如 `block_replace,random_scatter`
- `--domains`：领域过滤（逗号分隔）
- `--max-docs`：最多处理文档数
- `--concurrent`：并发请求数
- `--seed`：随机种子

### 单文本测试（single）

```bash
# 直接传文本
python run.py single --text "A. B. C. D. E."

# 指定参数
python run.py single --text "..." --ratio 0.6 --mode random_scatter --model qwen3.5-plus

# dry-run
python run.py single --text "..." --dry-run
```

也可以直接改 `run_single.py` 里的参数后运行：

```bash
python run_single.py
```

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
- `doc_ai_ratio_exact` 在标签计算阶段等同于 `lir`（写入记录时字段名为 `lir`）。

## 当前支持模型

- `qwen3.5-plus`（DashScope / OpenAI-compatible）
- `MiniMax-M2.5`（Anthropic-compatible）
- `gemini-3.1-flash-lite-preview`（google-genai）

模型列表由 `src/config.py` 的 `SUPPORTED_MODELS` 决定。
