# PACT: Proportion Assessment of Collaborative Texts between Human and AI

[English](README.md) | 中文

PACT 是一个面向开源发布的可控人类-AI 混合文本数据集与构建 pipeline。它从人类文本出发，按指定比例选择句子交给 LLM 改写，并输出带有句子级标签和文档级连续标签的 JSONL 数据集。

本仓库当前重点是：

- 发布并说明 PACT 数据集格式；
- 提供可复现的批量数据集构建 pipeline；
- 支持多 provider / 多模型的 LLM 改写实验。

单文本交互测试功能已暂时从公开入口移除，后续会重新添加。

## 数据集任务

每条样本包含一篇原始人类文本 `original_text` 和一篇混合文本 `mixed_text`。混合文本通过“只改写部分句子，其余句子保持人类原文”的方式生成。

当前默认构造设置：

- AI 句子比例：`0.0`, `0.2`, `0.4`, `0.6`, `0.8`, `1.0`
- 混合模式：`block_replace`, `random_scatter`
- 句子级标签：`0` 表示人类句子，`1` 表示 AI 改写句子
- 文档级连续标签：`lir`, `jaccard_distance`, `sentence_jaccard`, `cosine_distance`

## 人类文本来源

人类种子文本来自四类英文数据源：

- ArXiv abstracts
- OpenWebText
- XSum news documents
- DAIGT-v2 human essays

当前默认输入文件为：

```text
data/human_texts_1k.cleaned.jsonl
```

输入 JSONL 中每条记录至少需要包含：

```json
{"id": "arxiv_0902.3253", "text": "...", "sentence_count": 12}
```

## 输出格式

批量构建结果会写入 `output/` 目录。默认文件名格式为：

```text
output/mixed_dataset_<model-name>.jsonl
```

单条记录示例：

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

## Pipeline 流程

批量构建流程如下：

1. 读取清洗后的人类种子文本。
2. 对每篇文档分句。
3. 为每篇源文档分配一种混合模式。
4. 按目标 AI 比例生成多个变体。
5. 调用指定 LLM，只改写被选中的句子编号。
6. 校验 LLM 返回结果，跳过缺失、不完整或格式异常的样本。
7. 将改写句回填得到 `mixed_text`。
8. 计算句子级标签和文档级连续标签。
9. 将有效样本追加写入 JSONL，并维护 checkpoint。

断点续跑时，最终 JSONL 输出文件是唯一完成态来源；checkpoint 只保存运行统计，例如已处理源文档和 API token 消耗。

## 安装

使用 Conda：

```bash
conda env create -f environment.yml
conda activate ob_dataset
```

或使用 pip：

```bash
pip install -r requirements.txt
```

## API Key

复制环境变量模板并填写需要使用的 provider：

```bash
cp .env.example .env
```

当前支持 OpenAI、Anthropic、Gemini、MiniMax、DashScope、DeepSeek、Doubao/Ark、OpenRouter，以及 Ollama 兼容的本地接口。只需要配置你实际使用模型对应的 provider。

## 使用方式

查看支持的改写模型：

```bash
python run.py list-models
```

不调用 API，先验证 pipeline 流程：

```bash
python run.py batch --dry-run --max-docs 10
```

使用默认模型批量构建：

```bash
python run.py batch
```

指定模型批量构建：

```bash
python run.py batch --model MiniMax-M2.7
python run.py batch --model qwen3.5-flash
python run.py batch --model gemini-3.1-flash-lite-preview
python run.py batch --model claude-haiku-4.5
python run.py batch --model gpt-5.4
```

运行时覆盖生成参数和限速参数：

```bash
python run.py batch \
  --model MiniMax-M2.7 \
  --temperature 0.2 \
  --rpm 30 \
  --max-output-tokens 1024
```

## 配置

核心配置位于 `src/config.py`：

- `source_path`：默认输入 JSONL
- `output_dir`：输出目录
- `concurrent_requests`：异步并发数
- `random_seed`：随机种子
- `ai_ratios`：目标 AI 句子比例
- `mixing_modes`：混合模式
- `SUPPORTED_MODELS`：模型名、provider、真实 model id 和模型级覆盖配置

模型默认共享统一的 `temperature`、`max_output_tokens` 和 `requests_per_minute`。只有确实不同的模型才在模型表中单独覆盖。

## 当前范围

当前支持：

- 批量数据集构建；
- dry-run 流程验证；
- 多 provider 模型注册；
- JSONL 追加写入与断点恢复；
- 句子级标签和文档级连续标签计算。

暂未暴露：

- 单文本交互测试；
- 通过 `run.py` 命令行直接修改输入路径、输出目录、随机种子或比例列表。

更详细的构建方案见 `docs/dataset_plan.md`。
