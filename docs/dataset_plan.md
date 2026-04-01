# 数据集构建方案

人类文本从哪里来、如何筛选清洗、如何生成混合文本与标签。

## 1. 任务目标

构建句子级可控 AI 混合文本数据集。每条样本包含：

- 原始人类文本 `original_text`
- 混合后文本 `mixed_text`
- 句子级标签 `sentence_labels`（0=人类句，1=AI 改写句）
- 文档级连续标签（`lir`, `jaccard_distance`, `cosine_distance`）

## 2. 人类文本采样（来源与筛选）

### 2.1 数据来源

人类文本来自 4 个数据集：

1. ArXiv: `nick007x/arxiv-papers`（字段 `abstract`）
2. OpenWebText: `Skylion007/openwebtext`（字段 `text`，streaming）
3. XSum: `EdinburghNLP/xsum`（字段 `document`）
4. DAIGT-v2: `thedrcat/daigt-v2-train-dataset`（字段 `text`，仅保留 `label=0` 人类文本）

### 2.2 采样策略

使用脚本：

```bash
python scripts/sample_human_texts.py \
  --total 10000 \
  --seed 42 \
  --output data/human_texts_10k.jsonl \
  --min-sentences 8 \
  --max-sentences 20 \
  --max-chars 8000
```

采样规则：

- 总配额 `total` 按 4 个来源等比例分配（不能整除时前几个来源多分 1 条）。
- 各来源先 shuffle，再做无放回采样。
- 统一输出为 JSONL，每条至少包含：

```json
{"id": "arxiv_0902.3253", "text": "...", "sentence_count": 12}
```

### 2.3 采样过滤条件

文本需同时满足：

- 字符数 `<= max_chars`（默认 8000）
- 总词数在 `[100, 300]`
- 句子数在 `[min_sentences, max_sentences]`（默认 8 到 20）
- 每句词数在 `[5, 50]`

分句工具为 PySBD（`language="en", clean=False`）。

### 2.4 采样阶段文本清洗

采样时会先做轻量清洗：

- ArXiv 摘要做 LaTeX 残留清理（如 `\cite{}`、行内公式符号等）
- 所有来源统一做空白归一化（换行转空格、连续空格折叠）

## 3. 已采样文本清洗（构造前）

在混合构造前，对采样结果执行保守清洗：

```bash
python scripts/clean_human_texts.py \
  --input data/human_texts_10k.jsonl \
  --output data/human_texts_10k.cleaned.jsonl
```

该步骤主要移除高置信噪声句（链接尾巴、来源署名、社交账号尾注等），并更新 `sentence_count`。

最终用于构造任务的数据文件为清洗后的 JSONL（当前常用：`data/human_texts_1k.cleaned.jsonl`）。

## 4. 混合样本构造流程

批量入口：

```bash
python run.py batch \
  --source data/human_texts_1k.cleaned.jsonl \
  --model qwen3.5-plus \
  --seed 42
```

当前默认构造配置：

- `ai_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`
- `mixing_modes = ["block_replace", "random_scatter"]`

### 4.1 文档处理与模式分配

1. 读取输入 JSONL，抽取 `id` 与 `text`。
2. 全量样本按 `seed` 固定 shuffle。
3. 每篇文档只分配 1 个混合模式（在全体文档上近似 1:1 均衡）。

### 4.2 分句与目标句选择

对每篇文档：

1. PySBD 分句得到 `n` 句。
2. 对每个目标比例 `r` 计算 `k = round(r * n)`。
3. 按模式选择待改写句索引：
   - `block_replace`: 选连续 `k` 句
   - `random_scatter`: 随机无放回选 `k` 句
4. 当 `r=0.0` 或 `r=1.0` 时只保留 1 个等价变体。

### 4.3 LLM 差分改写

对选中句执行差分改写：

1. 输入“全文编号句子 + 待改写句号列表”。
2. 要求模型输出 JSON 映射：

```json
{"1": "rewritten sentence", "4": "rewritten sentence"}
```

3. 仅接受被选中句子的 key；其他 key 丢弃。
4. 若目标比例 `r>0` 且返回改写句数量少于请求数量，则该样本丢弃，不写入数据集。

### 4.4 回填与句级标签

1. 仅替换“返回且非空”的改写句。
2. 拼接得到 `mixed_text`。
3. 同步生成 `sentence_labels`：
   - 该句成功改写 -> 1
   - 否则 -> 0

## 5. 标签计算

每条样本统一计算 3 个文档级连续指标：

1. `LIR = T_LLM / T_total`
2. `Jaccard Distance = 1 - |A∩B| / |A∪B|`
3. `Cosine Distance = 1 - cosine_sim`（n-gram TF，默认 `n=2`）

其中：

- `T_LLM` 是所有 AI 句子的 token 数之和
- `T_total` 是整篇混合文本 token 总数

## 6. 输出样本格式

输出为 JSONL。单条记录示例：

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
  "cosine_distance": 0.1893,
  "extra": {}
}
```


参考命令：

```bash
# 1) 采样
python scripts/sample_human_texts.py --total 10000 --seed 42 --output data/human_texts_10k.jsonl --min-sentences 8 --max-sentences 20 --max-chars 8000

# 2) 清洗
python scripts/clean_human_texts.py --input data/human_texts_10k.jsonl --output data/human_texts_10k.cleaned.jsonl

# 3) 构造混合数据
python run.py batch --source data/human_texts_10k.cleaned.jsonl --model qwen3.5-plus --seed 42
```
