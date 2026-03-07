# 数据集构建方案（与当前代码一致）

## Step1: 人类文本筛选

1. 默认数据源：`data/raid.dev.json`
2. 仅保留人类文本：`content_source == "human"`（若字段存在）
3. 过滤条件：
   - 句子数在 `[min_sentences, max_sentences]`（默认 5~60）
   - 字符数不超过 `max_chars`（默认 8000）
   - 可按 `include_domains` 过滤领域

## Step2: 分句与比例采样

对于每篇文档：

1. 使用 NLTK 分句，得到 `S=[s1, s2, ..., sn]`
2. 对每个目标比例 `r` 计算改写句数 `k = round(r * n)`
3. 按混合模式选索引：
   - `block_replace`：连续区间
   - `random_scatter`：随机无放回
4. 特殊规则：
   - `r=0.0` 和 `r=1.0` 时两种模式等价，仅生成 1 份
5. 当前批量流程按文档交替分配模式，整体保持 1:1

## Step3: LLM 差分改写（Diff-based Editing）

1. 输入全文上下文 + 待改写句索引
2. 要求模型仅输出 JSON 映射：`{"句号(1-indexed)": "改写句子"}`
3. 解析后仅接受“被选中句子”的 key，其他 key 丢弃
4. 回填策略：仅替换返回且非空的句子
5. 数据质量保护：
   - 若 `target_ratio > 0` 且模型返回句数小于请求句数，该样本直接跳过，不写入数据集

## Step4: 标签计算

当前实现对所有模式统一计算三个指标：

1. `LIR = T_LLM / T_total`
2. `Jaccard Distance = 1 - |A∩B| / |A∪B|`
3. `Cosine Distance = 1 - cosine_sim`（基于 n-gram TF，默认 n=2）

说明：

- `doc_ai_ratio_exact` 在计算层面等同于 `LIR`
- 输出记录中主标签字段为 `lir`（未单独展开 `doc_ai_ratio_exact` 字段）
- 另输出句子级标签 `sentence_labels`（0/1）

## Step5: 输出与断点续传

1. 输出格式：JSONL（`output/mixed_dataset.jsonl`）
2. 断点文件：`output/checkpoints/run_{model}_{dataset}_{fingerprint}.json`
3. 指纹由关键配置计算，避免不同配置误复用同一个 checkpoint
