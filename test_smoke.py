"""快速冒烟测试，验证所有模块的基础逻辑。"""
import sys, random
sys.path.insert(0, "/Volumes/Mac/Project/ob")

from src.config import DatasetConfig, SUPPORTED_MODELS
from src.utils import extract_json_from_llm_response, AsyncRateLimiter
from src.data_loader import create_loader, RaidDocumentLoader
from src.sentence_processor import (
    split_into_sentences, create_sentence_selection,
    build_rewrite_prompt, enumerate_variants,
)
from src.label_calculator import (
    compute_lir, compute_jaccard_distance,
    compute_cosine_distance, compute_labels,
)
from src.dataset_writer import DatasetRecord, JsonlWriter, CheckpointManager, make_record_id
from src.pipeline import DatasetPipeline, VariantTask

print("✓ 所有模块导入成功")

# 测试分句
text = "The sky is blue. Water is wet. The sun is bright. Birds can fly. Cats meow. Dogs bark."
sents = split_into_sentences(text)
print(f"✓ 分句结果 ({len(sents)} 句): {sents}")

# 测试选取策略
rng = random.Random(42)
cfg = DatasetConfig(source_path="data/raid.dev.json")

sel = create_sentence_selection(sents, 0.4, "block_replace", rng)
print(f"✓ Block Replace  k={sel.k}, indices={sel.selected_indices}")

sel2 = create_sentence_selection(sents, 0.4, "random_scatter", rng)
print(f"✓ Random Scatter k={sel2.k}, indices={sel2.selected_indices}")

# 测试标签计算
mixed = list(sents)
mixed[sel.selected_indices[0]] = "The azure sky stretches above us beautifully."
labels = compute_labels(text, mixed, sel.selected_indices, "block_replace", cfg)
print(f"✓ LIR 标签: {labels}")

# 测试 JSON 解析
json_str = '{"2": "The cerulean sky is beautiful.", "4": "Avian creatures possess flight."}'
parsed = extract_json_from_llm_response(json_str)
print(f"✓ JSON 解析: {parsed}")

# 测试 record ID 生成
rid = make_record_id("raid_dev", "wiki-abc123", 0.4, "block_replace")
print(f"✓ Record ID: {rid}")

# 测试 Prompt 构建
prompt = build_rewrite_prompt(sents, sel.selected_indices)
print(f"✓ Prompt 构建成功 (前100字符): {prompt[:100]}...")

# 测试数据加载过滤
loader = create_loader(cfg)
docs = loader.load_filtered(shuffle=False, max_count=5)
print(f"✓ 加载了 {len(docs)} 篇文档，首条 domain={docs[0].domain}, n_chars={len(docs[0].text)}")

# 测试 enumerate_variants（单模式：每文档只分配一种模式）
doc_rng = random.Random(42)
variants = enumerate_variants(docs[0].text, [0.0, 0.4, 1.0], ["block_replace"], doc_rng)
print(f"✓ 变体生成（单模式）: {len(variants)} 个变体")
for v in variants:
    print(f"   ratio={v.target_ratio:.1f} mode={v.mode} k={v.k}/{v.n}")
assert len(variants) == 3, f"期望 3 个变体，实际 {len(variants)}"

# 测试完整 ratios 单模式
doc_rng2 = random.Random(42)
variants2 = enumerate_variants(docs[0].text, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ["random_scatter"], doc_rng2)
print(f"✓ 变体生成（6 档单模式）: {len(variants2)} 个变体")
assert len(variants2) == 6, f"期望 6 个变体，实际 {len(variants2)}"

# 测试回填
mock_rewrites = {idx: f"REWRITTEN:{sents[idx]}" for idx in sel.selected_indices}
mixed_text = sel.build_mixed_text(mock_rewrites)
sentence_labels = sel.sentence_label_array(mock_rewrites)
print(f"✓ 回填混合文本: {mixed_text[:80]}...")
print(f"✓ 句子级标签: {sentence_labels}")

print()
print("=" * 50)
print("全部基础测试通过 ✓")
