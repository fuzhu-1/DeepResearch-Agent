#!/usr/bin/env python3
"""混合检索 vs 纯向量 —— 隔离语料对比评测。

为什么需要隔离语料：
  真实知识库的文档语义高度重叠（都讲向量库选型），纯向量已能覆盖大部分查询，
  BM25 的增量价值体现不出来。本脚本构造一批"语义相近但含独特精确术语"的文档，
  精准暴露 BM25 的关键词召回能力。

用法:
    python tests/benchmark_hybrid_contrast.py
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONIOENCODING"] = "utf-8"

from app.rag.chunker import TextChunker
from app.rag.embedder import Embedder
from app.rag.retriever import RAGRetriever
from app.rag.vector_store import VectorStore
from app.config import settings


# ---------------------------------------------------------------------------
# 隔离语料：目标文档（含独特精确术语）+ 干扰文档（高度相似改写，无精确术语）
# ---------------------------------------------------------------------------
# 设计核心：干扰文档与目标文档"语义几乎相同"（改写自同一段话），
# 向量嵌入无法区分 → 纯向量 top-1/top-3 被噪声挤占；
# 但干扰文档不含精确术语 → BM25 能靠术语唯一锁定目标。
CORPUS = [
    # ---- 目标文档（各含独特精确术语）----
    ("HNSW 是一种基于图的向量索引算法，通过多层小世界图实现高效的近似最近邻搜索，适合高维向量的在线查询场景。HNSW 的核心是 NSW 图结构，支持动态增删。",
     "hnsw.txt", "目标"),
    ("DiskANN 是一种面向 SSD 存储的向量索引算法，通过图与内存压缩技术，在磁盘上实现大规模向量的近似最近邻搜索，适合十亿级向量的低成本部署。",
     "diskann.txt", "目标"),
    ("PQ 乘积量化是一种向量压缩技术，将高维向量切分成子空间分别量化，大幅降低存储与检索开销，常与倒排索引（IVF）配合用于大规模检索。",
     "pq.txt", "目标"),
    ("混合检索通常将向量检索与关键词检索结合，向量负责语义召回，关键词负责精确术语召回，两者融合提升整体召回率。",
     "hybrid_rag.txt", "目标"),
    ("Reranker 重排器在检索召回后对候选结果进行精排，使用交叉编码器对每个候选重新打分，显著提升最终排序的准确性。",
     "rerank.txt", "目标"),
    ("OpenHands Reviewer Agent 通过标签触发自动代码审查，获取代码变更后调用大模型进行审查，按严重级别分级输出意见。",
     "openhands_review.txt", "目标"),
    # ---- 干扰文档：语义与目标高度相似，但去掉精确术语 ----
    # 每组干扰围绕一个目标改写，让向量嵌入几乎重合
    *[
        (f"干扰{i}：一种基于图的向量索引算法，通过多层小世界图实现高效的近似最近邻搜索，适合高维向量的在线查询场景，其核心是图结构，支持动态增删。",
         f"noise_{i}.txt", "干扰")
        for i in range(15)
    ],
    *[
        (f"干扰{i}：一种面向大容量存储的向量索引算法，通过图与内存压缩技术，在磁盘上实现大规模向量的近似最近邻搜索，适合超大规模向量的低成本部署。",
         f"noise_dk_{i}.txt", "干扰")
        for i in range(15)
    ],
    *[
        (f"干扰{i}：一种向量压缩技术，将高维向量切分成子空间分别量化，大幅降低存储与检索开销，常与倒排索引配合用于大规模检索。",
         f"noise_pq_{i}.txt", "干扰")
        for i in range(15)
    ],
    *[
        (f"干扰{i}：一种检索增强生成方法，将向量检索与关键词检索结合，向量负责语义召回，关键词负责精确召回，两者融合提升整体召回率。",
         f"noise_rag_{i}.txt", "干扰")
        for i in range(15)
    ],
    *[
        (f"干扰{i}：一种排序优化器，在检索召回后对候选结果进行精排，对每个候选重新打分，显著提升最终排序的准确性。",
         f"noise_rr_{i}.txt", "干扰")
        for i in range(15)
    ],
    *[
        (f"干扰{i}：一种自动代码审查工具，通过触发自动化流程获取代码变更后调用大模型进行审查，按严重级别分级输出意见。",
         f"noise_oh_{i}.txt", "干扰")
        for i in range(15)
    ],
]

# ---------------------------------------------------------------------------
# 查询集：每项 (查询, 期望命中的文档 source)
# ---------------------------------------------------------------------------
# 精确术语查询：查询词与文档里的专有名词完全一致 → BM25 优势
# 语义查询：查询是自然语言描述 → 向量优势
QUERIES = [
    # 精确术语查询（预期 Hybrid > Dense）
    ("HNSW", "hnsw.txt"),
    ("DiskANN", "diskann.txt"),
    ("PQ 量化", "pq.txt"),
    ("BM25", "hybrid_rag.txt"),
    ("Reranker", "rerank.txt"),
    ("OpenHands", "openhands_review.txt"),
    # 语义查询（预期两者持平）
    ("怎么在高维空间里快速找到相似的向量", "hnsw.txt"),
    ("大规模向量怎么低成本存到磁盘上", "diskann.txt"),
    ("向量和关键词两种检索怎么结合", "hybrid_rag.txt"),
]

KEYWORD_QUERIES = {q for q, _ in QUERIES[:6]}  # 前 6 个是精确术语查询


def is_hit(result, expected_source):
    return (result.get("metadata") or {}).get("source", "") == expected_source


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    print("混合检索 vs 纯向量 —— 隔离语料对比评测")
    print("-" * 50)
    print(f"HYBRID_SEARCH_ENABLED = {settings.HYBRID_SEARCH_ENABLED}")

    # 强制开启混合检索（本评测需要对比 hybrid 路径）
    settings.HYBRID_SEARCH_ENABLED = True

    # 隔离的向量库（内存版，不污染真实知识库）
    vs = VectorStore(persist_dir=str(tempfile.mkdtemp(prefix="hybrid_contrast_")))
    retriever = RAGRetriever(
        chunker=TextChunker(chunk_size=200, chunk_overlap=0),
        vector_store=vs,
    )

    # 灌入语料
    for text, source, _topic in CORPUS:
        await retriever.ingest_document(text, source=source, doc_type="text")
    print(f"语料灌入完成: {len(CORPUS)} 篇文档\n")

    # 逐条对比
    results = []
    for query, expected in QUERIES:
        dense = await retriever.retrieve(query, k=args.top_k)
        hybrid = await retriever.hybrid_retrieve(query, top_k=args.top_k)

        dense_hit = any(is_hit(r, expected) for r in dense)
        hybrid_hit = any(is_hit(r, expected) for r in hybrid)
        dense_top1 = bool(dense) and is_hit(dense[0], expected)
        hybrid_top1 = bool(hybrid) and is_hit(hybrid[0], expected)

        results.append({
            "query": query,
            "expected": expected,
            "is_keyword": query in KEYWORD_QUERIES,
            "dense_hit": dense_hit,
            "hybrid_hit": hybrid_hit,
            "dense_top1": dense_top1,
            "hybrid_top1": hybrid_top1,
            "dense_results": [
                {"src": r.get("metadata", {}).get("source", "?"), "score": round(r.get("score", 0), 3)}
                for r in dense
            ],
            "hybrid_results": [
                {"src": r.get("metadata", {}).get("source", "?"), "score": round(r.get("score", 0), 3), "type": r.get("source_type", "?")}
                for r in hybrid
            ],
        })

        d_mark = "✓" if dense_hit else "✗"
        h_mark = "✓" if hybrid_hit else "✗"
        tag = "[词]" if query in KEYWORD_QUERIES else "[语]"
        print(f"  {tag} {query:<22} → {expected:<16} DenseTopK={d_mark} HybridTopK={h_mark}")

    # 汇总
    total = len(results)
    kw = [r for r in results if r["is_keyword"]]
    sem = [r for r in results if not r["is_keyword"]]

    def fmt(hits, n):
        return f"{hits}/{n}"

    print("\n" + "=" * 60)
    print("  对比汇总（top-k 命中）")
    print("=" * 60)
    print(f"  {'场景':<10} {'查询数':<6} {'Dense':<8} {'Hybrid':<8} {'提升'}")
    print("-" * 50)
    for label, group in [("全部", results), ("精确术语", kw), ("语义", sem)]:
        d = sum(1 for r in group if r["dense_hit"])
        h = sum(1 for r in group if r["hybrid_hit"])
        delta = f"+{h - d}" if h > d else ("持平" if h == d else f"{h - d}")
        print(f"  {label:<10} {len(group):<6} {fmt(d, len(group)):<8} {fmt(h, len(group)):<8} {delta}")

    # 明细（只打印有差异的）
    print("\n  --- 两者结果不同的查询（差异来源） ---")
    shown = 0
    for r in results:
        if r["dense_hit"] != r["hybrid_hit"]:
            shown += 1
            print(f"\n  [{r['query']}] 期望={r['expected']}")
            print(f"    Dense : {r['dense_results']}")
            print(f"    Hybrid: {r['hybrid_results']}")
    if shown == 0:
        print("  （本语料下两种方法结果一致）")

    if args.json:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hybrid_contrast_results.json")
        summary = {
            "date": __import__("datetime").datetime.now().isoformat(),
            "top_k": args.top_k,
            "total": total,
            "all": {"dense": sum(1 for r in results if r["dense_hit"]), "hybrid": sum(1 for r in results if r["hybrid_hit"])},
            "keyword": {"dense": sum(1 for r in kw if r["dense_hit"]), "hybrid": sum(1 for r in kw if r["hybrid_hit"])},
            "semantic": {"dense": sum(1 for r in sem if r["dense_hit"]), "hybrid": sum(1 for r in sem if r["hybrid_hit"])},
            "results": results,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 结果已保存到 {out}")


if __name__ == "__main__":
    asyncio.run(main())
