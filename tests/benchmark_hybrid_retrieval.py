#!/usr/bin/env python3
"""RAG 检索评测 — 混合检索 vs 纯向量检索对比。

对比 Dense 向量检索 (RAGRetriever.retrieve) 与混合检索 (RAGRetriever.hybrid_retrieve)
在同一批查询上的召回命中情况，量化混合检索带来的提升。

用法:
    python tests/benchmark_hybrid_retrieval.py
    python tests/benchmark_hybrid_retrieval.py --top-k 3
    python tests/benchmark_hybrid_retrieval.py --json  (输出 JSON 供记录)
"""

import argparse
import asyncio
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ["PYTHONIOENCODING"] = "utf-8"

from app.rag.retriever import RAGRetriever
from app.config import settings

# ---------------------------------------------------------------------------
# 评测查询集：每项 (查询, 期望命中的文档源)
# ---------------------------------------------------------------------------
# 设计原则：
#   1. 每条查询有明确的单一目标文档（避开多个文档内容交叉的歧义查询）
#   2. 语义查询 → 考察向量检索（query 与文档表述不同，但语义相关）
#   3. 精确术语查询 → 考察 BM25（query 含文档里的专有名词/精确词）
TEST_QUERIES = [
    # 语义查询（靠向量检索）——考察 Dense
    ("开源项目代码审查怎么做", "openhands"),
    ("大厂实习面试准备", "大厂实习入职攻略"),
    ("多智能体系统架构设计", "Agent框架对比"),
    ("检索增强生成的基本流程", "向量检索"),
    (r"Chroma 和 FAISS 定位区别", "向量存储选型"),
    # 关键词查询（靠 BM25 检索）——考察 Hybrid
    ("ChromaDB 对比", "向量数据库对比"),
    ("PQ 量化", "向量检索"),
    ("HNSW 索引", "向量数据库对比"),
    ("LangGraph", "Agent框架对比"),
]

# 哪些查询属于"BM25 关键词优势"场景（用于分组统计）
KEYWORD_QUERIES = {q for q, _ in TEST_QUERIES[5:]}  # 后 3 个


def is_hit(result, expected_source) -> bool:
    """判断一个检索结果是否命中期望文档。"""
    src = (result.get("metadata") or {}).get("source", "")
    return src == expected_source


def format_results(results, top_k):
    """格式化结果用于打印。"""
    lines = []
    for i, r in enumerate(results[:top_k], 1):
        src = (r.get("metadata") or {}).get("source", "?")
        st = r.get("source_type", "?")
        lines.append(f"      #{i} [{st}] score={r['score']:.3f} src={src}")
    return "\n".join(lines) if lines else "      (无结果)"


async def run_query(retriever, query, expected_source, top_k):
    """对单个查询跑两种检索，返回对比结果。"""
    # 纯向量检索
    dense = await retriever.retrieve(query, k=top_k)
    dense_hit = any(is_hit(r, expected_source) for r in dense)

    # 混合检索
    hybrid = await retriever.hybrid_retrieve(query, top_k=top_k)
    hybrid_hit = any(is_hit(r, expected_source) for r in hybrid)

    return {
        "query": query,
        "expected_source": expected_source,
        "is_keyword_query": query in KEYWORD_QUERIES,
        "dense_top1_hit": bool(dense) and is_hit(dense[0], expected_source),
        "dense_topk_hit": dense_hit,
        "hybrid_top1_hit": bool(hybrid) and is_hit(hybrid[0], expected_source),
        "hybrid_topk_hit": hybrid_hit,
        "dense_results": format_results(dense, top_k),
        "hybrid_results": format_results(hybrid, top_k),
    }


def print_summary(results, top_k):
    """打印汇总表格。"""
    total = len(results)
    dense_topk = sum(1 for r in results if r["dense_topk_hit"])
    hybrid_topk = sum(1 for r in results if r["hybrid_topk_hit"])
    dense_top1 = sum(1 for r in results if r["dense_top1_hit"])
    hybrid_top1 = sum(1 for r in results if r["hybrid_top1_hit"])

    # 分场景
    kw = [r for r in results if r["is_keyword_query"]]
    sem = [r for r in results if not r["is_keyword_query"]]

    print("\n" + "=" * 78)
    print(f"  RAG 检索评测报告  (top_k={top_k}, 查询 {total} 条)")
    print(f"  生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    print(f"\n  {'场景':<12} {'方法':<10} {'Top1 命中':<12} {'Top-k 命中':<12} {'查询数':<6}")
    print("-" * 55)

    def row(label, method, top1, topk, n):
        print(f"  {label:<12} {method:<10} {top1}/{n:<10} {topk}/{n:<10} {n:<6}")

    row("全部查询", "Dense", dense_top1, dense_topk, total)
    row("全部查询", "Hybrid", hybrid_top1, hybrid_topk, total)
    print("-" * 55)
    if kw:
        k_d = sum(1 for r in kw if r["dense_topk_hit"])
        k_h = sum(1 for r in kw if r["hybrid_topk_hit"])
        row("关键词场景", "Dense", sum(1 for r in kw if r["dense_top1_hit"]), k_d, len(kw))
        row("关键词场景", "Hybrid", sum(1 for r in kw if r["hybrid_top1_hit"]), k_h, len(kw))
    if sem:
        s_d = sum(1 for r in sem if r["dense_topk_hit"])
        s_h = sum(1 for r in sem if r["hybrid_topk_hit"])
        row("语义场景", "Dense", sum(1 for r in sem if r["dense_top1_hit"]), s_d, len(sem))
        row("语义场景", "Hybrid", sum(1 for r in sem if r["hybrid_top1_hit"]), s_h, len(sem))
    print("-" * 55)

    # 提升量
    improved = sum(1 for r in results if r["hybrid_topk_hit"] and not r["dense_topk_hit"])
    regressed = sum(1 for r in results if r["dense_topk_hit"] and not r["hybrid_topk_hit"])
    print(f"\n  混合检索相对纯向量：命中提升 {improved} 条，回落 {regressed} 条")

    # 逐条明细
    print("\n" + "-" * 78)
    print("  逐条明细")
    print("-" * 78)
    for r in results:
        d_mark = "✓" if r["dense_topk_hit"] else "✗"
        h_mark = "✓" if r["hybrid_topk_hit"] else "✗"
        tag = "[关键词]" if r["is_keyword_query"] else "[语义]"
        print(f"\n  {tag} 查询: {r['query']}")
        print(f"      期望来源: {r['expected_source']}")
        print(f"      Dense  Top-k {d_mark} | Hybrid Top-k {h_mark}")
        print(f"      --- Dense ---\n{r['dense_results']}")
        print(f"      --- Hybrid ---\n{r['hybrid_results']}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3, help="检索返回条数")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    print("RAG 混合检索 vs 纯向量检索 对比评测")
    print("-" * 40)

    # 关键检查：混合检索开关必须开启
    print(f"  HYBRID_SEARCH_ENABLED = {settings.HYBRID_SEARCH_ENABLED}")
    if not settings.HYBRID_SEARCH_ENABLED:
        print("  WARNING: 混合检索未开启，Hybrid 结果会降级为纯向量！")

    retriever = RAGRetriever()
    await retriever.rebuild_bm25_from_store()
    print(f"  BM25 索引重建完成: {len(retriever.embedder._bm25_chunks)} chunks\n")

    results = []
    for query, expected in TEST_QUERIES:
        r = await run_query(retriever, query, expected, args.top_k)
        results.append(r)
        mark = "✓" if r["hybrid_topk_hit"] else "✗"
        print(f"  [{mark}] {query[:30]:<32} → {expected}")

    print_summary(results, args.top_k)

    if args.json:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hybrid_retrieval_results.json")
        summary = {
            "date": __import__("datetime").datetime.now().isoformat(),
            "top_k": args.top_k,
            "total_queries": len(results),
            "dense_topk_hit": sum(1 for r in results if r["dense_topk_hit"]),
            "hybrid_topk_hit": sum(1 for r in results if r["hybrid_topk_hit"]),
            "dense_top1_hit": sum(1 for r in results if r["dense_top1_hit"]),
            "hybrid_top1_hit": sum(1 for r in results if r["hybrid_top1_hit"]),
            "results": [{k: v for k, v in r.items() if not k.endswith("_results")} for r in results],
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 结果已保存到 {out}")


if __name__ == "__main__":
    asyncio.run(main())
