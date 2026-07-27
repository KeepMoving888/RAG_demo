"""
RAG 检索质量评估脚本

功能:
1. 运行 ``RetrievalEvaluator.ablation_study`` 消融实验, 对比全部策略;
2. 输出对比表格到 stdout, 并写入 ``data/eval_results_{timestamp}.json``;
3. 支持 ``--strategy full`` 单策略运行, 仅输出该策略指标.

用法:
    python -m scripts.eval_rag                       # 消融实验
    python -m scripts.eval_rag --strategy full       # 单策略
    python scripts/eval_rag.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# 确保 backend 在 sys.path 中
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.rag.evaluator import ABLATION_STRATEGIES, RetrievalEvaluator  # noqa: E402
from app.utils.logger import logger  # noqa: E402


# ======================== 输出路径 ========================
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_DATA_DIR = _BACKEND_DIR / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    """生成时间戳字符串 (用于文件名)."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _run_ablation() -> dict:
    """运行消融实验, 返回完整结果."""
    evaluator = RetrievalEvaluator()
    logger.info("开始消融实验, 策略数={}, 数据集大小={}",
                len(ABLATION_STRATEGIES), evaluator.dataset_size)
    return evaluator.ablation_study()


def _run_single(strategy: str) -> dict:
    """运行单策略评估, 返回该策略指标."""
    if strategy not in ABLATION_STRATEGIES:
        raise ValueError(
            f"不支持的策略: {strategy}, 允许: {ABLATION_STRATEGIES}"
        )
    evaluator = RetrievalEvaluator()
    logger.info("开始单策略评估: strategy={}, 数据集大小={}",
                strategy, evaluator.dataset_size)
    return evaluator.evaluate(strategy=strategy)


def _write_results(payload: dict, suffix: str = "ablation") -> Path:
    """将结果写入 JSON 文件, 返回路径."""
    ts = _timestamp()
    out_path = _DATA_DIR / f"eval_results_{ts}_{suffix}.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("评估结果已写入: {}", out_path)
    return out_path


def _print_ablation(result: dict) -> None:
    """打印消融实验对比表到 stdout."""
    print("\n" + "=" * 80)
    print("RAG 检索质量消融实验")
    print("=" * 80)
    print()
    print(result.get("comparison_table", "(无对比表格)"))
    print()
    print(f"最优策略 (按 NDCG@5): {result.get('best_strategy', 'none')}")
    print()
    print("自动分析:")
    print(result.get("analysis", "(无)"))
    print()
    print("=" * 80)


def _print_single(metrics: dict) -> None:
    """打印单策略指标到 stdout."""
    print("\n" + "=" * 80)
    print(f"单策略评估: {metrics.get('strategy', '?')}")
    print("=" * 80)
    print(f"  样本数:        {metrics.get('sample_count', 0)}")
    print(f"  Recall@5:      {metrics.get('recall@5', 0.0):.4f}")
    print(f"  MRR:           {metrics.get('mrr', 0.0):.4f}")
    print(f"  NDCG@5:        {metrics.get('ndcg@5', 0.0):.4f}")
    print(f"  Precision@5:   {metrics.get('precision@5', 0.0):.4f}")
    print(f"  平均延迟 (ms): {metrics.get('avg_latency_ms', 0.0):.2f}")
    if metrics.get("error"):
        print(f"  错误: {metrics['error']}")
    print("=" * 80)


def main() -> int:
    """命令行入口.

    Returns:
        进程退出码 (0=成功, 1=失败).
    """
    parser = argparse.ArgumentParser(
        description="RAG 检索质量评估 (消融实验 / 单策略)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help=f"单策略运行, 可选: {ABLATION_STRATEGIES}; 缺省跑全部消融实验",
    )
    args = parser.parse_args()

    try:
        if args.strategy:
            metrics = _run_single(args.strategy)
            _print_single(metrics)
            out_path = _write_results(metrics, suffix=f"single_{args.strategy}")
            print(f"\n结果文件: {out_path}")
        else:
            result = _run_ablation()
            _print_ablation(result)
            out_path = _write_results(result, suffix="ablation")
            print(f"\n结果文件: {out_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("评估失败: {}", str(exc))
        print(f"\n评估失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
