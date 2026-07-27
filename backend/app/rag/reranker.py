"""app.rag.reranker —— Cross-Encoder 精排 + 三级降级链

设计要点
--------
召回阶段（向量 + BM25）追求高召回率，返回 top-50 候选；精排阶段用
Cross-Encoder 对 query 与每个候选做联合编码打分，选出最相关的 top-5。
Cross-Encoder 比双塔向量模型更准，因为它让 query 与 document 在 attention
层充分交互，能捕捉细粒度语义关系。

三级降级链
~~~~~~~~~~
1. **首选 Cross-Encoder 精排**：加载 ``BAAI/bge-reranker-v2-m3``，对
   ``(query, candidate)`` 对打分，按分数重排。这是质量最高的方案。
2. **模型加载失败 → RRF 分数排序**：若 Cross-Encoder 模型不可用（离线模式
   / GPU 不足 / 模型文件缺失），退化为按 RRF 融合分数排序。此时标记
   ``rerank_method="rrf_fallback"``，质量略降但链路不中断。
3. **候选为空 → 返回空列表**：无候选时直接返回，避免无意义计算。

离线模式
~~~~~~~~
``settings.is_offline_mode=True`` 时直接跳过 Cross-Encoder，走 RRF fallback，
避免在没有模型文件的环境下阻塞。这使本模块可在无 GPU 的开发/测试环境运行。

为何类级缓存
~~~~~~~~~~~~
Cross-Encoder 模型加载耗时数秒，整个进程共享一个实例即可，无需每个请求
重新加载。``get_reranker`` 单例 + 类级 ``_model`` 字段保证这一点。
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.config import settings
from app.utils.logger import logger

try:
    from sentence_transformers import CrossEncoder

    _CE_AVAILABLE = True
except ImportError:  # pragma: no cover
    CrossEncoder = None  # type: ignore
    _CE_AVAILABLE = False
    logger.warning("sentence-transformers 未安装，Cross-Encoder 精排将降级")


class CrossEncoderReranker:
    """Cross-Encoder 精排器（三级降级链）。

    Parameters
    ----------
    model_name : str | None
        模型名，默认取 settings.reranker_model。
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name: str = model_name or getattr(
            settings, "reranker_model", "BAAI/bge-reranker-v2-m3"
        )
        self._model: CrossEncoder | None = None
        self._model_loaded: bool = False
        self._load_attempted: bool = False

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """懒加载 Cross-Encoder 模型。"""
        if self._load_attempted:
            return
        self._load_attempted = True

        # 离线模式直接跳过
        if getattr(settings, "is_offline_mode", False):
            logger.info("离线模式，跳过 Cross-Encoder 加载，使用 RRF fallback")
            return

        if not _CE_AVAILABLE:
            logger.warning("sentence-transformers 未安装，Cross-Encoder 降级")
            return

        try:
            self._model = CrossEncoder(self._model_name, device=self._resolve_device())
            # FP16 半精度: GPU 下吞吐约 2x
            device = self._resolve_device()
            if getattr(settings, "embedding_use_fp16", False) and device.startswith("cuda"):
                try:
                    self._model.model = self._model.model.half()
                    logger.info("Cross-Encoder 已切换 FP16 半精度推理")
                except Exception as fp16_err:  # noqa: BLE001
                    logger.warning("Reranker FP16 切换失败, 沿用 FP32: %s", fp16_err)
            self._model_loaded = True
            logger.info("Cross-Encoder 模型加载成功: %s", self._model_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Cross-Encoder 模型加载失败，降级为 RRF fallback: %s", exc)
            self._model_loaded = False

    @staticmethod
    def _resolve_device() -> str:
        """解析推理设备: 复用 embedding_device 配置, CUDA 不可用时降级 cpu。"""
        device = getattr(settings, "embedding_device", "cpu")
        if device.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    logger.warning("Reranker 配置 device=%s 但 CUDA 不可用, 降级 cpu", device)
                    return "cpu"
            except ImportError:
                return "cpu"
        return device

    # ------------------------------------------------------------------
    # 精排
    # ------------------------------------------------------------------
    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
        content_field: str = "content",
    ) -> list[dict[str, Any]]:
        """对候选列表精排。

        执行三级降级链：
        1. 候选为空 → 返回空列表（rerank_method="none"）
        2. Cross-Encoder 可用 → 联合编码打分（rerank_method="cross_encoder"）
        3. Cross-Encoder 不可用 → 按 RRF 分数排序（rerank_method="rrf_fallback"）

        Parameters
        ----------
        query : str
            用户查询。
        candidates : list[dict]
            召回候选列表，每个 dict 须含 ``content_field`` 字段与 ``score``
            （RRF 分数，用于 fallback 排序）。
        top_k : int
            返回数量。
        content_field : str
            候选中用于打分的文本字段。

        Returns
        -------
        list[dict]
            精排后的 top-k 列表，每个 dict 新增 ``rerank_score`` 与
            ``rerank_method`` 字段。
        """
        # 第一级：候选为空
        if not candidates:
            logger.debug("Rerank 输入为空，返回空列表")
            return []

        # 第二级：Cross-Encoder 精排
        self._load_model()
        if self._model_loaded and self._model is not None:
            try:
                ranked = await self._rerank_with_cross_encoder(
                    query, candidates, top_k, content_field
                )
                for doc in ranked:
                    doc["rerank_method"] = "cross_encoder"
                logger.debug(
                    "Cross-Encoder 精排完成: 候选=%d, 返回=%d",
                    len(candidates),
                    len(ranked),
                )
                return ranked
            except Exception as exc:  # noqa: BLE001
                logger.error("Cross-Encoder 精排异常，降级为 RRF fallback: %s", exc)

        # 第三级：RRF fallback
        ranked = self._rerank_with_rrf_fallback(candidates, top_k)
        for doc in ranked:
            doc["rerank_method"] = "rrf_fallback"
        logger.debug("RRF fallback 排序完成: 候选=%d, 返回=%d", len(candidates), len(ranked))
        return ranked

    async def _rerank_with_cross_encoder(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
        content_field: str,
    ) -> list[dict[str, Any]]:
        """用 Cross-Encoder 对候选打分并排序。"""
        # 构造 (query, content) 对
        pairs = [(query, str(doc.get(content_field, ""))) for doc in candidates]

        # Cross-Encoder 的 predict 是同步 CPU 密集操作，放线程池
        # 显式 batch_size 控制显存峰值, 避免 50 候选 OOM
        batch_size = getattr(settings, "reranker_batch_size", 32)
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda p=pairs, b=batch_size: self._model.predict(p, batch_size=b),  # type: ignore[union-attr]
        )

        # 附加分数并排序
        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        # 回填排名
        for rank, doc in enumerate(ranked, start=1):
            doc["rank"] = rank
            # 统一 score 字段为精排分数
            doc["score"] = doc["rerank_score"]

        return ranked[:top_k]

    def _rerank_with_rrf_fallback(
        self,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """RRF fallback：直接用已有的 RRF 分数排序。

        候选列表在 RRF 融合后已按 ``score``（rrf_score）降序，但为保险起见
        重新排序。降级时 ``rerank_score`` 设为原 RRF 分数。
        """
        ranked = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
        for rank, doc in enumerate(ranked, start=1):
            doc["rank"] = rank
            doc["rerank_score"] = doc.get("score", 0.0)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        """模型是否已加载。"""
        return self._model_loaded

    @property
    def is_available(self) -> bool:
        """精排是否可用（含 fallback，总是返回 True）。"""
        return True

    @property
    def model_info(self) -> dict[str, Any]:
        """模型信息。"""
        return {
            "model_name": self._model_name,
            "loaded": self._model_loaded,
            "fallback": "rrf" if not self._model_loaded else "none",
        }


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
_reranker_instance: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    """获取全局精排器单例。"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance
