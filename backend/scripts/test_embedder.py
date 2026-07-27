"""Quick smoke test: verify BGE-M3 local model loads and produces real vectors."""
import asyncio
import sys
from pathlib import Path

# Ensure backend on path
_BACKEND = Path(__file__).resolve().parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.rag.embedder import get_embedder  # noqa: E402


async def main():
    embedder = get_embedder()
    queries = [
        "跨境电商选品指南",
        "Amazon FBA 运营规范",
        "BGE-M3 向量检索测试",
    ]
    vectors = await embedder.embed(queries)
    print(f"=== BGE-M3 Embedder Smoke Test ===")
    print(f"model_name: {embedder._model_name}")
    print(f"is_loaded:  {embedder.is_loaded}")
    print(f"is_available: {embedder.is_available}")
    print(f"query_count: {len(queries)}")
    for q, v in zip(queries, vectors):
        print(f"  query='{q}' dim={len(v)} first5={[round(x, 4) for x in v[:5]]}")
    # Verify vectors are real (not random): same query should produce same vector
    v1 = await embedder.embed_one(queries[0])
    v2 = await embedder.embed_one(queries[0])
    diff = sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5
    print(f"determinism_check (L2 diff of same query): {diff:.8f}")
    print(f"model_info: {embedder.model_info}")
    print("=== PASS ===" if embedder.is_loaded and diff < 1e-5 else "=== FALLBACK (random vectors) ===")


if __name__ == "__main__":
    asyncio.run(main())
