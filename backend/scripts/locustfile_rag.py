"""Locust 压测脚本: 企业 RAG 知识库检索 API.

针对 /api/v1/retrieval/search 端点做并发压测.
启动前需先启动后端 (uvicorn app.main:app --port 8765).

用法:
    # 方式 1: locust CLI (Web UI)
    locust -f scripts/locustfile_rag.py --host http://localhost:8765

    # 方式 2: 直接运行 (headless, 推荐)
    python -m scripts.locustfile_rag --users 20 --spawn-rate 2 --run-time 30s --host http://localhost:8765
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 确保能 import app
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


# ======================== 配置 ========================
DEFAULT_HOST = "http://localhost:8765"
DEFAULT_HOSTS = "http://localhost:8765,http://localhost:8766,http://localhost:8767,http://localhost:8768"
DEFAULT_USERS = 20
DEFAULT_DURATION = 30  # 秒
DEFAULT_SPAWN_RATE = 2  # 用户/秒

# 测试账号 (init_db.py 默认 admin)
TEST_EMAIL = "admin@semitech.cn"
TEST_PASSWORD = "admin123"

# 真实查询样本 (32 条 - 来自 rag_eval_dataset.json 的真实查询)
SAMPLE_QUERIES = [
    "NAND Flash 工作温度范围",
    "车规 eMMC 5.1 标准",
    "DDR5 SPD 信息",
    "LPDDR4X 内存颗粒",
    "UFS 3.1 顺序读取速度",
    "SSD 固态硬盘 TRIM 命令",
    "BSCI 社会责任认证",
    "ISO 9001 质量管理体系",
    "IATF 16949 车规认证",
    "CE 认证电磁兼容测试",
    "RoHS 有害物质限制",
    "FCC 认证美国市场",
    "UKCA 英国合格评定",
    "3D TLC NAND 闪存",
    "QLC SSD 高密度存储",
    "DRAM 内存时序",
    "eMMC 启动分区配置",
    "UFS HS Gear 4",
    "SSD 主控 Wear Leveling",
    "NAND Bad Block 管理",
    "DDR4 颗粒封装",
    "LPDDR5 VDDQ 电压",
    "eMMC 5.1 HS400 模式",
    "UFS 2.2 写入速度",
    "SSD Opal 自加密",
    "NAND ECC 纠错",
    "DRAM Refresh 周期",
    "车规 AEC-Q100 标准",
    "工业级 SSD 工作温度",
    "消费级 eMMC 寿命",
    "企业级 SSD DWPD 指标",
    "SSD TBW 写入寿命",
]


# ======================== JWT 登录 ========================
def login(host: str, email: str, password: str, timeout: int = 10) -> Optional[str]:
    """登录获取 JWT token."""
    url = f"{host}/api/v1/auth/login"
    body = json.dumps({"email": email, "password": password}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    try:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("data", {}).get("access_token")
    except Exception as e:
        print(f"登录失败: {e}")
        return None


# ======================== HTTP 客户端 ========================
def make_request(
    method: str,
    url: str,
    token: Optional[str] = None,
    body: Optional[dict] = None,
    timeout: int = 30,
) -> Tuple[int, float, str]:
    """发起单次 HTTP 请求, 返回 (status, latency_ms, error)."""
    data = json.dumps(body).encode("utf-8") if body else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    start = time.perf_counter()
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency = (time.perf_counter() - start) * 1000
            return resp.status, latency, ""
    except urllib.error.HTTPError as e:
        latency = (time.perf_counter() - start) * 1000
        return e.code, latency, ""
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return 0, latency, str(e)[:100]


def percentile(data: List[float], p: float) -> float:
    """计算百分位数."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


# ======================== 压测主流程 ========================
def run_benchmark(
    hosts: str,
    users: int,
    duration: int,
    spawn_rate: int = DEFAULT_SPAWN_RATE,
) -> Dict[str, Any]:
    """运行 RAG 检索 API 压测.

    流程:
    1. 登录获取 JWT token (admin@semitech.cn) — 对每个 host 都登录, 复用同一 token
    2. 启动 N 个并发线程, 每个线程循环:
       - 从样本查询池随机选一个
       - 从 host 池轮询选一个后端实例 (负载均衡)
       - 调用 GET /api/v1/retrieval/search?query=...&top_k=5
       - 记录延迟与状态
    3. 持续 duration 秒后停止
    4. 汇总统计: QPS / P50 / P95 / P99 / 错误率 / 成功率
    """
    host_list = [h.strip() for h in hosts.split(",") if h.strip()]
    primary_host = host_list[0]

    print(f"\n=== RAG 检索 API 压测启动 ===")
    print(f"目标: {len(host_list)} 个后端实例: {host_list}")
    print(f"并发: {users} | 持续: {duration}s | 启动速率: {spawn_rate}/s")

    # 登录 (用首个 host 登录, token 跨实例复用, 因为共享同一 JWT secret)
    print(f"\n[1/4] 登录获取 JWT (admin@semitech.cn)...")
    token = login(primary_host, TEST_EMAIL, TEST_PASSWORD)
    if not token:
        return {"error": "登录失败, 无法继续压测"}
    print(f"  JWT 获取成功 (长度 {len(token)})")

    # 预热每个实例 (首次检索会触发模型加载)
    print(f"\n[2/4] 预热 {len(host_list)} 个实例 (加载 BGE-M3 模型, 首次约 10-15s)...")
    # 注意: URL 中的中文必须 URL-encode, 否则 urllib.request.Request 会抛
    # UnicodeEncodeError (ascii codec can't encode), 状态返回 0 即时失败.
    warmup_query = urllib.parse.quote("NAND Flash 工作温度范围")
    for i, h in enumerate(host_list):
        warmup_url = f"{h}/api/v1/retrieval/search?query={warmup_query}&top_k=5"
        status, latency, _ = make_request(
            "GET",
            warmup_url,
            token=token,
            timeout=120,
        )
        print(f"  [{i+1}/{len(host_list)}] {h}: status={status} latency={latency:.0f}ms")

    # 压测
    print(f"\n[3/4] 启动 {users} 个并发线程, 持续 {duration}s...\n")
    results: List[Dict] = []
    end_time = time.time() + duration
    lock = threading.Lock()
    host_counter = [0]  # round-robin counter

    def worker(worker_id: int):
        local_results: List[Dict] = []
        rng = random.Random(time.time() + worker_id)
        while time.time() < end_time:
            query = rng.choice(SAMPLE_QUERIES)
            top_k = rng.choice([3, 5, 5, 5, 10])  # 5 占多数, 模拟真实分布
            # round-robin 选择 host (负载均衡)
            with lock:
                h = host_list[host_counter[0] % len(host_list)]
                host_counter[0] += 1
            url = f"{h}/api/v1/retrieval/search?query={urllib.parse.quote(query)}&top_k={top_k}"
            status, latency, error = make_request("GET", url, token=token, timeout=30)
            local_results.append({
                "worker_id": worker_id,
                "endpoint": "/api/v1/retrieval/search",
                "method": "GET",
                "status": status,
                "latency_ms": latency,
                "error": error,
                "query": query,
                "top_k": top_k,
                "host": h,
            })
        with lock:
            results.extend(local_results)

    # 渐进启动
    pool = ThreadPoolExecutor(max_workers=users)
    futures = []
    for i in range(users):
        # 每 1/spawn_rate 秒启动一个 worker
        time.sleep(1.0 / spawn_rate)
        futures.append(pool.submit(worker, i + 1))

    # 实时进度打印
    last_print = time.time()
    while time.time() < end_time:
        time.sleep(2)
        if time.time() - last_print >= 5:
            with lock:
                cur = len(results)
            elapsed = time.time() - (end_time - duration)
            qps = cur / elapsed if elapsed > 0 else 0
            print(f"  [{elapsed:.0f}s] 已完成 {cur} 请求, 当前 QPS ≈ {qps:.1f}")
            last_print = time.time()

    for f in as_completed(futures):
        f.result()
    pool.shutdown(wait=True)

    # 统计
    print(f"\n[4/4] 汇总统计...")
    total = len(results)
    by_status: Dict[int, List[Dict]] = defaultdict(list)
    for r in results:
        by_status[r["status"]].append(r)

    latencies = [r["latency_ms"] for r in results]
    successes = [r for r in results if 200 <= r["status"] < 300]
    success_latencies = [r["latency_ms"] for r in successes]

    stats = {
        "host": hosts,
        "instances": len(host_list),
        "endpoint": "/api/v1/retrieval/search",
        "concurrent_users": users,
        "duration_seconds": duration,
        "spawn_rate": spawn_rate,
        "total_requests": total,
        "total_success": len(successes),
        "total_errors": total - len(successes),
        "overall_error_rate": round((total - len(successes)) / total * 100, 2) if total else 0,
        "qps": round(total / duration, 2) if duration > 0 else 0,
        "success_qps": round(len(successes) / duration, 2) if duration > 0 else 0,
        # 全部请求 (含失败) 延迟
        "all_avg_ms": round(statistics.mean(latencies), 1) if latencies else 0,
        "all_p50_ms": round(percentile(latencies, 50), 1) if latencies else 0,
        "all_p95_ms": round(percentile(latencies, 95), 1) if latencies else 0,
        "all_p99_ms": round(percentile(latencies, 99), 1) if latencies else 0,
        "all_min_ms": round(min(latencies), 1) if latencies else 0,
        "all_max_ms": round(max(latencies), 1) if latencies else 0,
        # 成功请求延迟 (排除超时/错误的干扰)
        "success_avg_ms": round(statistics.mean(success_latencies), 1) if success_latencies else 0,
        "success_p50_ms": round(percentile(success_latencies, 50), 1) if success_latencies else 0,
        "success_p95_ms": round(percentile(success_latencies, 95), 1) if success_latencies else 0,
        "success_p99_ms": round(percentile(success_latencies, 99), 1) if success_latencies else 0,
        "success_min_ms": round(min(success_latencies), 1) if success_latencies else 0,
        "success_max_ms": round(max(success_latencies), 1) if success_latencies else 0,
        # 状态码分布
        "status_distribution": {
            str(code): len(items) for code, items in sorted(by_status.items())
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return stats


# ======================== 报告生成 ========================
def generate_report(stats: Dict[str, Any]) -> str:
    """生成 Markdown 报告."""
    lines = [
        "# RAG 检索 API 性能压测报告",
        "",
        f"> 压测时间: {stats['timestamp']}",
        f"> 目标: {stats['host']}{stats['endpoint']}",
        f"> 并发: {stats['concurrent_users']} 用户 | 持续: {stats['duration_seconds']}s | 启动速率: {stats['spawn_rate']}/s",
        "",
        "## 总体指标",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 总请求数 | {stats['total_requests']} |",
        f"| 成功请求 | {stats['total_success']} |",
        f"| 失败请求 | {stats['total_errors']} |",
        f"| 错误率 | {stats['overall_error_rate']}% |",
        f"| **QPS (总请求/s)** | **{stats['qps']}** |",
        f"| 成功 QPS | {stats['success_qps']} |",
        "",
        "## 延迟分布 (含全部请求)",
        "",
        "| 指标 | 数值 (ms) |",
        "|------|-----------|",
        f"| 平均 | {stats['all_avg_ms']} |",
        f"| P50 | {stats['all_p50_ms']} |",
        f"| **P95** | **{stats['all_p95_ms']}** |",
        f"| **P99** | **{stats['all_p99_ms']}** |",
        f"| 最小 | {stats['all_min_ms']} |",
        f"| 最大 | {stats['all_max_ms']} |",
        "",
        "## 延迟分布 (仅成功请求)",
        "",
        "| 指标 | 数值 (ms) |",
        "|------|-----------|",
        f"| 平均 | {stats['success_avg_ms']} |",
        f"| P50 | {stats['success_p50_ms']} |",
        f"| **P95** | **{stats['success_p95_ms']}** |",
        f"| **P99** | **{stats['success_p99_ms']}** |",
        f"| 最小 | {stats['success_min_ms']} |",
        f"| 最大 | {stats['success_max_ms']} |",
        "",
        "## 状态码分布",
        "",
        "| HTTP 状态码 | 次数 | 占比 |",
        "|-------------|------|------|",
    ]
    total = stats["total_requests"]
    for code, count in stats["status_distribution"].items():
        pct = round(count / total * 100, 1) if total else 0
        lines.append(f"| {code} | {count} | {pct}% |")

    lines.extend([
        "",
        "## 测试环境",
        "",
        f"- OS: Windows (TRAE Sandbox)",
        f"- Backend: FastAPI + Uvicorn (4 独立进程, 端口 8765-8768)",
        f"- GPU: NVIDIA RTX 4060 Ti 16GB (BGE-M3 + CrossEncoder FP16)",
        f"- DB: PostgreSQL 15 (5433) + Redis (6380, QA 缓存)",
        f"- Vector: Milvus 2.x + BGE-M3 (1024 维, GPU 加速)",
        f"- Graph: Neo4j 5.x",
        f"- 检索链路: BM25 + Milvus 向量 + RRF 融合 + Cross-Encoder 精排 (recall_k=20)",
        f"- 压测工具: Python `urllib` + `ThreadPoolExecutor` (无外部依赖)",
        "",
        "## 测试说明",
        "",
        "1. **多实例负载均衡**: 4 个独立 Uvicorn 进程, Round-Robin 轮询, 绕开 Windows SO_REUSEPORT 限制.",
        "2. **JWT 鉴权**: 测试前用 `admin@semitech.cn` 登录获取 JWT, 模拟真实用户.",
        "3. **查询样本**: 32 条真实存储行业查询 (NAND/DRAM/eMMC/SSD + 认证类).",
        "4. **top_k 分布**: 3/5/5/5/10 加权随机, 模拟真实用户检索习惯 (5 占多数).",
        "5. **QPS 含义**: 总请求/秒, 包含失败的请求; `success_qps` 仅算成功请求.",
        "6. **P95/P99**: 95%/99% 的请求延迟低于该值, 是企业级 RAG 系统关键 SLO 指标.",
        "7. **Redis 缓存**: 重复查询命中缓存 (TTL=1h), 显著降低 P95/P99 延迟.",
    ])
    return "\n".join(lines)


# ======================== CLI ========================
def main():
    parser = argparse.ArgumentParser(description="RAG 检索 API Locust 压测")
    parser.add_argument("--host", default=DEFAULT_HOSTS, help="目标主机(逗号分隔多实例负载均衡)")
    parser.add_argument("--hosts", default=None, help="覆盖 --host, 多实例用逗号分隔")
    parser.add_argument("--users", type=int, default=DEFAULT_USERS, help="并发用户数")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION, help="持续时间(秒)")
    parser.add_argument("--spawn-rate", type=int, default=DEFAULT_SPAWN_RATE, help="用户启动速率(/s)")
    parser.add_argument("--output", default="docs/load-test-200-4worker-gpu.md", help="报告输出路径")
    args = parser.parse_args()

    hosts = args.hosts if args.hosts else args.host
    stats = run_benchmark(hosts, args.users, args.duration, args.spawn_rate)

    if "error" in stats:
        print(f"\n压测失败: {stats['error']}")
        return 1

    # 控制台输出
    print(f"\n{'=' * 70}")
    print(f"RAG 检索 API 压测结果")
    print(f"{'=' * 70}")
    print(f"总请求: {stats['total_requests']} | 成功: {stats['total_success']} | 错误率: {stats['overall_error_rate']}%")
    print(f"QPS: {stats['qps']} (成功 QPS: {stats['success_qps']})")
    print(f"\n延迟分布 (全部请求):")
    print(f"  平均: {stats['all_avg_ms']}ms | P50: {stats['all_p50_ms']}ms | P95: {stats['all_p95_ms']}ms | P99: {stats['all_p99_ms']}ms")
    print(f"延迟分布 (成功请求):")
    print(f"  平均: {stats['success_avg_ms']}ms | P50: {stats['success_p50_ms']}ms | P95: {stats['success_p95_ms']}ms | P99: {stats['success_p99_ms']}ms")
    print(f"\n状态码分布: {stats['status_distribution']}")

    # 写入报告
    report = generate_report(stats)
    report_path = Path(__file__).resolve().parent.parent.parent / args.output
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\n报告已写入: {report_path}")

    # 也保存原始 JSON 数据 (供 README 渲染图表)
    json_path = report_path.parent / "load-test-data.json"
    json_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"原始数据: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
