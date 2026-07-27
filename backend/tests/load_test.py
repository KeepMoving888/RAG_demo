"""
Locust 性能压测脚本 —— Enterprise RAG Knowledge Base

仿照企业内部员工对 RAG 系统的典型访问行为:
1. 登录获取 JWT
2. 按权重随机调用 4 类接口:
   - 文档列表   (30%): GET /api/v1/documents
   - 问答检索   (40%): POST /api/v1/qa/ask
   - 图谱统计   (20%): GET /api/v1/graph/stats
   - 健康检查   (10%): GET /health

问答 query 从预定义企业知识库问题列表中随机选取, 贴近真实流量分布.

用法:
    locust -f tests/load_test.py
    locust -f tests/load_test.py --host http://localhost:8000
    locust -f tests/load_test.py --headless -u 50 -r 5 --run-time 60s \
        --host http://localhost:8000

可通过环境变量覆盖默认账号:
    LOAD_TEST_EMAIL=admin@semitech.cn LOAD_TEST_PASSWORD=admin123 locust -f tests/load_test.py
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

from locust import HttpUser, between, task

# 确保 backend 在 sys.path 中 (复用 app 的 logger / 配置, 非必需但便于调试)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


# ======================== 配置 (环境变量覆盖) ========================
_DEFAULT_EMAIL = os.getenv("LOAD_TEST_EMAIL", "admin@semitech.cn")
_DEFAULT_PASSWORD = os.getenv("LOAD_TEST_PASSWORD", "admin123")


# ======================== 预定义企业知识库问题 ========================
# 覆盖产品规格 / 认证 / 售后 / 供应商 / 选型 / 合规等典型咨询场景
EVAL_QUERIES: list[str] = [
    "车规 eMMC 的技术规格是什么",
    "ISO 9001 认证的范围和有效期",
    "通过 CE 认证的产品有哪些",
    "BSCI 社会责任审核的要求",
    "NAND Flash 的读取速度",
    "售后保修期多久",
    "退货流程是什么",
    "供应商如何评级",
    "DRAM 颗粒选型有哪些考量",
    "存储产品需要哪些产品认证",
]


class RagKnowledgeBaseUser(HttpUser):
    """仿照企业员工访问 RAG 知识库的虚拟用户.

    行为流程:
    1. 启动时调用 on_start 执行登录, 缓存 JWT;
    2. 每个迭代按权重随机执行 4 类 task;
    3. 所有 task 失败不阻断, locust 自动记录响应时间与状态码.
    """

    # 真实用户操作间隔: 1-3 秒
    wait_time = between(1, 3)

    # 每个虚拟用户持有的 JWT
    token: str = ""

    def on_start(self) -> None:
        """用户启动: 执行登录获取 JWT.

        登录失败不抛异常 (压测环境可能未初始化 admin), 后续请求会以 401 返回,
        locust 仍会记录这些响应, 便于发现环境问题.
        """
        email = _DEFAULT_EMAIL
        password = _DEFAULT_PASSWORD
        with self.client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": password},
            name="POST /auth/login",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                    data = payload.get("data") or {}
                    self.token = data.get("access_token", "")
                except ValueError:
                    self.token = ""
                if self.token:
                    resp.success()
                else:
                    resp.failure("登录响应缺少 access_token")
            else:
                resp.failure(f"登录失败 status={resp.status_code}")

    # ---------------- 认证头 ----------------
    def _auth_headers(self) -> dict:
        """构造 Bearer 鉴权头."""
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    # ======================== Tasks ========================
    # 权重分布: 文档列表 30 / 问答检索 40 / 图谱统计 20 / 健康检查 10
    @task(30)
    def list_documents(self) -> None:
        """文档列表查询 (分页). 仿照员工浏览知识库文档."""
        params = {
            "page": random.randint(1, 3),
            "page_size": random.choice([10, 20, 50]),
        }
        with self.client.get(
            "/api/v1/documents",
            params=params,
            headers=self._auth_headers(),
            name="GET /documents (列表)",
            catch_response=True,
        ) as resp:
            self._record(resp)

    @task(40)
    def ask_question(self) -> None:
        """智能问答检索. 从预定义问题中随机选取."""
        query = random.choice(EVAL_QUERIES)
        with self.client.post(
            "/api/v1/qa/ask",
            json={"query": query, "top_k": 5},
            headers=self._auth_headers(),
            name="POST /qa/ask (问答检索)",
            catch_response=True,
        ) as resp:
            self._record(resp)

    @task(20)
    def graph_stats(self) -> None:
        """知识图谱统计. 仿照管理员查看图谱规模."""
        with self.client.get(
            "/api/v1/graph/stats",
            headers=self._auth_headers(),
            name="GET /graph/stats (图谱统计)",
            catch_response=True,
        ) as resp:
            self._record(resp)

    @task(10)
    def health_check(self) -> None:
        """健康检查. 无需鉴权, 探测系统可用性."""
        with self.client.get(
            "/health",
            name="GET /health (健康检查)",
            catch_response=True,
        ) as resp:
            self._record(resp)

    # ---------------- 响应记录 ----------------
    @staticmethod
    def _record(resp) -> None:
        """统一处理响应: 标记 2xx 为成功, 其他为失败.

        locust 默认会记录响应时间与状态码, 这里通过 catch_response 显式控制
        成功/失败判定, 使非 2xx 响应在统计中被标记为失败.
        """
        if 200 <= resp.status_code < 300:
            resp.success()
        else:
            resp.failure(f"非预期状态码: {resp.status_code}")
