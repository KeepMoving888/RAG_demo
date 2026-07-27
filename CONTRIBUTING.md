# 贡献指南

感谢你对企业级 RAG 知识库项目的关注。本文档说明如何参与开发：从环境搭建、代码规范、Git 工作流、提交规范、PR 检查清单、项目结构到 Issue 模板与测试要求。请在提交 PR 前完整阅读本指南。

## 1. 开发环境搭建

### 1.1 前置依赖

- Python 3.11+
- Node.js 18+ / pnpm 8+
- Docker 24+ / Docker Compose 2.20+
- Make（Windows 建议通过 WSL2 或 Git Bash 提供）

### 1.2 一键启动开发环境

项目通过 `Makefile` 封装常用开发命令。克隆仓库后执行：

```bash
git clone <repo-url> enterprise-rag-kb
cd enterprise-rag-kb

# 安装后端与前端依赖
make install

# 拉起全部依赖服务（PostgreSQL/Redis/Milvus/Neo4j/ES/RabbitMQ）
make dev

# 初始化数据库与种子数据
make init-db
make seed
make graph-init
```

`make dev` 会以后端热重载模式启动，文件修改自动生效。开发期默认 `LLM_PROVIDER=offline`，无需配置大模型 API Key 即可跑通检索链路。

### 1.3 分服务启动

调试单一服务时可单独启动，避免全量拉起：

| 命令 | 作用 |
|------|------|
| `make backend` | 仅启动后端 API（FastAPI + uvicorn 热重载，端口 8080） |
| `make worker` | 仅启动 Celery Worker（文档入库流水线） |
| `make frontend` | 仅启动前端开发服务器（Vite，端口 3080） |
| `make dev` | 拉起依赖存储 + 后端 + Worker + 前端 |
| `make logs` | 查看全部服务日志 |
| `make down` | 停止全部服务 |

### 1.4 环境变量

复制 `.env.example` 为 `.env` 并按需修改。开发期默认值可直接使用，关键项说明见 [部署文档](./docs/deployment.md) 第 3 节。

## 2. 项目结构

```
enterprise-rag-kb/
├── backend/                      # 后端（FastAPI + Celery）
│   ├── app/
│   │   ├── api/                  # 路由层（auth/documents/qa/search/graph/admin）
│   │   ├── core/                 # 配置、安全、中间件（JWT/限流/审计）
│   │   ├── services/             # 业务服务层
│   │   ├── rag/                  # RAG 核心组件
│   │   │   ├── parsers/          # PDF/Word/图片解析器
│   │   │   ├── cleaners/         # 文档清洗
│   │   │   ├── chunkers/         # 语义分块
│   │   │   ├── embedders/        # 嵌入向量化
│   │   │   ├── retrievers/       # 向量/BM25/图谱检索
│   │   │   ├── fusion/           # RRF 融合
│   │   │   ├── rerankers/        # Cross-Encoder 精排
│   │   │   └── terminology.py    # 术语扩展
│   │   ├── graph/                # GraphRAG（Cypher 编排与安全）
│   │   ├── models/               # SQLAlchemy 数据模型
│   │   ├── schemas/              # Pydantic 请求/响应 schema
│   │   ├── tasks/                # Celery 异步任务（入库流水线）
│   │   ├── scripts/              # 初始化/种子/备份脚本
│   │   └── main.py
│   ├── tests/                    # 单元测试 + 集成测试 + 评测集
│   └── pyproject.toml
├── frontend/                     # 前端（React 18 + TypeScript + Vite）
│   ├── src/
│   │   ├── components/           # shadcn/ui 组件
│   │   ├── pages/                # 页面
│   │   ├── api/                  # 接口封装
│   │   ├── hooks/                # 自定义 hooks
│   │   └── stores/               # 状态管理
│   └── package.json
├── benchmarks/                   # 性能基准与消融评估脚本
├── deploy/                       # docker-compose.yml / Dockerfile / nginx
├── docs/                         # 文档
├── scripts/                      # 备份、迁移等运维脚本
├── Makefile
└── .env.example
```

## 3. 代码规范

### 3.1 后端（Python）

使用以下工具保证代码风格与类型一致，提交前必须全部通过：

| 工具 | 作用 | 配置位置 |
|------|------|----------|
| black | 代码格式化 | `pyproject.toml` |
| isort | import 排序 | `pyproject.toml` |
| flake8 | 风格检查 | `.flake8` |
| mypy | 静态类型检查 | `pyproject.toml` |

一键检查与格式化：

```bash
make lint        # 运行 flake8 + mypy
make format      # 运行 black + isort 自动修复
```

类型注解要求：新增公开函数与方法必须标注参数与返回类型，`mypy --strict` 在新增代码上必须通过。

### 3.2 前端（TypeScript）

| 工具 | 作用 |
|------|------|
| eslint | 代码规范检查 |
| prettier | 代码格式化 |
| tsc | TypeScript 类型检查 |

```bash
cd frontend
pnpm lint        # eslint 检查
pnpm format      # prettier 格式化
pnpm typecheck   # tsc --noEmit
```

前端禁用 `any`，确需使用时以 `// eslint-disable-next-line` 标注原因。

## 4. Git 工作流

项目采用 fork + PR 模式，主仓库保护 `main` 分支，禁止直接推送。

### 4.1 分支模型

- `main`：始终可发布的状态，所有 PR 经 review 合入。
- `feature/<scope>-<topic>`：功能分支，如 `feature/rag-terminology-expand`。
- `fix/<scope>-<topic>`：修复分支，如 `fix/graph-cypher-injection`。
- `docs/<topic>`：文档分支。

### 4.2 贡献流程

```bash
# 1. Fork 仓库到个人空间，克隆到本地
git clone git@github.com:<your-fork>/enterprise-rag-kb.git
cd enterprise-rag-kb
git remote add upstream git@github.com:<origin>/enterprise-rag-kb.git

# 2. 同步最新 main
git fetch upstream
git checkout main
git merge upstream/main

# 3. 创建功能分支
git checkout -b feature/rag-terminology-expand

# 4. 开发、提交（遵循提交规范）
git add <files>
git commit -m "feat(rag): add terminology expander for BM25 boost"

# 5. 推送到个人 fork 并发起 PR
git push origin feature/rag-terminology-expand
# 在 GitHub 上向 upstream/main 发起 Pull Request
```

### 4.3 Code Review 要求

- 至少一名 maintainer 批准（Approve）方可合入。
- 涉及检索/分块/图谱核心逻辑的 PR 需两名 maintainer 批准。
- Review 聚焦：正确性、测试覆盖、性能影响、安全（尤其 Cypher 与 SQL 拼接）、文档同步。
- PR 讨论修改后需重新请求 review，不得自行合入。

## 5. 提交规范

采用 [Conventional Commits](https://www.conventionalcommits.org/) 规范，便于自动生成变更日志与版本管理。

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 5.1 type 取值

| type | 含义 | 示例 |
|------|------|------|
| feat | 新功能 | `feat(rag): add terminology expander` |
| fix | 缺陷修复 | `fix(graph): parameterize cypher to prevent injection` |
| docs | 文档更新 | `docs(api): add SSE streaming protocol section` |
| refactor | 重构（无行为变化） | `refactor(retriever): unify retriever interface` |
| test | 测试相关 | `test(rag): add ablation eval for chunking strategies` |
| chore | 构建/工具/依赖 | `chore(deps): bump langchain to 0.2.0` |
| perf | 性能优化 | `perf(milvus): switch to HNSW index` |
| ci | CI 配置 | `ci: add mypy gate to workflow` |

### 5.2 scope 取值

`rag` / `graph` / `api` / `core` / `frontend` / `deploy` / `docs` / `deps` / `test`。

### 5.3 示例

```
feat(rag): add terminology expander for BM25 boost

Add TerminologyExpander to expand synonyms and boost term keywords
in BM25 retrieval. Dictionary is configurable per department.

Closes #142
```

## 6. PR 检查清单

提交 PR 前请逐项确认：

- [ ] 分支基于最新 `main` 创建，无冲突。
- [ ] 提交信息符合 Conventional Commits 规范。
- [ ] `make lint` 通过（后端 flake8 + mypy，前端 eslint + tsc）。
- [ ] `make format` 已执行，无格式问题。
- [ ] 新增/修改功能配套单元测试，且 `make test` 全部通过。
- [ ] 涉及 RAG 检索/分块/图谱的改动，附消融评估结果对比（见第 8 节）。
- [ ] 涉及接口变更的，已更新 [API 文档](./docs/api-reference.md)。
- [ ] 涉及配置项变更的，已更新 `.env.example` 与 [部署文档](./docs/deployment.md)。
- [ ] 无硬编码密钥、密码、Token 等敏感信息。
- [ ] 无大体积二进制文件入库（文档原文件走对象存储）。
- [ ] PR 描述说明：改动目的、实现方式、测试方法、对检索质量的影响。

## 7. 测试要求

### 7.1 测试分层

| 层级 | 目录 | 工具 | 覆盖要求 |
|------|------|------|----------|
| 单元测试 | `backend/tests/unit/` | pytest | 新增函数/类需配套测试，核心组件覆盖率 ≥ 80% |
| 集成测试 | `backend/tests/integration/` | pytest + testcontainers | 涉及存储交互的组件需集成测试 |
| API 测试 | `backend/tests/api/` | pytest + httpx | 新增/变更接口需 API 测试 |
| 前端测试 | `frontend/src/**/*.test.tsx` | vitest + testing-library | 关键交互逻辑需测试 |
| 评测集 | `backend/tests/eval/` | 自研评估框架 | RAG 改动需跑消融评估 |

### 7.2 运行测试

```bash
# 后端全量测试
make test

# 仅单元测试（快）
make test-unit

# 仅集成测试（需 Docker 拉起依赖）
make test-integration

# 前端测试
cd frontend && pnpm test
```

### 7.3 RAG 改动的消融评估

任何影响检索链路的改动（分块策略、向量化、检索融合、重排、术语、图谱抽取）必须在 PR 中附带消融评估结果，证明改动不会导致检索质量回归。

```bash
# 在评测集上对比改动前后的检索指标
python benchmarks/run_ablation.py \
  --strategies vector_only,bm25_only,rrf,full,full_with_terminology \
  --top-k 5 \
  --dataset internal_v3
```

PR 描述中需贴出改动前后 Recall@5 / MRR / NDCG@5 / Precision@5 对比表。若指标下降，需说明取舍理由（如换取延迟收益）并获 maintainer 确认。

### 7.4 测试编写规范

- 测试函数命名：`test_<被测行为>_<场景>`，如 `test_rrf_fusion_handles_empty_retriever`。
- 每个测试独立可运行，不依赖其他测试的执行顺序与状态。
- 使用 fixture 管理测试数据与外部依赖，避免测试间状态泄漏。
- 集成测试使用 testcontainers 拉起临时存储，测试结束自动清理。

## 8. Issue 模板

提 Issue 前请先搜索是否已有相同问题。选择对应模板填写。

### 8.1 Bug Report

```markdown
**问题描述**
简述遇到的问题。

**复现步骤**
1. 进入 '...'
2. 点击 '...'
3. 执行 '...'
4. 观察到错误

**预期行为**
描述期望发生的行为。

**实际行为**
描述实际发生的行为。

**环境**
- 部署方式：[docker compose / 源码]
- 版本/Commit：[e.g. v1.2.0 / abc1234]
- LLM_PROVIDER：[offline / deepseek / openai]
- 浏览器（前端问题）：[e.g. Chrome 120]

**日志/截图**
粘贴相关日志（脱敏）或截图。

**trace_id（若有）**
问答问题可附 trace_id 便于定位。
```

### 8.2 Feature Request

```markdown
**是否与现有问题相关**
关联 Issue 编号（如有）。

**要解决的问题**
描述当前无法满足的场景或痛点。

**期望的方案**
描述你期望的实现方式。若有多方案，列出各自取舍。

**替代方案**
考虑过的其他实现方式及为何未采用。

**影响范围**
预计影响的模块（rag / graph / api / frontend / deploy）。

**是否愿意提交 PR**
[是 / 否]
```

## 9. 文档维护

文档与代码同等重要。以下改动需同步更新文档：

| 改动类型 | 需更新文档 |
|----------|-----------|
| 新增/变更接口 | `docs/api-reference.md` |
| 新增/变更配置项 | `docs/deployment.md` + `.env.example` |
| 架构/数据流调整 | `docs/architecture.md` |
| 检索/分块/向量化调整 | `docs/rag-pipeline.md` |
| 图谱 Schema/Cypher 调整 | `docs/graphrag.md` |
| 性能数据更新 | `docs/benchmark.md` |

文档使用中文撰写，代码/配置/命令保留英文。新增文档需在相关文档的"相关文档"章节互相引用，保持导航完整。

## 10. 行为准则

- 尊重所有贡献者，就技术问题进行客观讨论。
- Issue 与 PR 中不夹带与项目无关的内容。
- 安全漏洞不通过公开 Issue 报告，请私下联系 maintainer。
- 不得提交包含真实敏感数据（真实账号、密钥、内部文档）的测试用例。

## 11. 相关文档

- [系统架构设计](./docs/architecture.md)
- [RAG 检索流水线设计](./docs/rag-pipeline.md)
- [GraphRAG 知识图谱增强](./docs/graphrag.md)
- [部署运维指南](./docs/deployment.md)
- [API 接口文档](./docs/api-reference.md)
- [性能基准测试](./docs/benchmark.md)
