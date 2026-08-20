# DeepResearch-Agent

> 基于多 Agent 协作的自动化研究分析系统

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)](https://langchain-ai.github.io/langgraph)

---

## 项目简介

DeepResearch-Agent 是一个基于多 Agent 协作架构的自动化研究分析系统。用户输入研究课题，系统自动完成：任务拆解 → 资料搜索 → 网页深度阅读 → 知识整理 → 报告撰写 → 质量审查的完整闭环。

### 演示

<video width="800" controls>
  <source src="./video/deepsearch-agent.mp4" type="video/mp4">
  您的浏览器不支持 video 标签，请直接下载视频 <a href="demo.mp4">demo.mp4</a>
</video>

### 核心能力

| 能力 | 说明 |
|------|------|
| 🤖 多 Agent 协作 | Planner-Researcher-Writer-Reviewer 四 Agent 系统 |
| 🔧 插件式 Tool 系统 | 搜索、网页浏览、Python 执行等工具的灵活路由 |
| 🧠 RAG 知识库 | 文档导入、向量化检索、检索增强生成、研究中引用知识来源 |
| 💾 双记忆系统 | Session Memory（任务上下文）+ Knowledge Memory（跨任务知识） |
| 📊 结构化报告 | Markdown + PDF 双格式下载 |
| 🔍 自动质量审查 | Reviewer Agent 迭代评分优化，最多 3 轮 |
| ⚡ 实时流式输出 | SSE 实时展示 Agent 执行轨迹 |
| 🌓 深色/浅色主题 | 一键切换，自动记忆偏好 |
| 📚 知识库管理 | 文档导入、检索测试、引用来源列表 |
| 🛠 自定义 Skills | 用户创建/编辑/启停结构化技能，按触发词注入对应 Agent 的 system prompt |
| 👤 用户特化 Agent | 用户档案（写作风格/领域/模型/附加指令）+ 技能按用户隔离 |
| 🔄 自进化模块 | 任务复盘提炼经验，生成技能草稿，用户确认后生效 |
| 📁 会话工作目录 | 每任务隔离工作目录 + 参考文件上传，路径与文件清单注入 Agent 提示词 |

---

## 会话工作目录

每个研究任务自动创建一个隔离工作目录（`data/workspaces/<task_id>/`）：

- 上传的参考文件（PDF/Markdown/TXT/CSV/JSON/DOCX）复制到该目录；
- 工作目录路径与文件清单自动注入各 Agent 的提示词，Agent 会优先阅读参考文件；
- 研究过程中，Agent 可通过 read_workspace 工具读取工作目录中的参考文件内容；
- 最终报告（Markdown/PDF）保存到该目录；
- API：`POST /api/research/{task_id}/upload` 上传附件，`GET /api/research/{task_id}/workspace` 浏览文件。

---

## 架构图

```
User → FastAPI → LangGraph Workflow → Tool Router → 外部服务
                 ├── Planner Agent
                 ├── Researcher Agent
                 ├── Writer Agent
                 └── Reviewer Agent
```

**工作流程：**

1. **Planner Agent** — 接收用户课题，拆解为可执行的子任务和研究计划
2. **Researcher Agent** — 执行搜索、浏览网页、读取文档，收集原始资料
3. **Writer Agent** — 基于收集的资料撰写结构化研究报告
4. **Reviewer Agent** — 对报告进行质量评分，触发最多 3 轮迭代优化

---

## 快速开始

### 前置条件

- Python >= 3.11
- Node.js >= 18（前端构建）
- Redis（可选，开发模式使用内存存储）

### 安装

1. 克隆仓库
```bash
git clone <repo-url>
cd deep-research-agent
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 填入 API Key
```

4. 启动服务
```bash
uvicorn app.main:app --reload
```

5. （可选）构建前端
```bash
cd app/web
npm install
npm run build
```

6. 访问
- 生产模式（使用构建后的前端）: http://localhost:8000
- 开发模式（Vite 热更新）: `cd app/web && npm run dev` → http://localhost:5173
- API 文档: http://localhost:8000/docs

### Docker 部署

```bash
docker-compose up --build
```

---

## 技术栈

### 后端

- **语言**: Python 3.11+
- **Web 框架**: FastAPI + Uvicorn
- **Agent 编排**: LangGraph
- **LLM**: OpenAI / Anthropic API
- **向量数据库**: ChromaDB
- **缓存**: Redis
- **浏览器**: Playwright / httpx+BeautifulSoup

### 前端

- **框架**: React 18 + Vite
- **样式**: Tailwind CSS + 深色/浅色主题切换
- **实时通信**: SSE (Server-Sent Events)
- **字体**: Geist Sans + Geist Mono
- **动画**: Motion
- **图标**: Phosphor Icons
- **特色功能**:
  - 深色/浅色一键切换，自动记忆偏好
  - Agent 执行轨迹实时时间线展示
  - 研究报告实时流式渲染
  - 可拖动浮动知识库管理窗口
  - 历史记录管理（刷新/批量删除）
  - 知识库检索测试

---

## API 文档

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/research` | 发起研究任务（支持 RAG 知识库开关） |
| POST | `/api/research/{id}/upload` | 上传参考文件到任务工作目录 |
| GET | `/api/research/{id}/workspace` | 浏览任务工作目录文件清单 |
| GET | `/api/research/{id}/stream` | SSE 实时流，推送 agent_status / report_chunk / completed 等事件 |
| GET | `/api/research/{id}` | 查询任务状态 |
| GET | `/api/reports/{id}` | 获取报告文件（Markdown/PDF） |
| GET | `/api/history` | 历史任务列表（无分页限制，返回全部） |
| POST | `/api/reports/batch-delete` | 批量删除历史报告 |
| POST | `/api/knowledge/ingest` | 导入文档到知识库 |
| GET | `/api/knowledge/search` | 检索知识库 |
| GET | `/api/knowledge/list` | 知识库文档列表 |
| DELETE | `/api/knowledge/docs` | 删除知识库文档 |
| GET | `/health` | 健康检查 |
| GET | `/api/skills` | 技能列表 |
| POST | `/api/skills` | 新建技能 |
| PUT | `/api/skills/{id}` | 更新技能（含启停） |
| DELETE | `/api/skills/{id}` | 删除技能 |
| POST | `/api/skills/match` | 输入任务文本，返回命中技能 |
| GET | `/api/profile` | 获取当前用户档案 |
| PUT | `/api/profile` | 更新当前用户档案 |
| PUT | `/api/skills/{id}/pref` | 设置全局技能对当前档案启用/停用 |
| GET | `/api/evolution/drafts` | 当前档案的进化草稿列表 |
| POST | `/api/evolution/drafts/{id}/accept` | 接受草稿并生成技能（可发布全局/带编辑） |
| POST | `/api/evolution/drafts/{id}/reject` | 拒绝草稿 |

---

## 项目亮点

1. **多 Agent 协作架构**: 基于 LangGraph 设计 Planner-Researcher-Writer-Reviewer 四 Agent 系统，实现复杂研究任务的自动拆解、执行和质量闭环。Reviewer 反射机制通过迭代评分提升输出质量，最多 3 轮自动优化。

2. **插件式 Tool 系统**: 统一的 Tool Router 路由层，Agent 按需动态调用搜索、网页浏览、Python 执行等工具。新增工具只需继承 BaseTool 并注册，零侵入扩展。

3. **RAG + 双记忆系统**: Session Memory 保存任务上下文实现多轮交互，Knowledge Memory 持久化研究成果实现跨任务知识复用。RAG 检索增强减少 LLM 幻觉，研究中自动标注 [来源: 文档名称]，报告末尾汇总知识库引用来源列表。

4. **实时流式前端**: 基于 SSE 的 Agent 执行轨迹实时展示，研究报告流式渲染，深色/浅色主题一键切换，知识库窗口可拖动管理，历史记录支持批量操作。


## 近期新增功能

### 2026年8月 — 会话工作目录与参考文件上传

- **每任务隔离工作目录**: 每个研究任务自动创建 `data/workspaces/<task_id>/`，上传的参考文件（PDF/MD/TXT/CSV/JSON/DOCX）复制到该目录
- **环境注入**: 工作目录路径与文件清单自动注入 Planner/Researcher/Writer 的提示词；PythonTool 沙箱暴露 `WORKSPACE_DIR` 供分析代码引用
- **报告落工作目录**: 最终报告（Markdown/PDF）与参考文件同目录存放
- **上传安全**: 文件名/扩展名白名单/大小三重校验，保留报告命名空间（`metadata.json`/`task_name.txt`/`rp_*`），拒绝路径穿越与目录逃逸
- **API**: `POST /api/research/{id}/upload` 上传附件，`GET /api/research/{id}/workspace` 浏览工作目录
- **检索兼容**: 报告读取自动回退到工作目录，历史报告（`data/reports/`）不受影响
- **read_workspace 工具**: 研究 Agent 可通过 `read_workspace` 工具直接读取上传的参考文件内容
- **上传文件数上限**: `UPLOAD_MAX_FILES` 在服务层原子性强制，超限返回 413
- **ARQ worker 工作目录**: ARQ worker 执行路径同样使用每任务工作目录
- **生命周期清理**: 删除任务（含重启后不在内存的历史任务）时同步清理工作目录

### 2026年7月 — 核心 Agent 性能优化（基准测试提升 80%）

**优化内容：**

- **搜索后自动浏览**（最高影响）: Researcher Agent 搜索完成后自动并行浏览前 3 个结果 URL 获取深度内容，工具调用量从 3 次/任务提升到 17 次/任务
- **来源追踪管道**: 每项研究发现携带完整来源信息（URL、title），确保 Writer 可准确引用，引用质量评分从 2.6→4.6
- **规划深度提升**: Planner Agent 子任务从 3-4 个扩展到 5-8 个，覆盖更多研究维度，结构完整度从 4.9→7.5
- **质量评审循环修复**: 改进 reviewer 阈值逻辑，中文评审反馈，使迭代真正生效
- **工具可靠性增强**: BrowserTool 增加 3 次重试 + 指数退避 + User-Agent 轮换，`timeout` 从 15s→60s，工具成功率从 67%→**100%**
- **结构化报告生成**: Writer Agent 强制 2000+ 字、`[来源: 标题](URL)` 引用格式

**优化后基准测试结果（5 课题，LLM-as-Judge 评分）：**

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 事实准确率 | 3.8/10 | **7.3/10** | +92% |
| 结构完整度 | 4.9/10 | **7.5/10** | +53% |
| 引用质量 | 2.6/10 | **4.6/10** | +77% |
| 工具成功率 | 67% | **100%** | +49% |
| 数据点/任务 | 3.0 | **7.2** | +140% |

### 2026年7月 — 前端全面重构

- **深色科技风 UI**: 深色/浅色主题一键切换，偏好自动持久化
- **玻璃态设计**: `backdrop-filter` 玻璃面板 + 微光 border 效果
- **Geist 字体**: 替换默认字体，Geist Sans + Geist Mono 搭配
- **Agent 执行时间线**: Agent 状态 + 工具调用子事件分组 + 渐变进度条
- **研究报告渲染**: 代码块语言标签 + 引用侧栏 + MD/PDF 下载按钮统一样式
- **可拖动的知识库窗口**: 拖动标题栏任意移动，关闭按钮始终可见
- **历史记录管理**: 刷新 + 管理模式 + 勾选批量删除
- **RAG 知识库引用**: 报告末尾自动列出引用来源文档名称
- **修复**: fetchHistory 变量提升避免 TDZ 崩溃、RAG 检索缺少 action 参数、PDF 下载从 MD 回退改为报错提示、有序列表空行不中断序号、PDF 生成 bulletType 参数错误

---

## 项目结构

```
deep-research-agent/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI 应用入口
│   ├── config.py                # 全局配置（pydantic-settings）
│   ├── middleware.py            # 中间件（日志、CORS、监控）
│   ├── agents/
│   │   ├── base.py              # Agent 基类
│   │   ├── planner.py           # Planner Agent：任务拆解与研究规划
│   │   ├── researcher.py        # Researcher Agent：资料搜索与收集
│   │   ├── writer.py            # Writer Agent：报告撰写
│   │   └── reviewer.py          # Reviewer Agent：质量审查与评分
│   ├── memory/
│   │   ├── session_memory.py    # Session Memory：单次任务上下文
│   │   └── knowledge_memory.py  # Knowledge Memory：跨任务知识复用
│   ├── models/
│   │   ├── state.py             # Agent 状态定义
│   │   ├── schemas.py           # Pydantic 请求/响应模型
│   │   ├── report.py            # 报告数据模型
│   │   └── tools.py             # Tool 调用数据模型
│   ├── rag/
│   │   ├── document_loader.py   # 文档加载器（PDF/MD/TXT/HTML）
│   │   ├── chunker.py           # 文档分块策略
│   │   ├── embedder.py          # 向量化嵌入
│   │   ├── vector_store.py      # ChromaDB 向量存储
│   │   └── retriever.py         # RAG 检索器
│   ├── services/
│   │   ├── llm_service.py       # LLM 调用封装（OpenAI/Anthropic）
│   │   ├── task_manager.py      # 研究任务生命周期管理（创建/执行/事件/持久化）
│   │   ├── report_service.py    # 报告生成（Markdown + PDF）
│   │   ├── workspace.py         # WorkspaceManager：每任务隔离工作目录（创建/清理/上传/清单）
│   │   ├── profile_service.py   # 用户档案与模型选择
│   │   ├── skill_service.py     # 技能匹配与注入
│   │   ├── evolution_service.py # 任务复盘 → 技能草稿
│   │   └── config_service.py    # 运行时 LLM 配置管理
│   ├── tools/
│   │   ├── base.py              # BaseTool 基类
│   │   ├── search.py            # 搜索工具（Tavily/DuckDuckGo/GitHub）
│   │   ├── browser.py           # 网页浏览工具
│   │   ├── python_executor.py   # Python 代码执行工具（受限沙箱）
│   │   ├── memory.py            # 记忆读写工具
│   │   ├── rag_retriever.py     # RAG 检索工具
│   │   ├── workspace_reader.py  # 工作目录文件读取工具（read_workspace）
│   │   └── router.py            # Tool Router 路由层
│   ├── utils/
│   │   ├── llm.py               # LLM 工具函数
│   │   ├── logger.py            # 结构化日志
│   │   ├── markdown_utils.py    # Markdown 处理工具
│   │   ├── pdf_utils.py         # PDF 生成工具（ReportLab）
│   │   ├── date_hint.py         # 当前日期提示（注入 Agent prompt）
│   │   ├── workspace_context.py # 工作目录环境指令构建（注入 Agent prompt）
│   │   ├── citation_validator.py # 引用来源校验
│   │   └── grounding.py         # 论断-证据基础校验
│   ├── workflow/
│   │   ├── graph.py             # LangGraph 工作流图定义
│   │   ├── nodes.py             # 工作流节点函数（含 RAG 引用来源收集）
│   │   ├── conditions.py        # 条件路由函数
│   │   └── events.py            # 事件发布机制
│   └── web/                     # React 前端
│       ├── index.html
│       ├── package.json
│       ├── vite.config.js
│       ├── tailwind.config.js
│       ├── postcss.config.js
│       └── src/
│           ├── main.jsx
│           ├── App.jsx          # 主应用（深色/浅色主题、历史管理）
│           ├── index.css         # 设计令牌、玻璃态样式、动效
│           ├── hooks/useSSE.js   # SSE 连接 Hook
│           ├── contexts/
│           └── components/
│               ├── InputPanel.jsx    # 研究输入面板（RAG 开关、知识库弹窗）
│               ├── AgentTrace.jsx    # Agent 执行轨迹时间线
│               └── ReportViewer.jsx  # 报告展示（引用侧栏、MD/PDF 下载）
├── data/
│   ├── chroma_db/               # ChromaDB 持久化数据
│   ├── knowledge/               # Knowledge Memory 存储
│   ├── reports/                 # 生成的报告文件
│   ├── uploads/                 # 用户上传文件
│   └── workspaces/              # 每任务隔离工作目录（含上传附件与报告）
├── tests/
│   ├── conftest.py              # 测试配置与 Fixtures
│   ├── test_agents.py           # Agent 单元测试
│   ├── test_memory.py           # 记忆系统测试
│   ├── test_rag.py              # RAG 系统测试
│   ├── test_tools.py            # Tool 系统测试
│   ├── test_workflow.py         # 工作流测试
│   ├── test_workspace.py        # 工作目录管理测试
│   ├── test_workspace_context.py # 工作目录提示词注入测试
│   ├── test_workspace_reader.py # 工作目录文件读取工具测试
│   ├── test_report_service.py   # 报告服务测试
│   ├── test_integration_api.py  # API 集成测试
│   ├── test_skills.py           # 技能系统测试
│   ├── test_evolution.py        # 进化模块测试
│   ├── test_profile.py          # 用户档案测试
│   ├── test_auth.py             # 认证测试
│   ├── test_settings.py         # 配置测试
│   ├── test_database.py         # 数据库测试
│   ├── test_citation_validator.py # 引用校验测试
│   ├── test_grounding.py        # 论断-证据校验测试
│   ├── test_search_backends.py  # 搜索后端测试
│   ├── test_middleware.py       # 中间件测试
│   ├── test_logger.py           # 日志测试
│   ├── test_markdown_utils.py   # Markdown 工具测试
│   └── test_pdf_utils.py        # PDF 工具测试
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

---

## 测试

项目包含 **437 个测试用例**，覆盖核心功能模块：

```bash
# 运行全部测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_agents.py -v
pytest tests/test_workflow.py -v
pytest tests/test_rag.py -v

# 运行测试并生成覆盖率报告
pytest tests/ --cov=app --cov-report=term-missing
```

---

## 许可证

MIT
