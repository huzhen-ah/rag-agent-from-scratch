# RAG & Agent From Scratch

这是一个覆盖检索、重排、生成与智能体编排核心链路的 RAG 与 Agent 项目。目前包含两套可对照验证的 RAG 实现、一套原生 Python 手写的 Graph Agent Runtime，以及对应的 LangGraph 重构版本：

- `handwritten-rag`：不使用 RAG 编排框架，手写 BM25、RRF 和评测流程。
- `langchain-rag`：使用 LangChain 与 Milvus Standalone 重构同一条 RAG 链路。
- `handwritten-agent`：不使用 LangGraph，手写 State、Reducer、Node、Edge、Router、Compiled Graph 和 Tool-Calling 循环。
- `langgraph-agent`：使用 LangGraph 重构手写 Agent 的同一套核心业务流程。

手写 Agent 与 LangGraph 重构版均已完成 Checkpoint、HITL、Memory、Streaming、Subgraph、Multi-Agent、RAG Tool 和 Skills，并针对同一业务流程提供可对照实现。

## 署名

- 项目文档由 OpenAI Codex 根据现有代码和实验结果起草。
- 文档内容由项目作者审核、修改并最终确认。

## 项目结构

```text
rag-agent-from-scratch/
├── handwritten-rag/
│   ├── documents/
│   ├── dataset/
│   ├── docs/
│   ├── bm25_retrieval.py
│   ├── dense_retrieval.py
│   ├── reranker.py
│   ├── rag.py
│   └── demo.py
├── langchain-rag/
│   ├── documents/
│   ├── docs/
│   ├── ingest.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── generator.py
│   ├── rag.py
│   └── demo.py
├── handwritten-agent/
│   ├── state.py
│   ├── runtime.py
│   ├── graph.py
│   ├── checkpoint.py
│   ├── nodes.py
│   ├── model.py
│   ├── parser.py
│   ├── tools.py
│   └── agent.py
├── langgraph-agent/
│   ├── state.py
│   ├── nodes.py
│   ├── routers.py
│   ├── model.py
│   ├── tools.py
│   └── agent.py
└── README.md
```

## 整体流程

两套实现遵循同一个核心流程：

```text
文档加载与切分
→ Dense检索 + BM25检索
→ RRF融合
→ CrossEncoder重排
→ 本地LLM生成答案
```

## Handwritten RAG

手写版直接实现核心算法和数据流：

- NumPy 保存并检索 Dense Embedding。
- Jieba 分词。
- 手写 BM25。
- 手写 RRF。
- Qwen3-Reranker-0.6B 重排。
- Qwen3-1.7B 生成答案。
- 使用 MRR 和 Recall@K 分别评测各检索阶段。

当前小型评测集的最终结果：

| MRR | Recall@1 | Recall@3 | Recall@5 |
|---:|---:|---:|---:|
| 1.00 | 0.90 | 1.00 | 1.00 |

评测集只有 10 个问题，且问题与原文措辞接近。该结果只用于验证流程和建立回归基线，不代表真实业务效果。

运行：

```bash
cd handwritten-rag
python demo.py
```

详细说明见 [handwritten-rag/README.md](handwritten-rag/README.md)。

## LangChain + Milvus RAG

LangChain 版用于验证框架组件、向量数据库与手写链路之间的工程映射：

- 自行解析 TXT、Markdown 和 PDF。
- 使用 LangChain `Document` 和文本切分器。
- 使用 Milvus 保存原文、Dense 与 Sparse 数据。
- 使用 Qwen3-Embedding-0.6B 生成 Dense 向量。
- 使用 Milvus 原生 BM25 Function 和 Sparse 索引。
- 使用 RRF 执行混合检索。
- 使用 LangChain CrossEncoder 组件重排。
- 使用 LCEL 编排检索、Prompt、模型和输出解析。

运行前需要启动 Milvus Standalone，然后执行：

```bash
cd langchain-rag
python demo.py
```

详细说明见 [langchain-rag/README.md](langchain-rag/README.md)。

## Handwritten Graph Agent

手写 Agent 用于理解主流 Graph Agent Runtime 的核心运行语义：

```text
User Input
→ State Update
→ ModelNode
→ Tool Calling
→ ToolNode
→ Reducer统一提交
→ Conditional Routing
→ 下一Super-step或END
```

当前已经实现：

- Provider-agnostic 的 Message 与 ToolCall 内部协议。
- Qwen3 Tool Calling 适配、解析和 ToolCall ID 标准化。
- Tool Registry、JSON Schema 参数校验和分层异常处理。
- Typed State、字段 Reducer 和 Partial State Update。
- `StateGraph` Builder、`CompiledStateGraph`、固定边和条件边。
- 多个 Node 读取同一份旧 State，并在 Super-step 末统一提交 Updates。
- `model_call_count` 加法 Reducer，支持多个模型 Node 累计调用次数。
- `StateSnapshot`、Checkpointer 接口、内存存储和 JSONL 本地持久化。
- 通过 `thread_id` 隔离执行历史，通过 UUID 和父 Checkpoint ID 维护版本血缘。
- Super-step 后保存 State 与下一批 Nodes，支持新输入续聊和无新输入恢复。
- JSONL 进程重启恢复，已完成 Node 不重复执行。

详细说明见 [handwritten-agent/README.md](handwritten-agent/README.md)。

## LangGraph Agent

LangGraph 版目前已经完成：

- `MessagesState` 与自定义 `model_call_count` Reducer。
- `StateGraph`、官方 `ToolNode` 和条件路由。
- SQLite Checkpointer 与基于 `thread_id` 的状态恢复。
- 基于 `interrupt()` 和 `Command(resume=...)` 的工具 HITL。
- 按 Policy 执行参数补充，以及工具调用的 approve、edit、reject。
- 使用 `Send` 将每个待执行 ToolCall 调度为独立任务。
- 将参数补充、人工审核和工具执行封装为 Tool Workflow SubGraph。
- 使用 SQLite Store 保存跨线程长期记忆，并通过 `stream_mode="updates"` 输出节点事件。
- 使用 Supervisor + `task` Tool 编排 Resume Agent 与 RAG Agent。
- RAG Agent 通过 HTTP 调用手写 RAG 服务，支持完整检索工具链。
- 支持 Skill Metadata 与 `read_skill` 按需加载 `SKILL.md`。

## 两个版本的对应关系

| 环节 | 手写版 | LangChain + Milvus 版 |
|---|---|---|
| 文档对象 | 自定义数据结构 | LangChain `Document` |
| Dense 存储 | NumPy / Pickle | Milvus |
| Dense 检索 | 矩阵乘法 | Milvus HNSW |
| BM25 | 手写实现 | Milvus 原生 BM25 |
| RRF | 手写实现 | Milvus 混合检索 |
| Reranker | 直接调用模型 | LangChain 组件包装 |
| 编排 | 普通 Python | LCEL |

## 本地模型

两个版本均使用：

```text
Qwen3-Embedding-0.6B
Qwen3-Reranker-0.6B
Qwen3-1.7B
```

模型权重不提交到 Git 仓库，需要分别放入各版本的 `models/` 目录。

## 当前边界

当前版本聚焦本地算法实现与 Agent Runtime 核心语义，尚未包含：

- API 服务和前端。
- 生产级并发、API 流式传输与限流。
- 增量索引。
- 权限控制。
- 生产级配置和监控。
- LangChain 版正式检索评测。

## 后续计划

```text
FastAPI与演示页面
→ 增量索引与权限过滤
→ 自动化回归测试、正式评测集与生产监控
```
