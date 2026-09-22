# RAG & Agent From Scratch

这是一个覆盖检索、重排、生成与智能体编排核心链路的 RAG 与 Agent 项目。目前包含两套可对照验证的 RAG 实现、一套原生 Python 手写的 Graph Agent Runtime，以及对应的 LangGraph 重构版本：

- `handwritten-rag`：不使用 RAG 编排框架，手写 BM25、RRF 和评测流程。
- `langchain-rag`：使用 LangChain 与 Milvus Standalone 重构同一条 RAG 链路。
- `handwritten-agent`：不使用 LangGraph，手写 State、Reducer、Node、Edge、Router、Compiled Graph 和 Tool-Calling 循环。
- `langgraph-agent`：使用 LangGraph 重构手写 Agent 的同一套核心业务流程。
- `appliance-support`：基于手写 RAG 与 Agent Runtime 构建的家电故障诊断应用，串联 Qwen3-8B LoRA、混合检索、工具调用和可视化执行追踪。

手写 Agent 与 LangGraph 重构版均已完成 Checkpoint、HITL、Memory、Streaming、Subgraph、Multi-Agent、RAG Tool 和 Skills，并针对同一业务流程提供可对照实现。

## 在线体验

家电故障诊断应用支持通过公网地址体验。由于完整检索链路需要在本地加载 Qwen3 Embedding 与 Reranker 模型，演示服务不会长期在线；如果想体验，请联系项目作者，由作者启动 RAG、Agent 和公网转发服务后提供访问地址。体验者不需要下载模型，也不需要提供 DeepSeek API Key。

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
│   ├── rag_service.py
│   ├── appliance_rag_service.py
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
│   ├── appliance_support.py
│   ├── appliance_support_ui.py
│   └── agent.py
├── langgraph-agent/
│   ├── state.py
│   ├── nodes.py
│   ├── routers.py
│   ├── model.py
│   ├── tools.py
│   └── agent.py
├── evaluation/
│   ├── data/appliance_retrieval_eval.jsonl
│   ├── data/appliance_agent_eval.jsonl
│   ├── evaluate_retrieval.py
│   ├── evaluate_agent.py
│   └── README.md
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

## 家电故障诊断 Agent

该应用面向“家电出现故障代码或异常现象后如何处理”的真实场景，将微调、RAG 和 Agent 串成一条可运行链路：

```text
Streamlit双栏界面
→ 手写Graph Agent Runtime
→ DeepSeek Tool Calling与多轮信息补全
→ query_rag HTTP Tool
→ Qwen3-Embedding + 手写BM25
→ RRF融合
→ Qwen3-Reranker精排
→ DeepSeek生成故障解释与安全建议
```

当前家电知识库包含 438 条结构化故障资料，覆盖 13 个品牌和洗衣机、洗碗机、烘干机、冰箱、烤箱/炉灶五类家电。每条资料作为独立 Chunk，保留品牌、市场版本、家电类型、故障代码、故障含义、处理建议与来源链接。

应用支持：

- 信息不足时由模型继续追问，信息充分后生成独立的 `query_rag` 查询。
- Dense、BM25 双路召回，RRF 融合后使用 CrossEncoder 精排并返回 Top-5。
- 通过 `thread_id` 和 Checkpoint 保留多轮会话状态；同一会话重复提问时，模型可以复用已有上下文而不重复检索。
- 右侧执行面板展示模型决策、参数检查、工具审核、RAG 返回文档、来源、分数、排名和节点耗时。
- Qwen3-8B 基座通过 bitsandbytes NF4 4-bit 加载，并叠加 PEFT LoRA Adapter；同一套 Transformers/PEFT 代码可运行在 Apple Silicon MPS 与 NVIDIA CUDA 环境。

### Agent评测

当前使用10条单轮样本进行基础工具路由评测。每条问题均已提供完整的品牌、家电类型和故障代码，评测Agent是否调用`query_rag`；10条样本全部调用成功，`query_rag`调用率为 **1.0000**。

当前评测入口使用 DeepSeek API，仅覆盖“信息完整时是否调用 RAG”这一基础场景，暂不代表信息缺失追问、多轮上下文融合、工具参数质量和最终回答质量。详细说明见 [evaluation/README.md](evaluation/README.md)。

### 运行家电应用

目录约定：`appliance-support-sft` 与 `rag-agent-from-scratch` 位于同一父目录。家电语料由前者提供；Embedding 与 Reranker 模型放在 `handwritten-rag/models/`，模型权重不提交到本仓库。

先启动检索服务：

```bash
conda activate ENV_rag
cd rag-agent-from-scratch/handwritten-rag
python appliance_rag_service.py
```

再启动双栏界面：

```bash
conda activate ENV_agent
cd rag-agent-from-scratch/handwritten-agent
export DEEPSEEK_API_KEY="你的 API Key"
streamlit run appliance_support_ui.py
```

也可以使用终端交互版：

```bash
python appliance_support_deepseek.py
```

## Handwritten RAG

手写版直接实现核心算法和数据流：

- NumPy 保存并检索 Dense Embedding。
- Jieba 分词。
- 手写 BM25。
- 手写 RRF。
- Qwen3-Reranker-0.6B 重排。
- Qwen3-1.7B 生成答案。
- 使用 Recall@K、MRR@K 和 nDCG@K 分别评测各检索阶段。

当前评测集包含 159 个问题，分别覆盖故障码、故障现象以及故障码与现象组合查询。各检索阶段总体结果如下：

| 检索阶段 | Recall@1 | Recall@3 | Recall@5 | MRR@1 | MRR@3 | MRR@5 | nDCG@1 | nDCG@3 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 0.9418 | 0.9921 | 1.0000 | 0.9686 | 0.9811 | 0.9827 | 0.9686 | 0.9844 | 0.9871 |
| BM25 | 0.8684 | 0.9502 | 0.9895 | 0.8868 | 0.9161 | 0.9249 | 0.8868 | 0.9247 | 0.9415 |
| RRF | 0.9481 | 0.9900 | 1.0000 | 0.9748 | 0.9843 | 0.9858 | 0.9748 | 0.9849 | 0.9889 |
| RRF + Reranker | 0.8831 | 0.9879 | 1.0000 | 0.9057 | 0.9486 | 0.9502 | 0.9057 | 0.9583 | 0.9635 |

RRF取得最佳总体结果。当前Reranker提升了症状查询指标，但降低了故障码和混合查询指标，未带来总体增益。评测集由测试文档按规则模板生成，用于检索阶段对比和回归验证，不代表完整真实用户分布。详细结果见 [evaluation/README.md](evaluation/README.md)。

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

RAG 使用：

```text
Qwen3-Embedding-0.6B
Qwen3-Reranker-0.6B
Qwen3-1.7B
```

家电故障诊断 Agent 额外使用：

```text
Qwen3-8B
Qwen3-8B Appliance Support LoRA Adapter
```

模型权重不提交到 Git 仓库，需要分别放入各版本的 `models/` 目录。

## 当前边界

当前版本聚焦本地算法实现与 Agent Runtime 核心语义，尚未包含：

- 生产级并发、API 流式传输与限流。
- 增量索引。
- 权限控制。
- 生产级配置和监控。
- LangChain 版正式检索评测。
- 家电场景的大规模多轮与多品牌困难样本评测；当前 LoRA 数据以单轮、单条正确资料为主，复杂上下文中仍可能出现证据混淆。

## 后续计划

```text
家电检索评测与多轮困难样本
→ 增量索引与权限过滤
→ 自动化回归测试、正式评测集与生产监控
```
