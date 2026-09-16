# LangChain + Milvus 混合检索 RAG

这是 `rag-agent-from-scratch` 的框架版 RAG，用于将手写版中已经验证过的检索流程迁移到 LangChain 与 Milvus：

```text
文档加载与切分
        ↓
Dense Retrieval ─┐
                 ├→ Milvus RRF → Qwen3 Reranker → Prompt → Qwen3 Generator
BM25 Retrieval ──┘
```

项目使用 Qwen3-Embedding 生成 Dense 向量，使用 Milvus 内置 BM25 Function 生成 Sparse 表示并建立倒排索引，再通过 RRF 融合两路排名。融合结果由 Qwen3-Reranker 重排，最终交给本地 Qwen3-1.7B 生成回答。

## 署名与贡献说明

- 本项目代码由项目作者在 OpenAI Codex 协助下完成。
- 项目文档由 OpenAI Codex 根据现有代码与实际流程起草。
- 文档内容由项目作者逐项审核、修改并最终确认。

## 与手写版的关系

`handwritten-rag` 用于理解算法和数据流：

```text
NumPy Dense Retrieval
+ 手写 BM25
+ 手写 RRF
+ CrossEncoder Reranker
```

当前 `langchain-rag` 使用框架与数据库完成相同主流程：

```text
LangChain组件编排
+ Milvus Dense/BM25混合检索
+ Milvus RRF
+ LangChain CrossEncoder重排
```

手写版展示算法理解，框架版展示 Milvus 数据建模、标准 Retriever 接口和端到端组件编排。

## 核心流程

### 1. 文档加载与切分

`ingest.py` 支持：

- 直接读取 TXT 和 Markdown 文件。
- 使用 `pypdf` 按 PDF 页面提取文本。
- 将原始内容保存为 LangChain `Document`。
- 使用 `RecursiveCharacterTextSplitter` 切分 Document。
- 通过 overlap 保留相邻 chunk 的部分上下文。
- 在 metadata 中保留 `source`、`page` 和 `start_index` 等信息。

当前示例文档：

```text
documents/THREE_WEEK_PLAN.md
```

默认切分参数：

```text
chunk_size = 500
chunk_overlap = 50
add_start_index = True
```

### 2. Milvus Collection

项目连接本地 Milvus Standalone：

```text
URI: http://localhost:19530
Database: rag_demo
Collection: rag_chunks
```

Collection 的核心字段：

| 字段 | 类型 | 作用 |
|---|---|---|
| `pk` | `INT64` | Milvus 自动生成的主键 |
| `text` | `VARCHAR` | chunk 原文，也是 BM25 Function 的输入 |
| `dense` | `FLOAT_VECTOR` | Qwen3-Embedding 生成的 Dense 向量 |
| `sparse` | `SPARSE_FLOAT_VECTOR` | Milvus BM25 Function 生成的 Sparse 表示 |

Collection 开启动态字段，用于保存 Document metadata。

### 3. Dense 与 BM25 入库

Dense 路线：

```text
Document.page_content
→ Qwen3-Embedding-0.6B
→ dense
```

BM25 路线：

```text
Document.page_content
→ Milvus中文Analyzer
→ Milvus BM25 Function
→ sparse
```

应用侧提交 Document 原文和 metadata，并在客户端生成 Dense 向量。Sparse 表示、语料统计和 BM25 检索由 Milvus服务端处理。

当前索引：

| 字段 | 索引 | Metric |
|---|---|---|
| `dense` | HNSW | COSINE |
| `sparse` | SPARSE_INVERTED_INDEX | BM25 |

### 4. 混合检索与 RRF

`retriever.py` 将 Milvus VectorStore 包装成标准 LangChain Retriever：

```text
query
├→ Qwen3 Embedding → Dense检索
└→ Milvus Analyzer → BM25检索
                         ↓
                        RRF
                         ↓
                    Top 10 Documents
```

默认参数：

```text
RRF结果数：10
RRF平滑常数：60
Dense search ef：64
```

项目使用 `Milvus(...).as_retriever()`，没有使用即将被弃用的 `MilvusCollectionHybridSearchRetriever`。

### 5. CrossEncoder 重排

`reranker.py` 使用 LangChain 当前的 CrossEncoder 重排组合：

```text
HuggingFaceCrossEncoder
→ CrossEncoderReranker
→ ContextualCompressionRetriever
```

各层职责：

```text
HuggingFaceCrossEncoder
→ 加载Qwen3-Reranker并计算(query, document)相关性分数

CrossEncoderReranker
→ 组织文本对、按分数排序并截取Top N

ContextualCompressionRetriever
→ 将基础Retriever和Reranker包装成新的Retriever
```

当前候选数量：

```text
RRF Top 10
→ Qwen3-Reranker Top 5
```

### 6. 本地生成

`generator.py` 的模型包装关系：

```text
ChatHuggingFace
└── HuggingFacePipeline（LangChain LLM适配器）
    └── transformers.pipeline
        ├── Qwen3-1.7B
        └── Qwen3 Tokenizer
```

Prompt 要求模型：

- 只能依据参考资料回答。
- 不编造资料中不存在的信息。
- 资料不足时返回“根据现有资料无法确定”。

Qwen3 默认开启 Thinking。当前 Human Prompt 追加 `/no_think`，用于关闭实际思考内容；在当前 Chat Template 包装方式下，输出仍可能保留一个空的 `<think></think>` 标签。

### 7. RAG Chain

`rag.py` 使用 LCEL 组合两条输入路线：

```python
{
    "context": rerank_retriever | format_documents,
    "question": RunnablePassthrough(),
}
```

含义：

```text
同一个question
├→ 检索、重排、格式化 → context
└→ 原样传递           → question
```

完整链路：

```text
question
→ Dense + BM25
→ RRF Top 10
→ Qwen3-Reranker Top 5
→ 格式化context
→ ChatPromptTemplate
→ Qwen3-1.7B
→ StrOutputParser
→ answer
```

## 目录结构

```text
langchain-rag/
├── demo.py                         # 一键端到端入口
├── ingest.py                       # 文档加载、切分、Collection与入库
├── retriever.py                    # Dense + BM25 + RRF Retriever
├── reranker.py                     # CrossEncoder重排Retriever
├── generator.py                    # 本地ChatModel初始化
├── rag.py                          # Prompt与LCEL RAG Chain
├── documents/
│   └── THREE_WEEK_PLAN.md          # 示例知识库
├── models/                         # 本地模型，不提交Git
└── README.md
```

## 环境要求

- Python 3.11 或兼容版本。
- 正在运行的 Milvus Standalone。
- 本地 Qwen3 Embedding、Reranker 和生成模型。
- Apple Silicon 可使用 MPS；其他环境需要调整代码中的 `device`。

主要 Python 依赖：

```text
torch
transformers
sentence-transformers
pypdf
pymilvus
langchain-core
langchain-text-splitters
langchain-huggingface
langchain-milvus
langchain-community
langchain-classic
```

安装示例：

```bash
pip install -U \
  torch \
  transformers \
  sentence-transformers \
  pypdf \
  pymilvus \
  langchain-core \
  langchain-text-splitters \
  langchain-huggingface \
  langchain-milvus \
  langchain-community \
  langchain-classic
```

`langchain-community` 当前仅用于官方本地 `HuggingFaceCrossEncoder` 适配器；Embedding 使用已拆分的 `langchain-huggingface`，Milvus 使用 `langchain-milvus`。

## 本地模型

在 `langchain-rag/models/` 中准备：

```text
models/
├── Qwen3-Embedding-0.6B
├── Qwen3-Reranker-0.6B
└── Qwen3-1.7B
```

下载示例：

```bash
cd rag-agent-from-scratch/langchain-rag

hf download Qwen/Qwen3-Embedding-0.6B \
  --local-dir models/Qwen3-Embedding-0.6B

hf download Qwen/Qwen3-Reranker-0.6B \
  --local-dir models/Qwen3-Reranker-0.6B

hf download Qwen/Qwen3-1.7B \
  --local-dir models/Qwen3-1.7B
```

模型目录已被仓库 `.gitignore` 忽略。

## 启动 Milvus

本项目要求 Milvus Standalone 已经监听：

```text
http://localhost:19530
```

代码使用：

```text
user = root
password = Milvus
```

如果本地 Milvus 启用了认证，需要保证账号密码一致；如果修改了 URI、Database、Collection 或认证信息，需要同步修改 `ingest.py` 和 `demo.py`。

Milvus 的部署文件目前不包含在 `langchain-rag` 目录中，需要提前使用自己的 Docker Compose 或其他方式启动 Milvus Standalone。

## 运行

以下命令从 `langchain-rag` 目录执行：

```bash
cd rag-agent-from-scratch/langchain-rag
```

### 一键端到端 Demo

```bash
python demo.py
```

首次运行时：

```text
检查Database
→ 创建rag_demo（不存在时）
→ 检查rag_chunks
→ 加载并切分documents
→ 创建Collection与索引
→ 写入Dense和BM25数据
→ 启动交互问答
```

后续运行时：

```text
检测到rag_chunks已存在
→ 跳过文档处理与入库
→ 连接已有Collection
→ 启动交互问答
```

输入以下内容退出：

```text
exit
```

### 单独重建知识库

```bash
python ingest.py
```

注意：当前 `ingest.py` 会删除已存在的 `rag_chunks` Collection，然后重新创建并写入全部文档。这是开发阶段的全量重建，不是增量更新。

### 单独验证各阶段

```bash
python retriever.py
python reranker.py
python generator.py
python rag.py
```

其中：

- `retriever.py` 验证 Dense + BM25 + RRF。
- `reranker.py` 验证混合召回后的 CrossEncoder 重排。
- `generator.py` 验证本地 ChatModel。
- `rag.py` 验证完整 RAG Chain。

## 当前限制

- `demo.py` 只根据 Collection 是否存在决定是否入库，不会自动检测本地文档是否改变。
- `ingest.py` 采用全量删除并重建 Collection，没有实现增量索引和文档版本管理。
- Milvus 连接参数、模型路径和设备当前直接写在代码中，没有抽离为配置文件或环境变量。
- 默认设备为 `mps`，非 Apple Silicon 环境需要调整。
- PDF 仅提取文本，不包含 OCR、表格恢复和复杂版面解析。
- 中文 BM25 使用 Milvus Chinese Analyzer，没有增加领域词典、同义词或停用词配置。
- 当前示例知识库规模较小，没有补充框架版的独立检索评测集。
- 当前为本地单进程演示，没有实现并发、流式输出、API 服务和权限过滤。

## 后续方向

```text
完善LangChain RAG文档
→ 手写最小Agent
→ LangGraph Agent
→ FastAPI与演示页面
```

该版本重点是跑通并理解：

```text
LangChain标准组件
+ Milvus原生混合检索
+ 本地Qwen3模型
```

而不是隐藏底层检索原理或追求生产级部署。
