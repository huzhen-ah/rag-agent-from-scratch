# 手撕混合检索 RAG

这是一个不依赖 RAG 编排框架的本地问答项目，用于理解并跑通完整检索增强生成流程：

```text
文档加载与切块
        ↓
Dense Retrieval ─┐
                 ├→ RRF 融合 → Qwen3 Reranker → Qwen3 Generator → Answer
BM25 Retrieval ──┘
```

项目使用 NumPy 保存和计算 dense embedding，手写 BM25 与 RRF，并使用 Qwen3-Reranker 完成候选重排。除本地完整问答外，当前版本还提供独立的 HTTP 检索服务，供手写 Agent 通过 Tool 调用。

## 模型

运行前需要在项目的 `models/` 目录准备以下本地模型：

```text
models/
├── Qwen3-Embedding-0.6B
├── Qwen3-Reranker-0.6B
└── Qwen3-1.7B
```

各模型用途：

| 模型 | 用途 |
|---|---|
| Qwen3-Embedding-0.6B | 为 chunks 和 query 生成归一化 dense embedding |
| Qwen3-Reranker-0.6B | 对 RRF 候选进行相关性重排 |
| Qwen3-1.7B | 根据最终检索上下文生成回答 |

## 核心流程

### 1. 文档加载与切块

`load.py` 负责：

- 加载 TXT、Markdown 和 PDF。
- 按固定字符长度切块。
- 使用 overlap 保留相邻 chunk 的上下文。
- 为 chunk 分配整数 `chunk_id`。

当前默认数据源位于：

```text
documents/THREE_WEEK_PLAN.md
```

### 2. Dense Retrieval

`embedding.py` 使用 Qwen3-Embedding-0.6B 生成归一化向量。

`dense_retrieval.py` 使用矩阵乘法计算 query 与全部 chunk 的相似度：

```python
scores = chunk_embeddings @ query_embedding
```

因为两侧向量都已归一化，所以点积等价于余弦相似度。

### 3. BM25 Retrieval

`bm25_retrieval.py` 使用 Jieba 分词，并手写实现：

- chunk 词频统计。
- 文档频率与 BM25 IDF。
- TF 饱和。
- chunk 长度归一化。
- 倒排候选召回与 BM25 排序。

BM25 配置保存在：

```text
bm25/bm25_config.pkl
```

### 4. RRF

`rag.py` 对 Dense 与 BM25 返回的候选取并集，然后按照两路排名计算：

```text
RRF(chunk) = Σ 1 / (c + rank)
```

默认使用：

```text
c = 60
```

RRF 不直接比较 Dense score 与 BM25 score，因为两种分数不在同一个量纲中。

### 5. Reranker

`reranker.py` 使用 Sentence Transformers 的 `CrossEncoder` 加载 Qwen3-Reranker-0.6B。

模型对每个 `(query, chunk)` 文本对计算相关性分数，并对 RRF 候选重新排序。

### 6. Generation

`generator.py` 使用 Qwen3-1.7B，根据重排后的 Top-K chunks 生成回答。Prompt 要求模型严格依据参考资料回答；资料不足时返回“根据现有资料无法确定”。

## 目录结构

```text
handwritten-rag/
├── demo.py                          # 端到端交互入口
├── load.py                          # 文档加载与切块
├── build_index.py                   # 构建 dense 索引
├── embedding.py                     # Embedding 模型封装
├── dense_retrieval.py               # Dense 检索
├── bm25_retrieval.py                # 手写 BM25
├── reranker.py                      # Reranker
├── generator.py                     # 生成模型封装
├── rag.py                           # RRF 与完整 RAG 编排
├── rag_service.py                   # FastAPI 检索服务，返回重排后的文档
├── documents/
│   └── THREE_WEEK_PLAN.md           # 当前示例知识库文档
├── evaluate_retrieval_dense.py      # Dense 评估
├── evaluate_retrieval_bm25.py       # BM25 评估
├── evaluate_retrieval_rrf.py        # RRF 评估
├── evaluate_retrieval_reranker.py   # 最终检索链路评估
├── dataset/
│   ├── data.pkl                     # chunks 与 dense embeddings
│   └── retrieval_eval.jsonl         # 检索评测集
└── bm25/
    └── bm25_config.pkl              # BM25 索引配置
```

## 环境依赖

项目使用 Python 3.11，主要依赖：

```text
torch
transformers
sentence-transformers
numpy
jieba
PyMuPDF
fastapi
uvicorn
```

本项目使用已有 Conda 环境：

```bash
conda activate ENV_rag
```

## 运行

以下命令均从 `handwritten-rag` 目录执行：

```bash
cd rag-agent-from-scratch/handwritten-rag
```

### 端到端问答

```bash
python demo.py
```

`demo.py` 会：

1. 检查 `dataset/data.pkl`。
2. 索引不存在时自动加载文档、切块并生成 embedding。
3. 初始化 Dense、BM25、RRF、Reranker 和 Generator。
4. 进入交互问答。

输入以下内容退出：

```text
exit
```

### 启动 Agent 使用的检索服务

```bash
python rag_service.py
```

服务监听 `http://127.0.0.1:8080/retrieve`，只加载 Embedding 与 Reranker，并返回 Top-K 文档；最终答案由 Agent 的模型生成。

当前端到端检索参数：

```text
Dense Top-15
BM25 Top-15
→ RRF Top-10，c=60
→ Reranker Top-5
→ Generator
```

### 启动家电故障诊断检索服务

`appliance_rag_service.py` 读取同级项目 `appliance-support-sft/data/ready/rag_corpus.jsonl`。每条家电故障记录直接构造成一个 Chunk，首次运行时生成 Dense Embedding 缓存，后续启动直接加载；BM25、RRF 与 Reranker 继续复用手写实现。

```bash
python appliance_rag_service.py
```

家电场景的检索链路为：

```text
438条家电故障资料
→ Dense Top-15 + BM25 Top-15
→ RRF Top-10，c=60
→ Qwen3-Reranker Top-5
→ /retrieve 返回文档、来源、分数和排名
```

服务启动后，由 `handwritten-agent/tools.py` 中的 `query_rag` Tool 调用。生成模型不在该服务中加载，避免 RAG 与 Agent Runtime 耦合。

### 单独构建 Dense 索引

```bash
python build_index.py
```

### 检索评估

```bash
python evaluate_retrieval_dense.py
python evaluate_retrieval_bm25.py
python evaluate_retrieval_rrf.py
python evaluate_retrieval_reranker.py
```

## 检索评估结果

评测集包含 10 个查询，标注字段为：

```json
{
  "question": "问题",
  "relevant_chunk_ids": [0, 1]
}
```

受控 Top-5 候选池实验结果：

| 检索方法 | MRR@5 | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|---:|
| Dense | 0.4817 | 0.25 | 0.50 | 0.85 |
| BM25 | 0.6833 | 0.35 | 0.95 | 0.95 |
| RRF | 0.9000 | 0.70 | 0.95 | 0.95 |
| RRF + Reranker | **1.0000** | **0.90** | 0.95 | 0.95 |

最终端到端配置使用：

```text
Dense Top-15
BM25 Top-15
→ RRF Top-10
→ Reranker Top-5
```

其结果为：

| 配置 | MRR@5 | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|---:|
| 15 → 15 → 10 → 5 | **1.0000** | **0.90** | **1.00** | **1.00** |

实验现象：

- Dense 能召回语义相关内容，但当前测试集上的前排排序较弱。
- BM25 对 `Day 10`、`LoRA`、`Agent` 等精确词和编号更敏感。
- RRF 利用两路排名互补，显著提高 MRR 和 Recall@1。
- Reranker 将 RRF 候选中的正确 chunk 进一步推到前排。

评测集规模较小，而且问题与原文措辞接近，因此这些指标用于验证流程和比较各阶段，不代表真实业务数据上的泛化效果。

受控 Top-5 表格记录的是开发过程中的统一候选深度实验；当前评估脚本已经调整为最终候选配置。详细参数口径见评测文档。

## 详细文档

- [手撕混合检索设计](docs/RETRIEVAL_DESIGN.md)
- [检索评测说明与实验结果](docs/EVALUATION.md)

## 当前限制

- Dense 索引使用 pickle 与 NumPy，不支持大规模向量检索。
- BM25 使用 Jieba 基础分词，没有停用词、同义词和领域词典优化。
- 文档使用固定字符长度切块，没有实现语义切块。
- `chunk_id` 是当前索引中的整数编号，重建索引后可能变化。
- 当前评测集仅包含 10 个问题。
- 模型均在单进程中本地加载，没有批处理、并发或服务化。
- 尚未实现增量索引、权限过滤、引用格式化和生成质量评估。

## 下一阶段

下一版本将使用 Milvus 与 LangChain 重写相同链路：

```text
Milvus dense + BM25 hybrid retrieval
→ RRF
→ Qwen3 Reranker
→ LangChain RAG
```

手写版用于理解底层数据流，框架版用于学习向量数据库、标准组件和工程化编排。
