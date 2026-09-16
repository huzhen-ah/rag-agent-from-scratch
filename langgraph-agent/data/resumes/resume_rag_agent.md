# RAG / Agent 工程师简历

## 基本信息

- 姓名：待填写
- 手机：待填写
- 邮箱：待填写
- 所在地：广州
- 求职方向：RAG 工程师 / Agent 应用工程师

## 核心优势

- 同时实现过不依赖 RAG 编排框架的手写版本，以及 LangChain + Milvus 工程版本。
- 能够解释 Dense、BM25、RRF、CrossEncoder 和检索评测的原理，而非只会调用框架接口。
- 具备智能客服对话系统经验，理解对话业务、知识查询和知识库更新场景。
- 能够使用本地 Qwen3 系列模型完成 Embedding、Reranker 和生成。

## 技术栈

```text
Python
PyTorch
Transformers
Sentence Transformers
LangChain
Milvus
Qwen3
NumPy
Jieba
Neo4j
```

## 项目经历

### 手写混合检索 RAG

- 自行定义文档加载与 Chunk 数据流程，支持 TXT、Markdown 和 PDF。
- 使用 Qwen3-Embedding-0.6B 生成 Dense Embedding。
- 使用 NumPy 完成向量归一化、相似度计算和 Top-K 召回。
- 使用 Jieba 分词，手写 BM25 的语料统计、评分与排序。
- 手写 RRF，对 Dense 与 BM25 候选并集进行排名融合。
- 使用 Qwen3-Reranker-0.6B 对 RRF 候选重新打分并排序。
- 使用 Qwen3-1.7B 根据最终上下文生成回答。
- 使用 MRR 和 Recall@K 分阶段评估 Dense、BM25、RRF 和 Reranker。

小型回归评测集结果：

```text
MRR：1.00
Recall@1：0.90
Recall@3：1.00
Recall@5：1.00
```

评测集只有 10 个问题，且问题与原文措辞接近，结果只作为流程验证和回归基线。

### LangChain + Milvus 混合检索 RAG

- 自行解析原始文件并构造 LangChain Document。
- 使用 RecursiveCharacterTextSplitter 完成 Chunking 和重叠切分。
- 显式创建 Milvus Database、Collection Schema、BM25 Function 和索引。
- 使用 HNSW + COSINE 完成 Dense 向量检索。
- 使用 Milvus 原生 BM25 Function 和 Sparse Inverted Index 完成关键词检索。
- 使用 RRF 融合 Dense 与 BM25 的排名结果。
- 使用 Hugging Face CrossEncoder 和 LangChain Reranker 组件保留 Top-N 文档。
- 使用 LCEL 组合检索、文档格式化、Prompt、本地生成模型和输出解析。
- 使用 Docker 运行 Milvus Standalone，并通过 Attu 检查 Schema、索引和数据。

### 智能客服对话系统

- 参与智能客服对话系统相关开发，理解对话系统中的意图、知识查询和流程控制。
- 接触 Neo4j 知识数据的定时与实时更新。
- 理解实时增量更新与定时全量对账结合的最终一致性方案。
- 公司、时间和具体业务指标待补充。

## 教育背景

- 学校：待填写
- 专业：计算机相关专业
- 学历：待填写
- 时间：待填写
- 毕业论文：隐马尔可夫模型相关应用

## 项目链接

- <https://github.com/huzhen-ah/rag-agent-from-scratch>
- <https://github.com/huzhen-ah/mini-llm-demo>
