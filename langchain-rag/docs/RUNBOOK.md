# 运行与排错手册

本文用于启动、验证和排查 `langchain-rag`。算法与架构说明见其他文档；这里关注实际操作。

## 1. 启动前检查

运行完整 Demo 前，需要确认：

```text
Python环境可用
Milvus Standalone正在运行
三个本地模型已准备
示例文档存在
当前工作目录是langchain-rag
```

当前目录应包含：

```text
langchain-rag/
├── demo.py
├── ingest.py
├── retriever.py
├── reranker.py
├── generator.py
├── rag.py
├── documents/
│   └── THREE_WEEK_PLAN.md
└── models/
    ├── Qwen3-Embedding-0.6B
    ├── Qwen3-Reranker-0.6B
    └── Qwen3-1.7B
```

## 2. Python 环境

建议为项目使用独立虚拟环境。

主要依赖：

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

检查关键包：

```bash
python -c "import pymilvus; print(pymilvus.__version__)"
python -c "import langchain_milvus; print('langchain_milvus OK')"
python -c "import langchain_huggingface; print('langchain_huggingface OK')"
python -c "import sentence_transformers; print(sentence_transformers.__version__)"
```

当前项目的包来源：

```text
Document / Prompt / Runnable
→ langchain-core

Text Splitter
→ langchain-text-splitters

Embedding / LLM / ChatModel
→ langchain-huggingface

Milvus VectorStore
→ langchain-milvus

HuggingFaceCrossEncoder
→ langchain-community

Reranker / ContextualCompressionRetriever
→ langchain-classic
```

## 3. 本地模型

模型目录：

```text
models/
├── Qwen3-Embedding-0.6B
├── Qwen3-Reranker-0.6B
└── Qwen3-1.7B
```

下载示例：

```bash
hf download Qwen/Qwen3-Embedding-0.6B \
  --local-dir models/Qwen3-Embedding-0.6B

hf download Qwen/Qwen3-Reranker-0.6B \
  --local-dir models/Qwen3-Reranker-0.6B

hf download Qwen/Qwen3-1.7B \
  --local-dir models/Qwen3-1.7B
```

快速检查：

```bash
ls models/Qwen3-Embedding-0.6B
ls models/Qwen3-Reranker-0.6B
ls models/Qwen3-1.7B
```

至少应存在模型配置、Tokenizer 配置和权重文件。

代码设置：

```python
local_files_only=True
```

因此本地模型不完整时会直接报错，不会自动联网补齐。

## 4. 启动 Docker Desktop

Milvus Standalone 运行在 Docker 中。执行 Compose 命令前，需要先启动 Docker Desktop。

检查 Docker：

```bash
docker info
```

如果 Docker Desktop 没有启动，通常会出现无法连接 Docker daemon 的错误。

## 5. 启动 Milvus Standalone

进入保存 Milvus `docker-compose.yml` 的目录：

```bash
cd /path/to/milvus-standalone
```

启动：

```bash
docker compose up -d
```

查看状态：

```bash
docker compose ps
```

预期核心容器：

```text
milvus-etcd
milvus-minio
milvus-standalone
milvus-attu
```

当前项目使用：

```text
Milvus URI: http://localhost:19530
Attu: http://localhost:8000
```

### `no configuration file provided`

如果执行：

```bash
docker compose ps
```

出现：

```text
no configuration file provided: not found
```

原因通常不是 Milvus 本身故障，而是当前目录没有：

```text
docker-compose.yml
compose.yml
```

解决：

```text
进入保存Compose文件的目录
→ 再执行docker compose命令
```

## 6. 检查 Milvus 端口

Milvus 默认连接地址：

```text
http://localhost:19530
```

查看容器端口：

```bash
docker ps
```

应看到类似：

```text
0.0.0.0:19530->19530/tcp
```

如果端口未映射，Python 程序无法通过 `localhost:19530` 连接。

## 7. Attu

浏览器打开：

[http://localhost:8000](http://localhost:8000)

当前 Docker Compose 中 Attu 通过 Docker 网络连接：

```text
standalone:19530
```

在 Attu 中可以查看：

```text
Database
Collection
Schema
Function
Index
数据行
```

BM25 Function 生成的 Sparse 具体向量值不能像普通标量字段一样直接取回。能看到 `sparse` Schema 和索引，但看不到完整 Sparse 值并不表示入库失败。

## 8. 一键运行

进入项目目录：

```bash
cd rag-agent-from-scratch/langchain-rag
```

运行：

```bash
python demo.py
```

首次运行：

```text
连接Milvus
→ 创建rag_demo（不存在时）
→ 检查rag_chunks
→ 加载documents
→ 切分
→ 创建Schema和索引
→ 写入Documents
→ 加载Embedding、Reranker和Generator
→ 启动问答
```

后续运行：

```text
检测到rag_chunks
→ 跳过入库
→ 连接已有Collection
→ 加载模型
→ 启动问答
```

退出：

```text
exit
```

## 9. 首次启动为什么较慢

首次启动需要：

- 加载 Qwen3-Embedding。
- 计算全部 chunk 的 Dense 向量。
- 创建 Collection 与索引。
- 生成 BM25 Sparse 表示。
- 加载 Qwen3-Reranker。
- 加载 Qwen3-1.7B。

后续启动虽然跳过入库，但仍然需要重新加载三个本地模型，因为当前程序没有常驻模型服务。

## 10. 单独重建知识库

当以下内容改变时，应重建 Collection：

- `documents/` 中的文件。
- `chunk_size` 或 `chunk_overlap`。
- Embedding 模型。
- Dense Dimension。
- Schema 字段。
- Analyzer。
- 索引类型或 Metric。
- BM25 Function 配置。

执行：

```bash
python ingest.py
```

当前行为：

```text
发现rag_chunks已存在
→ drop_collection
→ 重新创建
→ 全量写入
```

这是破坏性全量重建。旧 Collection 的：

- 数据。
- 自动主键。
- Schema。
- Function。
- 索引。

都会被替换。

## 11. `demo.py` 为什么不自动检测文档变化

当前判断：

```python
if not client.has_collection(collection_name):
    # 创建并入库
else:
    # 直接使用
```

它只判断 Collection 是否存在。

不会判断：

- 文档是否修改。
- 文档是否新增或删除。
- Collection 是否为空。
- Schema 是否过期。
- 模型是否替换。

因此：

```text
文档改变
≠ demo.py自动更新
```

需要显式运行：

```bash
python ingest.py
```

## 12. 分阶段验证

### 12.1 只验证入库

```bash
python ingest.py
```

预期输出：

```text
写入数量： N
```

### 12.2 只验证混合 Retriever

```bash
python retriever.py
```

验证：

```text
Dense
+ BM25
+ RRF
→ Document
```

### 12.3 验证 Reranker

```bash
python reranker.py
```

验证：

```text
RRF Top 10
→ CrossEncoder
→ Top 5
```

### 12.4 验证生成模型

```bash
python generator.py
```

验证：

```text
Tokenizer
→ Model
→ Transformers Pipeline
→ LangChain LLM
→ ChatModel
```

当前 `generator.py` 的独立测试没有自动添加 `/no_think`。如果需要禁止实际 Thinking 内容，应在测试问题中显式追加：

```text
/no_think
```

### 12.5 验证完整 Chain

```bash
python rag.py
```

验证：

```text
Retriever
→ Reranker
→ Prompt
→ ChatModel
→ String Answer
```

## 13. 使用 MilvusClient 验证 Collection

连接：

```python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    db_name="rag_demo",
    user="root",
    password="Milvus",
)
```

### 13.1 查看 Collection

```python
print(
    client.list_collections()
)
```

应包含：

```text
rag_chunks
```

### 13.2 查看数据量

```python
print(
    client.get_collection_stats(
        "rag_chunks"
    )
)
```

重点检查：

```text
row_count > 0
```

### 13.3 查看 Schema

```python
print(
    client.describe_collection(
        "rag_chunks"
    )
)
```

确认存在：

```text
pk
text
dense
sparse
bm25_function
```

### 13.4 查看索引

```python
print(
    client.list_indexes(
        "rag_chunks"
    )
)
```

确认存在：

```text
dense_hnsw_index
sparse_bm25_index
```

## 14. 验证纯 BM25

```python
client.load_collection("rag_chunks")

results = client.search(
    collection_name="rag_chunks",
    data=["第一周学习什么"],
    anns_field="sparse",
    search_params={
        "metric_type": "BM25",
        "params": {},
    },
    limit=5,
    output_fields=["text"],
)

print(results)
```

能够返回相关文本，说明：

```text
文本已写入
BM25 Function可用
Sparse索引可搜索
```

不要使用：

```python
output_fields=["sparse"]
```

尝试打印 BM25 Function 生成的 Sparse 具体值。

## 15. 验证 Dense

```python
query_vector = embeddings.embed_query(
    "第一周学习什么"
)

results = client.search(
    collection_name="rag_chunks",
    data=[query_vector],
    anns_field="dense",
    search_params={
        "metric_type": "COSINE",
        "params": {
            "ef": 64,
        },
    },
    limit=5,
    output_fields=["text"],
)
```

注意：

```text
Dense Search
→ data传向量

BM25 Search
→ data传原始Query文本
```

## 16. 连接错误

### 16.1 `Connection refused`

检查：

```text
Docker Desktop是否启动
milvus-standalone是否运行
19530端口是否映射
URI是否为http://localhost:19530
```

命令：

```bash
docker ps
```

### 16.2 认证错误

当前代码：

```text
user = root
password = Milvus
```

如果服务端：

- 开启了认证但账号密码不同，会认证失败。
- 未开启认证，示例中的凭据可能不会实际参与权限校验。

本地配置与代码必须一致。

生产环境不要使用硬编码管理员凭据。

### 16.3 Database 不存在

`demo.py` 和 `ingest.py` 会通过 `create_database()` 创建 `rag_demo`。

如果单独运行其他文件，而 Database 尚未创建，应先：

```bash
python ingest.py
```

或：

```bash
python demo.py
```

## 17. Schema 或 Metric 不匹配

典型原因：

```text
旧Collection仍然存在
但代码已经从vector改为dense
或新增了sparse字段
```

也可能是：

```text
Dense索引使用COSINE
搜索却传L2
```

或：

```text
Sparse索引使用BM25
搜索参数与字段不对应
```

处理：

```bash
python ingest.py
```

这会根据当前代码重建 Schema 和索引。

注意：它会删除旧 Collection 数据。

## 18. `show_progress_bar` 参数冲突

如果出现：

```text
SentenceTransformer.encode()
got multiple values for keyword argument
'show_progress_bar'
```

原因通常是同时在包装器和：

```python
encode_kwargs
```

中传入 `show_progress_bar`。

当前项目没有在 `encode_kwargs` 中传这个参数。

如果需要显示进度，应使用当前 `HuggingFaceEmbeddings` 版本支持的包装器级参数，不要在多个位置重复传递。

## 19. Reranker 导入问题

当前导入：

```python
from langchain_community.cross_encoders import (
    HuggingFaceCrossEncoder,
)

from langchain_classic.retrievers.document_compressors import (
    CrossEncoderReranker,
)

from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
```

如果报 `ModuleNotFoundError`，检查：

```bash
pip install -U \
  langchain-community \
  langchain-classic
```

当前本地 Hugging Face CrossEncoder 仍位于 `langchain-community`；不要把它误写成：

```python
from langchain_huggingface import (
    HuggingFaceCrossEncoder,
)
```

当前 `langchain-huggingface` 没有这个类。

## 20. Qwen3 输出 Thinking

Qwen3 默认开启 Thinking。

`rag.py` 的 Human Prompt 已添加：

```text
/no_think
```

因此正式 RAG Chain 不需要在调用时再次添加：

```python
rag_chain.invoke(
    "第一周学习什么"
)
```

不要写成：

```python
rag_chain.invoke(
    "第一周学习什么\n/no_think"
)
```

否则 Prompt 会重复出现 `/no_think`。

在当前 ChatHuggingFace 包装方式下，即使关闭实际思考内容，输出仍可能保留空的：

```text
<think>

</think>
```

## 21. MPS 问题

当前默认：

```text
device = mps
dtype = float32
```

如果设备不支持 MPS，需要改为：

```text
cuda
```

或：

```text
cpu
```

需要同步检查：

- Embedding。
- Reranker。
- Generator。

如果发生内存不足：

- 关闭其他占用统一内存的程序。
- 减少同时加载的模型。
- 降低生成长度。
- 根据硬件与模型支持情况评估更低精度。

不要在没有验证模型和设备兼容性的情况下随意修改 dtype。

## 22. Retriever 返回空结果

检查顺序：

```text
Collection是否存在
→ row_count是否大于0
→ dense和sparse字段是否存在
→ 两个索引是否存在
→ 纯Dense是否能搜索
→ 纯BM25是否能搜索
→ 混合Retriever是否能搜索
```

如果 Collection 存在但 `row_count=0`，`demo.py` 当前不会自动重新入库，因为它只检查 Collection 是否存在。

处理：

```bash
python ingest.py
```

## 23. Reranker 返回结果少于 `top_n`

```python
top_n=5
```

表示最多保留5条。

如果基础 Retriever 只返回3条，Reranker 不会凭空补足到5条。

结果数量满足：

```text
最终数量
≤ 基础Retriever返回数量
≤ top_n
```

更准确地写：

```text
最终数量
= min(实际候选数, top_n)
```

## 24. 模型回答与资料不一致

按以下顺序检查：

```text
1. 打印Retriever结果
2. 打印Reranker结果
3. 打印format_documents结果
4. 检查Prompt变量
5. 再检查生成模型
```

不要一开始只修改 Prompt。

可能原因：

- 文档解析丢失。
- Chunk 边界不合适。
- Dense 或 BM25 没召回。
- RRF 候选深度不足。
- Reranker 排序错误。
- Context 太长或包含冲突信息。
- 1.7B 模型能力不足。

## 25. 代码语法检查

不启动模型和 Milvus，只检查 Python 语法：

```bash
python -m py_compile \
  ingest.py \
  retriever.py \
  reranker.py \
  generator.py \
  rag.py \
  demo.py
```

这个检查不能验证：

- Milvus 是否可连接。
- 模型是否完整。
- Schema 是否匹配。
- 检索结果是否正确。
- 生成效果是否符合预期。

## 26. 推荐排错顺序

遇到端到端错误时，从底层向上检查：

```text
Docker
→ Milvus连接
→ Database
→ Collection
→ Schema与Index
→ row_count
→ Dense单路
→ BM25单路
→ RRF
→ Reranker
→ Generator
→ Prompt与LCEL
→ Demo
```

每次只验证一层，避免把数据库、模型和 Chain 错误混在一起。

## 27. 当前运行边界

当前 Demo 是本地学习版本：

- 没有常驻模型服务。
- 没有 FastAPI。
- 没有并发。
- 没有流式输出。
- 没有自动文档更新。
- 没有增量索引。
- 没有生产凭据管理。

它的目标是让以下流程可以独立验证：

```text
Milvus入库
→ 混合检索
→ RRF
→ Reranker
→ 本地生成
```
