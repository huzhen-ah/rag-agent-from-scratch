# Enterprise RAG

这是一个可以独立运行的企业级 RAG 最小实现。项目以家电故障知识库为示例，重点实现以下核心能力：

- PostgreSQL 管理租户、用户、知识库、文档、文档版本和索引状态。
- MinIO 保存原始文档。
- Milvus 同时完成 Dense 检索、BM25 稀疏检索和 RRF 融合。
- 基于租户、知识库、文档可见范围和文档所有者进行检索权限过滤。
- 支持创建知识库、上传 JSONL、建立索引和整篇文档版本更新。
- 对 Dense、BM25、RRF、Reranker 四个阶段分别进行评测。

本项目与同级的 `handwritten-rag` 相互独立，不会导入或调用另一个项目中的代码。

## 架构

```text
原始 JSONL 文档
      │
      ├── MinIO：保存原始文件
      │
      ├── PostgreSQL：保存业务元数据、权限字段、版本和索引状态
      │
      └── Embedding → Milvus
                       ├── Dense 向量索引
                       ├── BM25 稀疏索引
                       └── RRF 混合检索
                                │
                                └── Reranker → Top-K 文档
```

各组件的职责：

| 组件 | 职责 |
|---|---|
| PostgreSQL | 保存租户、用户、知识库、文档、版本及索引状态等结构化数据 |
| MinIO | 保存上传的原始文档，当前示例为 `rag_corpus.jsonl` |
| Milvus | 保存知识记录、Dense 向量和权限字段，并提供 Dense、BM25、RRF 检索 |
| etcd | 保存 Milvus 自身的元数据，由 Milvus 内部使用 |
| Attu | 查看 Milvus collection、schema 和数据的可视化界面 |
| FastAPI | 对外提供带权限过滤的检索接口 |

## 权限模型

当前只实现核心基础权限，不引入复杂角色系统：

```text
Tenant
├── User
└── KnowledgeBase
    └── Document
        ├── owner_id
        ├── visibility
        ├── DocumentVersion
        └── KnowledgeRecord
```

检索前会验证：

1. 用户存在。
2. 知识库存在。
3. 用户和知识库属于同一租户。
4. Milvus 只返回指定租户、指定知识库中，对当前用户可见的记录。

当前 Milvus 过滤表达式等价于：

```text
tenant_id == 当前用户租户
and knowledge_base_id == 当前知识库
and (visibility == "knowledge_base" or owner_id == 当前用户)
```

## 数据表

| 表 | 作用 | 关键字段 |
|---|---|---|
| `tenants` | 租户 | `tenant_id`, `name` |
| `users` | 用户 | `user_id`, `tenant_id`, `name` |
| `knowledge_bases` | 知识库 | `knowledge_base_id`, `tenant_id`, `name` |
| `documents` | 文档元数据 | `document_id`, `knowledge_base_id`, `owner_id`, `visibility`, `current_version` |
| `document_versions` | 原始文件版本 | `document_id`, `version`, `bucket_name`, `object_name`, `status` |
| `knowledge_records` | 文档解析后的知识记录 | `record_id`, `document_id`, `content`, `attributes`, `version`, `index_status` |

PostgreSQL 保存业务事实；Milvus 保存用于检索的副本。`index_status` 用于记录知识记录是否已经写入 Milvus。

## 目录结构

```text
enterprise-rag/
├── create_tables.py            # 明确创建 PostgreSQL 表
├── database.py                 # SQLAlchemy engine 和 Base
├── models.py                   # 数据表模型
├── document_service.py         # 文档创建、更新和索引流程
├── minio_storage.py            # MinIO 文件上传和下载封装
├── milvus_storage.py           # Milvus collection、索引和检索封装
├── embedding.py                # Embedding 模型封装
├── reranker.py                 # Reranker 模型封装
├── init_data.py                # 初始化示例数据并写入 Milvus
├── rag.py                      # 检索流程编排
├── rag_service.py              # FastAPI 检索服务
├── sample_data/                # 文档新增与更新样例
├── test_rag_interface.py       # 检索接口冒烟测试
├── docker-compose.yml          # PostgreSQL、MinIO、Milvus、etcd、Attu
├── requirements.txt            # Python 依赖
├── documents/                  # 项目内原始文档
└── evaluation/                 # 检索评测脚本、数据和结果说明
```

## 环境准备

以下命令都从 `enterprise-rag` 目录执行：

```bash
cd rag-agent-from-scratch/enterprise-rag
conda activate ENV_enterprise
pip install -r requirements.txt
```

本地模型放在：

```text
models/Qwen3-Embedding-0.6B
models/Qwen3-Reranker-0.6B
```

模型和数据库卷均已被 Git 忽略，不会提交到 GitHub。

## 启动基础设施

```bash
docker compose up -d
docker compose ps
```

服务地址：

| 服务 | 地址 |
|---|---|
| PostgreSQL | `127.0.0.1:5432` |
| MinIO API | `http://127.0.0.1:9000` |
| MinIO Console | `http://127.0.0.1:9001` |
| Milvus | `http://127.0.0.1:19530` |
| Attu | `http://127.0.0.1:3000` |

开发环境默认账号：

```text
PostgreSQL: rag / rag_dev_password
MinIO: minioadmin / minioadmin
```

这些账号只用于本地演示，部署时必须改为环境变量或密钥管理。

## 初始化数据

第一次运行时按顺序执行：

```bash
python create_tables.py
python minio_storage.py
python init_data.py
```

三个脚本分别完成：

1. 创建 PostgreSQL 数据表。
2. 创建 MinIO 的 `documents` 桶并上传原始 JSONL 文件。
3. 写入示例租户、用户、知识库、文档和知识记录，生成 Dense Embedding，并写入 Milvus。

`init_data.py` 会检查主键和 `index_status`，已经完成的记录不会重复插入或重复建立索引。

## 启动检索服务

```bash
python rag_service.py
```

服务监听 `http://127.0.0.1:8080`。

正式检索接口：

```bash
curl -X POST http://127.0.0.1:8080/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user-001","knowledge_base_id":"kb-fault-codes","question":"美的洗衣机显示E03是什么意思？"}'
```

查看每一个检索阶段：

```bash
curl -X POST http://127.0.0.1:8080/retrieve_every_stage \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user-001","knowledge_base_id":"kb-fault-codes","question":"美的洗衣机显示E03是什么意思？"}'
```

`/retrieve_every_stage` 分别返回 `dense`、`bm25`、`rrf` 和 `reranker` 的记录 ID，供评测使用。

上传接口为 `/register_and_upload`，Streamlit 页面会调用它创建租户、用户、知识库和文档，并在索引完成后自动切换到新知识库。页面启动命令：

```bash
cd ../handwritten-agent
streamlit run appliance_support_ui.py
```

## 运行评测

保持 RAG 服务运行，在另一个终端执行：

```bash
cd rag-agent-from-scratch/enterprise-rag/evaluation
python evaluate_retrieval.py
```

评测数据、指标定义和当前结果见 [evaluation/README.md](evaluation/README.md)。

## 当前边界

当前版本定位为企业 RAG 最小实现，暂不包含登录认证、复杂 RBAC、任务队列和生产级监控。
