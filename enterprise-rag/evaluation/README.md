# Enterprise RAG 检索评测

本目录用于比较同一批问题在四个检索阶段的效果：

```text
Dense
BM25
Dense + BM25 + RRF
Dense + BM25 + RRF + Reranker
```

## 评测数据

`data/appliance_retrieval_eval.jsonl` 包含 159 个问题，由 53 条测试文档生成三种查询：

- `code_lookup`：品牌、家电类型和故障码查询，共 53 条。
- `symptom_lookup`：品牌、家电类型和故障现象查询，共 53 条。
- `code_and_symptom`：故障码与故障现象组合查询，共 53 条。

样本格式：

```json
{
  "retrieve_id": 1,
  "question": "Beko洗衣机显示E03，是什么意思？",
  "query_type": "code_lookup",
  "relevant_record_ids": ["apdb-666be03e9b1a"]
}
```

`source/rag_documents.jsonl` 是生成评测集所需的本地源数据。生成脚本不会读取其他项目，因此 `enterprise-rag` 可以独立运行。

## 评测指标

- `Recall@1/3/5`：前 K 个结果召回的相关记录比例。
- `MRR@1/3/5`：第一个相关结果排名的倒数，再对全部问题取平均。
- `nDCG@1/3/5`：前 K 个结果的排序质量，相对理想排序进行归一化。

四个阶段使用相同的 Top-K 截断位置。

## 当前总体结果

### Recall@K

| 检索阶段 | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|
| Dense | 0.9418 | **0.9921** | **1.0000** |
| BM25 | 0.8684 | 0.9502 | 0.9895 |
| RRF | **0.9481** | 0.9900 | **1.0000** |
| RRF + Reranker | 0.8831 | 0.9879 | **1.0000** |

### MRR@K

| 检索阶段 | MRR@1 | MRR@3 | MRR@5 |
|---|---:|---:|---:|
| Dense | 0.9686 | 0.9811 | 0.9827 |
| BM25 | 0.8868 | 0.9161 | 0.9249 |
| RRF | **0.9748** | **0.9843** | **0.9858** |
| RRF + Reranker | 0.9057 | 0.9486 | 0.9502 |

### nDCG@K

| 检索阶段 | nDCG@1 | nDCG@3 | nDCG@5 |
|---|---:|---:|---:|
| Dense | 0.9686 | 0.9844 | 0.9871 |
| BM25 | 0.8868 | 0.9247 | 0.9415 |
| RRF | **0.9748** | **0.9849** | **0.9889** |
| RRF + Reranker | 0.9057 | 0.9583 | 0.9635 |

## 结论

- RRF 当前总体效果最好：Recall@1 为 0.9481，MRR@5 为 0.9858，nDCG@5 为 0.9889。
- Dense 对自然语言故障现象的效果明显优于 BM25。
- 当前 Reranker 提升了部分症状查询，但降低了故障码及混合查询的总体指标，因此尚未带来总体增益。
- Recall@5 已接近或达到 1.0，主要优化空间是把正确记录排到第一位。

该评测集由规则模板生成，适合检索阶段对比和回归测试，不等同于真实用户问题分布。

## 运行

先从项目根目录启动 RAG 服务：

```bash
cd rag-agent-from-scratch/enterprise-rag
conda activate ENV_enterprise
python rag_service.py
```

再打开一个终端运行评测：

```bash
cd rag-agent-from-scratch/enterprise-rag/evaluation
conda activate ENV_enterprise
python evaluate_retrieval.py
```

## 重新生成评测数据

```bash
cd rag-agent-from-scratch/enterprise-rag/evaluation
python build_eval_datasets.py
```

生成器读取本目录的 `source/rag_documents.jsonl` 和项目内的 `documents/.../rag_corpus.jsonl`，不会依赖其他项目。

脚本还会生成同一测试划分对应的 `appliance_agent_eval.jsonl` 和 `manifest.json`，用于保留数据来源和后续 Agent 联调；`evaluate_retrieval.py` 只读取 `appliance_retrieval_eval.jsonl`。
