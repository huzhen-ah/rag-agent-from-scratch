import requests
import json
from tqdm import tqdm
import numpy as np


def query_rag(question: str) -> dict:
    """
    从RAG知识库中检索与问题相关的参考资料。

    Args:
        question: 需要检索的问题。
    """

    ret = requests.post("http://127.0.0.1:8080/retrieve_every_stage",json={"question":question},timeout=120)
    ret.raise_for_status()
    ret = ret.json()
    return ret


def load_jsonl(path):
    ret = []
    with open(path,"r",encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ret.append(json.loads(line))
    return ret

def retrieve(data):
    ret = []
    for d in tqdm(iter(data),desc="...检索中..."):
        retrieve_id = d["retrieve_id"]
        question = d["question"]
        query_type = d["query_type"]
        relevant_chunk_ids = d["relevant_chunk_ids"]

        retrieve_ret = query_rag(question)
        _ = {
            "retrieve_id":retrieve_id,
            "question":question,
            "query_type":query_type,
            "relevant_chunk_ids":relevant_chunk_ids,
        }
        for stage,stage_ret in retrieve_ret.items():
            stage_chunk_ids = [chunk["chunk_id"] for chunk in stage_ret["documents"]]
            _["{}_chunk_ids".format(stage)] = stage_chunk_ids
        ret.append(_)
    return ret


def recall_at_K(retrieve_ret, stage, k, use_query_type=True):
    #recall的范围限制在k
    ret = {}
    for r in retrieve_ret:
        query_type = r["query_type"]
        if not use_query_type:
            query_type = "total"
        if query_type not in ret:
            ret[query_type] = []
        relevant_chunk_ids = r["relevant_chunk_ids"]
        stage_retrieve_chunk_ids = r[stage+"_chunk_ids"][:k]
        if not relevant_chunk_ids:
            score = 0
        else:
            score = len(set(stage_retrieve_chunk_ids) & set(relevant_chunk_ids)) / len(set(relevant_chunk_ids))
        ret[query_type].append(score)
    scores = {}
    for query_type,type_scores in ret.items():
        scores[query_type] = float(sum(type_scores) / len(type_scores))
    return scores

def mrr_at_K(retrieve_ret, stage, k, use_query_type=True):
    #第一个检索出来的文档中第一个正确的rank的倒数
    ret = {}
    for r in retrieve_ret:
        query_type = r["query_type"]
        if not use_query_type:
            query_type = "total"
        if query_type not in ret:
            ret[query_type] = []
        relevant_chunk_ids = set(r["relevant_chunk_ids"])
        stage_retrieve_chunk_ids = r[stage+"_chunk_ids"][:k]
        score = 0
        if relevant_chunk_ids:
            for rank, iid in enumerate(stage_retrieve_chunk_ids):
                if iid in relevant_chunk_ids:
                    score += 1/(rank+1)
                    break


        ret[query_type].append(score)
    scores = {}
    for query_type,type_scores in ret.items():
        scores[query_type] = float(sum(type_scores) / len(type_scores))
    return scores

def nDCG_at_K(retrieve_ret, stage, k, use_query_type=True):
    #DCG@K = Σ relevance(rank) / log₂(ranke + 1) rank >= 1
    ret = {}
    for r in retrieve_ret:
        query_type = r["query_type"]
        if not use_query_type:
            query_type = "total"
        if query_type not in ret:
            ret[query_type] = []
        relevant_chunk_ids = set(r["relevant_chunk_ids"])
        stage_retrieve_chunk_ids = r[stage+"_chunk_ids"][:k]
        dcg_score = 0
        idcg_score = 0

        if relevant_chunk_ids:
            for index, iid in enumerate(stage_retrieve_chunk_ids):
                if iid in relevant_chunk_ids:
                    rank = index + 1
                    dcg_score += 1/np.log2(rank+1)

        correct_num = min(len(relevant_chunk_ids), k)
        idcg_score = sum([1/np.log2(rank+1) for rank in range(1,correct_num+1)])
        if idcg_score == 0:
            score = 0
        else:
            score = dcg_score / idcg_score
        ret[query_type].append(score)
    scores = {}
    for query_type,type_scores in ret.items():
        scores[query_type] = float(sum(type_scores) / len(type_scores))
    return scores


if __name__ == "__main__":
    appliance_retrieval_eval_path = r"data/appliance_retrieval_eval.jsonl"
    appliance_retrieval_eval_data = load_jsonl(appliance_retrieval_eval_path)
    retrieve_ret = retrieve(appliance_retrieval_eval_data)
    print("\n\n")
    Ks = [1, 3, 5]
    for stage in ["dense", "bm25", "rrf", "reranker"]:

        for k in Ks:
            recall_at_k = recall_at_K(retrieve_ret, stage, k)
            print("{}_recall_at_{}: {}\n".format(stage, k, recall_at_k))
            total_recall_at_k = recall_at_K(retrieve_ret, stage, k ,use_query_type=False)
            print("total_{}_recall_at_{}: {}\n".format(stage, k, total_recall_at_k))
        print("\n\n")

        for k in Ks:
            mrr_at_k = mrr_at_K(retrieve_ret, stage, k)
            print("{}_mrr_at_{}: {}\n".format(stage, k, mrr_at_k))
            total_mrr_at_k = mrr_at_K(retrieve_ret, stage, k, use_query_type=False)
            print("total_{}_mrr_at_{}: {}\n".format(stage, k, total_mrr_at_k))
        print("\n\n")

        for k in Ks:
            nDCG_at_k = nDCG_at_K(retrieve_ret, stage, k)
            print("{}_nDCG_at_{}: {}\n".format(stage, k, nDCG_at_k))
            total_nDCG_at_k = nDCG_at_K(retrieve_ret, stage, k, use_query_type=False)
            print("total_{}_nDCG_at_{}: {}\n".format(stage, k, total_nDCG_at_k))
        print("\n\n")
