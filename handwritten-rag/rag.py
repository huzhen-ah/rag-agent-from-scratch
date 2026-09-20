#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 18:02:03 2026

@author: huzhen
"""
import torch
from dense_retrieval import DenseRetrieval
from bm25_retrieval import BM25Retrieval
from reranker import Reranker
from embedding import Embedder
from generator import Generator
import pickle


class RAG:
    def __init__(self,denseRetrieval,bm25Retrieval,reranker,generator):
        self.denseRetrieval = denseRetrieval
        self.bm25Retrieval = bm25Retrieval
        self.reranker = reranker
        self.generator = generator
        
    
    def rrf(self,dense_chunks_ret,bm25_chunks_ret,c=60,k=5):
        if len(dense_chunks_ret) == 0:
            return bm25_chunks_ret[:k]
        elif len(bm25_chunks_ret) == 0:
            return dense_chunks_ret[:k]

        dense_id2chunk = {}
        bm25_id2chunk = {}
        
        for _ in dense_chunks_ret:
            chunk_id = _["chunk"].chunk_id
            if chunk_id not in dense_id2chunk:
                dense_id2chunk[chunk_id] = _
                
        for _ in bm25_chunks_ret:
            chunk_id = _["chunk"].chunk_id
            if chunk_id not in bm25_id2chunk:
                bm25_id2chunk[chunk_id] = _
        
        chunk_ids = list(set(list(dense_id2chunk) + list(bm25_id2chunk)))#这里有一个问题，当2路分数相同时，排序可能不稳定，因为用了set,但无所谓。。。就是要有点波动
        scores = []
        
        for chunk_id in chunk_ids:
            score = 0
            if chunk_id in dense_id2chunk:
                dense_rank = dense_id2chunk[chunk_id]["rank"]
                score = score + 1 / (c+dense_rank)
            if chunk_id in bm25_id2chunk:
                bm25_rank = bm25_id2chunk[chunk_id]["rank"]
                score = score + 1 / (c+bm25_rank)
            scores.append(score)
        
        ids_scores = sorted(zip(chunk_ids,scores),key=lambda x : x[1],reverse=True)[:k]
        ret = []
        for i,id_score in enumerate(ids_scores):
            chunk_id,score = id_score
            if chunk_id in dense_id2chunk:
                _ = {"chunk":dense_id2chunk[chunk_id]["chunk"],"score":score,"rank":i+1}
            else:
                _ = {"chunk":bm25_id2chunk[chunk_id]["chunk"],"score":score,"rank":i+1}
            ret.append(_)
        return ret


    def retrieve(self, question, c, k):
        dense_chunks_ret = self.denseRetrieval.search_documents(question,k*3)
        bm25_chunks_ret = self.bm25Retrieval.search_documents(question,k*3)
        rrf_chunks_ret = self.rrf(dense_chunks_ret, bm25_chunks_ret,c,k*2)
        reranker_chunks_ret = self.reranker.rerank(question,rrf_chunks_ret,k)
        
        return reranker_chunks_ret

    def retrieve_every_stage(self, question, c, k):
        dense_chunks_ret = self.denseRetrieval.search_documents(question,k*3)
        bm25_chunks_ret = self.bm25Retrieval.search_documents(question,k*3)
        rrf_chunks_ret = self.rrf(dense_chunks_ret, bm25_chunks_ret,c,k*2)
        reranker_chunks_ret = self.reranker.rerank(question,rrf_chunks_ret,k)

        ret = {
                "dense"    : dense_chunks_ret,
                "bm25"    : bm25_chunks_ret,
                "rrf"      : rrf_chunks_ret,
                "reranker" : reranker_chunks_ret
                }
        return ret
        
        
    def answer_question(self,question,c,k):
        
        reranker_chunks = self.retrieve(question, c, k)
        
        prompt = "请严格按照以下参考资料回答问题，如果资料不全，请回答'根据现有资料无法确定'\n" + "\n".join(["参考资料_{}:{}".format(i,content) for i,content in enumerate([chunk["chunk"].content for chunk in reranker_chunks])])
        messages = [
            {
                "role": "system",
                "content": "你是一名RAG助手",
            },
            {
                "role": "user",
                "content": prompt+"\n"+question,
            },
        ]

        answer = self.generator.generate(messages)

       
        return answer
    
    
        
        
    
if __name__ == "__main__":
    if torch.cuda.is_available():
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    embedding_model_path = r"models/Qwen3-Embedding-0.6B"
    generate_model_path = r"models/Qwen3-1.7B"
    reranker_model_path = r"models/Qwen3-Reranker-0.6B"
    
    
    dense_dataset_path = r"dataset/data.pkl"
    bm25_config_path = r"bm25/bm25_config.pkl"
    with open(dense_dataset_path,"rb") as f:
        dataset = pickle.load(f)
    embedder = Embedder(embedding_model_path, device)
    generator = Generator(generate_model_path, device)
    
    
    denseRetrieval = DenseRetrieval(embedder, dataset["chunks"], dataset["chunk_embeddings"])
    bm25Retrieval = BM25Retrieval(dataset["chunks"],bm25_config_path)
    reranker = Reranker(reranker_model_path, device)
    rag = RAG(denseRetrieval, bm25Retrieval, reranker, generator)
    k = 5
    c = 60
    while True:
        question = str(input("我:")).strip()
        if not question:
            continue
        if question == "exit":
            break
        answer = rag.answer_question(question,c,k)
        print("assistant: ",answer)
