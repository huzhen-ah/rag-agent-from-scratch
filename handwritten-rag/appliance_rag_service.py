#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:36:06 2026

@author: huzhen
"""

import torch
from embedding import Embedder
from load import Chunk
import pickle
import os
import json
from dense_retrieval import DenseRetrieval
from bm25_retrieval import BM25Retrieval
from reranker import Reranker
from rag import RAG
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn



if torch.cuda.is_available():
        device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

corpus_path = "../../appliance-support-sft/data/ready/rag_corpus.jsonl"
chunk_config_path = r"dataset/appliance_chunk_config.pkl"
embedding_model_path = "models/Qwen3-Embedding-0.6B"
reranker_model_path = "models/Qwen3-Reranker-0.6B"
bm25_config_path = "bm25/appliance_bm25_config.pkl"

class BuildRAG:
    def __init__(self, corpus_path, chunk_config_path, bm25_config_path, embedding_model_path, reranker_model_path):
        self.corpus_path = corpus_path
        self.chunk_config_path = chunk_config_path
        self.bm25_config_path = bm25_config_path
        self.embedding_model_path = embedding_model_path
        self.reranker_model_path = reranker_model_path
        self.embedder = Embedder(self.embedding_model_path,device)

    def load_chunks(self):
        chunks = []
        with open(self.corpus_path, "r", encoding="utf8") as f:
            for chunk_id, line in enumerate(f):
                item = json.loads(line)
                content = item["content"]

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        content=content,
                        source=item["source_url"],
                        start=0,
                        end=len(content),
                        page=None,
                    )
                )

        return chunks

    def load_chunk_config(self):

        if os.path.isfile(self.chunk_config_path):
            with open(self.chunk_config_path, "rb") as f:
                chunk_config = pickle.load(f)
            chunks = chunk_config["chunks"]
            chunk_embeddings = chunk_config["embeddings"]
        else:
            chunks = self.load_chunks()
            chunk_embeddings = self.embedder.embed_chunks(chunks)
            with open(self.chunk_config_path, "wb") as f:
                config = {"chunks":chunks, "embeddings":chunk_embeddings}
                pickle.dump(config, f)
        return chunks, chunk_embeddings

    def build_rag(self):

        chunks, chunk_embeddings = self.load_chunk_config()
        denseRetrieval = DenseRetrieval(self.embedder, chunks, chunk_embeddings)
        bm25Retrieval = BM25Retrieval(chunks, self.bm25_config_path)
        reranker = Reranker(self.reranker_model_path, device)
        return RAG(denseRetrieval, bm25Retrieval, reranker, None)

class RetrieveRequest(BaseModel):
    question: str

app = FastAPI()
rag = BuildRAG(corpus_path, chunk_config_path, bm25_config_path, embedding_model_path, reranker_model_path).build_rag()

@app.post("/retrieve")
def retrieve(request: RetrieveRequest):
    results = rag.retrieve(question=request.question, c=60, k=5)

    documents = []
    for result in results:
        chunk = result["chunk"]
        rank = result["rank"]
        score = float(result["score"])
        document = {
                        "chunk_id" : chunk.chunk_id,
                        "content" : chunk.content,
                        "source" : chunk.source,
                        "page" : chunk.page,
                        "score" : score,
                        "rank" : rank
                   }
        documents.append(document)
    return {"documents" : documents}

@app.post("/retrieve_every_stage")
def retrieve_every_stage(request: RetrieveRequest):
    retrieve_results = rag.retrieve_every_stage(question=request.question, c=60, k=5)
    ret = {}
    for retrieve_type,type_results in retrieve_results.items():
        if retrieve_type not in ret:
            ret[retrieve_type] = {"documents":[]}
        for result in type_results:
            chunk = result["chunk"]
            rank = result["rank"]
            score = float(result["score"])
            document = {
                            "chunk_id" : chunk.chunk_id,
                            "content" : chunk.content,
                            "source" : chunk.source,
                            "page" : chunk.page,
                            "score" : score,
                            "rank" : rank
                       }
            ret[retrieve_type]["documents"].append(document)
    return ret


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080
    )
