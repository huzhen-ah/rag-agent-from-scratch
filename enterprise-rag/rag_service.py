#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:04:05 2026

@author: huzhen
"""

from rag import RAG
import os
import torch

from embedding import Embedder
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session
from database import engine
from models import User, KnowledgeBase
from reranker import Reranker
from milvus_storage import MilvusStorage
from minio_storage import MinioStorage
from document_service import DocumentService
from upload_api import UploadBodyLimitMiddleware, create_upload_router


def build_rag():
    if torch.cuda.is_available():
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    embedding_model_path = r"models/Qwen3-Embedding-0.6B"
    reranker_model_path = r"models/Qwen3-Reranker-0.6B"
    milvus_storage = MilvusStorage("http://127.0.0.1:19530")
    embedder = Embedder(embedding_model_path, device)
    collection_name = "knowledge_records"
    reranker = Reranker(reranker_model_path, device)
    rag = RAG(embedder, milvus_storage, reranker, collection_name)
    return rag

def build_filter_expression(user_id, knowledge_base_id):
    with Session(engine) as session:
        user = session.get(User, user_id)

        if user is None:
            raise HTTPException(status_code=403, detail="用户不存在或无权访问")

        knowledge_base = session.get(KnowledgeBase, knowledge_base_id)

        if knowledge_base is None:
            raise HTTPException(status_code=403, detail="知识库不存在或无权访问")

        if user.tenant_id != knowledge_base.tenant_id:
            raise HTTPException(status_code=403, detail="用户无权访问该知识库")

        filter_expression = 'tenant_id == "{}" and knowledge_base_id == "{}" and (visibility == "knowledge_base" or owner_id == "{}")'.format(user.tenant_id, knowledge_base.knowledge_base_id, user.user_id)

        return filter_expression

class RetrieveRequest(BaseModel):
    user_id: str
    knowledge_base_id: str
    question: str

app = FastAPI()
app.add_middleware(UploadBodyLimitMiddleware)
rag = build_rag()
minio_storage = MinioStorage(os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000"), os.getenv("MINIO_ACCESS_KEY", "minioadmin"), os.getenv("MINIO_SECRET_KEY", "minioadmin"))
document_service = DocumentService(engine, minio_storage, rag.milvus_storage, rag.embedder, rag.collection_name, os.getenv("MINIO_BUCKET", "documents"))


app.include_router(create_upload_router(document_service))

@app.post("/retrieve")
def retrieve(request: RetrieveRequest):
    filter_expression = build_filter_expression(request.user_id, request.knowledge_base_id)
    results = rag.retrieve(question=request.question, filter_expression=filter_expression, candidate_limit=20, result_limit=5)

    documents = []
    for result in results:
        entity = result["entity"]
        rank = result["rank"]


        document = {
                        "record_id" : result["record_id"],
                        "document_id": entity["document_id"],
                        "content" : entity["content"],
                        "attributes" : entity["attributes"],
                        "retrieval_score":result["retrieval_score"],
                        "reranker_score": result["reranker_score"],
                        "rank" : rank
                   }
        documents.append(document)
    return {"documents" : documents}

@app.post("/retrieve_every_stage")
def retrieve_every_stage(request: RetrieveRequest):
    filter_expression = build_filter_expression(request.user_id, request.knowledge_base_id)
    results = rag.retrieve_every_stage(question=request.question, filter_expression=filter_expression, candidate_limit=20, result_limit=5)

    ret = {}
    for retrieve_type,type_results in results.items():
        if retrieve_type not in ret:
            ret[retrieve_type] = {"documents":[]}
        for index,result in enumerate(type_results):

            document = {
                            "record_id" : result["record_id"]
                       }
            ret[retrieve_type]["documents"].append(document)
    return ret



if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080
    )
