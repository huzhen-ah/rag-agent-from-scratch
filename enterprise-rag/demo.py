#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:18:44 2026

@author: huzhen
"""
from minio_storage import MinioStorage
from milvus_storage import MilvusStorage
from document_service import DocumentService
from embedding import Embedder
from database import engine



endpoint = "127.0.0.1:9000"
access_key = "minioadmin"
secret_key = "minioadmin"
bucket_name = "documents"
file_path = r"sample_data/employee_handbook_v1.jsonl"

miniostorage = MinioStorage(endpoint, access_key, secret_key)


collection_name = "knowledge_records"
milvusstorage = MilvusStorage("http://127.0.0.1:19530")
milvusstorage.create_collection(collection_name, 1024)

model_path = r"models/Qwen3-Embedding-0.6B"
device = "mps"
embedder = Embedder(model_path, device)



documentservice = DocumentService(engine, miniostorage, milvusstorage, embedder, collection_name, bucket_name)

tenant_name = "天马行空"
user_name = "牛马1号"
knowledge_base_name = "员工手册"

tenant = documentservice.create_tenant(tenant_name)
user = documentservice.create_user(tenant.tenant_id, user_name)
knowledge_base = documentservice.create_knowledge_base(tenant.tenant_id, knowledge_base_name)
document = documentservice.create_document(user.user_id, knowledge_base.knowledge_base_id, file_path)

question = "员工申请年假时，最小可以申请多长时间？"

query_vector = embedder.embed_query(question)
filter_expression = f'tenant_id == "{tenant.tenant_id}" and knowledge_base_id == "{knowledge_base.knowledge_base_id}" and (visibility == "knowledge_base" or owner_id == "{user.user_id}")'
results = milvusstorage.hybrid_search(collection_name, question, query_vector, filter_expression)
print("type results: ",type(results))
print("results: ",results[0])

new_file_path = r"sample_data/employee_handbook_v2.jsonl"
document = documentservice.update_document(user.user_id, document.document_id, new_file_path)

question = "员工申请年假时，最小可以申请多长时间？"

query_vector = embedder.embed_query(question)
filter_expression = f'tenant_id == "{tenant.tenant_id}" and knowledge_base_id == "{knowledge_base.knowledge_base_id}" and (visibility == "knowledge_base" or owner_id == "{user.user_id}")'
results = milvusstorage.hybrid_search(collection_name, question, query_vector, filter_expression)
print("type results: ",type(results))
print("results: ",results[0])
