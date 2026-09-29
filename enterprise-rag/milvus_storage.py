#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:56:27 2026

@author: huzhen
"""

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker


class MilvusStorage:
    def __init__(self, uri):
        self.client = MilvusClient(uri=uri)

    def create_collection(self, collection_name, dense_dimension):
        if self.client.has_collection(collection_name):
            print("collection: {} 已存在".format(collection_name))
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)

        schema.add_field(field_name="record_id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(field_name="tenant_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="knowledge_base_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="document_version", datatype=DataType.INT64)
        schema.add_field(field_name="record_version", datatype=DataType.INT64)
        schema.add_field(field_name="owner_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="visibility", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True, analyzer_params={"type": "chinese"})
        schema.add_field(field_name="attributes", datatype=DataType.JSON)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dimension)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        bm25_function = Function(name="content_bm25", input_field_names=["content"], output_field_names=["sparse_vector"], function_type=FunctionType.BM25)
        schema.add_function(bm25_function)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_name="dense_index", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
        index_params.add_index(field_name="sparse_vector", index_name="sparse_index", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25", params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75})

        self.client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
        print("collection: {} 创建成功".format(collection_name))

    def upsert_records(self, collection_name, records):
        result = self.client.upsert(collection_name, records)
        print("写入Milvus {} 条".format(result["upsert_count"]))
        return result

    def dense_search(self, collection_name, query_vector, filter_expression, limit=20):
        results = self.client.search(collection_name=collection_name, data=[query_vector], anns_field="dense_vector", filter=filter_expression, search_params={"metric_type": "COSINE"}, limit=limit, output_fields=["record_id", "tenant_id", "knowledge_base_id", "document_id", "content", "attributes"])
        return results[0]

    def bm25_search(self, collection_name, question, filter_expression, limit=20):
        results = self.client.search(collection_name=collection_name, data=[question], anns_field="sparse_vector", filter=filter_expression, search_params={"metric_type": "BM25"}, limit=limit, output_fields=["record_id", "tenant_id", "knowledge_base_id", "document_id", "content", "attributes"])
        return results[0]

    def hybrid_search(self, collection_name, question, query_vector, filter_expression, candidate_limit=20, result_limit=20):
        dense_request = AnnSearchRequest(data=[query_vector], anns_field="dense_vector", param={"metric_type": "COSINE"}, limit=candidate_limit, expr=filter_expression)
        sparse_request = AnnSearchRequest(data=[question], anns_field="sparse_vector", param={"metric_type": "BM25"}, limit=candidate_limit, expr=filter_expression)
        results = self.client.hybrid_search(collection_name=collection_name, reqs=[dense_request, sparse_request], ranker=RRFRanker(), limit=result_limit, output_fields=["record_id", "tenant_id", "knowledge_base_id", "document_id", "content", "attributes"])
        return results[0]

if __name__ == "__main__":
    from embedding import Embedder

    collection_name = "knowledge_records"
    storage = MilvusStorage("http://127.0.0.1:19530")
    storage.create_collection(collection_name, 1024)

    model_path = r"models/Qwen3-Embedding-0.6B"
    device = "mps"
    embedder = Embedder(model_path, device)

    question = "美的洗衣机出现E03怎么回事？"

    query_vector = embedder.embed_query(question)
    filter_expression = 'tenant_id == "tenant-001" and knowledge_base_id == "kb-fault-codes" and (visibility == "knowledge_base" or owner_id == "user-001")'
    results = storage.hybrid_search(collection_name, question, query_vector, filter_expression)
    print("type results: ",type(results))
    print("results: ",results)
