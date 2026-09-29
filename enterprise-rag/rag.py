#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 18:02:03 2026

@author: huzhen
"""

class RAG:
    def __init__(self,embedder, milvus_storage, reranker, collection_name):
        self.embedder = embedder
        self.milvus_storage = milvus_storage
        self.reranker = reranker
        self.collection_name = collection_name

    def retrieve(self, question, filter_expression, candidate_limit=20, result_limit=5):
        query_vector = self.embedder.embed_query(question)
        rrf_chunks_ret = self.milvus_storage.hybrid_search(self.collection_name, question, query_vector, filter_expression, candidate_limit, candidate_limit)
        reranker_chunks_ret = self.reranker.rerank(question,rrf_chunks_ret,result_limit)

        return reranker_chunks_ret

    def retrieve_every_stage(self, question, filter_expression, candidate_limit=20, result_limit=5):

        query_vector = self.embedder.embed_query(question)
        dense_chunks_ret = self.milvus_storage.dense_search(self.collection_name, query_vector, filter_expression, candidate_limit)
        bm25_chunks_ret = self.milvus_storage.bm25_search(self.collection_name, question, filter_expression, candidate_limit)
        rrf_chunks_ret = self.milvus_storage.hybrid_search(self.collection_name, question, query_vector, filter_expression, candidate_limit, candidate_limit)
        reranker_chunks_ret = self.reranker.rerank(question, rrf_chunks_ret, result_limit)

        ret = {
                "dense"    : dense_chunks_ret,
                "bm25"    : bm25_chunks_ret,
                "rrf"      : rrf_chunks_ret,
                "reranker" : reranker_chunks_ret
                }
        return ret







if __name__ == "__main__":
    pass
