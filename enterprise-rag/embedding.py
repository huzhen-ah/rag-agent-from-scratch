#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:36:20 2026

@author: huzhen
"""

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self,model_path,device):
        self.model_path = model_path
        self.device = device
        self.model = self.load_embedding_model()


    def load_embedding_model(self):
        model = SentenceTransformer(
            self.model_path,
            device=self.device,
            local_files_only=True,
        )
        model.eval()
        return model

    def embed_texts(self, texts):
        text_embeddings = self.model.encode(
            texts,
            batch_size=8,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return text_embeddings

    def embed_chunks(self,chunks):
        chunk_embeddings = self.model.encode(
            [chunk.content for chunk in chunks],
            batch_size=8,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return chunk_embeddings


    def embed_query(self,query):
        query_embedding = self.model.encode(
            query,
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return query_embedding
