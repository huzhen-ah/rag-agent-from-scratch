#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 10:30:35 2026

@author: huzhen
"""

from sqlalchemy import select
from sqlalchemy.orm import Session
from database import engine
from models import Tenant, User, KnowledgeBase, Document, DocumentVersion, KnowledgeRecord
import json
from embedding import Embedder
from milvus_storage import MilvusStorage


model_path = r"models/Qwen3-Embedding-0.6B"
device = "mps"
embedder = Embedder(model_path, device)

milvus_url = r"http://127.0.0.1:19530"
collection_name = "knowledge_records"
milvus_storage = MilvusStorage(milvus_url)
milvus_storage.create_collection(collection_name, 1024)

with Session(engine) as session:
    tenant = session.get(Tenant, "tenant-001")
    if tenant is None:
        tenant = Tenant(
            tenant_id = "tenant-001",
            name = "家电售后公司"
        )
        session.add(tenant)
        session.flush()

    user = session.get(User, "user-001")
    if user is None:
        user = User(
            user_id = "user-001",
            tenant_id = "tenant-001",
            name = "hz"
        )
        session.add(user)
        session.flush()

    knowledgebase = session.get(KnowledgeBase, "kb-fault-codes")
    if knowledgebase is None:
        knowledgebase = KnowledgeBase(
            knowledge_base_id = "kb-fault-codes",
            tenant_id = "tenant-001",
            name = "kb-fault-codes"
        )
        session.add(knowledgebase)
        session.flush()

    document = session.get(Document, "doc-rag-corpus")
    if document is None:
        document = Document(
            document_id = "doc-rag-corpus",
            knowledge_base_id = "kb-fault-codes",
            owner_id = "user-001",
            name = "rag_corpus.jsonl",
            visibility = "knowledge_base"
        )
        session.add(document)
        session.flush()

    document_version_primary_key = {"document_id" : "doc-rag-corpus", "version" : 1}
    document_version = session.get(DocumentVersion, document_version_primary_key)
    if document_version is None:
        document_version = DocumentVersion(
            document_id = "doc-rag-corpus",
            version = 1,
            bucket_name = "documents",
            object_name = "tenant-001/kb-fault-codes/doc-rag-corpus/v1/rag_corpus.jsonl",
            status = "UPLOADED"
        )
        session.add(document_version)
        session.flush()

    with open(r"documents/tenant-001/kb-fault-codes/doc-rag-corpus/v1/rag_corpus.jsonl","r",encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            record_id = data["id"]
            record = session.get(KnowledgeRecord, record_id)
            if record is None:
                record = KnowledgeRecord(
                    record_id = record_id,
                    document_id = "doc-rag-corpus",
                    content = data["content"],
                    attributes = {k:v for k,v in data.items() if k not in ["id","content"]},
                    version = 1
                )
                session.add(record)
                session.flush()
    session.commit()
    statement = select(KnowledgeRecord, Document, KnowledgeBase).join(Document, KnowledgeRecord.document_id == Document.document_id)\
        .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.knowledge_base_id).where(KnowledgeRecord.index_status=="PENDING")
    rows = session.execute(statement).all()
    print("待写入Milvus {} 条".format(len(rows)))
    text_embeddings = embedder.embed_texts([knowledgerecord.content for knowledgerecord, document, knowledgebase in rows])

    milvus_records = []

    for index, (knowledgerecord, document, knowledgebase) in enumerate(rows):
        milvus_record = {
            "record_id": knowledgerecord.record_id,
            "tenant_id": knowledgebase.tenant_id,
            "knowledge_base_id": knowledgebase.knowledge_base_id,
            "document_id": document.document_id,
            "document_version": document_version.version,
            "record_version": knowledgerecord.version,
            "owner_id": document.owner_id,
            "visibility": document.visibility,
            "content": knowledgerecord.content,
            "attributes": knowledgerecord.attributes,
            "dense_vector": text_embeddings[index].tolist()
        }
        milvus_records.append(milvus_record)
    if milvus_records:
        upsert_result = milvus_storage.upsert_records(collection_name, milvus_records)
        print(upsert_result["cost"])

        if upsert_result["upsert_count"] != len(milvus_records):
            raise RuntimeError("Milvus写入数量不一致")

        for knowledgerecord, document, knowledgebase in rows:
            knowledgerecord.index_status = "READY"

        document_version.status = "READY"
        document.current_version = document_version.version
        session.commit()
