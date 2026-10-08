#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:46:15 2026

@author: huzhen
"""

import os
import uuid

from sqlalchemy import select, delete
from sqlalchemy.orm import Session
from models import Tenant, User, KnowledgeBase, Document, DocumentVersion, KnowledgeRecord
from upload_validation import load_jsonl

class DocumentService:

    def __init__(self, engine, minio_storage, milvus_storage, embedder, collection_name, bucket_name):
        self.engine = engine
        self.minio_storage = minio_storage
        self.milvus_storage = milvus_storage
        self.embedder = embedder
        self.collection_name = collection_name
        self.bucket_name = bucket_name

    def load_jsonl(self, file_path):
        return load_jsonl(file_path)

    def search_tenant(self, tenant_id):
        with Session(self.engine) as session:
            tenant = session.get(Tenant, tenant_id)
            return tenant

    def create_tenant(self, tenant_name):
        tenant_id = "tenant-{}".format(uuid.uuid4().hex)
        with Session(self.engine, expire_on_commit=False) as session:
            tenant = Tenant(tenant_id = tenant_id, name = tenant_name)
            session.add(tenant)
            session.commit()
            return tenant

    def search_user(self, user_id):
        with Session(self.engine) as session:
            user = session.get(User, user_id)
            return user

    def create_user(self, tenant_id, user_name):
        with Session(self.engine, expire_on_commit=False) as session:
            user = User(user_id = "user-{}".format(uuid.uuid4().hex), tenant_id = tenant_id, name = user_name)
            session.add(user)
            session.commit()
            return user

    def search_knowledge_base(self, knowledge_base_id):
        with Session(self.engine) as session:
            return session.get(KnowledgeBase, knowledge_base_id)

    def create_knowledge_base(self, tenant_id, knowledge_base_name):
        with Session(self.engine, expire_on_commit=False) as session:
            knowledge_base = KnowledgeBase(knowledge_base_id = "kb-{}".format(uuid.uuid4().hex), tenant_id = tenant_id, name = knowledge_base_name)
            session.add(knowledge_base)
            session.commit()
            return knowledge_base

    def add_document_metadata(self, session, metadata):
        session.add(metadata)
        session.flush()

    def add_knowledge_records(self, session, document_id, version, records):
        for data in records:
            record = KnowledgeRecord(
                record_id = uuid.uuid5(uuid.NAMESPACE_URL, "{}:{}".format(document_id, data["record_id"])).hex,
                document_id = document_id,
                content = data["content"],
                attributes = data["attributes"],
                version = version
            )
            session.add(record)

    def create_document_in_postgresql(self, document, document_version, records):
        with Session(self.engine, expire_on_commit=False) as session:
            self.add_document_metadata(session, document)
            self.add_document_metadata(session, document_version)
            self.add_knowledge_records(session, document.document_id, document_version.version, records)
            session.commit()


    def search_pending_records(self, document_id, version):

        with Session(self.engine, expire_on_commit=False) as session:
            statement = select(KnowledgeRecord).where(KnowledgeRecord.document_id == document_id, KnowledgeRecord.version == version, KnowledgeRecord.index_status=="PENDING")
            rows = session.scalars(statement).all()
            print("待写入Milvus {} 条".format(len(rows)))
            return rows

    def mark_knowledge_records_ready(self, session, document_id, version):
        statement = select(KnowledgeRecord).where(KnowledgeRecord.document_id == document_id, KnowledgeRecord.version == version, KnowledgeRecord.index_status=="PENDING")
        rows = session.scalars(statement).all()
        for knowledgerecord in rows:
            knowledgerecord.index_status = "READY"

    def mark_document_version_ready(self, session, document_id, version):
        document_version = session.get(DocumentVersion, {"document_id":document_id, "version":version})
        document_version.status = "READY"

    def update_document_current_version(self, session, document_id, version):
        document = session.get(Document, document_id)
        document.current_version = version

    def finalize_document_indexing(self, document_id, version):
        with Session(self.engine,expire_on_commit=False) as session:
            self.mark_knowledge_records_ready(session, document_id, version)
            self.mark_document_version_ready(session, document_id, version)
            self.update_document_current_version(session, document_id, version)
            session.commit()

    def search_document(self, document_id):
        with Session(self.engine,expire_on_commit=False) as session:
            document = session.get(Document, document_id)
            return document

    def replace_document_records_in_postgresql(self, document_version, records):
        with Session(self.engine, expire_on_commit=False) as session:
            statement = select(KnowledgeRecord.record_id).where(KnowledgeRecord.document_id == document_version.document_id)
            old_record_ids = session.scalars(statement).all()

            statement = delete(KnowledgeRecord).where(KnowledgeRecord.document_id == document_version.document_id)
            session.execute(statement)

            self.add_document_metadata(session, document_version)
            self.add_knowledge_records(session, document_version.document_id, document_version.version, records)
            session.commit()
            return old_record_ids

    def create_document(self, user_id, knowledge_base_id, file_path, visibility = "knowledge_base"):

        if not isinstance(user_id, str):
            raise TypeError("user_id必须是str类型")

        if not isinstance(knowledge_base_id, str):
            raise TypeError("knowledge_base_id必须是str类型")

        if not os.path.isfile(file_path):
            raise ValueError("file_path: {} 不存在".format(file_path))

        user = self.search_user(user_id)
        if user is None:
            raise ValueError("user: {} 不存在".format(user_id))


        tenant_id = user.tenant_id


        knowledge_base = self.search_knowledge_base(knowledge_base_id)
        if knowledge_base is None:
            raise ValueError("knowledge_base: {} 不存在".format(knowledge_base_id))


        if knowledge_base.tenant_id != tenant_id:
            raise ValueError("知识库: {} 不属于租户: {}".format(knowledge_base_id, tenant_id))

        records = self.load_jsonl(file_path)


        file_name = os.path.basename(file_path)
        # Milvus 字符串限制按 UTF-8 字节计算，中文文件名不能直接拼入主键。
        document_id = "doc-{}".format(uuid.uuid4().hex)
        object_name = f"{tenant_id}/{knowledge_base_id}/{document_id}/v1/{file_name}"
        #先上传到minio
        #object_name sample: "tenant-001/kb-fault-codes/doc-rag-corpus/v1/rag_corpus.jsonl"
        self.minio_storage.create_bucket(self.bucket_name)
        upload_result = self.minio_storage.upload_file(self.bucket_name, object_name, file_path)
        if upload_result is None:
            raise RuntimeError("MinIO文件上传失败")
        document = Document(
            document_id = document_id,
            knowledge_base_id = knowledge_base_id,
            owner_id = user_id,
            name = file_name,
            visibility = visibility
        )

        document_version = DocumentVersion(
            document_id = document_id,
            version = 1,
            bucket_name = self.bucket_name,
            object_name = object_name,
            status = "UPLOADED"
        )

        self.create_document_in_postgresql(document, document_version, records)

        self.generate_embedding_and_upsert_to_milvus(user, document, document_version)

        self.finalize_document_indexing(document_id, document_version.version)

        return self.search_document(document_id)

    def generate_embedding_and_upsert_to_milvus(self, user, document, document_version):
        pending_records = self.search_pending_records(document.document_id, document_version.version)
        text_embeddings = self.embedder.embed_texts([knowledgerecord.content for knowledgerecord in pending_records])
        milvus_records = []
        for index, knowledgerecord in enumerate(pending_records):
            milvus_record = {
                "record_id": knowledgerecord.record_id,
                "tenant_id": user.tenant_id,
                "knowledge_base_id": document.knowledge_base_id,
                "document_id": document.document_id,
                "document_version": document_version.version,
                "record_version": knowledgerecord.version,
                "owner_id": user.user_id,
                "visibility": document.visibility,
                "content": knowledgerecord.content,
                "attributes": knowledgerecord.attributes,
                "dense_vector": text_embeddings[index].tolist()
            }
            milvus_records.append(milvus_record)

        if milvus_records:
            upsert_result = self.milvus_storage.upsert_records(self.collection_name, milvus_records)
            print(upsert_result["cost"])

            if upsert_result["upsert_count"] != len(milvus_records):
                raise RuntimeError("Milvus写入数量不一致")

    def update_document(self, user_id, document_id, file_path):
        if not os.path.isfile(file_path):
            raise ValueError("file_path: {} 不存在".format(file_path))

        user = self.search_user(user_id)
        if user is None:
            raise ValueError("user: {} 不存在".format(user_id))

        document = self.search_document(document_id)
        if document is None:
            raise ValueError("document: {} 不存在".format(document_id))

        if document.owner_id != user_id:
            raise PermissionError("当前用户: {} 无权更新该文档".format(user_id))

        if document.current_version is None:
            raise RuntimeError("当前文档尚未完成索引")

        new_version = document.current_version + 1

        records = self.load_jsonl(file_path)

        file_name = os.path.basename(file_path)
        object_name = f"{user.tenant_id}/{document.knowledge_base_id}/{document_id}/v{new_version}/{file_name}"
        #先上传到minio
        #object_name sample: "tenant-001/kb-fault-codes/doc-rag-corpus/v2/rag_corpus.jsonl"
        upload_result = self.minio_storage.upload_file(self.bucket_name, object_name, file_path)
        if upload_result is None:
            raise RuntimeError("MinIO文件上传失败")

        document_version = DocumentVersion(
            document_id = document_id,
            version = new_version,
            bucket_name = self.bucket_name,
            object_name = object_name,
            status = "UPLOADED"
        )

        old_record_ids = self.replace_document_records_in_postgresql(document_version, records)
        self.milvus_storage.delete_records(self.collection_name, old_record_ids)

        self.generate_embedding_and_upsert_to_milvus(user, document, document_version)
        self.finalize_document_indexing(document_id, document_version.version)
        return self.search_document(document_id)
