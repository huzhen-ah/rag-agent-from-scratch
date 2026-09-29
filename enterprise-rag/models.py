#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 09:26:20 2026

@author: huzhen
"""

from sqlalchemy import ForeignKey, String, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB

from database import Base



class Tenant(Base):
    __tablename__ = "tenants"

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    knowledge_base_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.tenant_id"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.tenant_id"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

class Document(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.knowledge_base_id"),
        nullable=False,
    )

    owner_id: Mapped[str] = mapped_column(
        ForeignKey("users.user_id"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    visibility: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="knowledge_base",
    )

    current_version: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

class DocumentVersion(Base):
    __tablename__ = "document_versions"

    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id"),
        primary_key=True,
    )

    version: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    bucket_name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    object_name: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="UPLOADED",
    )

class KnowledgeRecord(Base):
    __tablename__ = "knowledge_records"

    record_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id"),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    attributes: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
    )

    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    index_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="PENDING",
    )
