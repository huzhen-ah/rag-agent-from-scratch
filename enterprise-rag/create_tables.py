#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 09:29:23 2026

@author: huzhen
"""

from database import Base, engine
from models import Tenant, KnowledgeBase, User, Document, DocumentVersion, KnowledgeRecord



Base.metadata.create_all(
    engine,
    tables=[
        Tenant.__table__,
        KnowledgeBase.__table__,
        User.__table__,
        Document.__table__,
        DocumentVersion.__table__,
        KnowledgeRecord.__table__
    ]
)

print("数据表创建完成")
