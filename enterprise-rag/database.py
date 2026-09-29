#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 09:18:59 2026

@author: huzhen
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


database_url = (
    "postgresql+psycopg://"
    "rag:rag_dev_password@"
    "127.0.0.1:5432/"
    "enterprise_rag"
)

engine = create_engine(database_url)

if __name__ == "__main__":


    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        print(result.scalar())
