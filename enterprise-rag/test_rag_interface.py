#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 11:17:23 2026

@author: huzhen
"""
import requests
import time

def query_rag(question: str) -> list[dict]:
    """
    从RAG知识库中检索与问题相关的参考资料。

    Args:
        question: 需要检索的问题。
    """

    ret = requests.post("http://127.0.0.1:8080/retrieve",json={"question":question, "user_id":"user-001", "knowledge_base_id":"kb-fault-codes"},timeout=120)
    ret.raise_for_status()
    ret = ret.json()

    return ret["documents"]

st = time.time()
response = query_rag("Beko洗衣机一直排不出去水，检查很久后还是停机，应该怎么办？")
print("用时： ",time.time()-st)
print(response)
