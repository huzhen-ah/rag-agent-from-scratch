#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 09:51:32 2026

@author: huzhen
"""

from dataclasses import dataclass
import os
import fitz

@dataclass
class Document:
    content:str
    source:str
    page:int | None = None
    

@dataclass
class Chunk:
    chunk_id:int
    content:str
    source:str
    start:int
    end:int
    page:int | None = None

def load_text(file):
    
    with open(file,"r",encoding="utf8") as f:
        content = f.read()
    document = Document(content, file)
    
    return [document]

def load_pdf(file):
    documents = []
    with fitz.open(file) as pdf:
        for page_num,page in enumerate(pdf,start=1):
            content = page.get_text("text")
            if content.strip():
                documents.append(Document(content, file,page=page_num))
    return documents


def load_documents(file_paths):
    documents = []
    for file in file_paths:
        file = os.path.abspath(file)
        suffix = os.path.splitext(file)[1].lower()
        if suffix in [".txt",".md"]:
            _ = load_text(file)
            documents.extend(_)
        elif suffix == ".pdf":
            _ = load_pdf(file)
            documents.extend(_)
    return documents

def cut_a_document(document,chunk_id,chunk_size=500,chunk_overlap=100):
    assert chunk_size > 0
    assert chunk_overlap >= 0
    assert chunk_size > chunk_overlap
    
    chunks = []
    content = document.content
    source = document.source
    page = document.page
    for i in range(0,len(content),chunk_size-chunk_overlap):
        chunk_content = content[i:i+chunk_size]
        start = i
        end = start + len(chunk_content)
        chunk = Chunk(chunk_id, chunk_content, source, start, end, page)
        chunks.append(chunk)
        if end == len(content):
            break
        chunk_id += 1
    return chunks

def cut_documents(documents, chunk_size=500, chunk_overlap=100):
    chunks = []
    
    for document in documents:
        chunk_id = len(chunks)
        _ = cut_a_document(document, chunk_id, chunk_size, chunk_overlap)
        chunks.extend(_)
    return chunks
        
        

if __name__ == "__main__":
    file_paths = [r"documents/THREE_WEEK_PLAN.md"]
    documents = load_documents(file_paths)
    print(documents[0].page)
    chunks = cut_documents(documents)
    print(chunks[1])
