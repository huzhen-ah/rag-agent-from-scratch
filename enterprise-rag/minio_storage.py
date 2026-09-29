#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 18:25:11 2026

@author: huzhen
"""

from minio import Minio



class MinioStorage:
    def __init__(
            self,
            endpoint,
            access_key,
            secret_key
    ):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )

    def create_bucket(self, bucket_name):
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
            print("bucket: {} 创建成功".format(bucket_name))
        else:
            print("bucket: {} 已存在".format(bucket_name))

    def upload_file(self, bucket_name, object_name, file_path):
        try:
            result = self.client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=file_path
            )
            print("upload success!")
            return result
        except Exception as err:
            print("err: ",err)

    def download_file(self, bucket_name, object_name, file_path):
        try:
            self.client.fget_object(
                bucket_name = bucket_name,
                object_name = object_name,
                file_path = file_path
            )
            print("download success!")
        except Exception as err:
            print("err: ",err)

if __name__ == "__main__":
    endpoint = "127.0.0.1:9000"
    access_key = "minioadmin"
    secret_key = "minioadmin"
    bucket_name = "documents"
    object_name="tenant-001/kb-fault-codes/doc-rag-corpus/v1/rag_corpus.jsonl"
    file_path = r"documents/tenant-001/kb-fault-codes/doc-rag-corpus/v1/rag_corpus.jsonl"

    ministorage = MinioStorage(endpoint, access_key, secret_key)
    ministorage.create_bucket(bucket_name)
    ministorage.upload_file(bucket_name, object_name, file_path)
