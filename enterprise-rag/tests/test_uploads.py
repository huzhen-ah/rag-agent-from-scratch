"""上传边界和请求失败后的恢复测试，不加载模型或连接数据库。"""

import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient
import upload_api
import upload_validation
from upload_api import UploadBodyLimitMiddleware, create_upload_router
from upload_validation import UploadValidationError, load_jsonl


def record(content="正常文本", record_id="r1", **extras):
    return json.dumps(dict(id=record_id, content=content, **extras), ensure_ascii=False).encode("utf-8") + b"\n"


class ValidationTests(unittest.TestCase):
    def parse(self, payload):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_bytes(payload)
            return load_jsonl(path)

    def test_valid_unicode_bom_blank_lines_and_attributes(self):
        result = self.parse(b"\xef\xbb\xbf" + record(source={"name": "资料"}) + b"\n  \n" + record(record_id="r2"))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["content"], "正常文本")
        self.assertEqual(result[0]["attributes"], {"source": {"name": "资料"}})

    def test_invalid_records_are_rejected(self):
        cases = [
            (b"", "文件中无记录"), (b" \n\n", "文件中无记录"),
            (b"\xff\n", "第 1 行编码错误"), (b"{oops}\n", "第 1 行第"),
            (b"[]\n", "必须是 JSON 对象"), (b"null\n", "必须是 JSON 对象"),
            (b'"text"\n', "必须是 JSON 对象"), (b'{"id":"r1"}\n', "缺少 id 或 content"),
            (record(record_id=1), "id 必须是非空字符串"),
            (record(record_id=None), "id 必须是非空字符串"),
            (record(record_id=[]), "id 必须是非空字符串"),
            (record(record_id="  "), "id 必须是非空字符串"),
            (record(record_id="a" * 257), "id 不能超过"),
            (record() + record(), "第 2 行的 id"),
            (record(content=""), "content 必须是非空文本"),
            (record(content=" \n\t"), "content 必须是非空文本"),
            (record(content=None), "content 必须是非空文本"),
            (record(content=42), "content 必须是非空文本"),
            (record(content={}), "content 必须是非空文本"),
            (record(content="a" * (16 * 1024 + 1)), "content 超过 16 KiB"),
            (record(content="中" * 5462), "content 超过 16 KiB"),
            (b'{"id":"r1","content":"ok","score":NaN}\n', "不允许 NaN"),
            (b'{"id":"r1","content":"ok","score":1e999}\n', "数值超出范围"),
            (b'{"id":"r1","id":"r2","content":"ok"}\n', "重复字段"),
            (b'{"id":"r1","content":"\\ud800"}\n', "无效的 Unicode"),
            (b'{"id":"r1","content":"ok","extra":"\\ud800"}\n', "无效的 Unicode"),
            (record(content="text\x00"), "不能包含空字符"),
            (record(extra={"invalid\x00": "value"}), "不能包含空字符"),
            (b"[" * 2000 + b"0" + b"]" * 2000, "嵌套过深"),
            (record(extra="a" * (64 * 1024)), "第 1 行超过 64 KiB"),
        ]
        for payload, message in cases:
            with self.subTest(message=message, payload=payload[:80]):
                with self.assertRaisesRegex(UploadValidationError, message):
                    self.parse(payload)

    def test_exact_content_limit_is_accepted(self):
        self.assertEqual(len(self.parse(record(content="a" * (16 * 1024)))[0]["content"]), 16 * 1024)

    def test_record_count_limit(self):
        with patch.object(upload_validation, "MAX_RECORDS", 2):
            self.assertEqual(len(self.parse(record() + record(record_id="r2"))), 2)
            with self.assertRaises(UploadValidationError) as caught:
                self.parse(record() + record(record_id="r2") + record(record_id="r3"))
            self.assertEqual(caught.exception.status_code, 413)

    def test_file_size_limit(self):
        payload = record()
        with patch.object(upload_validation, "MAX_UPLOAD_BYTES", len(payload)):
            self.assertEqual(len(self.parse(payload)), 1)
            with self.assertRaises(UploadValidationError) as caught:
                self.parse(payload + b"\n")
            self.assertEqual(caught.exception.status_code, 413)

    def test_existing_corpus_still_validates(self):
        project = Path(__file__).resolve().parents[1]
        files = [project / "evaluation/source/rag_documents.jsonl",
                 project / "documents/tenant-001/kb-fault-codes/doc-rag-corpus/v1/rag_corpus.jsonl"]
        for path in files:
            with self.subTest(path=path):
                if path.exists():
                    self.assertGreater(len(load_jsonl(path)), 0)


class FakeDocumentService:
    def __init__(self):
        self.writes = []
        self.paths = []
        self.fail_index = False
        self.entered = None
        self.release = None

    def load_jsonl(self, path):
        self.paths.append(path)
        return load_jsonl(path)

    def create_tenant(self, name):
        self.writes.append(("tenant", name))
        return SimpleNamespace(tenant_id="tenant-test")

    def create_user(self, tenant_id, name):
        self.writes.append(("user", name))
        return SimpleNamespace(user_id="user-test")

    def create_knowledge_base(self, tenant_id, name):
        self.writes.append(("knowledge_base", name))
        return SimpleNamespace(knowledge_base_id="kb-test")

    def create_document(self, user_id, knowledge_base_id, path):
        if self.entered is not None:
            self.entered.set()
            if not self.release.wait(5):
                raise RuntimeError("test timed out")
        if self.fail_index:
            raise RuntimeError("internal storage details must stay in server logs")
        return SimpleNamespace(document_id="doc-test")


class UploadApiTests(unittest.TestCase):
    def setUp(self):
        self.service = FakeDocumentService()
        self.app = FastAPI()
        self.app.add_middleware(UploadBodyLimitMiddleware)
        self.router = create_upload_router(self.service)
        self.app.include_router(self.router)
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def post(self, payload=None, filename="input.jsonl", **fields):
        names = dict(tenant_name="test tenant", user_name="test user", knowledge_base_name="test kb")
        names.update(fields)
        return self.client.post("/register_and_upload", data=names,
                                files={"file": (filename, record() if payload is None else payload, "application/jsonl")})

    def assert_cleaned(self):
        for path in self.service.paths:
            self.assertFalse(Path(path).parent.exists())

    def test_bad_uploads_do_not_write_and_next_upload_succeeds(self):
        for payload in (b"", b"{bad}\n", record(content=None), record(record_id=1), record() + record()):
            with self.subTest(payload=payload[:80]):
                response = self.post(payload)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(self.service.writes, [])
                self.assertEqual(self.client.get("/openapi.json").status_code, 200)
                self.assert_cleaned()
        self.assertEqual(self.post().status_code, 200)
        self.assert_cleaned()

    def test_names_and_filenames(self):
        for fields in ({"tenant_name": " "}, {"user_name": "x" * 101}, {"knowledge_base_name": "x" * 101}, {"tenant_name": "name\x00"}):
            self.assertEqual(self.post(**fields).status_code, 400)
        for name in ("input.txt", "a" * 256 + ".jsonl", "中" * 84 + ".jsonl"):
            self.assertEqual(self.post(filename=name).status_code, 400)
        self.assertEqual(self.service.writes, [])
        self.assertEqual(self.post(filename="使用说明.jsonl").status_code, 200)

    def test_file_size_is_checked_after_multipart_parsing(self):
        with patch.object(upload_api, "MAX_UPLOAD_BYTES", 100):
            self.assertEqual(self.post(b"x" * 101).status_code, 413)
        self.assertEqual(self.service.writes, [])
        self.assertEqual(self.post().status_code, 200)

    def test_unknown_file_size_is_bounded_and_file_closed(self):
        file = UploadFile(filename="input.jsonl", file=io.BytesIO(b"x" * 101))
        with patch.object(upload_api, "MAX_UPLOAD_BYTES", 100):
            with self.assertRaises(HTTPException) as caught:
                self.router.routes[0].endpoint(tenant_name="t", user_name="u", knowledge_base_name="k", file=file)
        self.assertEqual(caught.exception.status_code, 413)
        self.assertTrue(file.file.closed)
        self.assertEqual(self.service.writes, [])
        self.assertEqual(self.post().status_code, 200)

    def test_content_length_rejected_before_file_parsing(self):
        with patch.object(upload_api, "MAX_REQUEST_BYTES", 100):
            response = self.client.post("/register_and_upload", content=b"", headers={"content-length": "101"})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.service.paths, [])
        self.assertEqual(self.post().status_code, 200)

    def test_missing_or_forged_content_length_cannot_bypass_limit(self):
        for headers in ({}, {"content-length": "1"}):
            with self.subTest(headers=headers), patch.object(upload_api, "MAX_REQUEST_BYTES", 100):
                response = self.client.post("/register_and_upload", content=iter([b"x" * 80, b"y" * 80]),
                                            headers={"content-type": "multipart/form-data; boundary=test", **headers})
                self.assertEqual(response.status_code, 413, response.text)
        self.assertEqual(self.service.writes, [])
        self.assertEqual(self.post().status_code, 200)

    def test_storage_failure_returns_safe_error_and_releases_lock(self):
        self.service.fail_index = True
        with self.assertLogs("upload_api", level="ERROR"):
            response = self.post()
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("internal storage details", response.text)
        self.assert_cleaned()
        self.service.fail_index = False
        self.assertEqual(self.post().status_code, 200)

    def test_concurrent_upload_is_rejected_then_recovers(self):
        self.service.entered, self.service.release = Event(), Event()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.post)
            try:
                self.assertTrue(self.service.entered.wait(5))
                response = self.post()
                self.assertEqual(response.status_code, 429, response.text)
            finally:
                self.service.release.set()
            self.assertEqual(future.result(timeout=5).status_code, 200)
        self.assertEqual(self.post().status_code, 200)
        self.assert_cleaned()

    def test_chinese_filename_document_id_fits_milvus(self):
        from document_service import DocumentService
        minio = Mock()
        service = DocumentService(None, minio, None, None, "records", "documents")
        service.search_user = Mock(return_value=SimpleNamespace(user_id="u", tenant_id="t"))
        service.search_knowledge_base = Mock(return_value=SimpleNamespace(tenant_id="t"))
        service.create_document_in_postgresql = Mock()
        service.generate_embedding_and_upsert_to_milvus = Mock()
        service.finalize_document_indexing = Mock()
        service.search_document = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ("中文" * 20 + ".jsonl")
            path.write_bytes(record())
            service.create_document("u", "k", str(path))
        document = service.create_document_in_postgresql.call_args.args[0]
        self.assertLessEqual(len(document.document_id.encode("utf-8")), 64)
        self.assertTrue(document.name.startswith("中文"))

    def test_rag_service_wires_upload_protection_without_loading_models(self):
        path = Path(__file__).resolve().parents[1] / "rag_service.py"
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                    backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
                                    device=lambda name: name)
        modules = {
            "torch": fake_torch,
            "rag": SimpleNamespace(RAG=lambda *args: SimpleNamespace(milvus_storage=None, embedder=None, collection_name="records")),
            "embedding": SimpleNamespace(Embedder=lambda *args: None),
            "reranker": SimpleNamespace(Reranker=lambda *args: None),
            "milvus_storage": SimpleNamespace(MilvusStorage=lambda *args: None),
            "minio_storage": SimpleNamespace(MinioStorage=lambda *args: None),
        }
        spec = importlib.util.spec_from_file_location("rag_service_upload_test", path)
        service_module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(service_module)
        client = TestClient(service_module.app)
        response = client.post("/register_and_upload", data=dict(tenant_name="t", user_name="u", knowledge_base_name="k"),
                               files={"file": ("input.jsonl", record(content=None), "application/jsonl")})
        self.assertEqual(response.status_code, 400)
        self.assertIn("content 必须是非空文本", response.json()["detail"])
        with patch.object(upload_api, "MAX_REQUEST_BYTES", 100):
            response = client.post("/register_and_upload", content=b"", headers={"content-length": "101"})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(client.get("/openapi.json").status_code, 200)


if __name__ == "__main__":
    unittest.main()
