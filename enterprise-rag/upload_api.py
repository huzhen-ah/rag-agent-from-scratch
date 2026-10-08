"""上传接口独立于模型加载，便于校验和接口回归测试。"""

import logging
import os
import shutil
import tempfile
from threading import Lock

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.responses import JSONResponse

from upload_validation import MAX_REQUEST_BYTES, MAX_UPLOAD_BYTES, UploadValidationError

logger = logging.getLogger(__name__)


class UploadBodyLimitMiddleware:
    """在 multipart 解析之前限制上传请求体，包含无 Content-Length 的请求。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"].rstrip("/") != "/register_and_upload":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            content_length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "请求的 Content-Length 无效"}, status_code=400)(scope, receive, send)
        if content_length < 0:
            return await JSONResponse({"detail": "请求的 Content-Length 无效"}, status_code=400)(scope, receive, send)
        if content_length > MAX_REQUEST_BYTES:
            return await JSONResponse({"detail": "上传请求过大，文件大小不能超过 20 MiB"}, status_code=413)(scope, receive, send)
        received_bytes = 0

        async def limited_receive():
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > MAX_REQUEST_BYTES:
                    raise HTTPException(status_code=413, detail="上传请求过大，文件大小不能超过 20 MiB")
            return message

        await self.app(scope, limited_receive, send)


def create_upload_router(document_service):
    router = APIRouter()
    upload_lock = Lock()

    @router.post("/register_and_upload")
    def register_and_upload(tenant_name: str = Form(...), user_name: str = Form(...), knowledge_base_name: str = Form(...), file: UploadFile = File(...)):
        temp_directory = None
        acquired = False
        try:
            for label, value in (("租户名称", tenant_name), ("用户名称", user_name), ("知识库名称", knowledge_base_name)):
                if not value.strip():
                    raise UploadValidationError("{}不能为空".format(label))
                if len(value.strip()) > 100:
                    raise UploadValidationError("{}不能超过 100 个字符".format(label))
                if "\x00" in value:
                    raise UploadValidationError("{}包含无效字符".format(label))
            file_name = os.path.basename((file.filename or "").replace("\\", "/"))
            if not file_name.lower().endswith(".jsonl"):
                raise UploadValidationError("只支持 JSONL 文件，每行一个包含 id 和 content 的 JSON 对象")
            try:
                file_name_bytes = len(file_name.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise UploadValidationError("文件名包含无效的 Unicode 字符，请重命名后重试") from error
            if len(file_name) > 255 or file_name_bytes > 255 or any(ord(char) < 32 for char in file_name):
                raise UploadValidationError("文件名过长或包含无效字符，请缩短文件名后重试")
            if file.size is not None and file.size > MAX_UPLOAD_BYTES:
                raise UploadValidationError("文件大小不能超过 20 MiB", 413)
            acquired = upload_lock.acquire(blocking=False)
            if not acquired:
                raise HTTPException(status_code=429, detail="当前有文件正在上传或建立索引，请稍后重试")
            temp_directory = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_directory, file_name)
            total_bytes = 0
            with open(temp_file_path, "wb") as target:
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UPLOAD_BYTES:
                        raise UploadValidationError("文件大小不能超过 20 MiB", 413)
                    target.write(chunk)
            # 全文件检查通过后才能创建租户、用户和知识库。
            document_service.load_jsonl(temp_file_path)
            tenant = document_service.create_tenant(tenant_name.strip())
            user = document_service.create_user(tenant.tenant_id, user_name.strip())
            knowledge_base = document_service.create_knowledge_base(tenant.tenant_id, knowledge_base_name.strip())
            document = document_service.create_document(user.user_id, knowledge_base.knowledge_base_id, temp_file_path)
            return {
                "tenant_id": tenant.tenant_id,
                "user_id": user.user_id,
                "knowledge_base_id": knowledge_base.knowledge_base_id,
                "document_id": document.document_id,
            }
        except UploadValidationError as error:
            raise HTTPException(status_code=error.status_code, detail=str(error)) from error
        except HTTPException:
            raise
        except Exception as error:
            logger.exception("知识库上传或索引处理失败")
            raise HTTPException(status_code=500, detail="上传处理失败，存储或索引服务暂时异常，请稍后重试") from error
        finally:
            try:
                file.file.close()
            finally:
                if temp_directory is not None:
                    shutil.rmtree(temp_directory, ignore_errors=True)
                if acquired:
                    upload_lock.release()

    return router
