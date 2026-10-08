"""JSONL 校验与上传限制；在任何存储写入之前调用。"""

import json
import math
import os

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_RECORDS = 5000
MAX_LINE_BYTES = 64 * 1024
MAX_CONTENT_BYTES = 16 * 1024
MAX_ID_LENGTH = 256
# multipart 的表单字段、文件名和边界也占用请求体空间。
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024


class UploadValidationError(ValueError):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def _reject_constant(value):
    raise ValueError("不允许 NaN 或 Infinity")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON 对象包含重复字段：{}".format(key))
        result[key] = value
    return result


def _check_json_values(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("数值超出范围，不允许 NaN 或 Infinity")
    if isinstance(value, str) and "\x00" in value:
        raise ValueError("文本或字段名不能包含空字符（U+0000）")
    if isinstance(value, dict):
        for key, item in value.items():
            _check_json_values(key)
            _check_json_values(item)
    elif isinstance(value, list):
        for item in value:
            _check_json_values(item)


def load_jsonl(file_path):
    if os.path.getsize(file_path) > MAX_UPLOAD_BYTES:
        raise UploadValidationError("文件大小不能超过 20 MiB", 413)
    records, record_ids = [], set()
    total_bytes, line_num = 0, 0
    with open(file_path, "rb") as source:
        while True:
            raw_line = source.readline(MAX_LINE_BYTES + 1)
            if not raw_line:
                break
            line_num += 1
            total_bytes += len(raw_line)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise UploadValidationError("文件大小不能超过 20 MiB", 413)
            if len(raw_line) > MAX_LINE_BYTES:
                raise UploadValidationError("第 {} 行超过 64 KiB，请拆分记录或减少附加字段".format(line_num))
            try:
                line = raw_line.decode("utf-8-sig" if line_num == 1 else "utf-8").strip()
            except UnicodeDecodeError as error:
                raise UploadValidationError("第 {} 行编码错误，请使用 UTF-8 编码".format(line_num)) from error
            if not line:
                continue
            if len(records) >= MAX_RECORDS:
                raise UploadValidationError("记录数不能超过 5,000 条", 413)
            try:
                data = json.loads(line, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
                _check_json_values(data)
            except json.JSONDecodeError as error:
                raise UploadValidationError("第 {} 行第 {} 列 JSON 格式错误：{}".format(line_num, error.colno, error.msg)) from error
            except RecursionError as error:
                raise UploadValidationError("第 {} 行 JSON 嵌套过深".format(line_num)) from error
            except ValueError as error:
                raise UploadValidationError("第 {} 行：{}".format(line_num, error)) from error
            if not isinstance(data, dict):
                raise UploadValidationError("第 {} 行必须是 JSON 对象，每行一条记录".format(line_num))
            if "id" not in data or "content" not in data:
                raise UploadValidationError("第 {} 行缺少 id 或 content 字段".format(line_num))
            record_id, content = data["id"], data["content"]
            if not isinstance(record_id, str) or not record_id.strip():
                raise UploadValidationError("第 {} 行的 id 必须是非空字符串".format(line_num))
            if len(record_id) > MAX_ID_LENGTH:
                raise UploadValidationError("第 {} 行的 id 不能超过 256 个字符".format(line_num))
            if record_id in record_ids:
                raise UploadValidationError("第 {} 行的 id 与前面的记录重复".format(line_num))
            if not isinstance(content, str) or not content.strip():
                raise UploadValidationError("第 {} 行的 content 必须是非空文本".format(line_num))
            try:
                content_bytes = len(content.encode("utf-8"))
                # 包括 id 和附加字段，拦截 JSON 中转义出的孤立代理字符。
                json.dumps(data, ensure_ascii=False).encode("utf-8")
            except UnicodeEncodeError as error:
                raise UploadValidationError("第 {} 行包含无效的 Unicode 字符".format(line_num)) from error
            if content_bytes > MAX_CONTENT_BYTES:
                raise UploadValidationError("第 {} 行的 content 超过 16 KiB，请拆成多条记录".format(line_num))
            record_ids.add(record_id)
            records.append({
                "record_id": record_id,
                "content": content,
                "attributes": {key: value for key, value in data.items() if key not in ("id", "content")},
            })
    if not records:
        raise UploadValidationError("文件中无记录，请上传包含 id 和 content 的 JSONL 文件")
    return records
