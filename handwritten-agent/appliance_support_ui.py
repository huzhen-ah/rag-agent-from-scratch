#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:07:00 2026

@author: huzhen
"""
#!/usr/bin/env python3

import json
import time
import uuid

import requests
import streamlit as st

from appliance_support_deepseek import build_agent
from tools import build_query_rag_tool



generic_system_prompt = """
你是可靠的企业知识库问答助手。

用户询问知识库内容时，必须调用 query_rag。
调用工具时，将用户问题改写成完整、独立的问题。

工具返回资料后：
1. 只依据工具返回的资料回答。
2. 不得编造资料中不存在的信息。
3. 资料不足时，明确说明知识库中没有足够信息。
4. 回答应简洁、准确，并保留关键条件和数字。
""".strip()


system_prompt = """
你是谨慎、可靠的家电故障排查助手。

你可以调用 query_rag 工具查询家电故障资料。

当用户询问故障代码、故障现象或处理方法，并且已经明确品牌、家电类型，以及故障代码或具体故障现象时，调用 query_rag。

调用 query_rag 时：
1. 将当前问题与对话历史中已经确认的信息合并。
2. 将问题改写成脱离对话历史也能理解的完整问题。
3. 不得猜测品牌、家电类型、型号、故障代码或用户已经执行的操作。
4. 当前用户明确修改的信息优先于较早的对话历史。

信息不足时，直接向用户追问，不要调用工具。

工具返回资料后：
1. 只依据工具返回的资料回答。
2. 优先使用与品牌、家电类型、故障代码和现象完全匹配的资料。
3. 不要把不相关候选中的故障原因或处理方法混入答案。
4. 资料无法确定时，明确说明无法确定。
5. 涉及拆机、电气、燃气、制冷剂或其他危险操作时，提醒用户停止自行处理并联系专业人员。
""".strip()


DEFAULT_USER_ID = "user-001"
DEFAULT_KNOWLEDGE_BASE_ID = "kb-fault-codes"
RAG_SERVICE_URL = "http://127.0.0.1:8080"
# 默认 rag_corpus.jsonl 的第一条完整记录：apdb-673ddaac7225。
DEFAULT_EXAMPLE_QUESTION = "Beko 英国版洗衣机出现 E01 故障，提示未检测到门已关闭并停止运行，应该怎么办？"


st.set_page_config(page_title="企业知识库 Agent", page_icon="🛠️", layout="wide")


@st.cache_resource
def load_agent(user_id="user-001", knowledge_base_id="kb-fault-codes", use_appliance_prompt=True):
    rag_tool = build_query_rag_tool(user_id, knowledge_base_id, RAG_SERVICE_URL + "/retrieve")

    if use_appliance_prompt:
        agent_system_prompt = system_prompt
    else:
        agent_system_prompt = generic_system_prompt
    return build_agent(tools=[rag_tool], agent_system_prompt=agent_system_prompt)


def parse_content(content):
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content


def build_trace_item(event, elapsed_seconds):
    if event.event_type != "update":
        return None

    node_name = event.payload["node_name"]
    update = event.payload["update"]

    if node_name == "model_node":
        messages = update.get("messages", [])

        if not messages:
            return None

        message = messages[-1]
        tool_calls = message.get("tool_calls", [])

        if tool_calls:
            return {
                "title": "模型决定调用工具",
                "elapsed": elapsed_seconds,
                "detail": tool_calls,
            }

        return {
            "title": "模型生成最终回复",
            "elapsed": elapsed_seconds,
            "detail": "本节点未生成新的 Tool Call，当前流程结束。",
        }

    if node_name == "tool_args_completion_node":
        return {
            "title": "工具参数检查完成",
            "elapsed": elapsed_seconds,
            "detail": "当前工具参数不需要人工补充。",
        }

    if node_name == "tool_review_node":
        return {
            "title": "工具调用审核完成",
            "elapsed": elapsed_seconds,
            "detail": "query_rag 为只读检索工具，直接执行。",
        }

    if node_name == "tool_node":
        messages = update.get("messages", [])

        if not messages:
            return None

        message = messages[-1]

        return {
            "title": "RAG混合检索完成",
            "elapsed": elapsed_seconds,
            "detail": {
                "pipeline": "Dense + BM25 → RRF → Reranker → Top-5",
                "documents": parse_content(message.get("content", "")),
            },
        }

    return None


def render_trace(trace_items):
    if not trace_items:
        st.info("发送问题后，这里会展示 Agent 的内部执行过程。")
        return

    for index, item in enumerate(trace_items, start=1):
        st.markdown(
            "#### {}. {} · {:.2f}s".format(
                index,
                item["title"],
                item["elapsed"],
            )
        )

        detail = item["detail"]

        if isinstance(detail, (dict, list)):
            st.json(detail)
        else:
            st.write(detail)


def clear_conversation():
    for key in ["agent_state", "thread_id", "chat_history", "trace_items"]:
        st.session_state.pop(key, None)


if "active_user_id" not in st.session_state:
    st.session_state.active_user_id = DEFAULT_USER_ID

if "active_knowledge_base_id" not in st.session_state:
    st.session_state.active_knowledge_base_id = DEFAULT_KNOWLEDGE_BASE_ID

if "use_appliance_prompt" not in st.session_state:
    st.session_state.use_appliance_prompt = True


with st.sidebar:
    st.header("知识库")
    with st.form("select_knowledge_base"):
        st.caption("不修改下面两个ID时，使用默认家电故障知识库。")
        selected_user_id = st.text_input("User ID", value=st.session_state.active_user_id)
        selected_knowledge_base_id = st.text_input("Knowledge Base ID", value=st.session_state.active_knowledge_base_id)
        select_submitted = st.form_submit_button("应用", use_container_width=True)

    if select_submitted:
        if not selected_user_id.strip() or not selected_knowledge_base_id.strip():
            st.error("User ID和Knowledge Base ID不能为空")
            st.stop()

        st.session_state.active_user_id = selected_user_id.strip()
        st.session_state.active_knowledge_base_id = selected_knowledge_base_id.strip()
        st.session_state.use_appliance_prompt = selected_user_id.strip() == DEFAULT_USER_ID and selected_knowledge_base_id.strip() == DEFAULT_KNOWLEDGE_BASE_ID
        st.session_state.pop("created_knowledge_base", None)
        clear_conversation()
        st.rerun()

    st.divider()
    st.subheader("创建知识库")
    with st.form("upload_knowledge_base", clear_on_submit=False):
        tenant_name = st.text_input("租户名称", max_chars=100)
        user_name = st.text_input("用户名称", max_chars=100)
        knowledge_base_name = st.text_input("知识库名称", max_chars=100)
        uploaded_file = st.file_uploader("上传JSONL文件", type=["jsonl"], max_upload_size=20)
        st.caption("UTF-8 编码，每行一个 JSON 对象；id 为非空且不重复的字符串，content 为非空文本。单文件最多 20 MiB、5,000 条记录，单条正文最多 16 KiB。")
        upload_submitted = st.form_submit_button("创建并上传", use_container_width=True)

    if upload_submitted:
        if not tenant_name.strip() or not user_name.strip() or not knowledge_base_name.strip() or uploaded_file is None:
            st.error("请完整填写名称并选择JSONL文件")
            st.stop()

        if uploaded_file.size == 0:
            st.error("上传失败：文件为空，请选择包含记录的 JSONL 文件")
            st.stop()
        if uploaded_file.size > 20 * 1024 * 1024:
            st.error("上传失败：文件大小不能超过 20 MiB")
            st.stop()

        try:
            with st.spinner("正在创建知识库并生成索引……"):
                response = requests.post(
                    RAG_SERVICE_URL + "/register_and_upload",
                    data={"tenant_name": tenant_name, "user_name": user_name, "knowledge_base_name": knowledge_base_name},
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/jsonl")},
                    timeout=1800
                )
                response.raise_for_status()
                created = response.json()
        except requests.RequestException as error:
            if error.response is not None:
                try:
                    detail = error.response.json().get("detail")
                except (ValueError, AttributeError):
                    detail = None
                if not isinstance(detail, str) or not detail:
                    detail = "服务未能完成上传，请检查文件或稍后重试"
            elif isinstance(error, requests.Timeout):
                detail = "等待上传结果超时，服务可能仍在建立索引，请稍后确认后再重试"
            else:
                detail = "无法连接检索服务，请确认 RAG 已启动后重试"
            st.error("上传失败：{}".format(detail))
            st.stop()

        st.session_state.active_user_id = created["user_id"]
        st.session_state.active_knowledge_base_id = created["knowledge_base_id"]
        st.session_state.use_appliance_prompt = False
        st.session_state.created_knowledge_base = created
        clear_conversation()
        st.rerun()

    if "created_knowledge_base" in st.session_state:
        created = st.session_state.created_knowledge_base
        st.success("知识库已创建，当前对话已自动切换")
        st.code("user_id: {}\nknowledge_base_id: {}".format(created["user_id"], created["knowledge_base_id"]))


with st.spinner("正在初始化 Agent……"):
    agent = load_agent(st.session_state.active_user_id, st.session_state.active_knowledge_base_id, st.session_state.use_appliance_prompt)


if "agent_state" not in st.session_state:
    st.session_state.agent_state = agent.create_initial_state()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "thread_{}".format(uuid.uuid4().hex)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "trace_items" not in st.session_state:
    st.session_state.trace_items = []


title_column, button_column = st.columns([5, 1])

with title_column:
    st.title("企业知识库 Agent")

with button_column:
    if st.button("新建会话", use_container_width=True):
        st.session_state.agent_state = agent.create_initial_state()
        st.session_state.thread_id = "thread_{}".format(uuid.uuid4().hex)
        st.session_state.chat_history = []
        st.session_state.trace_items = []
        st.rerun()


chat_column, trace_column = st.columns([3, 2], gap="large")


with chat_column:
    st.subheader("对话")

    example_input = None
    if st.session_state.active_user_id == DEFAULT_USER_ID and st.session_state.active_knowledge_base_id == DEFAULT_KNOWLEDGE_BASE_ID:
        st.caption("点击示例问题，体验家电故障问答：")
        if st.button(DEFAULT_EXAMPLE_QUESTION, key="default_kb_example", use_container_width=True):
            example_input = DEFAULT_EXAMPLE_QUESTION

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.use_appliance_prompt:
        input_placeholder = "请输入品牌、家电类型、故障代码或故障现象"
    else:
        input_placeholder = "请输入要查询的知识库问题"
    user_input = st.chat_input(input_placeholder)
    if example_input is not None:
        user_input = example_input


with trace_column:
    st.subheader("内部执行流程")
    trace_placeholder = st.empty()

    with trace_placeholder.container():
        render_trace(st.session_state.trace_items)


if user_input:
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    st.session_state.trace_items = []
    started_at = time.perf_counter()

    with chat_column:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            answer_placeholder.markdown("正在处理……")

    try:
        event_generator = agent.stream(
            user_input=user_input,
            agent_state=st.session_state.agent_state,
            thread_id=st.session_state.thread_id,
            checkpoint_ns="",
            checkpoint_id=None,
        )

        for event in event_generator:
            elapsed_seconds = time.perf_counter() - started_at

            trace_item = build_trace_item(
                event,
                elapsed_seconds,
            )

            if trace_item is not None:
                st.session_state.trace_items.append(trace_item)

                with trace_placeholder.container():
                    render_trace(st.session_state.trace_items)

            if event.event_type == "final":
                st.session_state.agent_state = event.payload["output"]

        answer = st.session_state.agent_state["messages"][-1]["content"]

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        answer_placeholder.markdown(answer)
        st.rerun()

    except Exception as error:
        answer_placeholder.error(str(error))
