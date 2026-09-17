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

import streamlit as st

from appliance_support import build_agent


st.set_page_config(
    page_title="家电故障诊断 Agent",
    page_icon="🛠️",
    layout="wide",
)


@st.cache_resource
def load_agent():
    return build_agent()


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


with st.spinner("正在加载 Qwen3-8B 和 LoRA Adapter……"):
    agent = load_agent()


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
    st.title("家电故障诊断 Agent")

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

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input(
        "请输入品牌、家电类型、故障代码或故障现象"
    )


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