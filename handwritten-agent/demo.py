#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 18:50:39 2026

@author: huzhen
"""

from model import LocalChatModel
from checkpoint import JsonlCheckpointer
from agent import Agent
from tool_register import Register
from tools import read_resume_tool
from multi_agent import CompiledSubAgent, TaskTool
from skill import load_skill_metadata, create_read_skill_tool
import uuid
from hitl import Command
import json



chat_model = LocalChatModel(
    model_path="models/Qwen3-4B",
    device="mps",
)

checkpointer = JsonlCheckpointer(
    local_checkpoint_file="checkpoints/job_match_checkpoints.jsonl",
    local_pending_writes_file="checkpoints/job_match_pending_writes.jsonl",
)


resume_register = Register()
resume_register.register(read_resume_tool)

tool_hitl_policy = {"read_resume" : ("args_completion", "review")}
system_prompt = (
    "你是简历证据提取Agent。"
    "必须先调用read_resume读取指定简历。"
    "然后针对输入中的每个requirement_id，从简历原文中提取证据。"
    "简历中没有明确证据时返回空列表，不得编造。"
)

resume_agent = Agent(
    chat_model = chat_model,
    register = resume_register,
    system_prompt = system_prompt,
    checkpointer = checkpointer,
    tool_hitl_policy = tool_hitl_policy,
    max_steps = 10
)

compiled_resume_agent = CompiledSubAgent(
    name = "resume_agent",
    description = "负责读取简历，并根据岗位招聘要求从简历中找出相关证据",
    runnable = resume_agent
)

tasktool = TaskTool([compiled_resume_agent])

skills = load_skill_metadata(r"skills")
read_skill_tool = create_read_skill_tool(skills)

supervisor_register = Register()
supervisor_register.register(tasktool)
supervisor_register.register(read_skill_tool)

supervisor_system_prompt = (
    "你是求职匹配Supervisor Agent。"
    "根据用户提供的简历和招聘要求，协调工具与子Agent完成分析。"
    "只能依据招聘要求原文和子Agent返回的简历证据作出判断，"
    "不得编造候选人经历；信息不足时必须明确说明。"
)

supervisor_agent = Agent(
    chat_model = chat_model,
    register = supervisor_register,
    system_prompt = supervisor_system_prompt,
    checkpointer = checkpointer,
    tool_hitl_policy = {"task" : ("args_completion", )},
    skills = skills,
    max_steps = 15
)

job_description = """
    岗位：Agent工程师

    岗位要求：
    1. 具备RAG或Agent项目经验。
    2. 理解Embedding、BM25、Reranker和检索评估。
    3. 熟悉LangGraph、工具调用、HITL和Multi-Agent。
    4. 具备Python和深度学习工程经验。
    5. 有智能客服或知识图谱经验者优先。
"""
user_input = (
    "请判断候选人是否适合投递下面的岗位.\n\n"
    "候选人的简历是main\n\n"
    "招聘要求:{}"
).format(job_description)

supervisor_state = supervisor_agent.create_initial_state()
thread_id = "thread_{}".format(uuid.uuid4().hex)

supervisor_state = supervisor_agent.invoke(
    user_input = user_input,
    agent_state = supervisor_state,
    thread_id = thread_id,
    checkpoint_ns = "",
    checkpoint_id = None
)

while "__interrupt__" in supervisor_state:
    resume_map = {}

    for interrupt_data in supervisor_state["__interrupt__"]:
        interrupt_request = json.dumps(interrupt_data.value,ensure_ascii=False,indent=4)
        print("interrupt_request: ",interrupt_request)
        resume_value_text = input("请输入resume_value(json): ")
        resume_value = json.loads(resume_value_text)
        resume_map[interrupt_data.id] = resume_value
    supervisor_state = supervisor_agent.invoke(
        user_input = Command(resume=resume_map),
        agent_state = supervisor_state,
        thread_id = thread_id,
        checkpoint_ns = "",
        checkpoint_id = None
    )
print("匹配报告: ",supervisor_state["messages"][-1]["content"])
