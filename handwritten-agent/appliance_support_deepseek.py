#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:00:52 2026

@author: huzhen
"""

import os
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
# 默认 MPS 高水位限制装不下完整的 8B FP16 权重；2.0 比完全禁用限制更稳妥。
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "2.0"
# 8B 模型直接加载到 MPS 时，Transformers 的多线程加载会造成瞬时内存峰值。


from agent import Agent
from checkpoint import InMemoryCheckpointer
from model import DeepSeekModel
from tool_register import Register



def build_agent(tools, agent_system_prompt):
    register = Register()
    for tool in tools:
        register.register(tool)

    chat_model = DeepSeekModel()

    agent = Agent(
        chat_model = chat_model,
        register = register,
        system_prompt = agent_system_prompt,
        checkpointer = InMemoryCheckpointer(),
        tool_hitl_policy = {},
        max_steps = 6
    )
    return agent
