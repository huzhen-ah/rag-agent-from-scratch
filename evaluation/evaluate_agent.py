#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:00:52 2026

@author: huzhen
"""
import sys
sys.path.append("../handwritten-agent")
import os
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
# 默认 MPS 高水位限制装不下完整的 8B FP16 权重；2.0 比完全禁用限制更稳妥。
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "2.0"
# 8B 模型直接加载到 MPS 时，Transformers 的多线程加载会造成瞬时内存峰值。


import uuid
from agent import Agent
from checkpoint import InMemoryCheckpointer
from model import PeftChatModel
from tool_register import Register
from tools import query_rag_tool
import torch
import json
from tqdm import tqdm



if torch.cuda.is_available():
        device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


model_path = "../../appliance-support-sft/models/Qwen3-8B"
adapter_path = "../../appliance-support-sft/outputs/qwen3-8b-appliance-lora/final_adapter"
chat_template_path = "../../appliance-support-sft/qwen3_training.jinja"


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


def build_agent():
    register = Register()
    register.register(query_rag_tool)

    chat_model = PeftChatModel(model_path, adapter_path, chat_template_path, device=device)

    agent = Agent(
        chat_model = chat_model,
        register = register,
        system_prompt = system_prompt,
        checkpointer = InMemoryCheckpointer(),
        tool_hitl_policy = {},
        max_steps = 6
    )
    return agent


def load_jsonl(path):
    ret = []
    with open(path,"r",encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ret.append(json.loads(line))
    return ret

def evaluate(agent,evaluate_data):
    correct = 0
    total = 0

    for d in tqdm(iter(evaluate_data),desc="...评测中..."):
        total += 1
        thread_id = "thread_{}".format(uuid.uuid4().hex)
        question = d["user"]
        agent_state = agent.invoke(
            user_input=question,
            agent_state=agent.create_initial_state(),
            thread_id=thread_id,
            checkpoint_ns="",
            checkpoint_id=None,
        )

        for message in reversed(agent_state["messages"]):
            if message["role"] == "assistant":
                tool_calls = message.get("tool_calls",[])
                if len(tool_calls) == 0:
                    continue
                if tool_calls[0]["name"] == "query_rag":
                    correct += 1
                break
    acc = correct / total
    return acc



if __name__ == "__main__":
    evaluate_agent_data_path = r"data/appliance_agent_eval.jsonl"
    agent = build_agent()
    evaluate_data = load_jsonl(evaluate_agent_data_path)
    acc = evaluate(agent, evaluate_data)
    print("acc: ",acc)
