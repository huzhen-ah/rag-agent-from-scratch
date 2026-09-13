#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:55:15 2026

@author: huzhen
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from parser import parse_model_response
import uuid

class LocalChatModel:
    def __init__(self, model_path, device="mps"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            local_files_only=True, 
            dtype=torch.float16).to(device)

    def generate(self, messages, tool_definitions, max_new_tokens=800):
        model_inputs = self.tokenizer.apply_chat_template(
            messages, 
            tools=tool_definitions, 
            add_generation_prompt=True, 
            enable_thinking=False, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        model_inputs = model_inputs.to(self.model.device)
        outputs = self.model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False
        )

        input_length = model_inputs["input_ids"].shape[1]
        generated_ids = outputs[0, input_length:]

        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return response
    
    def parse_and_normalize_response(self,response):
        response = parse_model_response(response)
        for i in range(len(response["tool_calls"])):
            tool_call_id = uuid.uuid4().hex
            response["tool_calls"][i]["id"] = r"call_{}".format(tool_call_id)
            arguments = response["tool_calls"][i]["arguments"]
            del response["tool_calls"][i]["arguments"]
            response["tool_calls"][i]["args"] = arguments
        return response
    
    def to_qwen_format(self,messages):
        qwen_messages = []
        """
        role
        content
        toolcalls
        """
        for message in messages:
            qwen_message = {}
            qwen_message["role"] = message["role"]
            qwen_message["content"] = message["content"]
            if "tool_calls" in message:
                qwen_message["tool_calls"] = [{"name":tool_call["name"],"arguments":tool_call["args"]} for tool_call in message["tool_calls"]]
            qwen_messages.append(qwen_message)
        return qwen_messages
    
    def invoke(self,messages, tool_definitions, max_new_tokens=800):
        qwen_messages = self.to_qwen_format(messages)
        response = self.generate(qwen_messages,tool_definitions,max_new_tokens=max_new_tokens)
        response_parsed = self.parse_and_normalize_response(response)
        return response_parsed
    
if __name__ == "__main__":
    from tool_register import Register
    from tools import read_resume_tool

    register = Register()
    register.register(read_resume_tool)

    chat_model = LocalChatModel(model_path="models/Qwen3-4B", device="mps")

    messages = [
        {
            "role": "system",
            "content": "你是一个求职助手。需要外部信息时请调用工具，如果工具需要参数，但是你拿不到，就把工具列出来，参数先缺失。",
        },
        {
            "role": "user",
            "content": "请读取main的简历，告诉我求职方向。",
        },
    ]

    response = chat_model.invoke(messages=messages, tool_definitions=register.get_tool_definitions())

    print(response)
