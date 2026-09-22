#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:55:15 2026

@author: huzhen
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from parser import parse_model_response
import uuid
from openai import OpenAI
import os
import json


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
        input_device = self.model.get_input_embeddings().weight.device
        model_inputs = model_inputs.to(input_device)
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
    
class PeftChatModel(LocalChatModel):
    def __init__(
        self,
        model_path,
        adapter_path,
        chat_template_path,
        device="mps"
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.chat_template_path = chat_template_path
        self.device = device

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        with open(self.chat_template_path, "r", encoding="utf-8") as file:
            self.tokenizer.chat_template = file.read()

        if self.device == "cuda":
            device_map = "auto"
        else:
            device_map = {"": self.device}

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            dtype=torch.bfloat16,
            device_map=device_map,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            self.adapter_path,
            local_files_only=True,
        )
        self.model.eval()

class DeepSeekModel:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )

    def generate(self, messages, tool_definitions, max_new_tokens=800):

        response = self.client.chat.completions.create(
            model="deepseek-flash",
            messages=messages,
            tools=tool_definitions,
            tool_choice="auto",
            stream=False,
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            },
        )

        response = response.choices[0].message


        return response

    def parse_and_normalize_response(self,message):
        new_response = {"content":message.content or "", "tool_calls":[]}
        for tool_call in message.tool_calls or []:

            tool_call = tool_call.model_dump()
            new_tool_call = {}
            new_tool_call["id"] = tool_call["id"]
            new_tool_call["name"] = tool_call["function"]["name"]
            new_tool_call["args"] = json.loads(tool_call["function"]["arguments"])
            new_tool_call["type"] = "tool_call"
            new_response["tool_calls"].append(new_tool_call)

        return new_response

    def to_deepseek_format(self,messages):
        deepseek_messages = []
        """
        role
        content
        toolcalls
        """
        for message in messages:
            deepseek_message = {}
            deepseek_message["role"] = message["role"]
            deepseek_message["content"] = message["content"]
            if "tool_calls" in message and message["tool_calls"]:
                deepseek_message["tool_calls"] = [{"id":tool_call["id"],"function":{"arguments":json.dumps(tool_call["args"]),"name":tool_call["name"]},"type":"function"} for tool_call in message["tool_calls"]]
            if message["role"] == "tool":
                deepseek_message["tool_call_id"] = message["tool_call_id"]
            deepseek_messages.append(deepseek_message)
        return deepseek_messages

    def invoke(self,messages, tool_definitions, max_new_tokens=800):
        deepseek_messages = self.to_deepseek_format(messages)
        response = self.generate(deepseek_messages,tool_definitions,max_new_tokens=max_new_tokens)
        response_parsed = self.parse_and_normalize_response(response)
        return response_parsed



if __name__ == "__main__":
    from tool_register import Register
    from tools import read_resume_tool

    register = Register()
    register.register(read_resume_tool)

    chat_model = DeepSeekModel()

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
