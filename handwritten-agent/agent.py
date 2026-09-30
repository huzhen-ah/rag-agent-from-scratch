#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 19:48:47 2026

@author: huzhen
"""

import uuid
from graph import StateGraph,START,END
from state import AgentState
from nodes import ModelNode, ToolNode, ToolArgsCompletionNode, ToolReviewNode, MemoryWriteNode
from routers import router_after_model, router_after_tool_review
from hitl import Command
import json
from skill import build_skills_prompt


class Agent:
    def __init__(self, chat_model, register, system_prompt, checkpointer, tool_hitl_policy, skills=None, max_steps=10, memory_store=None, memory_tool=None):
        self.chat_model = chat_model
        self.register = register
        self.system_prompt = system_prompt
        if skills:
            skills_prompt = build_skills_prompt(skills)
            self.system_prompt += "\n\n" + skills_prompt
        self.checkpointer = checkpointer
        self.tool_hitl_policy = tool_hitl_policy
        self.max_steps = max_steps
        self.memory_store = memory_store
        self.memory_tool = memory_tool
        if self.memory_store is not None:
            if self.memory_tool is None:
                raise ValueError("启用MemoryStore时必须传入memory_tool")
            self.memory_write_node = MemoryWriteNode(self.chat_model, self.memory_tool)
        else:
            self.memory_write_node = None
        self.model_node = ModelNode(self.chat_model,self.register.get_tool_definitions())
        self.tool_node = ToolNode(self.register)
        self.tool_args_completion_node = ToolArgsCompletionNode(self.register, self.tool_hitl_policy)
        self.tool_review_node = ToolReviewNode(tool_hitl_policy)
        self.compiled_state_graph = self.init_graph()
        
    def init_graph(self):
        state_graph = StateGraph(AgentState)
        state_graph.add_node("model_node", self.model_node)
        state_graph.add_node("tool_node", self.tool_node)
        state_graph.add_node("tool_args_completion_node", self.tool_args_completion_node)
        state_graph.add_node("tool_review_node", self.tool_review_node)

        if self.memory_write_node is not None:
            state_graph.add_node("memory_write_node", self.memory_write_node)
            state_graph.add_edge("memory_write_node", END)
            finished_target = "memory_write_node"
        else:
            finished_target = END

        state_graph.add_edge(START, "model_node")
        model_path_map = {
                        "need_tools":"tool_args_completion_node",
                        "finished":finished_target
                   }
        
        state_graph.add_conditional_edges("model_node", router_after_model,model_path_map)
        tool_args_completion_path_map = {
                        "needs_more_args" : "tool_args_completion_node",
                        "tool_review_node" : "tool_review_node"
                   }
        
        state_graph.add_conditional_edges("tool_args_completion_node", self.tool_args_completion_node.router_after_toolArgsCompletionNode, tool_args_completion_path_map)
        state_graph.add_conditional_edges("tool_review_node", router_after_tool_review)
        state_graph.add_edge("tool_node", "model_node")

        return state_graph.compile(self.checkpointer, self.memory_store)
    
    def create_initial_state(self):
        system_message = {
                            "id":r"msg_{}".format(uuid.uuid4().hex),
                            "role":"system",
                            "content":self.system_prompt
                         }
        agent_state = {
                            "messages":[system_message],
                            "model_call_count":0
                      }
        return agent_state
    
    def invoke(self, user_input, agent_state, thread_id, checkpoint_ns, checkpoint_id, checkpoint_map=None, context=None):
        
        if isinstance(user_input, Command):
            return self.compiled_state_graph.invoke(user_input, None, thread_id, checkpoint_ns, checkpoint_id, checkpoint_map, recursion_limit=self.max_steps, context=context)
        
        
        if user_input is not None:
            user_message = {
                                "id":r"msg_{}".format(uuid.uuid4().hex),
                                "role":"user",
                                "content":user_input
                           }
            input_update = {"messages":[user_message]}
        else:
            input_update = None

        agent_state = self.compiled_state_graph.invoke(agent_state, input_update, thread_id, checkpoint_ns, checkpoint_id, checkpoint_map, recursion_limit=self.max_steps, context=context)


        return agent_state

    def stream(self, user_input, agent_state, thread_id, checkpoint_ns, checkpoint_id, context=None):
        if isinstance(user_input, Command):
            event_item_generator = self.compiled_state_graph.stream(user_input, None, thread_id, checkpoint_ns, checkpoint_id, recursion_limit=self.max_steps, context=context)
            return event_item_generator
        
        if user_input is not None:
            user_message = {
                                "id":r"msg_{}".format(uuid.uuid4().hex),
                                "role":"user",
                                "content":user_input
                           }
            input_update = {"messages":[user_message]}
        else:
            input_update = None

        event_item_generator = self.compiled_state_graph.stream(agent_state, input_update, thread_id, checkpoint_ns, checkpoint_id, recursion_limit=self.max_steps, context=context)


        return event_item_generator
        


if __name__ == "__main__":
    from model import LocalChatModel
    from tool_register import Register
    from tools import read_resume_tool, search_project_evidence_tool, update_user_profile_tool, build_query_rag_tool, read_file_tool
    from checkpoint import JsonlCheckpointer
    from memory import JsonlStore
    from skill import load_skill_metadata
    
    
    
    register = Register()
    query_rag_tool = build_query_rag_tool()
    register.register(read_resume_tool)
    register.register(search_project_evidence_tool)
    register.register(query_rag_tool)
    register.register(read_file_tool)
    
    chat_model = LocalChatModel(
        model_path="models/Qwen3-4B",
        device="mps",
    )

    system_prompt = (
        "你是一个求职助手。"
        "需要读取简历或查询项目证据时，"
        "必须调用相应工具。"
        "需要查询知识库中的资料时，必须调用query_rag工具。"
        "只能依据工具返回的内容回答，"
        "不得编造经历。"
    )
    
    tool_hitl_policy = {
        "read_resume": ("args_completion","review"),
        "search_project_evidence": ("review",)
    }
    
    skills = load_skill_metadata(r"skills")
    
    checkpointer = JsonlCheckpointer()

    memory_store = JsonlStore()
    agent = Agent(
        chat_model= chat_model,
        register= register,
        system_prompt= system_prompt,
        checkpointer= checkpointer,
        tool_hitl_policy= tool_hitl_policy,
        skills= skills,
        max_steps= 10,
        memory_store= memory_store,
        memory_tool= update_user_profile_tool
    )
    print("prompt: ",agent.system_prompt)
    """
        user_id：用户，谁
        thread_id：具体会话，哪次会话
        checkpoint_ns：会话里的具体哪张图
        checkpoint_id：这张图的具体哪个状态版本，即哪个super-step结束之后的statesnapshot
        write_key: (task_id, channel) 这个checkpoint版本下面哪条任务写入,用task_id & channel来唯一write
    """
    user_A_agent_state = agent.create_initial_state()
    thread_id = "thread_{}".format(uuid.uuid4().hex)
    checkpoint_ns = ""
    checkpoint_id = None
    context = {"user_id":"user_A"}
    
    use_streaming = False
    if not use_streaming:
        while True:
            user_input = input("用户: ")
            user_input = user_input.strip()
            if user_input == "exit":
                break
            if user_input == "":
                user_input = None
            user_A_agent_state = agent.invoke(user_input,user_A_agent_state, thread_id, checkpoint_ns, checkpoint_id, context=context)
            # print(user_A_agent_state)
            while "__interrupt__" in user_A_agent_state:
                interrupts = user_A_agent_state["__interrupt__"]
                command_resume = {}
                for interrupt in interrupts:
                    print(
                            "interrupt_request: ",
                            json.dumps(
                                interrupt.value,
                                ensure_ascii=False,
                                indent=2,
                                )
                          )
                    resume_text = input("请输入resume_value(json): ")
                    resume_value = json.loads(resume_text)
                    command_resume[interrupt.id] = resume_value
                resume_command = Command(resume=command_resume)
                user_A_agent_state = agent.invoke(resume_command, user_A_agent_state, thread_id, checkpoint_ns, checkpoint_id, context=context)
                    
            print("assistant: ",user_A_agent_state["messages"][-1]["content"])
    else:
        while True:
            user_input = input("用户: ")
            user_input = user_input.strip()
            if user_input == "exit":
                break
            if user_input == "":
                user_input = None
            current_input = user_input

            while True:
                event_item_generator = agent.stream(current_input,user_A_agent_state, thread_id, checkpoint_ns, checkpoint_id, context=context)
                for event_item in event_item_generator:
                    print("event_item: ",event_item)
                    if event_item.event_type in {"interrupt","final"}:
                        user_A_agent_state = event_item.payload["output"]

                if "__interrupt__" not in user_A_agent_state:
                    break

                interrupts = user_A_agent_state["__interrupt__"]
                command_resume = {}
                for interrupt in interrupts:
                    print(
                            "interrupt_request: ",
                            json.dumps(
                                interrupt.value,
                                ensure_ascii=False,
                                indent=2,
                                )
                          )
                    resume_text = input("请输入resume_value(json): ")
                    resume_value = json.loads(resume_text)
                    command_resume[interrupt.id] = resume_value
                current_input = Command(resume=command_resume)

            print("assistant: ",user_A_agent_state["messages"][-1]["content"])
