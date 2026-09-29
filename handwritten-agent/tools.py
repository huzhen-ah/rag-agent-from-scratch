#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:54:10 2026

@author: huzhen
"""
from transformers.utils import get_json_schema
import json
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from exceptions import ToolInvocationException, ResumeNotFoundException, ToolExecutionException
import requests


class Tool:
    def __init__(self,function):
        self.function = function
        self.definition = get_json_schema(function)
        self.definition["function"]["parameters"]["additionalProperties"] = False
        self.name = self.definition["function"]["name"]
        
    def validate_parameters_pre_fun(self,instance):
        try:
            validate(instance=instance, schema=self.definition["function"]["parameters"])
        except ValidationError as error:
            raise ToolInvocationException("Tool-{}校验参数失败: {}".format(self.name,error.message)) from error
        
    def run(self,arguments):
        self.validate_parameters_pre_fun(arguments)
        return self.function(**arguments)
        
        
def read_resume(resume_id: str) -> str:
    """
    根据简历ID读取简历
    
    Args:
        resume_id: 需要读取的简历ID
    """
    
    resume_files = {
        "main":r"data/resumes/resume_main.md",
        "dialogue_nlp":r"data/resumes/resume_dialogue_nlp.md",
        "rag_agent":r"data/resumes/resume_rag_agent.md"
        }
    
    if resume_id not in resume_files:
        raise ResumeNotFoundException("resume_id: {} 不存在".format(resume_id))
        
    resume_file = resume_files[resume_id]
    with open(resume_file,"r",encoding="utf8") as f:
        data = f.read()
    
    return data

def search_project_evidence(requirement: str) -> list[dict]:
    """
    搜索项目相关证据
    
    Args:
        requirement: 一项项目要求
    """
    with open(r"data/evidence/project_evidence.json","r",encoding="utf8") as f:
        project_evidences = json.load(f)
    ret = []
    requirement = requirement.lower()
    for proj_evi in project_evidences:
        matched_keywords = []
        for word in proj_evi["keywords"]:
            if word.lower() in requirement:
                matched_keywords.append(word)
        if len(matched_keywords) == 0:
            continue
        ret.append({"project":proj_evi["project"],"matched_keywords":matched_keywords,"evidence":proj_evi["evidence"]})
    return ret


def update_user_profile(
            preferred_city: str | None = None,
            target_role: str | None = None,
            skills: list[str] | None = None,
            years_of_experience: int | None = None,
            fields_to_delete: list[str] | None = None
):
    """
    提取用户明确表达的长期求职信息。

    Args:
        preferred_city: 用户长期偏好的求职城市。
        target_role: 用户长期目标岗位。
        skills: 用户明确拥有的技能。
        years_of_experience: 用户明确说明的工作年限。
        fields_to_delete: 用户明确要求遗忘的字段名称。
    """
    profile_updates = {
                            "preferred_city" : preferred_city,
                            "target_role" : target_role,
                            "skills" : skills,
                            "years_of_experience" : years_of_experience
                      }
    
    if fields_to_delete is None:
        fields_to_delete = []
    
    unknown_fields = set(fields_to_delete) - set(profile_updates)
    if unknown_fields:
        raise ValueError("不能删除未知的长期记忆字段:{}".format(unknown_fields))
    profile_updates = {k:v for k,v in profile_updates.items() if v is not None}
    
    conflicting_fields = set(fields_to_delete).intersection(set(profile_updates))
    if conflicting_fields:
        raise ValueError("字段不能同时删除与更新: {}".format(conflicting_fields))
        
    ret = {"updates":profile_updates,"fields_to_delete":fields_to_delete}
    return ret
    
    
def query_rag(question: str) -> list[dict]:
    """
    从RAG知识库中检索与问题相关的参考资料。

    Args:
        question: 需要检索的问题。
    """
    try:
        ret = requests.post("http://127.0.0.1:8080/retrieve",json={"question":question, "user_id":"user-001", "knowledge_base_id":"kb-fault-codes"},timeout=120)
        ret.raise_for_status()
        ret = ret.json()
    except requests.RequestException as error:
            raise ToolExecutionException(
                "调用RAG服务失败: {}".format(error)
            ) from error

    return ret["documents"]


def read_file(file_path: str) -> str:
    """
    读取系统提示中列出的Skill文件。当用户请求与某个Skill说明匹配时，必须先调用本工具读取对应SKILL.md，再执行其他业务工具。
    
    Args:
        file_path: 需要读取的文件路径
    """
    with open(file_path,"r",encoding="utf8") as f:
        return f.read()
       
    
        
        

read_resume_tool = Tool(function=read_resume)
search_project_evidence_tool = Tool(function=search_project_evidence)

update_user_profile_tool = Tool(function=update_user_profile)
query_rag_tool = Tool(function=query_rag)
read_file_tool = Tool(function=read_file)