#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 20:47:49 2026

@author: huzhen
"""

from typing import TypedDict
import os
from tools import Tool

class SkillMetadata(TypedDict):#skill元数据
    name: str #skill名字
    description: str #skill的描述，就是功能介绍，用来给模型看，什么时候需要这个skill
    path: str #skill详细地址，通过这个地址拿到完整说明

    
    

def load_skill_metadata(skills_fold):
    skills = {}
    
    for skill_name in os.listdir(skills_fold):
        skill_file = "{}/{}/SKILL.md".format(skills_fold,skill_name)
        if not os.path.isfile(skill_file):
            continue
        with open(skill_file,"r",encoding="utf8") as f:
            data = f.read()
        
        data = data.split("---")[1]
        skill = {}
        for line in data.split("\n"):
            if not line.strip():
                continue
            if ":" not in line:
                continue
            key,value = line.split(":")
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if not value:
                continue
            skill[key] = value
        if not skill:
            continue
        skill["path"] = skill_file
        skills[skill_name] = skill
    return skills
        
def create_read_skill_tool(skills):
    def read_skill(skill_name: str) -> str:
        """
        根据Skill名称读取完整的任务执行说明。

        Args:
            skill_name: 需要读取的Skill名称。
        """
        if skill_name not in skills:
            raise ValueError("skill: {} 不存在".format(skill_name))
        skill = skills[skill_name]
        skill_path = skill["path"]
        with open(skill_path, "r", encoding="utf8") as f:
            return f.read()
    return Tool(read_skill)

def build_skills_prompt(skills):
    skills_prompt = "可用skills\n"
    for skill in skills.values():
        skills_prompt += "\n名称: {}\n".format(skill["name"])
        skills_prompt += "说明: {}\n".format(skill["description"])
    skills_prompt += (
                        "\n处理用户请求时，必须先检查请求是否与上述某个skill的说明匹配。"
                        "如果匹配，必须先调用read_skill，并传入对应的skill名称。"
                        "读取完整说明后，再严格按照其中的流程执行；"
                        "读取skill之前不要调用其他业务工具。\n"
                     )
    return skills_prompt
if __name__ == "__main__":
    skills = load_skill_metadata(r"skills")
    print(skills)
    print("\n\n\n")
    skills_prompt = build_skills_prompt(skills)
    print(skills_prompt)
