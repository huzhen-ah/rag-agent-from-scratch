#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 09:59:18 2026

@author: huzhen
"""


from runtime import get_state_reducers, apply_updates, NodeRuntime
from collections.abc import Hashable
from checkpoint import StateSnapshot, Checkpointer, PendingWrite
import time
import uuid
from hitl import Command, create_interrupt_id
from task_execution import Task, TaskResult, PregelScratchpad
from send import Send
from execution_context import task_execution_context_var, GraphCheckpointContext, TaskExecutionContext
from exceptions import GraphInterrupt
from memory import BaseStore
import inspect
from threading import Thread
from queue import Queue
from streaming import StreamEvent




START = "__start__"
END = "__end__"

class FixedTransition:
    def __init__(self,targets):
        self.targets = sorted(targets) #某个source对应的targets
        
    def resolve_targets(self,state):
        return self.targets
    
class ConditionalTransition:
    def __init__(self,router,node_names,path_map=None):
        self.router = router
        self.node_names = node_names
        self.path_map = path_map
        
    def resolve_targets(self,state):
        router_result = self.router(state)
        if isinstance(router_result, (list,tuple)):
            router_results = router_result
        else:
            router_results = [router_result]
            
        targets = []
        for router_result in router_results:
            if isinstance(router_result, Send):
                if router_result.node not in self.node_names:
                    raise ValueError("Send指向了未注册的Node: {}".format(router_result.node))
                targets.append(router_result)
                continue
            if not isinstance(router_result, Hashable):
                raise TypeError("Router返回的普通路由router_result必须是Hashable")
        
        
            if self.path_map is None:

                if not isinstance(router_result, str):
                    raise TypeError("当path_map是None的时候，返回值类型必须是str")
                if router_result not in self.node_names and router_result != END:
                    raise ValueError("当path_map是None的时候,返回值必须是node_name or END")
                target = router_result
            else:
                if router_result not in self.path_map:
                    raise ValueError("Router返回了不存在于path_map中的key:{}".format(router_result))
                target = self.path_map[router_result]
            targets.append(target)
        return targets
        
    

class StateGraph:
    def __init__(self,state_schema):
        self.state_schema = state_schema
        self.nodes = {}
        self.fixed_edges = {}
        self.conditional_edges = {}
        
    def add_node(self,node_name,node):
        if not isinstance(node_name, str):
            raise TypeError("Node的名称必须是字符串")
        if not node_name.strip():
            raise ValueError("node_name不能为空")
        if node_name in {START,END}:
            raise ValueError("{}是Graph保留的名称".format(node_name))
        if not callable(node):
            raise TypeError("{}对应的Node必须是callable".format(node_name))
        if node_name in self.nodes:
            raise ValueError("Node-{}已经注册".format(node_name))
            
        self.nodes[node_name] = node
        return self
    
    def add_edge(self,source,target):
        if not isinstance(source, str) or not isinstance(target, str):
            raise TypeError("Edge的source or target必须是字符串")
            
        if not source.strip() or not target.strip():
            raise ValueError("Edge的source or target不能为空")
        
        if source == END:
            raise ValueError("END不能作为Edge的source")
            
        if target == START:
            raise ValueError("START不能作为Edge的target")
            
        if source != START and source not in self.nodes:
            raise ValueError("source Node-{}暂未注册".format(source))
        
        if target != END and target not in self.nodes:
            raise ValueError("target Node-{}暂未注册".format(target))
            
        if source not in self.fixed_edges:
            self.fixed_edges[source] = set()
        
        if target in self.fixed_edges[source]:
            raise ValueError("Edge {}->{} 已存在".format(source,target))
        self.fixed_edges[source].add(target)
        
        return self
    
    def add_conditional_edges(self,source,router,path_map=None):
        if not isinstance(source, str):
            raise TypeError("Edge的source or target必须是字符串")
        
        if source != START and source not in self.nodes:
            raise ValueError("Node-{}尚未添加".format(source))
            
        if not source.strip():
            raise ValueError("Edge的source or target不能为空")
            
        if source == END:
            raise ValueError("END不能作为Edge的source")
            
        if source in self.conditional_edges:
            raise ValueError("{}已经存在于conditional_edges".format(source))
        
        if not callable(router):
            raise TypeError("router必须是callable")

        self.conditional_edges[source] = {"router":router,"path_map":path_map}

        return self
    
    def get_reached_nodes_from_edges(self,source,reached_nodes):
        if source in self.fixed_edges:
            source_reached_nodes = self.fixed_edges[source]
        elif source in self.conditional_edges:
            path_map = self.conditional_edges[source]["path_map"]
            if path_map is None:
                source_reached_nodes = set(self.nodes) | {END}
            else:
                source_reached_nodes = set(path_map.values())
        else:
            return reached_nodes
        
        for child_node in source_reached_nodes:
            if child_node in reached_nodes:
                continue
            reached_nodes.add(child_node)
            if child_node != END:
                reached_nodes = self.get_reached_nodes_from_edges(child_node,reached_nodes)
            
        return reached_nodes
        
        
    def compile(self, checkpointer: Checkpointer, memory_store: BaseStore = None):
        if START not in self.fixed_edges and START not in self.conditional_edges:
            raise ValueError("START必须至少存在于fixed_edges和conditional_edges中的一个，其实只能存在于一个")
        overlapping_sources = set(self.fixed_edges).intersection(set(self.conditional_edges))
        if overlapping_sources:
            raise ValueError("node不能同时拥有固定边与条件边，以下node同时拥有固定边与条件边：{}".format(overlapping_sources))
        
        nodes = self.nodes.copy()
        
        transitions = {}
        
        for source,targets in self.fixed_edges.items():
            transitions[source] = FixedTransition(targets.copy())
        
        for source,conditional_edge in self.conditional_edges.items():
            router = conditional_edge["router"]
            
            path_map = conditional_edge["path_map"]
            if path_map is not None and not isinstance(path_map, dict):
                raise TypeError("path_map要么是None,要么是dict")
            if isinstance(path_map, dict):
                for router_key,target in path_map.items():
                    if not isinstance(target, str):
                        raise TypeError("target 类型必须是str")
                    
                    if target != END and target not in self.nodes:
                        raise ValueError("source-{} 的router_key-{} 指向了未注册的Node-{}".format(source,router_key,target))
                path_map = path_map.copy()
            
            transitions[source] = ConditionalTransition(router,set(self.nodes),path_map)
            
        
        reached_nodes = set()
        reached_nodes = self.get_reached_nodes_from_edges(START, reached_nodes)
        unreached_nodes = set(self.nodes) - reached_nodes
        if unreached_nodes:
            raise ValueError("从START开始，存在未到达的nodes:{}".format(unreached_nodes))

        return CompiledStateGraph(self.state_schema, nodes, transitions, checkpointer, memory_store)
    
class CompiledStateGraph:
    def __init__(self, state_schema, nodes, transitions, checkpointer=None, memory_store=None):
        self.state_schema = state_schema
        self.nodes = nodes
        self.transitions = transitions
        self.checkpointer = checkpointer
        self.memory_store = memory_store
        self.key2reducer = get_state_reducers(state_schema)
        
    def get_created_at(self):
        return str(time.time())

    def merge_updates(self,old_state,update_states):
        return apply_updates(old_state, update_states, self.key2reducer)

    def split_targets(self, targets):
        pending_pull_node_names = []
        pending_sends = []
        for target in targets:
            if isinstance(target, Send):
                pending_sends.append(target)
            elif target != END:
                pending_pull_node_names.append(target)
        pending_pull_node_names = sorted(set(pending_pull_node_names))
        return pending_pull_node_names, pending_sends

    def execute_task(self, task, graph_checkpoint_context, resume_values=(), node_runtime=None):
        scratchpad = PregelScratchpad(resume=list(resume_values))
        task_execution_context = TaskExecutionContext(graph_checkpoint_context, task, scratchpad)
        token = task_execution_context_var.set(task_execution_context)
        try:
            node = self.nodes[task.node_name]
            if "node_runtime" in inspect.signature(node).parameters:
                update = node(task.input, node_runtime=node_runtime)
            else:
                update = node(task.input)
            task_result = TaskResult(task=task, channel="update",value=update)
            return task_result
        except GraphInterrupt as error:
            task_result = TaskResult(task=task, channel="interrupt", value=error.args[0])
            return task_result
        except Exception as error:
            task_result = TaskResult(task=task, channel="error", value=error)
            return task_result
        finally:
            task_execution_context_var.reset(token)
      
    def is_interrupt_created_by_task(self, interrupt_id, checkpoint_id, task_id):
        tmp_interrupt_id = create_interrupt_id(checkpoint_id, task_id)
        return interrupt_id == tmp_interrupt_id
    
        
    def execute_tasks(
            self, 
            tasks,
            thread_id, 
            checkpoint_ns, 
            checkpoint_id, 
            checkpoint_map, 
            context, 
            memory_store,
            stream_writer,
            forward_task_id_2_interrupt_resume_map
    ):
        graph_checkpoint_context = GraphCheckpointContext(thread_id, checkpoint_ns, checkpoint_id, checkpoint_map)
        saved_task_id_2_updates = {}
        saved_task_id_2_resumes = {}
        saved_task_id_2_interrupts = {}
        pending_writes = self.checkpointer.get_writes(thread_id, checkpoint_ns, checkpoint_id)
        for write in pending_writes:
            if write.channel == "update":
                task_id = write.task_id
                saved_task_id_2_updates[task_id] = write.value
            elif write.channel == "resume":
                task_id = write.task_id
                saved_task_id_2_resumes[task_id] = write.value
            elif write.channel == "interrupt":
                task_id = write.task_id
                saved_task_id_2_interrupts[task_id] = write.value
                
                
        task_results = []
        for task in tasks:
            if task.task_id in saved_task_id_2_updates:
                task_result = TaskResult(task=task, channel="update",value=saved_task_id_2_updates[task.task_id])
                task_results.append(task_result)
                continue
            task_forward_resume_map = forward_task_id_2_interrupt_resume_map.get(task.task_id, {})
            task_node_runtime = NodeRuntime(
                                context = context, 
                                memory_store = memory_store, 
                                stream_writer = stream_writer,
                                resume_map = task_forward_resume_map
                             )
            
            task_resume_values = saved_task_id_2_resumes.get(task.task_id,())
            task_result = self.execute_task(task, graph_checkpoint_context, task_resume_values, task_node_runtime)
            task_results.append(task_result)
            # if task_result.channel == "error":
            #     raise task_result.value
            writes = self.task_result_to_writes(task_result)
            self.checkpointer.put_writes(thread_id, checkpoint_ns, checkpoint_id, writes)
        return tuple(task_results)
    
    def task_result_to_writes(self, task_result):
        task_id = task_result.task.task_id
        channel = task_result.channel
        if channel not in {"update", "interrupt", "error"}:
            raise RuntimeError("channel 必须是update|interrupt|error")
        value = task_result.value
        return (PendingWrite(task_id, channel, value), )
        
        
    def save_snapshot(
            self,
            thread_id,
            checkpoint_ns,
            parent_checkpoint_id,
            super_step,
            state,
            pending_pull_node_names,
            pending_sends
    ):
        checkpoint_id = r"checkpoint_{}".format(uuid.uuid4().hex)
        stateSnapshot_dict = {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": checkpoint_id,
                                "parent_checkpoint_id": parent_checkpoint_id,
                                "super_step": super_step,
                                "state": state,
                                "pending_pull_node_names": pending_pull_node_names,
                                "pending_sends": pending_sends,
                                "created_at": self.get_created_at()
                             }
        stateSnapshot = StateSnapshot(**stateSnapshot_dict)
        self.checkpointer.put(stateSnapshot)
        return stateSnapshot
    
    def create_tasks(self, checkpoint_id, state, pending_pull_node_names, pending_sends):
        tasks = []
        for node_name in pending_pull_node_names:
            task_identity = "{}:pull:{}".format(checkpoint_id, node_name)
            task_id = str(uuid.uuid5(namespace=uuid.NAMESPACE_OID, name=task_identity))
            tasks.append(Task(task_id, node_name, state))

        for send_index, send in enumerate(pending_sends):
            task_identity = "{}:push:{}:{}".format(checkpoint_id, send_index, send.node)
            task_id = str(uuid.uuid5(namespace=uuid.NAMESPACE_OID, name=task_identity))
            tasks.append(Task(task_id, send.node, send.arg))
        return tasks
    
    def save_and_route_resume_values(self, thread_id, checkpoint_ns, checkpoint_id, command_resume):
        pending_writes = self.checkpointer.get_writes(thread_id, checkpoint_ns, checkpoint_id)
        interrupt_writes = [write for write in pending_writes if write.channel == "interrupt"]
        resume_writes = [write for write in pending_writes if write.channel == "resume"]
        if isinstance(command_resume, dict):#多个interrrupt同时更新来了
            interrupt_id_2_task_id = {}
            for write in interrupt_writes:
                task_id = write.task_id
                for interrupt_data in write.value:
                    interrupt_id = interrupt_data.id
                    interrupt_id_2_task_id[interrupt_id] = task_id
            unknown_interrupt_ids = set(command_resume) - set(interrupt_id_2_task_id)
            if unknown_interrupt_ids:
                raise RuntimeError("提供的以下interrupt_ids不是这次interrupt需要的: {}".format(unknown_interrupt_ids))
            task_id_2_resume_value = {}
            for write in resume_writes:
                task_id_2_resume_value[write.task_id] = list(write.value)
            forward_task_id_2_interrupt_resume_map = {}
            for interrupt_id, value in command_resume.items():
                task_id = interrupt_id_2_task_id[interrupt_id]
                #基于writes记录来看，这个interrupt_id应该是对应的这个task_id,
                if not self.is_interrupt_created_by_task(interrupt_id, checkpoint_id, task_id):
                    # 说明这个 Interrupt 不是当前 Task 直接创建的，而是当前 SubGraphNode Task 承载的，因此继续向子图转发
                    if task_id not in forward_task_id_2_interrupt_resume_map:
                        forward_task_id_2_interrupt_resume_map[task_id] = {}
                    forward_task_id_2_interrupt_resume_map[task_id][interrupt_id] = value
                    continue
                   
                
                if task_id not in task_id_2_resume_value:
                    task_id_2_resume_value[task_id] = []
                task_id_2_resume_value[task_id].append(value)
            writes = tuple(PendingWrite(task_id, "resume", value) for task_id, value in task_id_2_resume_value.items())
            self.checkpointer.put_writes(thread_id, checkpoint_ns, checkpoint_id, writes)
            return forward_task_id_2_interrupt_resume_map
        else:
            raise RuntimeError(r"command_resume是Command的resume字段值，必须是{interrupt_id:single_resume_value}")
        
    def invoke(self, 
             graph_input, 
             input_update=None, 
             thread_id=None, 
             checkpoint_ns="",
             checkpoint_id=None, 
             checkpoint_map=None,
             recursion_limit=25, 
             context=None
    ):
        for event in self.stream(graph_input, input_update, thread_id, checkpoint_ns, checkpoint_id, checkpoint_map, recursion_limit, context):
            if event.event_type in {"interrupt","final"}:
                return event.payload["output"]
            
            
    
    def stream(self,
             graph_input, 
             input_update=None, 
             thread_id=None,
             checkpoint_ns="",
             checkpoint_id=None, 
             checkpoint_map=None,
             recursion_limit=25, 
             context=None
    ):
        event_queue = Queue()
        
        def run_graph():
            try:
                output = self._run(
                                    graph_input,
                                    input_update,
                                    thread_id,
                                    checkpoint_ns,
                                    checkpoint_id,
                                    checkpoint_map,
                                    recursion_limit,
                                    context,
                                    stream_writer=event_queue.put
                                  )
                if "__interrupt__" in output:
                    event_type = "interrupt"
                    payload = {
                                "output" : output,
                                "interrupts" : output["__interrupt__"]
                              }
                    terminal_event = StreamEvent(event_type, payload)
                else:
                    event_type = "final"
                    payload = {"output" : output}
                    terminal_event = StreamEvent(event_type, payload)
                event_queue.put(terminal_event)
                    
            except BaseException as error:
                event_queue.put(error)
            

        worker = Thread(target=run_graph)
        worker.start()
        while True:
            item = event_queue.get()

            if isinstance(item, BaseException):
                raise item
            
            if item.event_type in {"interrupt","final"}:
                yield item
                break
                
            yield item
            
    
    def _run(self, 
             graph_input, 
             input_update=None, 
             thread_id=None,
             checkpoint_ns="",
             checkpoint_id=None, 
             checkpoint_map=None,
             recursion_limit=25, 
             context=None,
             stream_writer=None
    ):
        """
        graph_input:只有2种可能：
        1.init_state
        2.resume_command
        """
        
        if thread_id is None:
            raise ValueError("必须传入thread_id")
        if checkpoint_map is None:
            checkpoint_map = {}
        
            
        
        resume_command = graph_input if isinstance(graph_input, Command) else None
        
        
        stateSnapshot = self.checkpointer.get(thread_id, checkpoint_ns, checkpoint_id)
        if resume_command is not None:
            if stateSnapshot is None:
                raise RuntimeError("Command.resume必须依附已有checkpoint")
            if input_update is not None:
                raise RuntimeError("恢复中断时不能同时传入input_update")
        #每轮tasks开始之前都要保存一个checkpoint,后面的pendding,memory，hitl之类的要用
        #如果开始的时候没有拿到checkpoint，就生成一个保存
        #如果有用户输入，就把用户输入更新到state，保存checkpoint.
        #如果能拿到历史checkpoint且无用户输入，那就不用保存新的checkpoint
        if stateSnapshot:
            parent_checkpoint_id = stateSnapshot.checkpoint_id
            state = stateSnapshot.state
            if input_update is not None:
                state = self.merge_updates(state, [input_update])
                start_transition = self.transitions[START]
                targets = start_transition.resolve_targets(state)
                pending_pull_node_names, pending_sends = self.split_targets(targets)
                stateSnapshot = self.save_snapshot(thread_id, checkpoint_ns, parent_checkpoint_id, stateSnapshot.super_step, state, pending_pull_node_names, pending_sends)
            else:
                pending_pull_node_names = stateSnapshot.pending_pull_node_names
                pending_sends = stateSnapshot.pending_sends
            super_step = stateSnapshot.super_step
            recursion_limit += super_step
        else:
            parent_checkpoint_id = None
            state = graph_input
            if input_update is not None:
                state = self.merge_updates(state, [input_update])
            start_transition = self.transitions[START]
            targets = start_transition.resolve_targets(state)
            pending_pull_node_names, pending_sends = self.split_targets(targets)
            super_step = 0
            stateSnapshot = self.save_snapshot(thread_id, checkpoint_ns, parent_checkpoint_id, super_step, state, pending_pull_node_names, pending_sends)

        parent_checkpoint_id = stateSnapshot.checkpoint_id
        if resume_command is not None:
            #这里是比较奇葩的地方，它的interrupt反馈不是传进去的，是写到一个地方，然后另一段代码从那个地方拿出来的
            #所以如果发现有interrupt反馈，就要把这个反馈先保存起来。这里的保存，不是说直接放到制定地方，让对方拿，仅仅是先保存。
            #当前层直接创建的 resume 保存为 PendingWrite；不是当前层直接创建的则返回并继续向子图转发
            #那不是当前层的，怎么办呢？？隐藏的代码来了：通过resume_map传给node_runtime。。。subgraphnode通过node_runtime拿到resume_value,包装成command继续传给子图
            forward_task_id_2_interrupt_resume_map = self.save_and_route_resume_values(thread_id, stateSnapshot.checkpoint_ns, stateSnapshot.checkpoint_id, resume_command.resume)
        else:
            forward_task_id_2_interrupt_resume_map = {}
        
        while pending_pull_node_names or pending_sends:
            
            if super_step >= recursion_limit:
                raise RuntimeError("Graph 超过最大super-step步数:{}".format(recursion_limit))

            tasks = self.create_tasks(stateSnapshot.checkpoint_id, state, pending_pull_node_names, pending_sends)
            
            task_results = self.execute_tasks(
                                                tasks, 
                                                thread_id, 
                                                stateSnapshot.checkpoint_ns, 
                                                stateSnapshot.checkpoint_id, 
                                                checkpoint_map, 
                                                context = context,
                                                memory_store = self.memory_store,
                                                stream_writer = stream_writer,
                                                forward_task_id_2_interrupt_resume_map = forward_task_id_2_interrupt_resume_map
                                              )
            
            errors = [task_result.value for task_result in task_results if task_result.channel == "error"]
            if len(errors) > 0:
                raise errors[0]
            
                
            interrupts = tuple(interrupt for task_result in task_results if task_result.channel=="interrupt" for interrupt in task_result.value)
            if interrupts:
                output = dict(state)
                output["__interrupt__"] = interrupts
                return output
            
            update_states = [task_result.value for task_result in task_results if task_result.channel=="update"]
            state = self.merge_updates(state, update_states)
            
            active_node_names = sorted(set([task.node_name for task in tasks]))#这里用set的意思是：举个例子，所有toolcall汇总结果之后，只需要调用一次modelnode来分析汇总之后的信息
            targets = []
            for node_name in active_node_names:
                if node_name in self.transitions:
                    targets.extend(self.transitions[node_name].resolve_targets(state))
            pending_pull_node_names, pending_sends = self.split_targets(targets)

            super_step += 1
            if self.checkpointer and thread_id:
                stateSnapshot = self.save_snapshot(thread_id, checkpoint_ns, parent_checkpoint_id, super_step, state, pending_pull_node_names, pending_sends)
                parent_checkpoint_id = stateSnapshot.checkpoint_id
            if stream_writer is not None:
                for task_result in task_results:
                    if task_result.channel == "update":
                        event_type = "update"
                        payload = {
                                        "super_step" : super_step,
                                        "checkpoint_ns" : checkpoint_ns,
                                        "node_name" : task_result.task.node_name,
                                        "update" : task_result.value
                                   }
                        update_event = StreamEvent(event_type, payload)
                        stream_writer(update_event)
        
        return state
        
        
        

        

if __name__ == "__main__":
    
    from state import AgentState
    from nodes import ModelNode,ToolNode
    from routers import router_after_model
    from checkpoint import JsonlCheckpointer
    
    model_node = ModelNode(None, None)
    tool_node = ToolNode(None)
    graph = StateGraph(AgentState)
    
    graph.add_node("model_node", model_node)
    graph.add_node("tool_node", tool_node)
    
    graph.add_edge(START, "model_node")
    graph.add_edge("tool_node", "model_node")
    
    path_map={
                "need_tools":"tool_node",
                "finished":END
             }
    
    graph.add_conditional_edges("model_node", router_after_model,path_map)
    
    print(graph.nodes)
    print(graph.fixed_edges)
    print(graph.conditional_edges)
    
    jsonlCheckpointer = JsonlCheckpointer()
    compiledStateGraph = graph.compile(jsonlCheckpointer)
    
    print(compiledStateGraph.nodes is graph.nodes)
