# Handwritten Tool-Calling Agent

这是一个不依赖 LangChain、LangGraph 等 Agent 编排框架，使用原生 Python 手写的 Graph-based Tool-Calling Agent。

项目目标不是堆叠 API，而是实现并理解主流 Agent Runtime 的核心运行语义：

```text
Model 决策
→ Tool Calling
→ State Update
→ Conditional Routing
→ 下一轮执行或结束
```

当前版本已经完成 **Checkpoint + HITL + Long-term Memory + Graph Event Streaming + Subgraph + Multi-Agent + RAG Tool + Skills 核心闭环**。它能够运行完整的 Model → Tool → Model 循环，支持可恢复执行、跨会话记忆、事件流、嵌套图、`agent_as_tool` 多 Agent 协作，并可调用独立 RAG 检索服务。Skills 采用渐进式加载：模型先看到元数据，匹配后再通过 `read_skill` 读取完整 `SKILL.md`。

在该 Runtime 之上，项目进一步实现了家电故障诊断应用：既支持 Qwen3-8B + LoRA 本地推理，也支持通过 DeepSeek API 运行同一套 Agent Runtime；通过 `query_rag` 调用独立混合检索服务，并使用 Streamlit 双栏页面同时展示用户对话与 Graph 内部执行事件。

## 家电故障诊断应用

### 在线体验

完整演示通过公网地址按需开放。由于 RAG 服务需要在作者本地加载 Qwen3 Embedding 与 Reranker 模型，服务不会长期在线；如果想体验，请联系项目作者，由作者启动 RAG、Agent 和公网转发服务后提供访问地址。体验者无需下载本地模型，也无需提供 DeepSeek API Key。

```text
User
→ ModelNode判断信息是否充分
→ query_rag Tool Call
→ Dense + BM25 + RRF + Reranker
→ ToolMessage
→ ModelNode生成故障解释、处理步骤与安全提醒
```

相关入口：

| 文件 | 职责 |
|---|---|
| `appliance_support.py` | 装配 Qwen3-8B LoRA、Tool Registry、手写 Agent 与终端交互入口 |
| `appliance_support_deepseek.py` | 使用 DeepSeek API 驱动同一套手写 Agent Runtime 的终端入口 |
| `appliance_support_ui.py` | Streamlit 双栏界面；左侧对话，右侧展示节点事件、Tool Call、检索文档和耗时 |
| `model.py` | 提供 `LocalChatModel`、4-bit NF4 + PEFT LoRA 的 `PeftChatModel`，以及负责消息与 Tool Call 格式转换的 `DeepSeekModel` |
| `tools.py` | `query_rag` 通过 HTTP 调用 `http://127.0.0.1:8080/retrieve` |

### 安装依赖

Agent 与 RAG 使用各自独立的依赖文件。本目录执行：

```bash
pip install -r requirements.txt
```

### 运行测试

先在 `handwritten-rag` 目录启动家电检索服务：

```bash
cd ../handwritten-rag
pip install -r requirements.txt
python appliance_rag_service.py
```

保持 RAG 服务运行，另开终端进入 `handwritten-agent`，设置 DeepSeek API Key，并使用 Streamlit 启动测试页面：

```bash
cd ../handwritten-agent
export DEEPSEEK_API_KEY="你的 API Key"
streamlit run appliance_support_ui.py
```

浏览器打开 Streamlit 给出的本地地址，即可测试完整的 `DeepSeek → Tool Call → RAG → 最终回答` 链路。

终端版本：

```bash
python appliance_support.py
```

DeepSeek API 版本需要先设置 `DEEPSEEK_API_KEY`，再运行：

```bash
export DEEPSEEK_API_KEY="你的 API Key"
python appliance_support_deepseek.py
```

在当前机器上的交互测试中，DeepSeek API 版本的响应速度约为本地 Qwen3-8B + LoRA 版本的 10 倍。该数字是当前环境下的体验值，不是严格基准测试结果。

同一 `thread_id` 会保留完整消息历史。模型在历史中已经存在有效检索资料时，可能直接复用上下文回答，而不会再次调用 RAG；点击界面的“新建会话”可以创建新的 `thread_id`。

## 1. 当前实现范围

当前版本已经实现：

- Provider-agnostic 的内部 Message 与 ToolCall 协议。
- Qwen3 Tool Calling 格式适配。
- DeepSeek Chat Completions 与 Tool Calling 格式适配。
- 模型原始输出解析与标准化。
- Runtime 生成并维护 `tool_call_id`。
- Tool Schema 生成、Tool Registry 和 Tool 调用。
- 基于 JSON Schema 的 Tool 参数校验。
- Tool 调用错误与可恢复执行错误的分类处理。
- Typed State、Reducer 和 State Update。
- 同一 Super-step 内多个 Node Update 的统一提交。
- `CompiledStateGraph` 统一持有 Reducer 映射并负责合并外部输入与 Node Updates。
- 使用加法 Reducer 累计模型调用次数，允许同一 Super-step 中多个模型 Node 分别提交调用增量。
- `StateGraph` Builder 与 `CompiledStateGraph`。
- Fixed Edge、Conditional Edge、Router 和 `path_map`。
- `START`、`END`、循环执行和最大 Super-step 限制。
- Graph 编译期结构校验与 Node 可达性校验。
- `StateSnapshot`、`Checkpointer`、`InMemoryCheckpointer` 和本地 `JsonlCheckpointer`。
- 使用 `thread_id` 隔离会话，使用 UUID `checkpoint_id` 和 `parent_checkpoint_id` 维护快照血缘。
- Super-step 成功提交后保存 `state + pending_pull_node_names + pending_sends`，支持从最新或指定 Checkpoint 恢复。
- 新 UserMessage 从历史 State 重新经过 `START`；没有新 Input Update 时从 Snapshot 中保存的待执行 PULL/PUSH Tasks 继续。
- JSONL 落盘与进程重启恢复，一个 Agent 实例可以服务多个独立 thread。
- `Task`、`TaskResult`、`TaskExecutionContext`、`PregelScratchpad` 和基于 `ContextVar` 的 Task 执行上下文。
- `interrupt(value)`、`Interrupt`、`GraphInterrupt` 和 `Command.resume`。
- task-level pending writes：保存 `update`、`interrupt`、`error` 和 `resume`。
- Super-step 中部分 Task 中断时不提交 State；恢复时复用已经完成的兄弟 Task，只重跑尚未完成的 Task。
- 工具执行前的参数补全与人工审核，支持 approve、edit 和 reject。
- `BaseStore`、`InMemoryStore` 和追加写入的 `JsonlStore`。
- Runtime 向 Node 注入每次 invoke 的 `context` 与共享 Store。
- 使用 `user_id` 构建 Memory namespace，使长期记忆跨 thread 共享并在用户之间隔离。
- ModelNode 从 Store 读取长期记忆，只注入本次模型输入的消息副本，不写入 State。
- MemoryWriteNode 通过专用 Tool Schema 提取长期求职信息，支持新增、覆盖和显式遗忘。
- `StreamEvent`、同步 Queue 和 Graph 工作线程组成单次调用的事件通道。
- `CompiledStateGraph.stream()` 输出已提交的 `update` 事件，以及 `interrupt` 或 `final` 终止事件。
- `CompiledStateGraph.invoke()` 在内部消费同一事件流，只向调用者返回最终 Graph Output。
- `Agent.stream()` 支持普通用户输入和 `Command.resume`，CLI 流式模式能够完成包含多次 HITL 恢复的完整生命周期。
- Router 可以返回普通 Node 目标或 `Send(node, arg)`；Runtime 分别创建读取完整 State 的 PULL Task 和读取独立 `Send.arg` 的 PUSH Task。
- 每个 ToolCall 通过一个 `Send("tool_node", tool_call)` 形成独立 PUSH Task；同一 Super-step 的 Tool Updates 统一提交后，只进入一次后续 ModelNode。
- `SubGraphNode` 将 Compiled Graph 作为普通 Node 嵌入父图，通过 input/output mapper 完成父子 State 协议转换。
- 每层图使用独立 `checkpoint_ns`，并通过 `checkpoint_map` 保存祖先 Checkpoint 地址；嵌套图支持 Checkpoint、HITL、Memory 和 Streaming 继续向下工作。
- Multi-Agent 采用 `agent_as_tool`：`TaskTool` 根据 `subagent_type` 选择 `CompiledSubAgent`，调用子 Agent，并把最终 Assistant 内容返回为父 Agent 的 Tool 结果。
- 子 Agent 的 Interrupt 可以穿过 Tool 层冒泡到调用者，`Command.resume` 再按 Task 和 namespace 路由回真正产生 Interrupt 的子图。
- 已验证同一轮生成多个 `task` ToolCalls、多个子 Agent Task 分别中断与恢复、结果汇合后由 Supervisor 统一总结。
- `query_rag` 通过 HTTP 调用手写 RAG 的 `/retrieve` 接口，将检索结果作为 ToolMessage 返回模型。
- Skills 加载 `name + description + path` 元数据并注入 System Prompt，模型按需调用 `read_skill` 获取完整工作流说明。
- `demo.py` 串联 Skill、Supervisor、Resume Agent、HITL 和本地简历，完成简历与招聘要求的匹配分析。

当前求职 Agent 注册了四个 Tool：

- `read_resume`：按照 `resume_id` 读取本地简历。
- `search_project_evidence`：按照岗位要求检索项目证据。
- `query_rag`：调用独立 RAG 服务查询知识库。
- `read_skill`：按照 Skill 名称读取完整工作流说明，文件路径不交给模型拼接。

## 2. 核心运行流程

```text
User Input
    ↓
Agent.invoke(user_input, agent_state, thread_id, checkpoint_ns, checkpoint_id, context)
    ↓
构造包含 UserMessage 的 input_update
    ↓
CompiledStateGraph.invoke(state, input_update, thread_id, checkpoint_ns, checkpoint_id)
    ↓
按 thread_id 读取最新或指定 StateSnapshot
    ↓
通过 Reducer 将 input_update 合并到 State
    ↓
START → ModelNode
             ↓
       LocalChatModel
             ↓
       Parser / Adapter
             ↓
      AssistantMessage
        ├── 无 ToolCall → MemoryWriteNode → END
        └── 有 ToolCall → ToolArgsCompletionNode
                              ├── 仍缺少必需参数 → interrupt → 恢复后重新检查
                              └── 参数完整 → ToolReviewNode
                                                   ├── 需要审核 → interrupt → approve/edit/reject
                                                   └── 无需审核或审核完成
                                                               ↓
                                                    每个 ToolCall 生成一个 Send
                                                               ↓
                                                    多个 ToolNode PUSH Tasks
                                                               ↓
                                                    参数校验与 Tool/TaskTool 执行
                                                               ↓
                                                    ToolMessages 统一提交
                                                               ↓
                                                         一个 ModelNode
```

每个 Graph Super-step 遵循：

```text
Plan：确定本轮可执行 Nodes
→ Execute：各 Node 读取同一份旧 State 并返回 Partial Update
→ Update：一次性通过 Reducer 提交全部 Updates
→ Route：根据提交后的新 State 计算下一批 Nodes
```

Node 不直接修改共享 State。它的标准接口是：

```text
State → Partial State Update
```

## 3. 核心模块

| 文件 | 职责 |
|---|---|
| `state.py` | Message、ToolCall、AgentState、AgentStateUpdate 和 `add_messages` Reducer |
| `runtime.py` | 提取 State Reducer、按照 Super-step 规则合并 Node Update，并定义向 Node 注入 context/store/stream_writer 的 Runtime |
| `graph.py` | StateGraph Builder、Graph 编译、Task 调度、Transition、Router、interrupt 恢复和同步事件流 |
| `checkpoint.py` | StateSnapshot、PendingWrite、Checkpointer 接口、内存快照与 JSONL 本地持久化 |
| `hitl.py` | Interrupt、Command、Send、Task、TaskExecutionContext、PregelScratchpad、ContextVar 和 `interrupt()` |
| `memory.py` | MemoryItem、BaseStore、InMemoryStore 和 JSONL 长期记忆持久化 |
| `streaming.py` | 定义统一的 StreamEvent 事件协议 |
| `subgraph.py` | SubGraphNode、嵌套 Checkpoint namespace、父子 State 映射与跨层 Interrupt |
| `multi_agent.py` | CompiledSubAgent 元数据、统一 TaskTool、子 Agent 调用与结果转换 |
| `multi_agent_demo.py` | Supervisor + Resume Agent 的 `agent_as_tool` 组装与多 Task/HITL 演示 |
| `demo.py` | 简历与招聘要求匹配的完整业务演示 |
| `skill.py` | 加载 Skill 元数据并构造渐进式 Skills 提示词 |
| `skills/` | 保存各个 Skill 的 `SKILL.md` 与相关资源 |
| `model.py` | Qwen3 Model Adapter、4-bit PEFT LoRA 加载、消息格式转换和 ToolCall ID 标准化 |
| `parser.py` | 解析模型原始输出，提取 `content`、`name` 和 `arguments` |
| `nodes.py` | ModelNode、ToolArgsCompletionNode、ToolReviewNode、ToolNode 和 MemoryWriteNode |
| `routers.py` | 根据最新 State 决定进入 ToolNode 或结束 |
| `tools.py` | Tool 抽象、Tool Schema、参数校验、业务 Tool 和异常协议 |
| `tool_register.py` | Tool 注册、按名称查找以及 Tool Definition 导出 |
| `agent.py` | 组装 Resume Agent，提供 invoke/stream 接口，并把 User Input、thread、checkpoint 和 context 交给 Runtime |

## 4. Message 与 ToolCall 协议

内部 ToolCall 使用统一结构：

```python
{
    "id": "call_8f31",
    "name": "read_resume",
    "args": {
        "resume_id": "main",
    },
}
```

模型适配层负责把内部的 `args` 转换成 Qwen Chat Template 使用的 `arguments`。

Tool 执行后生成 ToolMessage：

```python
{
    "id": "msg_9a12",
    "role": "tool",
    "content": "...",
    "tool_call_id": "call_8f31",
    "name": "read_resume",
    "status": "success",
}
```

其中：

- `ToolCall.id` 标识一次具体 Tool 调用。
- `ToolMessage.tool_call_id` 将执行结果关联回原 ToolCall。
- `ToolMessage.status` 是 Runtime 内部的 `success/error` 状态。
- 当前 Qwen Adapter 不向裸模型发送 `status`；模型通过 ToolMessage 的 `content` 理解执行结果。

## 5. State 与 Reducer

`AgentState` 是 Graph 在某一时刻的已提交状态：

```python
from operator import add


class AgentState(TypedDict, total=True):
    messages: Annotated[list[Message], add_messages]
    model_call_count: Annotated[int, add]
```

字段更新分为两类：

- 带 Reducer 的字段：通过 Reducer 将旧值与一个或多个 Update 合并。
- 不带 Reducer 的字段：本轮无写入时保持不变；只有一个写入时直接覆盖；多个 Node 同时写入时拒绝提交。

`messages` 使用 `add_messages`：

- 新 ID 追加消息。
- 已存在的 ID 替换对应消息。
- 合并前复制旧消息，避免破坏上一版 State，为后续 Checkpoint 保留正确语义。

`model_call_count` 使用标准库的 `operator.add`：

- ModelNode 每完成一次模型调用，返回增量 `1`，而不是返回旧值加一后的总数。
- Runtime 从 `Annotated` 中取得 `add`，并在提交 State Update 时主动调用它。
- 同一 Super-step 中多个模型 Node 可以分别返回增量，最终统一累加到旧值上。

Reducer 映射只由 `CompiledStateGraph` 持有。`Agent` 只构造 `input_update`，不解析 State Schema，也不直接调用底层 `apply_updates()`。

## 6. Graph 与 Transition

`StateGraph` 是声明 Graph 的 Builder，负责注册：

```text
Nodes
+ Fixed Edges
+ Conditional Edges
```

Builder 注册阶段与 `compile()` 阶段共同完成核心结构校验；`compile()` 最终将可变 Builder 编译为独立的 `CompiledStateGraph`：

- Graph 必须存在 `START` 入口。
- Node 不能同时拥有 Fixed Edge 与 Conditional Edge。
- Edge 端点必须合法。
- `path_map` 的目标必须是已注册 Node 或 `END`。
- 所有注册 Node 必须能够从 `START` 到达。

编译后，Fixed Edge 和 Conditional Edge 都通过统一接口工作：

```python
transition.resolve_targets(state)
```

条件转移支持两种核心语义：

```text
有 path_map：Router 返回 router_key，再映射到 target Node
无 path_map：Router 直接返回已注册 Node 名称或 END
```

## 7. Tool 参数与错误处理

Tool Definition 中的 `function.parameters` 是 JSON Schema。它既用于告诉模型如何生成参数，也用于 Runtime 在 Tool 执行前校验参数。

```text
arguments
→ jsonschema.validate()
→ 合法：执行 Tool
→ 非法：ToolInvocationException
```

当前参数校验覆盖：

- 缺少必填参数。
- 参数类型错误。
- 出现 Tool Schema 未声明的额外参数。

可恢复错误分为两类：

```text
ToolInvocationException
    模型生成的 Tool 参数不符合 Schema

ToolExecutionException
    Tool 执行期间可预期、可以反馈给模型的业务错误
```

具体业务异常继承 `ToolExecutionException`。例如：

```text
ToolExecutionException
└── ResumeNotFoundException
```

ToolNode 将这些可恢复异常转换成 `status="error"` 的 ToolMessage，保留原 `tool_call_id`，再让 Graph 回到 ModelNode。未声明为可恢复错误的程序 Bug 或系统故障继续向上抛出，不会被伪装成正常 Tool 结果。

## 8. Checkpoint 与会话 State

`Agent` 本身保存的是可共享的组件：

```text
Model
+ Tool Registry
+ Compiled Graph
+ System Prompt
```

Checkpoint 开启后，每次调用必须提供 `thread_id`。Runtime 先查询对应历史：

```text
没有历史Snapshot
→ 使用initial_state

存在历史Snapshot
→ 使用Snapshot.state，忽略传入的initial_state
```

若本次存在新的 `input_update`，Runtime 将其合并到 State 并从 `START` 开始新一轮执行；若 `input_update=None`，Runtime 直接从 Snapshot 的 `pending_pull_node_names` 和 `pending_sends` 恢复，不重复执行已经完成的 Task。

每个成功提交的 Super-step 生成一个 Snapshot：

```text
thread_id
checkpoint_ns
checkpoint_id
parent_checkpoint_id
super_step
state
pending_pull_node_names
pending_sends
created_at
```

当前提供两种存储实现：

- `InMemoryCheckpointer`：用于快速调试和进程内恢复。
- `JsonlCheckpointer`：一行保存一个完整 Snapshot，支持进程重启后恢复。

JSONL 采用追加写入；`get/list` 逐行扫描，因此查询复杂度是 `O(n)`。当前版本面向单机小规模学习场景，不实现并发写锁、索引和文件压缩。

已通过最小恢复验证：ModelNode 完成后中断，恢复时只执行待执行的 ToolNode；再次恢复后继续 ModelNode，已提交的 Node 不会重复运行。

## 9. Long-term Memory

Checkpoint 和 Memory 解决不同层级的问题：

```text
Checkpoint：thread_id 范围内的 Graph 执行状态
Memory：user_id 范围内跨 thread 共享的长期信息
```

Agent 在每次 invoke 时通过 context 传入用户身份：

```python
context = {"user_id": "user_A"}
```

当前 Profile 使用固定定位：

```python
namespace = (user_id, "memories")
key = "profile"
```

读取流程：

```text
ModelNode
→ 按 namespace 查询 Store
→ 将 MemoryItem 序列化后追加到 SystemMessage 副本
→ 调用主对话模型
```

写入流程：

```text
ModelNode 生成最终回答
→ MemoryWriteNode 读取最新 UserMessage 和已有 Profile
→ 记忆提取模型调用专用 update_user_profile Tool
→ Schema 校验
→ Store.put 新增/覆盖，或 Store.delete 删除空 Profile
```

当前 Profile 保存用户明确表达的求职城市、目标岗位、技能和工作年限。用户明确要求忘记信息时，提取模型通过 `fields_to_delete` 提交字段级删除请求；没有提到某个字段不代表删除。模型只生成结构化的修改请求，最终的字段白名单校验和 Store 写入由代码执行。

Memory 不写入 AgentState，因此不会随每个 Checkpoint 重复复制。当前提供进程内的 `InMemoryStore` 和追加写入的 `JsonlStore`。

## 10. Graph Event Streaming

当前 Streaming 暴露 Graph 的执行过程，同时保留原有的同步 `invoke()` 调用方式：

```text
Agent.stream()
→ CompiledStateGraph.stream() 创建 Queue 和 Graph 工作线程
→ _run() 执行 Graph，并通过 Runtime.stream_writer 写入已提交的 update 事件
→ 工作线程根据 _run() 的返回结果写入 interrupt 或 final 终止事件
→ 调用者遍历 StreamEvent 生成器
```

事件统一使用 `StreamEvent(event_type, payload)`：

- `update`：某个 Task 所在 Super-step 已成功提交，携带 `super_step`、`node_name` 和 Partial State Update。
- `interrupt`：本次执行因 HITL 暂停，携带 Interrupt 数据和当前 Graph Output。
- `final`：本次 Graph 正常完成，携带最终 Graph Output；当前 Agent 的 Output 是完整 State。
- 工作线程中的异常不包装为业务事件，而是通过 Queue 交回调用线程并重新抛出。

`CompiledStateGraph.invoke()` 消费同一个 `stream()`，忽略中间事件并返回终止事件中的 Output。因此 `invoke()` 与 `stream()` 共用唯一的 `_run()` 执行逻辑。

当前本地模型使用阻塞式 `model.generate()`，不产生 `token` 事件。本项目将 Streaming 核心边界定义为 Graph 执行事件流；模型 token 级输出主要属于推理适配与前端体验，可在后续 FastAPI 演示阶段按需接入模型 Streamer，而不影响当前 Runtime 的 Streaming 语义。

## 11. Subgraph

`SubGraphNode` 把一个已经编译的 Graph 适配成父图中的普通 Node：

```text
Parent State
→ input_mapper
→ Child Graph
→ output_mapper
→ Parent State Update
```

子图不是父图 State 的简单函数调用。每次 SubGraphNode Task 都生成独立的 namespace segment：

```text
{node_name}:{task_id}
```

它与父图的 `checkpoint_ns` 拼接后形成子图地址；`checkpoint_map` 保存从根图到直接父图的祖先 Checkpoint 地址。每层图都维护自己的 Snapshot 和 pending writes，因此深层 Interrupt 恢复时只需要重建未完成的嵌套 Task，不需要从最外层重新执行全部工作。父图通过 NodeRuntime 把 `context`、`stream_writer` 和尚未由本层消费的 `resume_map` 继续传给子图。

## 12. Multi-Agent

当前 Multi-Agent 使用与 DeepAgents 主流实现一致的 `agent_as_tool` 形态：

```text
Supervisor ModelNode
→ task(description, subagent_type) ToolCall
→ ToolNode PUSH Task
→ TaskTool 选择 CompiledSubAgent
→ 子 Agent CompiledStateGraph
→ 子 Agent 最终 Assistant 内容
→ 父图 ToolMessage
→ Supervisor ModelNode
```

`CompiledSubAgent` 保存子 Agent 的名称、用途描述和可执行 Agent；`TaskTool` 向 Supervisor 暴露统一的 `task` Tool Schema。Supervisor 不直接看到子 Agent 内部的普通 Tools，只根据 `subagent_type` 委派完整任务。子 Agent 仍拥有自己的 Model、Tools、Graph 和 HITL policy。

同一轮的多个独立 `task` ToolCalls 会形成多个 PUSH Tasks。它们读取各自的 ToolCall 输入，拥有不同的 Task ID 和子图 namespace；全部完成后，多个 ToolMessage 在同一个 Super-step 中统一合并，再只调度一次 Supervisor ModelNode。当前 Runtime 保留这种并行执行语义，但底层 Python 循环仍顺序调用各 Task，不实现线程池、进程池或分布式并发。

## 13. 运行方式

当前运行环境需要：

```text
Python 3.11
PyTorch
Transformers
PEFT
bitsandbytes
jsonschema
requests
streamlit
```

进入项目目录：

```bash
cd handwritten-agent
```

激活环境：

```bash
conda activate ENV_agent
```

确认本地模型位于：

```text
models/Qwen3-4B
```

当前本地模型使用贪心生成，以提高 ToolCall 名称和参数的稳定性。

家电故障诊断应用还需要同级 `appliance-support-sft` 项目中的 Qwen3-8B、LoRA Adapter 与训练 Chat Template，路径配置见 `appliance_support.py`。

如需使用 `query_rag`，先在 `handwritten-rag` 目录启动检索服务：

```bash
conda activate ENV_rag
python rag_service.py
```

运行：

```bash
python agent.py
```

运行 Multi-Agent 演示：

```bash
python multi_agent_demo.py
```

运行简历与招聘要求匹配演示：

```bash
python demo.py
```

输入：

```text
exit
```

结束交互。

代码默认使用 `device="mps"`。没有可用 MPS 的环境需要在 `agent.py` 中改为 `device="cpu"` 或其他可用设备。

## 14. 当前边界

当前版本有意不实现以下生产能力：

- 数据库、远程 Checkpointer、并发写入和分布式一致性。
- Pending writes 的异步持久化。
- 分布式执行与并发 Tool 调度。
- 完整 Agent 服务化、鉴权、配额和 Tool 沙箱。
- 完整的自动化测试、评测和可观测性体系。
- 向量化 Memory 检索、TTL、自动压缩、异步写入和数据库 Store。
- 模型 token 级输出以及 SSE/WebSocket 前端传输。
- PULL/PUSH Tasks 目前顺序执行，只实现同一 Super-step 的并行状态语义，不提供真实并发调度。
- Multi-Agent 当前实现统一 `task` Tool、静态子 Agent 注册和最终文本返回，不实现动态创建 Agent、远程 Agent 或跨进程协作。

当前 HITL 只接受结构化的 `Command(resume={interrupt_id: resume_value})`。自然语言反馈需要在调用 Runtime 前由规则或 LLM 转换为结构化 `resume_value`。`Command.update`、`Command.goto` 和 `Command.graph` 暂未实现。

## 15. 设计文档

- [`ARCHITECTURE.md`](ARCHITECTURE.md)：完整架构设计与协议说明。
- [`HITL_DESIGN.md`](HITL_DESIGN.md)：HITL Runtime、工具补参与审核流程及当前边界。
- [`ADVANCED_CORE_PLAN_2026-08-06.md`](ADVANCED_CORE_PLAN_2026-08-06.md)：Checkpoint、HITL、Memory、Streaming、Subgraph 和 Multi-Agent 计划。
- [`TODAY_PLAN_2026-08-04.md`](TODAY_PLAN_2026-08-04.md)：Graph Runtime 核心实现计划。

## 16. 项目定位

这不是生产级 Agent Framework，也不是 LangGraph 的源码复刻。

它是一个规模可控但运行语义完整的学习型 Runtime，用于直接观察并解释：

```text
模型如何提出 Tool 调用
Runtime 如何赋予调用身份
Tool Result 如何回到消息历史
State 如何按 Reducer 更新
Graph 如何按 Super-step 执行
Router 如何决定下一跳
错误如何转化为可恢复 Observation
Checkpoint 如何保存并恢复 State 与下一批 Nodes
Task 如何保存部分执行结果并在中断后恢复
interrupt 如何暂停 Node 并通过 Command.resume 返回人工反馈
Runtime 如何向 Node 注入 context 和 Store
Checkpoint 与跨 thread 长期 Memory 如何分工
长期信息如何被提取、读取、更新和显式遗忘
普通目标与 Send 如何分别生成 PULL/PUSH Tasks
子图如何建立独立 Checkpoint namespace 并跨层恢复
Supervisor 如何通过统一 task Tool 委派多个子 Agent Tasks
Agent 如何通过 HTTP Tool 调用独立 RAG 检索服务
模型如何发现 Skill 元数据并按需读取完整 SKILL.md
```

完成高级核心能力后，再使用 LangGraph 重构同一业务流程，对照理解框架为这些底层机制提供的抽象。
