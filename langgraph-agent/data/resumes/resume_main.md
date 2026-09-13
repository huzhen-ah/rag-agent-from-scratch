# 胡真｜大模型应用 / RAG / Agent 工程师

## 基本信息

- 所在地：广州
- GitHub：huzhen-ah

## 专业概述

具备多年 NLP 与算法工程经验，覆盖智能客服、语音识别、声纹识别与传感器时序建模，能够独立完成数据建设、模型训练、推理服务和端侧部署。近期完成 Mini LLM、RAG 与 Agent 项目的系统实践。

## 个人项目

### Mini LLM 核心训练与推理全链路｜2026

- 项目地址：https://github.com/huzhen-ah/mini-llm-demo
- 分别使用 Keras / TensorFlow 与 PyTorch 实现 Decoder-only Transformer 及核心组件。
- 完成 Byte-level BPE、Next-token 预训练、LoRA-SFT 与 LoRA-DPO 的完整训练链路。
- 实现 Prefill / Decode 推理与 KVCache，支持不同 Prompt 长度的批量生成。

### RAG 与 Agent Runtime 实践｜2026

- 项目地址：https://github.com/huzhen-ah/rag-agent-demo
- 实现 Dense Retrieval、BM25、RRF、Reranker 与检索评估，并封装 RAG HTTP 服务。
- 使用纯 Python 实现包含持久化恢复、HITL、Memory、Streaming、Subgraph、Multi-Agent 与 Skills 的 Graph Agent Runtime。
- 使用 LangGraph 重构同一业务流程，并将 RAG 服务接入多 Agent 协作链路。

## 工作经历

### 云米｜算法工程师｜2021.06-2026.06

#### 智能家居 NLP 与语音算法

- 负责领域数据清洗、BERT 意图识别模型训练及 Sanic 推理服务；后续参与智能家居语音识别与声纹识别预研。
- 参考 WeNet 使用 Keras 实现端到端 ASR，引入 AISHELL 等中文开源语料；后转用 FunASR 进行领域微调，加入设备名称与控制指令热词，并使用 Sanic 封装 HTTP 服务。
- 使用 VoxCeleb、CN-Celeb、AISHELL 及领域录音，根据论文复现 ECAPA-TDNN 声纹模型并使用 AAM-Softmax 训练；同设备验证集 EER 约 2%。

#### 60 GHz 毫米波雷达血压估计

- 参与公司、医院及养老院数百名受试者的数据采集，负责数据清洗、时间对齐、样本构建和模型训练。
- 对比多种时序模型和训练方案，并使用 PulseDB 进行预训练实验；最终采用轻量化 1D U-Net 同时预测 SBP 与 DBP，按受试者划分数据避免数据泄漏。
- 测试集 SBP、DBP 均达到 BHS A 级；完成 INT8 量化和 TFLite 部署，芯片上单个 10 秒窗口推理约 50 ms，并集成至数十台雷达样机。

#### 60 GHz 毫米波雷达心率检测预研

- 参与数百名受试者的数据采集，负责异常数据清洗、时间戳对齐与训练样本构建；使用信号处理团队提供的特征训练 1D U-Net 心率回归模型。
- 完成模型评估、TFLite 转换与终端交付，为传统周期信号处理方案提供学习型模型验证。

#### 超声波杯满即停感知预研

- 独立采集 50 多种杯型数据，覆盖不同材质、摆放位置、水流状态及伸手干扰；将复数回波拆分为实部、虚部双通道，训练无杯、高杯、低杯、伸手四分类模型。
- 设计约 1.6 万参数的轻量 ResNet，完成 INT8 量化与 TFLite 转换；模型在 ESP32 上单次推理约 100 ms，对未参与训练的新杯型保持稳定效果。

### 健客网｜NLP 算法工程师｜2018.11-2021.06

#### 大健客服智能问答系统｜BERT、Neo4j、Redis、Bottle、IVF-PQ

- 面向 Web 与 App 医药售前、售后咨询，提供药品、疾病、优惠及物流等自动问答能力，包含知识图谱与 FAQ 两类检索链路。
- 独立负责数据标注、BERT 意图识别与实体识别模型训练、上线和迭代。
- 设计药品-疾病知识图谱，通过实体模糊匹配、别名字典与 Cypher 模板完成 Neo4j 查询。
- 自研关键词倒排索引，封装 IVF-PQ 向量检索组件，实现 FAQ 关键词及语义匹配。
- 实现对话流程、固定话术、Redis 会话隔离、兜底与人工转接机制。
- 使用 Bottle 封装 HTTP 服务，并实现无需重启服务的模型在线更新。
- 系统日均承接约 10 万-20 万次请求，核心业务指标为药品购买转化率。
