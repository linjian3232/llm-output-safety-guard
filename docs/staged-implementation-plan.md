# 中文大模型回复安全检测：分阶段实战计划

## 原则

每次只完成一个可观察、可运行的阶段。先解释该阶段在真实工程中解决什么问题，再实现必要代码并完成验证。每个完成阶段单独提交 Git。

## 阶段 0：独立环境

- 建立独立的 `pyproject.toml`、`uv.lock` 和 `.venv`，不与 Iris 项目共享依赖。
- 验证 PyTorch、Transformers、PEFT 和 Datasets 可以导入。
- 忽略原始数据、处理数据、模型产物和实验报告。

**状态：已完成。**

## 阶段 1：下载并认识预训练模型

- 下载 `hfl/chinese-roberta-wwm-ext` 及其 Tokenizer，并记录模型 revision。
- 使用一条中文模型回复观察 token、`input_ids`、`attention_mask` 和隐藏向量。
- 不训练、不下载安全数据。

## 阶段 2：收集与审计原始安全数据

- 下载 XGuard 数据集并记录数据版本。
- 只读统计 `response`、`stage`、`label`、语言和文本长度分布。
- 生成不包含原始敏感正文的数据审计报告。

## 阶段 3：构建可复现训练集

- 仅保留 `sample_type=general`、`stage=r`、非空且以中文为主的回复。
- 每个标签最多抽取 400 条，固定 `seed=42`。
- 按标签分层切分 train/validation/test 为 80/10/10，保存数据版本、样本 ID 和标签映射。

## 阶段 4：冻结编码器基线

- 使用中文 RoBERTa 序列分类模型，冻结编码器，仅训练分类头。
- 输出可训练参数数量、验证集 Macro-F1、混淆矩阵和逐类别指标。

## 阶段 5：LoRA / PEFT 微调

- 将 LoRA 注入 BERT attention 的 `query` 与 `value`，分类头保持可训练。
- 默认参数为 `r=8`、`lora_alpha=16`、`lora_dropout=0.05`。
- 保存 adapter、Tokenizer、标签映射和训练配置，而不是复制基础模型。

## 阶段 6：阈值与外部评测

- 计算 `unsafe_score = 1 - P(sec)`。
- 在验证集上选择安全文本误拦率不超过 10%、且不安全回复召回率最高的阈值。
- 在 CCAC track2 上做独立安全/不安全评测，绝不混入训练数据。

## 阶段 7：Adapter 推理与 API

- FastAPI 在启动时加载基础模型、LoRA adapter 和版本化阈值配置。
- `POST /moderate` 返回 `allow/block`、风险类别、类别置信度、`unsafe_score` 与模型版本。
- 提供 `/health`，并验证真实 HTTP 调用。

## 资源与边界

- 当前机器仅以 CPU 训练；2GB 显存不用于本地微调生成式 Guard 模型。
- 数据和模型仅用于合法的内容安全研究；原始数据与训练产物不提交 Git。
- 后续使用云 GPU 时，再扩展到生成式 Guard 模型和 QLoRA。
