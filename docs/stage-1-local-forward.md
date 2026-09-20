# 阶段 1：uv 环境与本地前向推理

## 为什么做

模型文件已固定，但 Python 还需要能运行它的库。独立环境让本项目的 PyTorch 和 Transformers 版本不影响其他项目；锁文件让另一台机器按同一套依赖版本复现。第一次前向推理还要验证：配置、权重和 Tokenizer 能在无 Hugging Face 网络连接时配合使用。

本阶段运行的是预训练模型原有的**中文掩码填空**任务。它只证明本地加载与前向计算可用，输出的词不是安全分类结果；安全分类头要在后续训练阶段建立。

## 三个项目文件各做什么

| 文件或目录 | 作用 | 是否提交 Git |
| --- | --- | --- |
| `pyproject.toml` | 声明项目需要 Python 3.12、PyTorch 和 Transformers | 是 |
| `uv.lock` | 记录解析出的具体依赖版本与下载校验信息 | 是 |
| `.venv/` | 当前电脑安装这些依赖的独立环境 | 否 |

`uv init --bare --no-workspace` 最初只建立 `pyproject.toml`。`uv add --no-sync 'torch>=2.6,<3' 'transformers>=4.57,<5'` 写入直接依赖并生成 `uv.lock`，但先不安装。本机第一次解析得到 46 个包；实际版本以仓库的锁文件为准。新电脑从 Git 获取项目时已有前两份文件，不必再运行 `uv init` 或 `uv add`。

## 由学习者运行：安装与观察

在 PowerShell 中进入项目目录：

```powershell
Set-Location 'D:\projectStore\codeXProject\python-basic-learning\projects\llm-output-safety-guard'
uv sync --locked --no-python-downloads
Test-Path .\.venv
```

`uv sync` 按锁文件创建 `.venv` 并安装依赖。`--locked` 要求锁文件已经与项目依赖声明一致，不悄悄重写它；`--no-python-downloads` 使用本机已有的 Python。预期安装成功，最后一行是 `True`。首次安装 PyTorch 可能需要下载较大的包。

然后亲自运行本地前向推理：

```powershell
uv run --no-sync --no-python-downloads python .\scripts\offline_forward.py
```

`uv run` 用项目的 `.venv` 运行命令；`--no-sync` 表示这次直接使用刚才同步好的环境。脚本将 `HF_HUB_OFFLINE=1` 设在加载库之前，并对 Tokenizer 与模型都传入本地目录和 `local_files_only=True`，缺文件时不会在线下载。

预期输出包含实际模型类 `BertForMaskedLM`、设备 `cpu`、输入文本、两个张量形状以及一个预测 token。`input_ids` 的形状为 `[样本数, token 数]`；`logits` 的形状为 `[样本数, token 数, 词表大小]`。本模型 `config.json` 中词表大小为 21128，因此最后一维应为 21128。预测的具体词以你真实运行的输出为准。

运行时请保留完整命令输出和报错文本，便于核对与排障。本次实际验收记录如下。

## 本次验收记录（2026-09-20）

- uv 使用本机 CPython 3.12.7；`uv add --no-sync` 解析 46 个包，锁文件记录 `torch 2.14.0`、`transformers 4.57.6`。
- 学习者运行前向推理后，项目 `.venv/Scripts/python.exe` 已存在；模型目录仍是本地 `artifacts/base-model/`。
- 实际模型类：`BertForMaskedLM`；设备：`cpu`；输入：`今天的天气很[MASK]。`。
- `input_ids` 形状为 `(1, 10)`：一条输入，共 10 个 token（包括特殊 token）。`logits` 形状为 `(1, 10, 21128)`：每个位置对词表 21128 个 token 各有一个分数。掩码位置预测为 `好`（ID `1962`）。

加载时报告 4 个未使用的权重：`bert.pooler.dense.bias`、`bert.pooler.dense.weight`、`cls.seq_relationship.bias`、`cls.seq_relationship.weight`。这是当前加载类与权重内容的差异：已安装的 Transformers 4.57.6 中，`BertForMaskedLM` 不建立 pooler，只保留掩码词预测头；下一句预测头也不属于本次填空任务。警告未阻止本地加载和前向计算；不要为消除警告而改写基础权重。[对应版本的 BERT 实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/bert/modeling_bert.py)

本阶段已完成本地加载验收。它仍未产生安全分类分数或训练后的模型。
