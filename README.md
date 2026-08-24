# LLM Output Safety Guard

一个按阶段推进的中文大模型回复安全检测实战项目。

当前已完成阶段 1：独立环境、依赖锁定，以及中文 RoBERTa 基座模型的本地观察。数据下载、训练和 API 会在后续阶段逐步加入。

## 阶段 1：观察预训练编码器

在 PowerShell 中执行下面的命令。前两行只影响当前终端，用于避免 GBK 代码页把中文 token 显示为乱码：

```powershell
chcp 65001
$env:PYTHONIOENCODING = "utf-8"
uv run python scripts\inspect_pretrained_model.py
```

脚本固定下载 `hfl/chinese-roberta-wwm-ext` 的 revision
`5c58d0b8ec1d9014354d691c538661bf00bfdb44` 到 Git 忽略的
`artifacts/huggingface`，然后打印：

- `tokens`：Tokenizer 切分出的词表单元；
- `input_ids`：每个 token 对应的词表整数 ID；
- `attention_mask`：真实 token 为 `1`，补齐位置才为 `0`；
- `hidden_state_shape`：`(batch, sequence_length, hidden_size)`。

对于示例文本，输出形状为 `(1, 24, 768)`：一条样本、24 个 token、每个 token
对应 768 维上下文向量。加载时出现的 `UNEXPECTED` Masked-LM 层是预训练预测头；本阶段
故意使用 `AutoModel`，只保留编码器，因此该提示符合预期。它也说明基座模型尚未具备安全分类能力。
