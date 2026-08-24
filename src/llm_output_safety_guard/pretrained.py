"""下载并观察后续安全分类任务要复用的中文 RoBERTa 基座模型。

本模块刻意只做一次前向推理，不做训练也不输出“安全/不安全”结论。
这样可以先把“文本如何变成 Transformer 可计算的张量”看清楚；阶段 4
才会在这个编码器之上加入安全分类头，阶段 5 再将训练方式改为 LoRA。
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

# 模型名只是一个可变的远端标签；固定到 commit hash 才能保证另一台电脑
# 下载到完全相同的 tokenizer 配置与模型权重。这是实验可复现的第一层保障。
MODEL_ID = "hfl/chinese-roberta-wwm-ext"
MODEL_REVISION = "5c58d0b8ec1d9014354d691c538661bf00bfdb44"

# 权重体积大、可重新下载，属于本机缓存而非源代码，因此放入 Git 忽略的 artifacts 目录。
DEFAULT_CACHE_DIR = Path("artifacts/huggingface")


@dataclass(frozen=True)
class PretrainedModelInspection:
    """一次编码器前向推理中，最值得观察的输入与输出事实。

    此类没有保存真正的张量或模型权重，只保存可 JSON 序列化的摘要。因此脚本
    可以输出它供人阅读，而不必把数百 MB 的模型对象或 PyTorch 张量写进报告。
    """

    model_id: str
    model_revision: str
    tokens: list[str]
    input_ids: list[int]
    attention_mask: list[int]
    hidden_state_shape: tuple[int, int, int]


def build_inspection(
    *,
    tokens: list[str],
    input_ids: list[int],
    attention_mask: list[int],
    hidden_state_shape: tuple[int, int, int],
) -> PretrainedModelInspection:
    """组装可序列化的观察结果，并把模型身份与观察内容绑定。

    单独提取这个纯函数有两个目的：一是单元测试无需下载模型也能验证输出契约；
    二是让学习重点聚焦在 ``tokens / input_ids / mask / shape`` 的对应关系上。
    """
    return PretrainedModelInspection(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokens=tokens,
        input_ids=input_ids,
        attention_mask=attention_mask,
        hidden_state_shape=hidden_state_shape,
    )


def inspect_pretrained_model(
    text: str,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_length: int = 128,
) -> PretrainedModelInspection:
    """加载固定版本的模型，对一条文本完成推理并返回教学用摘要。

    ``max_length`` 是单条文本允许进入模型的 token 上限。超长回复会被截断；
    当前只是观察，后续训练阶段会把它作为可版本化的训练配置，而不是散落在代码中。
    """
    # Tokenizer 与模型必须来自同一 revision：两者的词表和 embedding 行号才能对齐。
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )

    # return_tensors="pt" 将 Python 列表直接转为 PyTorch 张量。
    # input_ids 是词表索引，并不代表数值大小上的语义；模型会据此查 embedding 表。
    # attention_mask 中 1 表示真实 token，0 表示批处理时为对齐长度补上的 padding。
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)

    # eval() 关闭 dropout 等训练期随机行为，使同一输入的观察结果稳定。
    model.eval()
    # inference_mode() 明确告诉 PyTorch 不需要反向传播图，节省内存和计算。
    # 这与后续微调不同：微调必须保留梯度，才能根据 loss 更新分类头或 LoRA 参数。
    with torch.inference_mode():
        output = model(**encoded)

    # 张量第一维是 batch。示例只输入一条文本，所以取 [0] 转成便于打印的 Python 列表。
    input_ids = encoded["input_ids"][0].tolist()
    return build_inspection(
        # 反查词表，帮助我们把抽象整数 ID 与原始文本切分结果建立联系。
        tokens=tokenizer.convert_ids_to_tokens(input_ids),
        input_ids=input_ids,
        attention_mask=encoded["attention_mask"][0].tolist(),
        # last_hidden_state 是编码器的核心产物：(批大小, token 数, 隐藏维度)。
        # 它是每个 token 融合上下文后的向量，还不是安全类别或类别概率。
        hidden_state_shape=tuple(output.last_hidden_state.shape),
    )
