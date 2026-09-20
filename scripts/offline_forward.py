"""阶段 1：只用本地文件，让预训练模型完成一次中文填空前向计算。"""

import os
from pathlib import Path

# 在导入 Transformers 前关闭 Hugging Face Hub 网络请求；下面还会要求加载器只读本地文件。
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


# __file__ 是当前脚本路径；向上两层就是项目根目录，不依赖运行命令时所在的位置。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "artifacts" / "base-model"


def main() -> None:
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"本地模型目录不存在：{MODEL_DIR}")

    # Tokenizer 将文字转换成 token ID；模型根据本地配置和 PyTorch 权重建立网络。
    # local_files_only=True 保证这两个加载动作不会去 Hub 补下载缺失的文件。
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR, local_files_only=True)
    model.to("cpu")
    model.eval()  # 关闭 dropout 等只用于训练的行为，使本次观察更稳定。

    # 从 Tokenizer 读取掩码符号，避免对另一个模型硬编码「[MASK]」。
    if tokenizer.mask_token is None or tokenizer.mask_token_id is None:
        raise ValueError("这个模型的 Tokenizer 没有掩码 token，不能做本次填空演示")
    sentence = f"今天的天气很{tokenizer.mask_token}。"
    inputs = tokenizer(sentence, return_tensors="pt")

    # input_ids 是整数张量：第一维是样本数，第二维是每个样本的 token 数。
    mask_positions = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(
        as_tuple=True
    )[0]
    if mask_positions.numel() != 1:
        raise ValueError(f"预期恰好一个掩码，实际找到 {mask_positions.numel()} 个")
    mask_position = int(mask_positions.item())

    # 前向计算只读取权重；inference_mode 不记录训练所需的梯度，也不更新参数。
    with torch.inference_mode():
        logits = model(**inputs).logits

    # logits 的第三维为词表中每个候选 token 的分数，取掩码位置分数最高的 token。
    predicted_id = int(logits[0, mask_position].argmax().item())
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_id)

    print(f"模型目录：{MODEL_DIR}")
    print(f"实际模型类：{type(model).__name__}")
    print(f"设备：{next(model.parameters()).device}")
    print(f"输入文本：{sentence}")
    print(f"input_ids 形状：{tuple(inputs['input_ids'].shape)}")
    print(f"logits 形状：{tuple(logits.shape)}")
    print(f"掩码位置预测 token：{predicted_token}（ID={predicted_id}）")


if __name__ == "__main__":
    main()
