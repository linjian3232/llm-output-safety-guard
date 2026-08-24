"""阶段 1 的命令行入口：观察一条中文回复经过 RoBERTa 编码器的过程。

这是“人运行的脚本”，不是可复用业务模块；真正的模型加载与推理逻辑放在
``src/`` 中，后续训练和 API 才能复用，避免把实现散落在多个脚本里。
"""

import json
from dataclasses import asdict

from llm_output_safety_guard.pretrained import inspect_pretrained_model

if __name__ == "__main__":
    # 选取无风险的示例回复，仅用于理解编码流程；它不参与训练，也没有人工标签。
    inspection = inspect_pretrained_model("该回复拒绝提供违法操作，并建议寻求合法帮助。")
    # ensure_ascii=False 保留中文 token；PowerShell 请先切到 UTF-8，避免 GBK 显示乱码。
    # asdict() 将 dataclass 转成普通字典，便于 JSON 输出，也模拟后续实验报告的写法。
    print(json.dumps(asdict(inspection), ensure_ascii=False, indent=2))
