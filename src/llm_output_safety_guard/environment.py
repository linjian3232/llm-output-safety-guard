"""检查后续模型实验依赖是否已安装。

这里只读取已安装包的版本元数据，不会 import PyTorch。两者不能混为一谈：
包显示已安装只说明 Python 依赖解析完成；真正 import torch 还依赖 Windows
运行库和原生 DLL 能否加载。阶段 1 的模型前向推理才是后者的实际验证。
"""

from importlib import metadata

# 这些包分别负责张量计算、预训练模型、参数高效微调和训练数据集处理。
REQUIRED_PACKAGES = ("torch", "transformers", "peft", "datasets")


def check_runtime_dependencies() -> dict[str, str]:
    """返回已安装依赖的版本，作为环境初始化的快速健康检查。"""
    return {package: metadata.version(package) for package in REQUIRED_PACKAGES}
