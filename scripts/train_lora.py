"""阶段 5 入口：沿用已学的训练流程，将参数训练策略改为 LoRA + 分类头。"""

from train_frozen_baseline import main


if __name__ == "__main__":
    main(method="lora")
