# 阶段 5：LoRA + 分类头训练

## 操作顺序：5 步试跑，再完整训练

继续使用阶段 3 的 train 640 条、validation 80 条，长度 128、batch size 8；加载同一固定版本本地基础模型。先核对 LoRA 的实际注入位置和可训练参数，再完成 5 次更新并验证。真实试跑现已核验，当前下一小目标为完整 3 epoch 训练；真实数据训练仍由学习者执行，test 不读取。

本次从 `artifacts/base-model/` 独立开始，分类头重新初始化；阶段 4 保存的模型保留用于比较，不作为 LoRA 的起点。这样两种方法都从相同基础权重和任务数据出发。

## 先理解新增的部分

| 部分 | 阶段 4 基线 | 阶段 5 LoRA |
| --- | --- | --- |
| 原有编码器权重 | 冻结 | 冻结 |
| 编码器内新增的小分支 A/B | 无 | 可训练 |
| 分类头 | 可训练 | 可训练，并随 adapter 保存 |
| 训练、验证数据与指标 | 已固定 | 复用同一数据接口和计算方式 |

LoRA 将原有线性计算加上一条小分支：`原有结果 + (alpha/r) × B(A(输入))`。原有权重保持不变，A/B 的参数在训练中改变。编码器最终表示因此能够适应当前任务。

### 本项目的配置

| 设置 | 值 | 必需理解的意义 |
| --- | ---: | --- |
| `r` | 8 | A/B 中间维度；决定新增参数容量，是固定超参数 |
| `alpha` | 16 | 小分支结果的缩放设置，当前缩放为 alpha/r=2；是固定超参数 |
| `dropout` | 0.05 | 训练时对 LoRA 分支输入做随机置零，评估时关闭；不是冻结 5% 的参数 |
| 目标模块 | attention 的 query/value | 只在核对过的线性模块中加入分支 |
| `modules_to_save` | classifier | 确保分类头也训练并被写入 adapter |
| LoRA 学习率 | 0.0002 | 更新 A/B 的优化器设置 |
| 分类头学习率 | 0.001 | 与阶段 4 分类头学习率一致 |

`r/alpha/dropout` 在本轮训练中不自动改变；真正被优化器更新的是 A/B 和分类头的数值。配置含义依据 [PEFT LoRA 文档](https://huggingface.co/docs/peft/package_reference/lora)。

脚本先遍历实际模型，检查每个 BERT 层确实有 query/value 线性模块，再把完整模块路径交给 PEFT。当前模型 12 层，预期共 24 个目标模块。每个 768×768 权重对应两个 LoRA 矩阵，参数量为 768×8+8×768=12,288；24×12,288=294,912，加分类头 1,538，总可训练参数预期 296,450。实际以用户运行报告为准。

参数高效指相对于全参数微调少更新很多参数。与只训练 1,538 个分类头参数的基线相比，LoRA 需要反向传播穿过编码器内计算，通常耗时和内存更多，不保证 CPU 上更快或指标更好。

## 指令与作用

### 1. 同步锁定依赖

本阶段已执行 `uv add --no-sync --no-python-downloads 'peft>=0.17,<1'` 更新 pyproject.toml 和 uv.lock。解析结果 PEFT=0.21.2，新增 accelerate=1.15.0、psutil=7.2.2；torch=2.14.0、Transformers=4.57.6 未变。工程验证时已同步本机环境。学习者或换机器时按锁文件确认环境：

```powershell
uv sync --locked --no-python-downloads
```

这一步安装 Python 依赖，需要可用包源；`--locked` 不自行修改依赖版本，`--no-python-downloads` 使用已有 Python。已同步的本机重复执行通常只核对环境。模型和数据始终从本地读取，不经由 uv 下载。

### 2. 执行 5 步 LoRA 训练

```powershell
uv run --no-sync --offline --no-python-downloads python .\scripts\train_lora.py --max-steps 5
```

`train_lora.py` 是 LoRA 入口，复用阶段 4 的数据加载、编码、训练、验证和报告逻辑。默认仍是 5 步，不会自动开始完整 3 轮。阶段 4 的入口仍按原冻结基线策略运行。

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--max-steps` | 5 | 当前只更新 5 次；0 才会进入完整 epoch 模式 |
| `--batch-size` | 8 | 5 步处理当前训练集中的 40 条样本 |
| `--max-length` | 128 | 与基线保持一致；已知长文本截断局限仍存在 |
| `--epochs` | 3 | 完整模式轮数；当前会先被 5 步上限截停 |
| `--lora-r` | 8 | 新增矩阵的中间维度 |
| `--lora-alpha` | 16 | 小分支缩放设置 |
| `--lora-dropout` | 0.05 | LoRA 分支 dropout |
| `--learning-rate` | 0.0002 | LoRA 参数学习率 |
| `--head-learning-rate` | 0.001 | 分类头学习率 |
| `--seed` / `--threads` | 42 / 4 | 随机种子和 CPU 线程数，与基线相同 |

分类头仍可能出现新初始化提示，属于独立实验的新任务头；加载异常或额外模块不匹配则需保留日志排查。

### 3. 查看无原文报告

```powershell
Get-Content .\reports\stage-5-smoke.json
```

报告继承阶段 4 的数据和模型文件指纹、形状相关配置、训练/验证指标，并增加 PEFT 版本、LoRA 配置、目标模块路径、LoRA/分类头参数量和更新检查。可训练参数全名放在报告中；终端只打印名称示例，减少无关输出。

## 本次验收观察点

- 目标应为 24 个 query/value 线性模块，可训练参数预期 296,450（LoRA 294,912 + 分类头 1,538）。
- 可训练参数名称应包含 lora_A/lora_B 和 classifier 的 modules_to_save；不应包含其他未计划的原有权重。
- 首批形状仍为 input_ids=(8,128)、labels=(8,)、logits=(8,2)。LoRA 改变适配方式，数据接口和分类任务不变。
- 完成 5 个有限 loss，验证使用 80 条；观察实际 CPU 耗时，不要求短运行指标超过完整基线。
- `LoRA 参数已更新`、`分类头已更新`、`冻结参数梯度均为空`应为 True。新分支初始化后，不代表 A/B 每个数值在第一步都一定改变；检查的是分支参数中存在真实更新。

只训练分类头时，基线将编码器设为 eval；LoRA 使用模型的训练行为，保留 LoRA dropout 和编码器常规 dropout。冻结控制的是参数更新，train/eval 控制的是运行行为，两者不同。本次是两套明确训练配置的比较，不把结果差异解释成严格排除了所有其他因素的因果结论。

学习者已提供终端输出，已核对本地 stage-5-smoke.json。真实结果记录如下，完整效果尚待实际训练后观察。

## 真实 5 步试跑验收（2026-10-16）

已核对报告中的训练/验证文件哈希、基础模型文件指纹，均与阶段 4 相同；累计 5 步处理 40 条训练样本，尚未完成第一轮；验证 80 条，未保存 adapter（checkpoint=null）。CPU，PEFT=0.21.2；r=8、alpha=16、dropout=0.05，LoRA 学习率 0.0002、分类头学习率 0.001，seed=42、threads=4。

| 项目 | 本次实际值 |
| --- | --- |
| 目标模块 | 24 个 query/value 线性模块 |
| 模型类 | PeftModelForSequenceClassification |
| 可训练参数 | 296,450 / 102,565,636，其中 LoRA 294,912、分类头 1,538 |
| 可训练参数名称 | 48 个 lora_A/B 参数张量＋classifier 的 weight/bias，未发现额外可训练范围 |
| 更新检查 | LoRA 更新=True、分类头更新=True、冻结参数梯度均为空=True |
| 首批形状 | input_ids=(8,128)、labels=(8,)、logits=(8,2) |
| 5 步 train loss | 0.883473、0.606152、1.031173、0.729524、0.787323 |
| 平均训练 loss | 0.807529 |
| 验证 loss / Accuracy / Macro-F1 | 0.662392 / 0.6625 / 0.662025 |
| 报告耗时 | 约 38.57 秒 |

| 真值（行）/预测（列） | unsafe | safe |
| --- | ---: | ---: |
| unsafe | 28 | 12 |
| safe | 15 | 25 |

当前规则下正确 53/80 条；不安全召回 28/40=70%，安全误拦 15/40=37.5%。这些是 5 步后的 argmax 结果，不是业务阈值选取结果；不能与完整 240 步基线直接比较训练方法优劣，也不能据单次短运行确认稳定改进。

总参数比基础分类模型多 296,450 个，其中包括 294,912 个 LoRA 参数和 PEFT 保留的额外分类头副本（1,538）；不是原有整个编码器都变成可训练。参数张量个数与其中的数值个数不同：50 个可训练张量合计包含 296,450 个数值。

截断统计与基线相同：train 523/640（81.72%）、validation 64/80（80%）超过 128 token；LoRA 不会补回输入阶段已经截掉的内容。

## adapter 是什么，什么时候保存

本次 5 步试跑只保存报告，不保存 adapter。工程测试使用合成微型 BERT，已验证原有权重保持固定、LoRA/分类头发生更新，以及 adapter 保存重载后能恢复相同预测；当前真实模型仍待完整训练并保存。

接下来的完整模式按验证 Macro-F1 保存最优 adapter 到 `artifacts/lora-safety/adapter/`：

- adapter_config.json：LoRA 配置、目标模块及基础模型引用等。
- adapter_model.safetensors：LoRA 参数和可训练分类头，不是完整基础模型。
- task_config.json：标签映射、max_length、基础模型本地路径和文件 SHA256。
- Tokenizer 文件：配套文本编码规则。

重新推理时，明确加载本地固定基础模型并挂载 adapter，同时读取标签与长度设置。只有 adapter 文件而缺少匹配基础模型，不能重建同一次模型。

## 当前下一小目标：完整 3 epoch（尚未执行）

```powershell
uv run --no-sync --offline --no-python-downloads python .\scripts\train_lora.py `
  --max-steps 0 `
  --epochs 3 `
  --batch-size 8 `
  --output-dir .\artifacts\lora-safety `
  --report .\reports\stage-5-lora.json
```

`--max-steps 0` 取消试跑的 5 步上限；每轮仍是 640÷8=80 步，3 轮共 240 次更新。两个学习率、r/alpha/dropout 和长度沿用试跑设置。试跑没有保存参数，因此完整命令会重新加载基础模型、初始化分类头和 LoRA，不是从前 5 步续训。完整训练的真实 CPU 耗时和效果以实际输出为准。

完成后查看无原文报告，并提供终端输出：

```powershell
Get-Content .\reports\stage-5-lora.json
```

每轮验证并按 Macro-F1 选最优 adapter；不默认选择最后一轮。完整 adapter 已存在时需指定新 output-dir，避免覆盖旧实验。当前不进入最终测试集评测。
