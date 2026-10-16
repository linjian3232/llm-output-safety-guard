# 阶段 4：冻结编码器基线

## 本阶段目标与操作顺序

阶段 3 已得到 train 640 条、validation 80 条、test 80 条。本阶段把前两份数据送入本地中文模型，先完成 5 步 CPU 训练，观察编码后的形状、loss、可训练参数和验证结果；核对短运行后，再由学习者运行完整 3 epoch 基线。两个目标现已实际完成，命令与结果保留如下。

为什么先做基线：先确定模型的语言表示能否支持当前二分类任务，同时获得一个简单比较对象。基线只训练分类头；后续 LoRA 在同一份数据上增加编码器内部的可训练参数，才能比较适配方法带来的变化。

## 当前数据在训练中经过什么

```text
train.jsonl：640 条 text + label
→ Tokenizer：文本转为 token ID，观察长度，再截断/补齐到 128
→ DataLoader：每次取 8 条作为一个 batch
→ 冻结的中文编码器：每条回复产生表示
→ 可训练分类头：每条回复输出 unsafe/safe 两个 logits
→ 与整数标签比较得到交叉熵 loss
→ 反向传播计算分类头梯度 → AdamW 更新分类头
```

模型加载入口由阶段 1 的 `AutoModelForMaskedLM` 改为 `AutoModelForSequenceClassification`。当前安装的 BERT 实现用 `classifier` 线性层产生两类分数；本地配置 `hidden_size=768`，所以这个头预期有 1,538 个参数（768×2 个权重＋2 个偏置）。编码器含 pooler 的参数都冻结，优化器只接收分类头参数。

冻结由 `requires_grad=False` 控制。另把编码器设为 eval，固定其 dropout 行为；分类头仍以训练模式运行。仅调用 `model.eval()` 不能代替冻结，冻结梯度也不能代替设置 dropout 行为。

## 由学习者运行

在项目 PowerShell 目录执行：

```powershell
uv run --no-sync --offline --no-python-downloads python .\scripts\train_frozen_baseline.py --max-steps 5
```

这次使用已有 torch 和 transformers，不添加依赖。模型与 Tokenizer 只从 `artifacts/base-model/` 加载，`HF_HUB_OFFLINE=1` 和 `local_files_only=True` 约束本地加载。数据只读 `train.jsonl` 与 `validation.jsonl`，不读取 test。

| 参数 | 默认值 | 作用与意义 |
| --- | ---: | --- |
| `--max-steps` | 5 | 每一步更新一次分类头；正数为短运行上限，0 才会执行完整 epoch 训练 |
| `--batch-size` | 8 | 一次前向/反向处理 8 条回复；5 步在本数据上共处理 40 条训练样本 |
| `--max-length` | 128 | 包含特殊 token 的总序列长度上限；过长截断，短文本用 padding 补齐 |
| `--epochs` | 3 | 完整运行时最多遍历训练集 3 轮；短运行仍受 max-steps 限制 |
| `--learning-rate` | 0.001 | 优化器更新幅度的设置；本次仅训练新初始化的小分类头，后续不能机械照搬到 LoRA/全参数训练 |
| `--seed` | 42 | 固定分类头初始化、训练集洗牌等随机过程；不承诺不同平台完全逐位一致 |
| `--threads` | 4 | PyTorch CPU 计算线程数，真实速度由本机观察 |
| `--model-dir` | `artifacts/base-model` | 本地基础模型目录 |
| `--data-dir` | `data/processed/stage-3` | 训练与验证 JSONL 所在目录 |
| `--data-manifest` | `reports/stage-3-data-manifest.json` | 核对标签映射和文件条数 |
| `--report` | 按运行模式选择 | 短运行默认输出 `reports/stage-4-smoke.json` |

AdamW 的 weight decay 当前设为 0.01。学习率与 weight decay 是训练超参数，不是模型权重；第一轮先固定它们跑通，不进行超参数搜索。

## 预期观察与加载提示

- 实际模型类应为 `BertForSequenceClassification`，设备为 CPU。
- 可训练参数名称应只有 `classifier.weight`、`classifier.bias`；预期数量为 1,538。
- Tokenizer 输出完整 token 长度 p50/p90，以及超过 128 的条数。长度包含特殊 token；完整长度只用于统计，不送入模型。
- 首批 `input_ids=(8,128)`、`labels=(8,)`、`logits=(8,2)`。阶段 1 的 `(1,10,21128)` 是逐位置填空分数，这里每条回复只有两个类别分数。
- 打印 5 个训练 loss，然后在 80 条 validation 上计算验证 loss、Accuracy、Macro-F1。具体数值以真实运行为准，loss 不保证逐步下降。
- 最后打印“分类头已更新：True”“冻结参数梯度均为空：True”。工程测试已用小型合成模型验证编码器未被更新，真实短运行仍需学习者观察。

基础文件含掩码预测相关参数，而现在建立分类任务头，因此可能出现旧头未使用、classifier 新初始化等提示。新分类头从随机参数开始，提示需要训练属于预期；遇到其他缺失、形状不匹配或加载失败时保留完整日志排查，不把所有警告都一概忽略。

最大长度 128 表示当前任务头看到的是截断后的内容，可能丢失回复后半段的风险线索。先记录实际截断比例；观察 tokenizer 统计并不意味着已证明 128 合理或效果足够。

## 报告和验收

```powershell
Get-Content .\reports\stage-4-smoke.json
```

请提供终端输出与这份报告。报告仅包含配置、版本、输入资源哈希、token 长度统计、参数数量、训练/验证指标和耗时，不保存回复正文或逐条预测。核对 5 步成功、冻结范围正确、loss 为有限数、验证使用 80 条数据后，再进入完整基线运行。

5 步只说明训练链路可运行，不能作为模型效果验收；验证结果也不能代替阶段 6 的最终测试评测。短运行与随后完整训练的记录分列如下。

## 真实 5 步短运行验收（2026-10-15）

已核对学习者终端输出、本地 `stage-4-smoke.json`、训练/验证文件哈希以及报告内计数：完成 5 步，训练样本共 40 条，第一轮未结束；验证用 80 条。设备为 CPU，torch 为 2.14.0+cpu、Transformers 为 4.57.6，报告记录耗时约 28.86 秒。

- 可训练参数：1,538 / 102,269,186，仅 classifier.weight 与 classifier.bias。分类头已变化；冻结参数没有梯度，与优化器只更新分类头的设计一致。
- 首批形状：input_ids=(8,128)、labels=(8,)、logits=(8,2)。
- 逐步训练 loss：1.009693、0.645610、0.956196、0.766417、0.831486；5 步样本加权平均为 0.841881。
- 验证 loss=0.664485、Accuracy=0.6000、Macro-F1=0.587363。训练 loss 随 batch 波动，不能要求不同批次之间逐步下降，也不能只用最后一批训练 loss 与完整验证集直接比较。

| 真实标签（行）/预测标签（列） | unsafe | safe |
| --- | ---: | ---: |
| unsafe | 17 | 23 |
| safe | 9 | 31 |

正确数为 17+31=48，48/80=60%。40 条不安全回复识别 17 条，召回率 42.5%；40 条安全回复正确放行 31 条，召回率 77.5%。按当前 argmax 分类规则，不安全漏判 23 条、安全误拦 9 条，安全误拦比例为 22.5%。这不是已经在验证集选好的业务阈值，阈值选择留待阶段 6。

| 数据 | token 长度 p50 | p90 | 最大 | 超过 128 的条数/占比 |
| --- | ---: | ---: | ---: | --- |
| train | 552 | 1,492 | 3,386 | 523/640，81.72% |
| validation | 524 | 1,468 | 2,221 | 64/80，80% |

这些占比表示“被截断的回复比例”，不是“丢失的 token 比例”。模型只看到前缀，完整回复的风险线索可能在后面；保留完整回复标签会带来潜在监督偏差。当前沿用既定 128 做首个流程基线，将局限显式记录；不能据此宣称模型能够检测完整长回复。后续如比较长度或分段策略，需要单独记录配置和成本，并使用验证数据比较，不靠测试集调方案。

短运行链路验收通过。报告 checkpoint=null，未保存模型；下一次完整命令会重新从基础模型和新初始化分类头开始，不会从这 5 步自动续训。

## 完整基线命令（已执行）

```powershell
uv run --no-sync --offline --no-python-downloads python .\scripts\train_frozen_baseline.py `
  --max-steps 0 `
  --epochs 3 `
  --output-dir .\artifacts\frozen-baseline `
  --report .\reports\stage-4-baseline.json
```

本数据每轮为 640÷8=80 步，3 轮共 240 次更新。每轮后评估 validation，按验证 Macro-F1 保存最优 checkpoint；不是默认选择最后一轮。产物在 `artifacts/frozen-baseline/model/`，包含模型配置、权重和 Tokenizer，报告在 `reports/`，均被 Git 忽略。已有完整模型目录时必须指定新的输出目录，避免混用旧产物。

## 真实完整训练验收（2026-10-16）

学习者已执行上述命令。核对本地 `reports/stage-4-baseline.json`、训练/验证文件 SHA256、保存配置与 safetensors 文件结构：完整 3 轮，每轮处理 640 条、80 步，共 240 次更新；CPU，batch=8，max_length=128，lr=0.001，seed=42，threads=4。仅 classifier 的 1,538 个参数可训练，报告记录分类头已改变、冻结参数梯度为空。耗时约 405.99 秒（6 分 46 秒）。

| epoch | 累计更新步数 | 整轮平均训练 loss | 验证 loss | 验证 Accuracy | 验证 Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 80 | 0.707751 | 0.609691 | 0.6250 | 0.587912 |
| 2 | 160 | 0.627337 | 0.563276 | 0.7000 | 0.695238 |
| 3 | 240 | 0.574879 | 0.572706 | 0.7125 | 0.695314 |

整轮平均训练 loss 来自报告中全部 batch 的样本加权平均，不是终端最后一批 loss。它持续下降，说明模型在训练目标上的平均拟合有所改善；验证 loss 在第 3 轮回升，不能据三轮和 80 条验证样本就断言过拟合。loss 衡量概率分配，Accuracy 统计最终类别是否判对，两者不要求同步变化。

| epoch | 不安全识别/漏判（共 40） | 安全放行/误拦（共 40） | 不安全召回 | 安全误拦率 |
| --- | --- | --- | ---: | ---: |
| 1 | 37 / 3 | 13 / 27 | 92.5% | 67.5% |
| 2 | 33 / 7 | 23 / 17 | 82.5% | 42.5% |
| 3 | 19 / 21 | 38 / 2 | 47.5% | 5.0% |

第 3 轮比第 2 轮只多判对 1 条，但错误类型变化很大：更多安全文本被放行，同时漏掉更多不安全文本。二分类总体指标不能替代业务取舍判断；当前是 argmax 分类规则，尚未按验证集搜索业务阈值。

既定保存标准为验证 Macro-F1 最大，因此保存第 3 轮；它只比第 2 轮高约 0.0000757，不能据此宣称第 3 轮稳定优于第 2 轮。不得在看到测试效果后重新挑轮次；后续阈值与方法比较仍用验证集。

产物核验：

- 路径：`artifacts/frozen-baseline/model/`，包含 config.json、model.safetensors 与 Tokenizer 文件。
- 保存架构为 BertForSequenceClassification，标签映射 unsafe=0、safe=1。
- classifier.weight 的形状为 [2,768]，classifier.bias 为 [2]；safetensors 头部及数据范围与文件大小一致。本次收尾未重新执行模型前向推理。
- 权重文件 409,100,240 字节；SHA256：`7beef5acb0eef009ab798744e654e50b32ae1f4e9f1b9b45004e7e3cb3228595`。

虽然只训练了分类头，save_pretrained 仍保存完整分类模型（冻结编码器＋分类头）；文件大小不等于可训练参数大小。基础模型目录未作为输出路径，原始权重不被覆盖。

阶段 4 的训练流程与产物验收完成，安全检测效果尚未最终验收。验证集只有 80 条，长回复截断局限仍存在，test 未读取。下一阶段 LoRA 开始前，先讲清基础权重、可训练低秩参数与 adapter 的关系，不直接要求学习者逐行研究工程脚本。
