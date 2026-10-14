# 阶段 3：制作回复安全二分类数据集

## 本阶段唯一目标

把已审计的本地 XGuard JSONL 转成可复现的 `train`、`validation`、`test` 三个数据文件，并生成一份**不含回复正文**的 manifest。训练、模型 Tokenizer 和训练代码留到后续阶段。

这一步可迁移到其他项目的主干是：定义一条样本、核对标签语义、显式记录清洗/去重规则、避免重复样本跨集合、固定随机种子、分层切分，并保存数据来源指纹。更换数据集时通常要替换字段适配和标签映射；切分与泄漏检查的原则可以复用。

## 已确认的数据规则

1. **任务单位：单条模型回复。** 只保留 `sample_type=general`、`stage=r`、`response` 为非空字符串的记录。`q` 是提问，`qr` 是问答对；它们不是本轮回复分类任务的同一种样本。
2. **中文候选：含汉字且不含假名、韩文。** 这是你选择的简单、可复现文字规则。它能排除常见日文/韩文混杂，但不保证语义上一定是中文：汉字很少的中英混合内容仍会保留，纯英文夹杂汉字也可能通过。它是当前数据集的过滤约定，不是可直接迁移到所有语言任务的检测器。
3. **二分类标签：** 固定 XGuard 标签模板把 `sec` 定义为 Safe；本项目映射为 `safe=1`。其余 28 个已知内容类别合并为 `unsafe=0`。[固定版本标签模板](https://huggingface.co/datasets/Alibaba-AAIG/XGuard-Train-Open-200K/blob/88d30c0d80abde49663f97d1bdbbec519995d79a/examples/prompt_template.txt)。这是本次安全检测任务的标签设计，不意味着所有数据集都应把除 Safe 外的标签都合并成不安全。
4. **重复与冲突：** 先对候选文本做 Unicode NFKC、英文大小写折叠和连续空白归一化，再按归一化文本去重。相同二分类标签的重复记录保留第一条；同一归一化文本既有 safe 又有 unsafe 标签时，整组移除，避免任意选择一个标签造成错误监督。
5. **抽样与切分：** 对 safe、unsafe 分别用 `seed=42` 洗牌，每类最多取 400 条；再在每类内部按 80/10/10 切分。若每类最终都至少 400 条，预期总量为 800，三个集合分别为 640/80/80，且各自 safe/unsafe 数量各半。400 是上限，不足时不会重复填充；每类少于 10 条时脚本会停止，因为三个集合无法都包含该类。

归一化只用于判断重复；写入数据集的 `text` 保留首条回复的原始正文（去掉首尾空白）。每行只写 `text` 和整数 `label`，不会写 ID、prompt、explanation 或来源标签。near-duplicate（只改几个词的相似文本）尚未做语义去重，本轮明确只处理归一化后的精确重复。

## 你来运行：生成数据集

在 PowerShell 项目目录运行：

```powershell
uv run --no-sync --offline --no-python-downloads python .\scripts\prepare_dataset.py
```

可选参数：

```powershell
uv run --no-sync --offline --no-python-downloads python .\scripts\prepare_dataset.py `
  --input .\data\raw\xguard_train_open_200k.jsonl `
  --output-dir .\data\processed\stage-3 `
  --manifest .\reports\stage-3-data-manifest.json `
  --max-per-class 400 `
  --seed 42
```

命令中的 `--max-per-class` 指**二分类的每类上限**，不是 29 个原始标签各取 400。`--seed` 固定伪随机抽样顺序；同样的输入、代码版本和种子可以复现同样的样本划分。参数化路径和上限，也让之后换数据文件时不必改脚本源代码。

成功后会创建：

```text
data/processed/stage-3/train.jsonl
data/processed/stage-3/validation.jsonl
data/processed/stage-3/test.jsonl
reports/stage-3-data-manifest.json
```

三个数据文件被 `.gitignore` 排除；manifest 只记录来源文件名与 SHA256、筛选规则、去重数量、seed、标签映射和每份集合的类别数，不包含回复正文。请只把 manifest 内容和脚本终端输出发给我，不要粘贴 JSONL 正文。

### 关键输出怎么读

- `按规则筛选并去重后` 是安全/不安全两类的可用唯一文本数。此前审计的 6,456 条只是粗略文字系统候选，去重、冲突处理之后会更少。
- `最终样本数` 是抽样后 3 个集合之和。若两类各有至少 400 个有效样本，预期为 800；否则以实际 manifest 为准。
- `train` 用于更新模型参数；`validation` 用于训练过程比较和后续阈值选择；`test` 留到最后做一次独立验收。不能把 test 用来反复挑规则、阈值或模型。
- 三个集合类别数相同或近似，是这里的**分层切分**；它不能保证语义、来源主题或近似表达也完全隔离。
- manifest 中 `source_sha256` 将本地加工结果绑定到具体输入文件。模型版本、数据版本与训练参数都会在后续实验记录中共同决定结果。

## 验收时请提供

把脚本终端输出和 `Get-Content .\reports\stage-3-data-manifest.json` 的结果发来。回复正文不会出现在 manifest 中。我们会检查类别数量、去重/冲突计数、切分比例、标签映射、SHA256 和是否生成三个本地文件，然后将真实结果记录进笔记并完成阶段 3。你确认观察结果后才进入阶段 4。

## 实际运行与验收（2026-10-14）

学习者已运行真实 XGuard 文件并提供 manifest。输入 SHA256 仍为 `58ca381b907165d08de5c5d56d9bded3103db3ee4f9361a382355ca19c69cb18`，与阶段 2 固定文件一致。

| 处理步骤 | 安全 | 不安全 | 合计 |
| --- | ---: | ---: | ---: |
| 满足回复与中文候选规则 | 2,564 | 3,892 | 6,456 |
| 去重且无标签冲突 | 2,546 | 3,870 | 6,416 |
| 固定种子抽样 | 400 | 400 | 800 |
| train | 320 | 320 | 640 |
| validation | 40 | 40 | 80 |
| test | 40 | 40 | 80 |

原始 200,000 行中，69,346 行满足 `general + r + 非空回复`；其中 62,890 行未通过当前文字规则，因此留下 6,456 条候选。重复记录共折叠 40 条：安全减少 18 条，不安全减少 22 条。检测到的 safe/unsafe 标签冲突组为 0；这只说明相同归一化文本没有跨二分类标签冲突，不能证明所有标注都正确。

只读核验了实际生成文件：每行均包含非空 `text` 和合法整数 `label`，三份文件数量与 manifest 一致；每份集合内部归一化文本均唯一，train/validation、train/test、validation/test 的交集均为 0。阶段 3 数据集制作验收完成；尚未开始模型训练。

本轮抽样刻意让两类各半，以便先跑通 CPU 训练及指标解释。这种比例不代表上线回复的真实风险分布，后续误拦和漏判指标需要结合使用场景理解；28 个风险子类别的覆盖也没有由二分类分层保证。Tokenizer token 长度和最大长度 128 的截断影响尚未检查，将在阶段 4 的训练输入准备中观察。
