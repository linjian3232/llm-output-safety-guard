# 学习笔记：从本地预训练模型到微调

这份笔记为本项目的复习材料，随着实操阶段推进而更新。它围绕「快速掌握可迁移的模型微调流程」组织，只补充理解当前工程所需的 Python、机器学习和深度学习知识。

## 1. 这条学习路线最终要会什么

目标不是记住 `hfl/chinese-roberta-wwm-ext` 的下载命令，而是能把同一套判断方法迁移到同类项目：

```text
本地模型与数据
    → 核对文件、版本与标签
    → 清洗并切分数据
    → 建立冻结编码器基线
    → 用 LoRA/PEFT 微调
    → 在验证集和测试集评测
    → 本地推理与 API
```

每一步先问：**我要解决什么问题？输入和输出是什么？哪些结果能证明它成功？** 模型架构与任务会改变具体类名和配置，但数据检查、锁定依赖、训练集划分、验证指标和本地加载等工程方法可以复用。

当前已完成阶段 0（模型文件）、阶段 1（本地前向推理）、阶段 2（原始数据审计）和阶段 3（真实数据集制作与无交叉核验）。阶段 3 的实际观察记录见[数据集制作说明](stage-3-dataset-preparation.md)。

## 2. uv 项目里几个文件分别管什么

可以把它们类比成 Java 项目里的构建配置和本地构建环境，但这只是职责上的类比：

| 文件/目录 | 记住什么 | 是否提交 Git |
| --- | --- | --- |
| `pyproject.toml` | 项目元信息、Python 版本范围、直接依赖的版本范围 | 是 |
| `uv.lock` | 解析后的具体依赖版本和安装信息，用于复现 | 是 |
| `.venv/` | 当前电脑上实际安装的隔离环境 | 否 |

本项目的 `pyproject.toml` 声明 Python `>=3.12,<3.13`、PyTorch `>=2.6,<3` 和 Transformers `>=4.57,<5`。版本范围表达「允许哪些版本」；锁文件记录本项目实际解析到的版本（当前记录为 PyTorch `2.14.0`、Transformers `4.57.6`）。换机器时，提交锁文件让 uv 重建环境，不提交整份 `.venv`。

### 常用命令

在 PowerShell 中先进入项目目录：

```powershell
Set-Location 'D:\projectStore\codeXProject\python-basic-learning\projects\llm-output-safety-guard'
```

| 命令 | 在本项目里的作用 |
| --- | --- |
| `uv init --bare --no-workspace` | 初始化最小项目配置；`--bare` 不生成教学项目不需要的样板代码，`--no-workspace` 不把项目并入父目录的 uv 工作区。已经初始化后不要重复运行。 |
| `uv add --no-sync --no-python-downloads 'torch>=2.6,<3' 'transformers>=4.57,<5'` | 把直接依赖写入 `pyproject.toml` 并解析锁文件；`--no-sync` 暂不安装，`--no-python-downloads` 使用本机 Python。 |
| `uv sync --locked --no-python-downloads` | 检查锁文件与项目声明一致，按锁文件建立或更新 `.venv`；`--locked` 发现不一致时会报错，不自行改锁文件。 |
| `uv run --no-sync --no-python-downloads python .\scripts\offline_forward.py` | 使用项目环境运行脚本；`--no-sync` 表示本次不自动安装/调整依赖，适合刚完成 `uv sync` 后运行。 |
| `uv run --no-cache --no-sync --offline --no-python-downloads ...` | 需要完全离线运行某个 uv 命令时使用。通常的本地推理命令按阶段说明执行即可。 |

`uv run` 默认会检查并同步项目环境。加 `--no-sync` 后，若 `.venv` 尚未安装所需包，Python 会报缺少模块；这时先运行 `uv sync`。uv 管理的是 **Python 包和项目环境**，模型权重与训练数据是独立的大文件，通过本地文件目录管理。

要记住 `--offline` 的边界：它约束 uv 自己的网络访问；不能单独保证脚本或模型库不联网。阶段 1 的代码还设置 `HF_HUB_OFFLINE=1`，并给模型和 Tokenizer 加 `local_files_only=True`，共同确保加载本地模型文件。

uv 官方资料：[项目指南](https://docs.astral.sh/uv/guides/projects/)、[锁定与同步](https://docs.astral.sh/uv/concepts/projects/sync/)、[运行项目命令](https://docs.astral.sh/uv/concepts/projects/run/)。

## 3. 模型目录里为什么有不同文件

模型仓库不是「一个权重文件就能运行」。本项目固定版本的文件职责大致如下：

- `config.json`：描述模型结构和结构参数，例如词表大小；加载器据此构造网络。
- `pytorch_model.bin`：保存预训练模型学到的参数，也就是网络里需要读取的数值权重。
- `vocab.txt`、`tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`、`added_tokens.json`：共同支持 Tokenizer 将文本切成 token，并将 token 映射到模型认识的整数 ID。

具体文件组合随模型架构与仓库而变。换模型时，不要照抄一份文件清单；要根据模型卡、配置、加载器要求和实际目录确定，并固定版本、核对 SHA256。**预训练权重不等于已经训练好的安全分类器**：它学过通用语言表示，安全分类任务还要有适配的分类头和标注数据。

大权重和数据不进入 Git；Git 保存代码、文档、锁文件和少量经过审查的配置。这样既可复现流程，也避免提交体积过大的模型和原始文本。

## 4. 一条中文输入在阶段 1 里经过了什么

阶段 1 用的句子是：

```text
今天的天气很[MASK]。
```

Tokenizer 不是模型本身。它会把文字转换为模型词表中的 token 和整数 ID。比如这一句被分成 10 个 token（包括 `[CLS]`、`[MASK]`、`[SEP]` 等特殊 token），随后模型处理的是整数，不是直接处理中文字符串。

```python
inputs = tokenizer(sentence, return_tensors="pt")
with torch.inference_mode():
    logits = model(**inputs).logits
```

`inputs` 是类似字典的 `BatchEncoding`，通常包含：

- `input_ids`：每个 token 的整数 ID。阶段 1 形状是 `(1, 10)`，即 1 条文本、10 个 token。
- `attention_mask`：标记有效 token，批量补齐时让模型区分真实 token 和填充位置。
- `token_type_ids`：某些 BERT 类模型用它区分句子 A/B；具体是否返回取决于模型。

Python 的 `model(**inputs)` 会把字典键值展开，效果近似于 `model(input_ids=..., attention_mask=..., token_type_ids=...)`。这样 Tokenizer 的输出就按模型接口传入了模型。

## 5. `return_tensors="pt"` 的意思

`pt` 是 PyTorch 的缩写。`return_tensors="pt"` 要求 Tokenizer 返回 PyTorch 张量，而不是普通 Python 列表。这个项目用的是 PyTorch 模型，所以这里要让输入数据符合 PyTorch 接口。

```python
inputs = tokenizer(sentence, return_tensors="pt")
print(type(inputs["input_ids"]))  # torch.Tensor
print(inputs["input_ids"].shape)  # torch.Size([1, 10])
```

这个参数决定**编码结果的承载类型**，不决定用哪种模型，也不启动训练。Transformers 的 Tokenizer 文档说明了 `return_tensors` 与 PyTorch `torch.Tensor` 的关系：[Tokenizer 文档](https://huggingface.co/docs/transformers/main_classes/tokenizer)。

## 6. 张量和矩阵的关系

「张量就是矩阵」是一个好起点，但矩阵只涵盖二维情况。可以先按维度理解：

| 维度 | 常见叫法 | 小例子 |
| --- | --- | --- |
| 0 维 | 标量 | `3.14` |
| 1 维 | 向量 | `[101, 791, 1921]` |
| 2 维 | 矩阵 | 一批文本的 token ID 表 |
| 3 维 | 三维张量 | 每条文本、每个位置、每个候选 token 的分数 |

阶段 1 观察到：

```text
input_ids 形状 = (1, 10)
logits    形状 = (1, 10, 21128)
```

`logits` 三个轴依次表示：**样本条数、token 位置、词表候选数**。这里词表大小 `21128` 来自本模型配置。`logits[0, mask_position, :]` 是 `[MASK]` 位置对 21128 个候选 token 给出的分数；`argmax` 取分数最大的 ID，再由 Tokenizer 转回 token，因此本次预测结果是“好”（ID `1962`）。

张量也有数据类型（如整数 ID、浮点分数）和所在设备（本次是 CPU）。它和 Java 的 `long[][]` / `float[][][]` 在形状上相似，但 PyTorch 会为张量计算提供高效的 CPU/GPU 运算，并可记录训练需要的梯度。并不是所有张量都需要梯度：`input_ids` 是整数索引，模型参数通常才是训练时要更新的对象。[PyTorch 张量说明](https://docs.pytorch.org/docs/stable/tensors.html)、[张量属性](https://docs.pytorch.org/docs/stable/tensor_attributes)。

## 7. 前向传播、推理和训练不是同一个动作

### 前向传播

前向传播就是把输入送进模型，按当前权重逐层计算输出。可以简记为：

```text
输入 x + 当前参数 θ  →  模型计算 fθ(x)  →  输出
```

阶段 1 的 `model(**inputs)` 就是一次前向传播。它读取模型参数来计算 `logits`，本身不会更新参数。

### 阶段 1 为什么是推理

代码先调用 `model.eval()`，让 dropout 等训练时使用的层进入评估行为；然后用 `torch.inference_mode()` 执行前向传播，不记录训练所需的梯度。这减少了推理的额外开销。`argmax` 只是从掩码位置的分数中挑出最大值，并没有计算安全概率。

因此，这次输出“好”只表示模型按它原本的**掩码填空任务**补出了一个词。它不能证明回复安全，也不能证明安全分类头已训练。

### 训练会多哪些步骤

训练时，除了文本还要给出正确标签：

```text
输入 → 前向传播得到预测分数 → loss 衡量预测与真值的差距
    → 反向传播计算梯度 → optimizer 更新可训练参数
```

`loss` 是训练目标的误差度量；梯度指示参数往哪个方向改变可以降低误差；`optimizer.step()` 执行参数更新。阶段 1 没有标签、loss、反向传播或优化器，所以它没有训练模型。[PyTorch 自动求导教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial)、[优化模型参数教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial)。

## 8. `AutoTokenizer` 和 `AutoModelForMaskedLM` 怎么迁移到别的任务

可迁移的是「按本地配置加载匹配的 Tokenizer 和模型类」这一方法；类名要随任务改变。

| 当前要做的事 | 常见加载入口 | 说明 |
| --- | --- | --- |
| 文本编码 | `AutoTokenizer` | 将字符串编码为模型输入；要与权重配套。 |
| 填 `[MASK]` | `AutoModelForMaskedLM` | 阶段 1 使用；带掩码语言建模任务头。 |
| 文本分类 | `AutoModelForSequenceClassification` | 后续安全分类阶段考虑；输出类别分数，需检查分类头是否已训练或新初始化。 |
| 自回归文本生成 | `AutoModelForCausalLM` | 生成任务常用，不是当前项目的第一轮目标。 |
| 获取基础隐藏表示 | `AutoModel` | 不自带特定任务头，后续需要自己接任务层。 |

`Auto...` 会根据配置选择具体架构；`For...` 表示接了与任务对应的输出头。加载时应检查具体模型类、输出形状、缺失/未使用权重提示，并确认是否从本地路径加载。阶段 1 的 `AutoModelForMaskedLM` 不是微调的通用固定步骤。

本次出现的 4 个「未使用权重」是 pooler 和下一句预测相关参数。当前 `BertForMaskedLM` 不使用这些层，所以加载后提示它们未被使用；本次加载和前向计算仍成功。遇到权重提示时要结合模型类和任务判断，不要为了让日志安静就随意改权重。

## 9. 传统机器学习调参和 LoRA 微调

两者都可能调整实验配置，但做的事情不同：

- **传统机器学习的超参数调节**：例如 Iris 项目中比较逻辑回归的 `C`、`solver`，是在训练前选择算法设置，再根据验证结果比较模型。
- **深度模型微调**：从预训练网络开始，用任务数据训练网络中的参数。冻结编码器基线只更新分类头；LoRA/PEFT 在指定模块加入小型可训练参数，以较少参数适配任务。

后续会先做简单分类头基线，再做 LoRA；这样才能看清基线、可训练参数和 adapter 的关系。它们不等同于只搜索一组传统算法配置。

## 10. 当前输出速记与常见误读

阶段 1 的实际观察值：

```text
模型类：BertForMaskedLM
设备：cpu
input_ids：(1, 10)
logits：(1, 10, 21128)
[MASK] 预测：好，ID=1962
```

- `input_ids` 是词表 ID，不是原文字串。
- `logits` 是原始分数，不是已经归一化的概率。
- `(1, 10, 21128)` 不是 21128 条预测结果，而是这条文本每个 token 位置对整个词表的分数。
- 预测出通顺的词不代表做过安全分类或训练。
- `uv.lock` 锁定 Python 包；模型仓库 commit 与 SHA256 锁定大文件来源，两者解决不同的复现问题。

## 11. 自测题

1. `pyproject.toml` 和 `uv.lock` 为什么都要提交？
2. `.venv` 为什么不提交？换电脑时怎样重建？
3. `model(**inputs)` 在阶段 1 做了什么？它会改权重吗？
4. 为什么当前用 `AutoModelForMaskedLM`，未来安全分类时要重新选择模型头？
5. `(1, 10, 21128)` 的三个数字各表示什么？
6. `return_tensors="pt"` 改变了什么？它是否代表开始训练？
7. 为什么模型输出“好”不能当作安全分类结果？

### 答案检查

1. 前者表达项目约束和直接依赖，后者记录解析到的具体版本，合起来才能复现意图和结果。
2. `.venv` 是本机安装环境；用提交的 `pyproject.toml` 和 `uv.lock` 运行 `uv sync --locked` 重建。
3. 它按当前参数做一次前向计算并返回分数，不更新权重。
4. 掩码填空和文本分类要求的输出结构不同，需选与任务匹配的模型头。
5. 一条样本、十个 token 位置、每个位置的 21128 个候选词分数。
6. 它让 Tokenizer 返回 PyTorch 张量以匹配模型输入接口；不代表训练。
7. “好”是填 `[MASK]` 的预测词，模型头执行的是语言建模任务，不是安全标签分类。

## 12. 每阶段如何沉淀后续笔记

每阶段完成后，围绕实际运行过的代码补充：核心概念、输入与输出形状、命令作用、真实观察值、常见误读和一个迁移到其他模型时的判断方法。没有亲自运行或尚未用真实文件验证的内容会标记为计划或待实操，不记成实验结论。

阶段 1 的详细命令和验收记录见[本地前向推理说明](stage-1-local-forward.md)。

## 13. 阶段 2：原始数据审计学到了什么

本次本地文件为 564,525,162 字节，SHA256 与固定来源版本一致。审计脚本逐行读了 200,000 个 JSON 对象，未发现空行、坏 JSON 或非对象行。这里的“审计通过”只说明拿到的文件与来源相符、结构可解析；它不代表数据已经适合直接训练。

### 空值要结合字段语义解释

`prompt` 为空 71,925 条，恰好等于 `stage=r` 的数量；`response` 为空 49,819 条，恰好等于 `stage=q` 的数量；`policy` 为空 190,000 条，恰好等于 `sample_type=general` 的数量。这些不是解析失败，而是与问答阶段、样本类型相符的字段缺省。数据质量不能只看“空值很多”，还要问字段在该类记录中是否本来就应该存在。

### 筛选条件决定训练样本是什么

本项目暂定的回复候选筛选为 `sample_type=general`、`stage=r`、`response` 是非空字符串，得到 69,346 条。`stage=q` 是提问，`qr` 是问答对；如果不显式选 `r`，就会把不同任务粒度混在一起。这个子集仍然不是最终训练集，不能把“满足条件”误解成“已经是中文安全分类数据”。

### 类别不平衡会限制抽样与评测

候选回复有 29 个原始标签。`sec` 有 37,015 条，`md` 只有 24 条，`ps` 只有 89 条。阶段 3 已明确采用二分类，把非 sec 的已知标签合并为 unsafe；“每类最多 400 条”因此指 safe/unsafe 两类。原始风险子类别仍然不平衡：按二分类抽样与分层不能保证各风险子类别都有足够覆盖，后续也不能据二分类结果宣称所有细类效果都好。

### 字符长度与模型 token 长度不同

目标候选回复字符数 p50 是 851，p90 是 3,283，p99 是 6,948，最大 14,183。模型的最大长度按 Tokenizer 的 token 计，不按 Python 字符数计；中文、英文、标点和特殊符号会有不同切分。token 长度与截断比例尚未实测，将在阶段 4 的训练输入准备中观察，再解释默认最大长度 128 的取舍。

### 文字系统信号不是语言识别器

审计统计到 9,138 条含汉字的候选；排除同时含假名或韩文的记录后，粗略候选为 6,456 条。汉字也用于日文，中文回复可能夹英文，短回复也可能没有汉字。因此这个数字只能帮助估算和抽样检查，不能当作最终“中文样本数”。真正筛选前还要用可解释规则查看代表性样例，同时避免在审计报告或 Git 中暴露原文。

本次完整统计值和文件指纹记录在[阶段 2 审计说明](stage-2-raw-data-audit.md)；原始数据和包含统计的报告仍只保存在本机忽略目录。

## 14. 阶段 3：把原始记录变成可训练样本

阶段 3 已由学习者运行真实 XGuard 文件并验收。每条 `general + r` 非空 `response` 是一个候选样本，经过筛选、去重、抽样后最终得到 800 条。Tokenizer 和模型训练尚未参与这个阶段。

### 标签映射是任务设计，不是数据清洗

XGuard 原始数据有 29 个类别。上游模板把 `sec` 说明为 Safe；为适配二分类安全检测，本项目定义 `unsafe=0`、`safe=1`，并把其余 28 个已知类别映射到 unsafe。相当于数据适配层把细分类标签映射为业务任务标签。换一个数据集时，必须重新看标签定义和业务目标，不能只按字符串名字猜。

### 过滤和去重要先于抽样与切分

阶段顺序是：选任务样本 → 应用约定的中文候选规则 → 归一化并处理重复/冲突 → 每类最多抽样 → 分层切分。若重复文本分别进入 train 和 test，测试结果会因为模型见过相同文本而显得过于乐观。对相同归一化文本，脚本将 safe/unsafe 标签都出现的冲突组移除；同一二分类标签的重复只保留一条。近似语义重复还未处理，要把它看作本轮方法的边界。

文本归一化只用于生成去重 key：Unicode NFKC 统一兼容表示，`casefold()` 折叠大小写，空白序列压成一个空格。它不会把所有语义相同句子合并。训练文件保留首条原文的主体内容，以供模型学习。

### 为什么在二分类内部切分

先按二分类标签分别洗牌，再按 80/10/10 分配，可以让 train、validation、test 都包含 safe 与 unsafe，并减少随机切分导致的比例漂移。`seed=42` 让伪随机顺序可复现；它不是“更正确”的魔法数字，改 seed 会得到另一份样本划分。固定输入、筛选代码、依赖解释器版本和 seed 才能复跑同一次加工。

train 用于更新参数，validation 用于比较模型与选择阈值，test 留作最后的独立检查。即便类别比例一致，分层也不能保证领域、话题、模板或近义表达完全隔离，因此 test 指标仍应结合数据来源局限解释。

阶段 3 的具体规则、命令、预期输出与文件位置见[数据集制作说明](stage-3-dataset-preparation.md)。原始数据、三个 split 和 manifest 都在 `.gitignore` 保护的本地目录，不要把回复正文粘贴到对话或提交 Git。

### 用本次真实数字复习流程

200,000 行原始记录 → 69,346 条目标回复 → 按文字规则留下 6,456 条 → 折叠 40 条重复，留下 6,416 条 → safe/unsafe 各抽 400 条 → train 640、validation 80、test 80。

safe 从 2,564 条变为 2,546 条，说明折叠了 18 条重复；unsafe 从 3,892 条变为 3,870 条，说明折叠了 22 条。两者相加等于 manifest 中的 40 条。冲突组为 0，仅表示这次没有发现同文 safe/unsafe 冲突；不能推导为数据标注没有任何错误。

train 里 safe/unsafe 各 320 条，validation 和 test 各 40 条，这就是本次可观察的二分类分层切分。实际文件已核对：集合内无归一化重复，任意两个集合的归一化文本交集为 0。它避免了同一归一化文本跨集合的泄漏，但没有保证近似语义或回复模板完全隔离。

最后 800 条两类各半，是为了首轮 CPU 实验容易运行和比较，不能把 50% 风险比例当作真实线上分布。后续评测要结合误拦、漏判和样本来源解释；仅靠这批小样本不能证明上线效果。
