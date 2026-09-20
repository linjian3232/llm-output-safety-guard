# 阶段 0：取得并核对预训练模型文件

## 这一步在真实项目里解决什么问题

训练前必须先确定**用哪一版基础模型**。仓库名只是一个位置，`main` 分支可能继续变化；固定 commit 才能让不同电脑使用同一组文件。内网机器不必连接 Hugging Face：在能访问模型仓库的电脑上下载文件，再转移到本地目录。SHA256 是传输后核对文件内容的指纹；它不是模型质量分数，也不保证来源天然可信。

本阶段**不训练**。这里拿到的是中文预训练编码器，不是已经学会判断暴力、色情或违法内容的安全分类器。后续会添加分类头，用有标签的安全数据训练，再评测其错误类型。

## 1. 在 Hugging Face 识别要取的文件

- 模型仓库：[`hfl/chinese-roberta-wwm-ext`](https://huggingface.co/hfl/chinese-roberta-wwm-ext)。
- 固定 commit：`5c58d0b8ec1d9014354d691c538661bf00bfdb44`。
- 请打开[这个固定版本的文件页](https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/5c58d0b8ec1d9014354d691c538661bf00bfdb44)，确认地址中是上述 commit，而不是默认的 `main`。

模型仓库名称含“RoBERTa”，但实际加载哪一种网络结构要由下载后的 `config.json` 与模型文件决定，不能只看名称猜测；阶段 1 会实地查看这一点。

从该页逐个打开文件并使用页面的下载功能，保留原文件名。若下载浏览器在能联网的另一台电脑上，之后把文件复制到当前项目。第一轮需要这 7 个文件，全部放在**同一目录的顶层**：

| 文件 | 作用 |
| --- | --- |
| `config.json` | 描述模型架构，例如层数和隐藏维度；加载器据此构建网络 |
| `pytorch_model.bin` | PyTorch 权重，即预训练得到的大量数值参数；该文件约 412 MB |
| `tokenizer.json` | Fast Tokenizer 的完整序列化配置 |
| `tokenizer_config.json` | Tokenizer 的加载选项 |
| `vocab.txt` | 中文词表及 token ID 的对应关系 |
| `special_tokens_map.json` | `[CLS]`、`[SEP]` 等特殊 token 的映射 |
| `added_tokens.json` | 额外 token 的定义；即使内容很少也保留，避免不同电脑加载行为不一致 |

文件页还列出 TensorFlow 的 `tf_model.h5` 和 Flax 的 `flax_model.msgpack`。本项目用 PyTorch，**不需要**再下载这两份额外权重。也不要把 Git LFS 的几百字节指针文件误当成约 412 MB 的 `pytorch_model.bin`；下载后要看文件大小。

> 安全提醒：`.bin` 属于 PyTorch 序列化权重，加载来源不明的此类文件有安全风险。这里只使用已确认仓库和固定版本的文件，不接收不明来源的“镜像权重”。

## 2. 放到项目约定的本地目录

在 **PowerShell** 中执行；`Set-Location` 相当于切换工作目录，后续的 `.\` 路径都相对于这里：

```powershell
Set-Location 'D:\projectStore\codeXProject\python-basic-learning\projects\llm-output-safety-guard'
New-Item -ItemType Directory -Force .\artifacts\base-model | Out-Null
```

`New-Item` 创建目录，`-Force` 让已存在目录不会报错；`Out-Null` 只是不显示创建结果。将下载的 7 个文件复制到 `artifacts\base-model\`。不需要保持 Hugging Face 的网页目录或缓存目录结构。

在项目目录查看文件名和字节数：

```powershell
Get-ChildItem -LiteralPath .\artifacts\base-model -File |
    Sort-Object Name |
    Select-Object Name, Length
```

这里的 `|` 将前一个命令的输出交给下一个命令；`Length` 是字节数。应看到上表 7 个文件，且 `pytorch_model.bin` 为数亿字节。如果权重文件只有几百字节，通常下载的是指针或网页，不要进入下一阶段。

## 3. 计算 SHA256 并反馈结果

```powershell
Get-ChildItem -LiteralPath .\artifacts\base-model -File |
    Sort-Object Name |
    ForEach-Object {
        Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
    } |
    Format-List Path, Hash
```

`Get-FileHash` 只读取文件，不修改它。`$_` 表示管道当前处理的文件；这类似 Java Stream 中处理每个元素。请把**文件清单、大小和 SHA256 输出**发给我。我会检查文件是否齐全、权重大小是否合理，再把来源版本和真实校验值写进仓库中的来源记录并提交。这一步由你实际完成下载与核对，随后我们再进入阶段 1 的 uv 环境和本地前向推理。

`artifacts/` 已在 `.gitignore` 中；权重文件不进入 Git。SHA256 和来源说明可以进 Git，因为它们不包含模型或训练数据本身。
