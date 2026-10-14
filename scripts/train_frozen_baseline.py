"""阶段 4：本地 CPU 冻结编码器基线；默认仅做 5 步训练验证流程。"""

import argparse
import hashlib
import json
import math
import os
import random
import time
import unicodedata
from collections import Counter
from pathlib import Path

# 必须在导入模型库之前设置；加载器还会指定 local_files_only=True。
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL2ID = {"unsafe": 0, "safe": 1}


def file_sha256(path: Path) -> str:
    """记录实际资源指纹，不把文件正文放进报告。"""
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    """只读取训练或验证文件；失败时仅给文件名和行号，不回显原文。"""
    rows = []
    with path.open("r", encoding="utf-8") as reader:
        for number, line in enumerate(reader, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"{path.name} 第 {number} 行不是有效 JSON") from None
            if (not isinstance(row, dict) or not isinstance(row.get("text"), str)
                    or not row["text"].strip() or type(row.get("label")) is not int
                    or row["label"] not in (0, 1)):
                raise ValueError(f"{path.name} 第 {number} 行样本格式或标签不正确")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path.name} 没有可用样本")
    return rows


def normalized_keys(rows: list[dict]) -> set[str]:
    return {" ".join(unicodedata.normalize("NFKC", row["text"]).casefold().split())
            for row in rows}


class EncodedDataset(Dataset):
    """让 DataLoader 按索引取一条已编码样本，随后自动堆成一个 batch。"""
    def __init__(self, tensors: dict):
        self.tensors = tensors

    def __len__(self):
        return self.tensors["labels"].shape[0]

    def __getitem__(self, index):
        return {name: tensor[index] for name, tensor in self.tensors.items()}


def encode_rows(tokenizer, rows: list[dict], max_length: int) -> tuple[EncodedDataset, dict]:
    texts = [row["text"] for row in rows]
    # 先仅由 Tokenizer 测实际 token 长度，包含特殊 token；不把长序列送入模型。
    full = tokenizer(texts, truncation=False, padding=False, verbose=False)
    lengths = sorted(len(ids) for ids in full["input_ids"])
    del full
    truncated = sum(length > max_length for length in lengths)
    summary = {
        "count": len(lengths), "min": lengths[0], "max": lengths[-1],
        "p50": lengths[math.ceil(len(lengths) * 0.5) - 1],
        "p90": lengths[math.ceil(len(lengths) * 0.9) - 1],
        "over_max_length": truncated, "truncation_fraction": truncated / len(lengths),
    }
    # 第一轮固定补齐到 128，让形状容易观察。截断与 padding 是不同动作。
    tensors = tokenizer(texts, truncation=True, padding="max_length",
                        max_length=max_length, return_tensors="pt")
    tensors["labels"] = torch.tensor([row["label"] for row in rows], dtype=torch.long)
    return EncodedDataset(tensors), summary


def freeze_encoder(model) -> list[str]:
    """针对本项目的 BERT 类分类器，只允许 classifier 的参数训练。"""
    if not hasattr(model, "classifier"):
        raise ValueError("当前模型没有 classifier 模块，需要核对新架构的分类头")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def train_one_batch(model, optimizer, batch) -> tuple[float, torch.Tensor]:
    model.train()
    # 冻结梯度不等于关闭 dropout：基线同时固定编码器的评估行为，分类头仍处于训练模式。
    model.base_model.eval()
    optimizer.zero_grad(set_to_none=True)  # 梯度默认累加，因此每步先清掉上一步梯度。
    output = model(**batch)  # batch 包含 labels，模型据此计算交叉熵 loss。
    if not torch.isfinite(output.loss):
        raise ValueError("训练 loss 非有限值，请停止并检查数据、模型和学习率")
    output.loss.backward()  # 反向传播，只为可训练参数计算梯度。
    optimizer.step()        # 更新分类头，冻结编码器不在优化器参数列表中。
    return output.loss.item(), output.logits.detach()


def classification_metrics(labels: list[int], predictions: list[int]) -> dict:
    """二分类指标；混淆矩阵行是真值、列是预测，顺序均为 unsafe/safe。"""
    if not labels or len(labels) != len(predictions):
        raise ValueError("指标输入数量不一致或为空")
    matrix = [[0, 0], [0, 0]]
    for truth, prediction in zip(labels, predictions):
        matrix[truth][prediction] += 1
    f1_scores = []
    recalls = {}
    for name, index in LABEL2ID.items():
        tp = matrix[index][index]
        fn = sum(matrix[index]) - tp
        fp = sum(row[index] for row in matrix) - tp
        recalls[name] = tp / (tp + fn) if tp + fn else 0.0
        f1_scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"accuracy": (matrix[0][0] + matrix[1][1]) / len(labels),
            "macro_f1": sum(f1_scores) / 2, "recall": recalls,
            "confusion_matrix": matrix}


def evaluate(model, loader) -> dict:
    model.eval()
    truths, predictions = [], []
    total_loss = 0.0
    with torch.inference_mode():
        for batch in loader:
            output = model(**batch)
            total_loss += output.loss.item() * batch["labels"].shape[0]
            truths.extend(batch["labels"].tolist())
            predictions.extend(output.logits.argmax(dim=-1).tolist())
    metrics = classification_metrics(truths, predictions)
    return {"loss": total_loss / len(truths), "samples": len(truths), **metrics}


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="本地 CPU 冻结编码器基线，默认 5 步试跑")
    parser.add_argument("--model-dir", type=Path, default=root / "artifacts/base-model")
    parser.add_argument("--data-dir", type=Path, default=root / "data/processed/stage-3")
    parser.add_argument("--data-manifest", type=Path, default=root / "reports/stage-3-data-manifest.json")
    parser.add_argument("--report", type=Path, help="默认按短运行/完整训练选择对应报告文件")
    parser.add_argument("--output-dir", type=Path, default=root / "artifacts/frozen-baseline")
    parser.add_argument("--max-steps", type=int, default=5, help="默认 5；0 表示完整 epoch 训练")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU 计算线程数")
    args = parser.parse_args()
    if (args.max_steps < 0 or args.epochs < 1 or args.batch_size < 1
            or args.max_length < 2 or args.threads < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("步数、epoch、batch、长度、线程数或学习率参数不合法")
    smoke = args.max_steps > 0
    if args.report is None:
        args.report = root / "reports" / ("stage-4-smoke.json" if smoke else "stage-4-baseline.json")
    checkpoint_dir = args.output_dir / "model"
    if not smoke and checkpoint_dir.exists():
        parser.error("完整训练输出目录已存在；请用 --output-dir 指定新的实验目录")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    # 不读取 test，完整训练的模型选择也只使用 validation。
    train_path, val_path = args.data_dir / "train.jsonl", args.data_dir / "validation.jsonl"
    try:
        train_rows, val_rows = read_rows(train_path), read_rows(val_path)
        manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
        if manifest["label2id"] != LABEL2ID:
            raise ValueError("数据标签映射与训练入口不一致")
        for name, rows in (("train", train_rows), ("validation", val_rows)):
            counts = Counter(row["label"] for row in rows)
            expected = manifest["split_counts"][name]
            if (len(rows), counts[0], counts[1]) != (expected["total"], expected["unsafe"], expected["safe"]):
                raise ValueError(f"{name} 文件数量与阶段 3 manifest 不一致")
            if len(normalized_keys(rows)) != len(rows):
                raise ValueError(f"{name} 文件内存在归一化重复")
        if normalized_keys(train_rows) & normalized_keys(val_rows):
            raise ValueError("训练与验证文件存在归一化文本交叉")
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"数据准备检查失败：{error}" if not isinstance(error, json.JSONDecodeError)
                     else "数据 manifest 不是有效 JSON")

    print(f"模式：{'少量步数验证（不保存模型）' if smoke else '完整基线训练'}；设备：cpu", flush=True)
    print(f"train={len(train_rows)}，validation={len(val_rows)}，不读取 test", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, local_files_only=True, num_labels=2,
        label2id=LABEL2ID, id2label={value: key for key, value in LABEL2ID.items()},
    ).to("cpu")
    if args.max_length > model.config.max_position_embeddings:
        parser.error("max-length 超过模型位置编码容量")
    trainable_names = freeze_encoder(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in trainable)
    print(f"模型类：{type(model).__name__}；可训练参数：{trainable_params:,}/{total_params:,}", flush=True)
    print(f"可训练参数名称：{trainable_names}", flush=True)
    initial_head = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    train_data, train_lengths = encode_rows(tokenizer, train_rows, args.max_length)
    val_data, val_lengths = encode_rows(tokenizer, val_rows, args.max_length)
    for name, summary in (("train", train_lengths), ("validation", val_lengths)):
        print(f"{name} token 长度：p50={summary['p50']}，p90={summary['p90']}，"
              f"超过 {args.max_length}：{summary['over_max_length']}/{summary['count']}", flush=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    steps, best_f1 = 0, -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        weighted_loss, seen = 0.0, 0
        for batch in train_loader:
            loss, logits = train_one_batch(model, optimizer, batch)
            steps += 1
            size = batch["labels"].shape[0]
            seen += size
            weighted_loss += loss * size
            if steps == 1:
                print(f"首批形状：input_ids={tuple(batch['input_ids'].shape)}，"
                      f"labels={tuple(batch['labels'].shape)}，logits={tuple(logits.shape)}", flush=True)
            if smoke or steps % 10 == 0:
                print(f"epoch={epoch} step={steps} train_loss={loss:.6f}", flush=True)
            if smoke and steps >= args.max_steps:
                break
        validation = evaluate(model, val_loader)
        item = {"epoch": epoch, "steps_total": steps, "train_samples_seen_this_epoch": seen,
                "epoch_complete": seen == len(train_data),
                "mean_train_loss": weighted_loss / seen, "validation": validation}
        history.append(item)
        print(f"验证：loss={validation['loss']:.6f}，accuracy={validation['accuracy']:.4f}，"
              f"macro_f1={validation['macro_f1']:.4f}", flush=True)
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            if not smoke:
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
        if smoke and steps >= args.max_steps:
            break

    head_changed = any(not torch.equal(initial_head[name], p.detach())
                       for name, p in model.named_parameters() if name in initial_head)
    frozen_have_no_grad = all(p.grad is None for p in model.parameters() if not p.requires_grad)
    report = {
        "mode": "smoke" if smoke else "full_baseline", "device": "cpu",
        "model_class": type(model).__name__, "label2id": LABEL2ID,
        "config": {name: getattr(args, name) for name in
                   ("epochs", "batch_size", "max_length", "learning_rate", "seed", "threads", "max_steps")},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "data": {"source_sha256": manifest["source_sha256"],
                 "train_sha256": file_sha256(train_path), "validation_sha256": file_sha256(val_path),
                 "train_rows": len(train_rows), "validation_rows": len(val_rows),
                 "train_token_lengths": train_lengths, "validation_token_lengths": val_lengths},
        "model_files_sha256": {path.name: file_sha256(path) for path in args.model_dir.iterdir()
                               if path.is_file() and path.suffix in (".json", ".bin", ".safetensors", ".txt")},
        "parameters": {"total": total_params, "trainable": trainable_params,
                       "trainable_names": trainable_names, "classifier_changed": head_changed,
                       "frozen_parameters_have_no_grad": frozen_have_no_grad},
        "steps_completed": steps, "history": history, "best_validation_macro_f1": best_f1,
        "checkpoint": str(checkpoint_dir) if not smoke else None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"分类头已更新：{head_changed}；冻结参数梯度均为空：{frozen_have_no_grad}", flush=True)
    print(f"无原文报告：{args.report}", flush=True)


if __name__ == "__main__":
    main()
