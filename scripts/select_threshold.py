"""阶段 6 的第一步：重载本地模型，只在 validation 上比较并选择拦截阈值。"""

import argparse
import gc
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 同时支持直接运行脚本和由工程测试按包导入。
if __package__:
    from .train_frozen_baseline import (
        LABEL2ID, classification_metrics, encode_rows, file_sha256, normalized_keys, read_rows,
    )
else:
    from train_frozen_baseline import (
        LABEL2ID, classification_metrics, encode_rows, file_sha256, normalized_keys, read_rows,
    )


def operating_metrics(labels: list[int], predictions: list[int]) -> dict:
    metrics = classification_metrics(labels, predictions)
    matrix = metrics["confusion_matrix"]
    unsafe_count, safe_count = sum(matrix[0]), sum(matrix[1])
    if not unsafe_count or not safe_count:
        raise ValueError("阈值选择需要同时包含安全和不安全的验证样本")
    return {**metrics, "unsafe_recall": matrix[0][0] / unsafe_count,
            "safe_false_positive_rate": matrix[1][0] / safe_count,
            "safe_false_positive_count": matrix[1][0],
            "unsafe_false_negative_count": matrix[0][1]}


def validate_scores(labels: list[int], scores: list[float]) -> None:
    if not labels or len(labels) != len(scores) or set(labels) != {0, 1}:
        raise ValueError("标签/分数数量不一致，或验证数据缺少任一类别")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("模型分数必须是 [0,1] 范围内的有限值")


def metrics_at_threshold(labels: list[int], scores: list[float], threshold: float) -> dict:
    validate_scores(labels, scores)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("阈值必须在 [0,1] 范围内")
    predictions = [0 if score >= threshold else 1 for score in scores]
    return operating_metrics(labels, predictions)


def choose_threshold(labels: list[int], scores: list[float], max_safe_fpr: float) -> dict | None:
    """先满足误拦约束，再最大化不安全召回；不接触 test，也不训练模型。"""
    validate_scores(labels, scores)
    if not math.isfinite(max_safe_fpr) or not 0 <= max_safe_fpr <= 1:
        raise ValueError("最大安全误拦率必须在 [0,1] 范围内")
    # 只需检查预测会变化的实际分数边界，加上 0、0.5、1；不猜某个固定阈值。
    candidates = sorted({0.0, 0.5, 1.0, *scores})
    feasible = []
    for threshold in candidates:
        metrics = metrics_at_threshold(labels, scores, threshold)
        if metrics["safe_false_positive_rate"] <= max_safe_fpr:
            feasible.append({"threshold": threshold, "metrics": metrics})
    if not feasible:
        return None  # 分数饱和等边界情况下，明确报告不可行，不偷偷突破约束。
    # 规则事先固定：召回最高 → 误拦更低 → 两项均相同时选择较低阈值。
    best = max(feasible, key=lambda item: (
        item["metrics"]["unsafe_recall"], -item["metrics"]["safe_false_positive_rate"],
        -item["threshold"],
    ))
    return {**best, "candidates_evaluated": len(candidates)}


def load_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise ValueError(f"{path.name} 不是有效 JSON") from None
    if not isinstance(result, dict):
        raise ValueError(f"{path.name} 顶层不是 JSON 对象")
    return result


def snapshot_files(folder: Path) -> dict[str, str]:
    """固定实际加载的配置、权重、Tokenizer 与任务配置，供下一步测试核对。"""
    if not folder.is_dir():
        raise ValueError(f"本地模型目录不存在：{folder}")
    return {path.name: file_sha256(path) for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix in (".json", ".bin", ".safetensors", ".txt")}


def verify_files(folder: Path, expected: dict[str, str]) -> None:
    if not expected:
        raise ValueError("缺少基础模型文件指纹，不能核对 adapter 配套版本")
    for name, expected_sha in expected.items():
        # 元数据中的文件名不得跳出明确指定的基础模型目录。
        if Path(name).name != name or not (folder / name).is_file():
            raise ValueError("基础模型文件清单不合法或缺少文件")
        if file_sha256(folder / name) != expected_sha:
            raise ValueError(f"基础模型文件指纹不匹配：{name}")


def score_rows(model, tokenizer, rows: list[dict], max_length: int, batch_size: int) -> dict:
    dataset, lengths = encode_rows(tokenizer, rows, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.to("cpu")
    model.eval()
    scores, predictions = [], []
    total_loss = 0.0
    with torch.inference_mode():
        for batch in loader:
            output = model(**batch)
            if not torch.isfinite(output.logits).all() or not torch.isfinite(output.loss):
                raise ValueError("模型输出或验证 loss 不是有限数，停止阈值选择")
            # label 0 是 unsafe；双精度 softmax 减少临界分数被舍入到 0.5 的情况。
            probabilities = torch.softmax(output.logits.double(), dim=-1)
            scores.extend(probabilities[:, LABEL2ID["unsafe"]].tolist())
            predictions.extend(output.logits.argmax(dim=-1).tolist())
            total_loss += output.loss.item() * len(batch["labels"])
    labels = [row["label"] for row in rows]
    return {"scores": scores, "argmax": {"loss": total_loss / len(rows),
            **operating_metrics(labels, predictions)}, "token_lengths": lengths}


def run_validation(args) -> dict:
    started = time.perf_counter()
    if args.decision_file.exists():
        raise ValueError("阈值配置已存在；避免覆盖，请用 --decision-file 指定新实验文件")
    rows = read_rows(args.validation)
    labels = [row["label"] for row in rows]
    counts = Counter(labels)
    if len(normalized_keys(rows)) != len(rows):
        raise ValueError("验证集存在归一化重复")
    manifest = load_json(args.manifest)
    if manifest["label2id"] != LABEL2ID:
        raise ValueError("数据标签映射不匹配")
    expected = manifest["split_counts"]["validation"]
    if (len(rows), counts[0], counts[1]) != (expected["total"], expected["unsafe"], expected["safe"]):
        raise ValueError("验证集类别数量与阶段 3 manifest 不匹配")
    validation_sha = file_sha256(args.validation)
    baseline_report, lora_report = load_json(args.baseline_report), load_json(args.lora_report)
    if baseline_report["mode"] != "full_baseline" or lora_report["mode"] != "full_lora":
        raise ValueError("应使用完整训练报告，而不是 5 步试跑报告")
    for metadata in (baseline_report, lora_report):
        if metadata["label2id"] != LABEL2ID or metadata["data"]["validation_sha256"] != validation_sha:
            raise ValueError("标签映射或验证文件与训练时记录不一致")
    task = load_json(args.adapter_dir / "task_config.json")
    if task["label2id"] != LABEL2ID or task["max_length"] != lora_report["config"]["max_length"]:
        raise ValueError("adapter 的任务标签或长度配置不匹配")
    if task["base_model_files_sha256"] != lora_report["model_files_sha256"]:
        raise ValueError("adapter 任务配置与训练报告的基础文件版本不一致")
    verify_files(args.base_model_dir, task["base_model_files_sha256"])

    limit_count = math.floor(args.max_safe_fpr * counts[1] + 1e-12)
    print(f"只使用 validation={len(rows)}：unsafe={counts[0]}，safe={counts[1]}；不读取 test", flush=True)
    print(f"安全误拦率上限 {args.max_safe_fpr:.1%}，在本验证集上最多误拦 {limit_count} 条", flush=True)
    result_models, decisions = {}, {}
    for name, metadata in (("baseline", baseline_report), ("lora", lora_report)):
        max_length = metadata["config"]["max_length"]
        if name == "baseline":
            files = snapshot_files(args.baseline_dir)
            model = AutoModelForSequenceClassification.from_pretrained(args.baseline_dir, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(args.baseline_dir, local_files_only=True)
            artifact = {"kind": "full_model", "model_dir": str(args.baseline_dir.resolve()),
                        "model_files_sha256": files}
        else:
            from peft import PeftModel
            files = snapshot_files(args.adapter_dir)
            base = AutoModelForSequenceClassification.from_pretrained(
                args.base_model_dir, local_files_only=True, num_labels=2,
                label2id=LABEL2ID, id2label={value: key for key, value in LABEL2ID.items()},
            )
            model = PeftModel.from_pretrained(base, args.adapter_dir, local_files_only=True, is_trainable=False)
            del base
            tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, local_files_only=True)
            artifact = {"kind": "lora", "base_model_dir": str(args.base_model_dir.resolve()),
                        "base_model_files_sha256": task["base_model_files_sha256"],
                        "adapter_dir": str(args.adapter_dir.resolve()), "adapter_files_sha256": files}
        if model.config.label2id != LABEL2ID:
            raise ValueError("加载后模型的类别顺序与任务标签不一致")
        print(f"重载 {name} 并计算验证分数，模型参数不更新……", flush=True)
        scored = score_rows(model, tokenizer, rows, max_length, args.batch_size)
        best_epoch = max(metadata["history"], key=lambda item: item["validation"]["macro_f1"])
        if scored["argmax"]["confusion_matrix"] != best_epoch["validation"]["confusion_matrix"]:
            raise ValueError(f"{name} 重载后的预测计数与训练报告最优轮次不一致，请先核查 checkpoint")
        selected = choose_threshold(labels, scored["scores"], args.max_safe_fpr)
        result_models[name] = {"selected_training_epoch": best_epoch["epoch"],
                               "argmax_matches_training_report": True,
                               "original_argmax": scored["argmax"], "max_length": max_length,
                               "token_lengths": scored["token_lengths"], "selected": selected}
        original = scored["argmax"]
        print(f"{name} 原规则：误拦 {original['safe_false_positive_count']}/{counts[1]}，"
              f"不安全召回 {original['unsafe_recall']:.2%}", flush=True)
        if selected:
            metrics = selected["metrics"]
            decisions[name] = {**artifact, "threshold": selected["threshold"],
                               "max_length": max_length, "validation_metrics": metrics}
            print(f"{name} 阈值={selected['threshold']:.8f}：误拦 "
                  f"{metrics['safe_false_positive_count']}/{counts[1]}，"
                  f"不安全召回 {metrics['unsafe_recall']:.2%}", flush=True)
        else:
            print(f"{name} 没有满足约束的 [0,1] 阈值，不生成可交付决策", flush=True)
        del model, tokenizer, scored
        gc.collect()  # CPU 上逐个加载，避免同时持有两套完整模型。

    validation = {"path": str(args.validation.resolve()), "sha256": validation_sha,
                  "samples": len(rows), "unsafe": counts[0], "safe": counts[1]}
    selection_rule = "maximize unsafe_recall; then minimize safe_false_positive_rate; then lower threshold"
    ready = len(decisions) == 2
    report = {"evaluation_split": "validation", "label2id": LABEL2ID,
              "validation": validation, "max_safe_false_positive_rate": args.max_safe_fpr,
              "decision_rule": "unsafe_score >= threshold => unsafe", "selection_rule": selection_rule,
              "models": result_models, "decision_ready": ready,
              "decision_file": str(args.decision_file) if ready else None,
              "elapsed_seconds": time.perf_counter() - started}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if ready:
        plan = {"schema_version": 1, "selection_split": "validation", "label2id": LABEL2ID,
                "validation": validation, "max_safe_false_positive_rate": args.max_safe_fpr,
                "decision_rule": report["decision_rule"], "selection_rule": selection_rule,
                "models": decisions}
        args.decision_file.parent.mkdir(parents=True, exist_ok=True)
        args.decision_file.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"已保存阈值与模型指纹：{args.decision_file}", flush=True)
    print(f"无原文验证报告：{args.report}", flush=True)
    return report


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="仅在本地 validation 上重载两种模型、选择阈值")
    parser.add_argument("--validation", type=Path, default=root / "data/processed/stage-3/validation.jsonl")
    parser.add_argument("--manifest", type=Path, default=root / "reports/stage-3-data-manifest.json")
    parser.add_argument("--baseline-dir", type=Path, default=root / "artifacts/frozen-baseline/model")
    parser.add_argument("--base-model-dir", type=Path, default=root / "artifacts/base-model")
    parser.add_argument("--adapter-dir", type=Path, default=root / "artifacts/lora-safety/adapter")
    parser.add_argument("--baseline-report", type=Path, default=root / "reports/stage-4-baseline.json")
    parser.add_argument("--lora-report", type=Path, default=root / "reports/stage-5-lora.json")
    parser.add_argument("--report", type=Path, default=root / "reports/stage-6-validation.json")
    parser.add_argument("--decision-file", type=Path, default=root / "artifacts/evaluation/thresholds.json")
    parser.add_argument("--max-safe-fpr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or not math.isfinite(args.max_safe_fpr) or not 0 <= args.max_safe_fpr <= 1:
        parser.error("batch/线程数或安全误拦率上限不合法")
    protected = {path.resolve() for path in (args.validation, args.manifest, args.baseline_report, args.lora_report)}
    if args.report.resolve() in protected or args.decision_file.resolve() in protected or args.report.resolve() == args.decision_file.resolve():
        parser.error("输出文件不能覆盖输入文件，也不能相互覆盖")
    torch.set_num_threads(args.threads)
    try:
        run_validation(args)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
