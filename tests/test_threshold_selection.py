"""用手工可核对的分数验证阈值约束；不读取真实安全文本。"""

import importlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace


class ThresholdSelectionTest(unittest.TestCase):
    def evaluator(self):
        try:
            return importlib.import_module("scripts.select_threshold")
        except ModuleNotFoundError:
            self.fail("尚未实现验证集阈值选择入口")

    def test_maximizes_unsafe_recall_under_safe_false_positive_constraint(self):
        evaluation = self.evaluator()
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        scores = [0.9, 0.8, 0.6, 0.4, 0.7, 0.5, 0.2, 0.1]
        result = evaluation.choose_threshold(labels, scores, max_safe_fpr=0.25)
        # threshold=0.6 时，四条风险中识别三条，四条安全中误拦一条。
        self.assertEqual(result["threshold"], 0.6)
        self.assertEqual(result["metrics"]["unsafe_recall"], 0.75)
        self.assertEqual(result["metrics"]["safe_false_positive_rate"], 0.25)
        self.assertEqual(result["metrics"]["confusion_matrix"], [[3, 1], [1, 3]])
        stricter = evaluation.choose_threshold(labels, scores, max_safe_fpr=0.0)
        self.assertEqual(stricter["threshold"], 0.8)
        self.assertEqual(stricter["metrics"]["unsafe_recall"], 0.5)
        self.assertEqual(stricter["metrics"]["safe_false_positive_rate"], 0.0)

    def test_threshold_boundary_is_inclusive_and_invalid_inputs_are_rejected(self):
        evaluation = self.evaluator()
        metrics = evaluation.metrics_at_threshold([0, 1], [0.5, 0.49], threshold=0.5)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0], [0, 1]])
        for labels, scores in (([0, 1], [float("nan"), 0.1]),
                               ([0, 1], [1.1, 0.1]), ([0], [0.3]),
                               ([0, 1], [0.3])):
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                evaluation.choose_threshold(labels, scores, max_safe_fpr=0.1)

    def test_reports_no_feasible_probability_threshold_instead_of_breaking_constraint(self):
        evaluation = self.evaluator()
        # 分数恰好饱和为 1，>=threshold 且阈值限于 [0,1] 时不能把安全样本放行。
        self.assertIsNone(evaluation.choose_threshold([0, 1], [1.0, 1.0], max_safe_fpr=0.0))

    def test_offline_reload_of_both_models_freezes_decisions_without_test_file_or_text_output(self):
        evaluation = self.evaluator()
        import torch
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
        from scripts.train_frozen_baseline import attach_lora

        torch.set_num_threads(1)
        temporary_root = Path(__file__).resolve().parent.parent / ".test-tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temp_dir:
            root = Path(temp_dir)
            base, adapter = root / "base", root / "adapter"
            base.mkdir()
            vocab = base / "vocab.txt"
            vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\n", encoding="utf-8")
            tokenizer = BertTokenizer(vocab_file=str(vocab))
            tokenizer.save_pretrained(base)
            config = BertConfig(vocab_size=7, hidden_size=16, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=32,
                                max_position_embeddings=32, num_labels=2,
                                label2id={"unsafe": 0, "safe": 1},
                                id2label={0: "unsafe", 1: "safe"})
            model = BertForSequenceClassification(config)
            # 零分类头令所有 logits=[0,0]，预期 argmax 总是 unsafe，可手工验证矩阵。
            with torch.no_grad():
                model.classifier.weight.zero_()
                model.classifier.bias.zero_()
            model.save_pretrained(base)
            base_hashes = evaluation.snapshot_files(base)
            model, _ = attach_lora(model, rank=8, alpha=16, dropout=0.05)
            model.save_pretrained(adapter)
            tokenizer.save_pretrained(adapter)
            (adapter / "task_config.json").write_text(json.dumps({
                "label2id": {"unsafe": 0, "safe": 1}, "max_length": 8,
                "base_model_files_sha256": base_hashes,
            }), encoding="utf-8")
            validation = root / "validation.jsonl"
            validation.write_text("\n".join(json.dumps({
                "text": f"PRIVATE_SAMPLE_{i} a", "label": i % 2,
            }) for i in range(4)), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "label2id": {"unsafe": 0, "safe": 1},
                "split_counts": {"validation": {"total": 4, "unsafe": 2, "safe": 2}},
            }), encoding="utf-8")
            metadata = {"label2id": {"unsafe": 0, "safe": 1},
                        "data": {"validation_sha256": evaluation.file_sha256(validation)},
                        "config": {"max_length": 8}, "model_files_sha256": base_hashes,
                        "history": [{"epoch": 1, "validation": {
                            "macro_f1": 1 / 3, "confusion_matrix": [[2, 0], [2, 0]],
                        }}]}
            baseline_report, lora_report = root / "baseline.json", root / "lora.json"
            baseline_report.write_text(json.dumps({**metadata, "mode": "full_baseline"}), encoding="utf-8")
            lora_report.write_text(json.dumps({**metadata, "mode": "full_lora"}), encoding="utf-8")
            args = SimpleNamespace(validation=validation, manifest=manifest, baseline_dir=base,
                                   base_model_dir=base, adapter_dir=adapter,
                                   baseline_report=baseline_report, lora_report=lora_report,
                                   decision_file=root / "thresholds.json", report=root / "result.json",
                                   max_safe_fpr=0.1, batch_size=2)
            with redirect_stdout(io.StringIO()) as output:
                report = evaluation.run_validation(args)
            self.assertTrue(report["decision_ready"])
            for name in ("baseline", "lora"):
                result = report["models"][name]
                self.assertEqual(result["original_argmax"]["confusion_matrix"], [[2, 0], [2, 0]])
                self.assertEqual(result["selected"]["threshold"], 1.0)
                self.assertEqual(result["selected"]["metrics"]["safe_false_positive_rate"], 0.0)
            self.assertTrue(args.decision_file.is_file())
            self.assertNotIn("PRIVATE_SAMPLE", output.getvalue() + args.report.read_text(encoding="utf-8"))
            with self.assertRaises(ValueError):
                evaluation.run_validation(args)  # 不可静默覆盖已经写出的决策配置。
            with self.assertRaises(ValueError):
                evaluation.verify_files(base, {"config.json": "not-the-real-hash"})


if __name__ == "__main__":
    unittest.main()
