"""用合成微型 BERT 验证 LoRA 更新边界，以及 adapter 是否连同分类头正确保存。"""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from scripts import train_frozen_baseline as training


class LoraTrainingTest(unittest.TestCase):
    def test_lora_and_head_update_while_base_stays_fixed_and_adapter_roundtrips(self):
        self.assertTrue(hasattr(training, "attach_lora"), "尚未实现 LoRA 配置入口")
        torch.manual_seed(42)
        torch.set_num_threads(1)
        config = BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=32,
                            max_position_embeddings=32, num_labels=2)
        original = BertForSequenceClassification(config)
        original_state = {name: value.detach().clone()
                          for name, value in original.state_dict().items()}
        model, info = training.attach_lora(original, rank=8, alpha=16, dropout=0.05)
        self.assertEqual(len(info["target_modules"]), 2)
        self.assertEqual(info["lora_parameters"], 512)
        self.assertEqual(info["classifier_parameters"], 34)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        batch = {"input_ids": torch.tensor([[1, 2, 3, 4], [1, 5, 6, 7]]),
                 "attention_mask": torch.ones((2, 4), dtype=torch.long),
                 "labels": torch.tensor([0, 1], dtype=torch.long)}
        loss, _ = training.train_one_batch(model, optimizer, batch, freeze_encoder_behavior=False)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(model.base_model.training)  # LoRA 及其 dropout 必须保留训练行为。
        changed = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                self.assertTrue(torch.equal(before[name], parameter), name)
                self.assertIsNone(parameter.grad)
            elif not torch.equal(before[name], parameter):
                changed.append(name)
        self.assertTrue(any("lora_B" in name for name in changed))
        self.assertTrue(any("classifier" in name for name in changed))

        temporary_root = Path(__file__).resolve().parent.parent / ".test-tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temp_dir:
            model.eval()
            with torch.inference_mode():
                expected = model(**batch).logits
            model.save_pretrained(temp_dir)
            saved = load_file(str(Path(temp_dir) / "adapter_model.safetensors"))
            self.assertTrue(any("lora_A" in name for name in saved))
            self.assertTrue(any("classifier" in name for name in saved))
            self.assertFalse(any("embeddings" in name for name in saved))
            fresh = BertForSequenceClassification(config)
            fresh.load_state_dict(original_state)
            reloaded = PeftModel.from_pretrained(fresh, temp_dir, local_files_only=True)
            reloaded.eval()
            with torch.inference_mode():
                actual = reloaded(**batch).logits
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_shared_runner_keeps_baseline_mode_and_saves_lora_task_metadata(self):
        torch.set_num_threads(1)
        temporary_root = Path(__file__).resolve().parent.parent / ".test-tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temp_dir:
            root = Path(temp_dir)
            base, data = root / "base", root / "data"
            base.mkdir()
            data.mkdir()
            vocab = base / "vocab.txt"
            vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\n", encoding="utf-8")
            BertTokenizer(vocab_file=str(vocab)).save_pretrained(base)
            BertForSequenceClassification(BertConfig(
                vocab_size=7, hidden_size=16, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=32,
                max_position_embeddings=32, num_labels=2,
            )).save_pretrained(base)
            original_hash = training.file_sha256(base / "model.safetensors")
            for name, count in (("train", 4), ("validation", 2)):
                rows = [{"text": f"{name} a {i}", "label": i % 2} for i in range(count)]
                (data / f"{name}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "label2id": {"unsafe": 0, "safe": 1}, "source_sha256": "synthetic",
                "split_counts": {"train": {"total": 4, "unsafe": 2, "safe": 2},
                                 "validation": {"total": 2, "unsafe": 1, "safe": 1}},
            }), encoding="utf-8")
            for method, steps in (("frozen", "1"), ("lora", "0")):
                report = root / f"{method}.json"
                args = ["training", "--model-dir", str(base), "--data-dir", str(data),
                        "--data-manifest", str(manifest), "--report", str(report),
                        "--output-dir", str(root / method), "--max-steps", steps,
                        "--epochs", "1", "--batch-size", "2", "--max-length", "8", "--threads", "1"]
                with patch.object(sys, "argv", args), redirect_stdout(io.StringIO()):
                    training.main(method=method)
                result = json.loads(report.read_text(encoding="utf-8"))
                self.assertTrue(result["parameters"]["classifier_changed"])
                if method == "frozen":
                    self.assertEqual(result["steps_completed"], 1)
                    self.assertIsNone(result["checkpoint"])
                else:
                    self.assertEqual(result["steps_completed"], 2)
                    self.assertTrue(result["parameters"]["lora_changed"])
                    self.assertEqual(result["lora"]["lora_parameters"], 512)
                    task = json.loads((root / "lora/adapter/task_config.json").read_text(encoding="utf-8"))
                    self.assertEqual(task["label2id"], {"unsafe": 0, "safe": 1})
                    self.assertEqual(task["base_model_files_sha256"]["model.safetensors"], original_hash)
            self.assertEqual(training.file_sha256(base / "model.safetensors"), original_hash)


if __name__ == "__main__":
    unittest.main()
