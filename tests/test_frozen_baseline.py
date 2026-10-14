"""只使用随机初始化的小型 BERT 和合成输入，检查冻结与指标的真实行为。"""

import importlib
import unittest

import torch
from transformers import BertConfig, BertForSequenceClassification


class FrozenBaselineTest(unittest.TestCase):
    def load_training_module(self):
        try:
            return importlib.import_module("scripts.train_frozen_baseline")
        except ModuleNotFoundError:
            self.fail("尚未实现冻结编码器训练模块")

    def test_optimizer_updates_classifier_but_leaves_entire_encoder_unchanged(self):
        training = self.load_training_module()
        torch.manual_seed(42)
        torch.set_num_threads(1)
        model = BertForSequenceClassification(BertConfig(
            vocab_size=32, hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32, max_position_embeddings=32,
            num_labels=2,
        ))
        training.freeze_encoder(model)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.01
        )
        batch = {"input_ids": torch.tensor([[1, 2, 3, 4], [1, 5, 6, 7]]),
                 "attention_mask": torch.ones((2, 4), dtype=torch.long),
                 "labels": torch.tensor([0, 1], dtype=torch.long)}
        loss, logits = training.train_one_batch(model, optimizer, batch)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertFalse(model.base_model.training)
        changed = []
        for name, parameter in model.named_parameters():
            if name.startswith("bert."):
                self.assertTrue(torch.equal(before[name], parameter), name)
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
            elif not torch.equal(before[name], parameter):
                changed.append(name)
        self.assertIn("classifier.weight", changed)

    def test_metrics_use_true_rows_and_prediction_columns(self):
        training = self.load_training_module()
        metrics = training.classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["recall"], {"unsafe": 0.5, "safe": 1.0})
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 0.8) / 2)


if __name__ == "__main__":
    unittest.main()
