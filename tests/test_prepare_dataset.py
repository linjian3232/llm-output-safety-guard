"""阶段 3 的合成数据验收：不得读取或输出真实 XGuard 回复正文。"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "prepare_dataset.py"


class PrepareDatasetTest(unittest.TestCase):
    def test_filters_deduplicates_maps_labels_and_stratifies_splits(self) -> None:
        rows = []
        for index in range(12):
            rows.append({"sample_type": "general", "stage": "r", "label": "sec",
                         "response": f"安全中文回复编号{index}"})
            rows.append({"sample_type": "general", "stage": "r", "label": "ha",
                         "response": f"风险中文回复编号{index}"})
        # 同标签重复只留一条；安全与风险标签冲突的同文样本应整体剔除。
        rows.extend([
            {"sample_type": "general", "stage": "r", "label": "sec", "response": "重复中文安全"},
            {"sample_type": "general", "stage": "r", "label": "sec", "response": "冲突中文回复"},
            {"sample_type": "general", "stage": "r", "label": "pc", "response": "冲突中文回复"},
            {"sample_type": "general", "stage": "q", "label": "sec", "response": "中文提问"},
            {"sample_type": "dynamic_policy", "stage": "r", "label": "sec", "response": "中文动态规则"},
            {"sample_type": "general", "stage": "r", "label": "sec", "response": "English only"},
            {"sample_type": "general", "stage": "r", "label": "sec", "response": "日本語かな"},
            {"sample_type": "general", "stage": "r", "label": "sec", "response": "한국어 한글"},
        ])

        temporary_root = PROJECT_ROOT / ".test-tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temp_dir:
            root = Path(temp_dir)
            source = root / "input.jsonl"
            output_dir = root / "processed"
            manifest_path = root / "manifest.json"
            source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                              encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--input", str(source), "--output-dir",
                 str(output_dir), "--manifest", str(manifest_path), "--max-per-class", "10",
                 "--seed", "42"],
                capture_output=True, text=True, encoding="utf-8",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"}, check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["selected_rows"], 20)
            self.assertEqual(manifest["label2id"], {"unsafe": 0, "safe": 1})
            self.assertEqual(manifest["split_counts"], {
                "train": {"unsafe": 8, "safe": 8, "total": 16},
                "validation": {"unsafe": 1, "safe": 1, "total": 2},
                "test": {"unsafe": 1, "safe": 1, "total": 2},
            })
            self.assertEqual(manifest["deduplication"]["conflicting_text_groups_removed"], 1)

            split_texts = {}
            for split in ("train", "validation", "test"):
                split_rows = [json.loads(line) for line in
                              (output_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
                split_texts[split] = {row["text"] for row in split_rows}
                self.assertTrue(all(set(row) == {"text", "label"} for row in split_rows))
            self.assertFalse(split_texts["train"] & split_texts["validation"])
            self.assertFalse(split_texts["train"] & split_texts["test"])
            self.assertFalse(split_texts["validation"] & split_texts["test"])
            all_output = "".join(path.read_text(encoding="utf-8") for path in output_dir.glob("*.jsonl"))
            self.assertNotIn("冲突中文回复", all_output)
            self.assertNotIn("重复中文安全", all_output)
            self.assertNotIn("English only", all_output)
            self.assertNotIn("日本語かな", all_output)
            self.assertNotIn("한국어 한글", all_output)


class ChineseCandidateUnicodeTests(unittest.TestCase):
    def test_accepts_astral_han_character(self):
        sys.path.insert(0, str(SCRIPT.parent))
        try:
            from prepare_dataset import is_chinese_candidate
            self.assertTrue(is_chinese_candidate("𠀀"))
        finally:
            sys.path.pop(0)


if __name__ == "__main__":
    unittest.main()
