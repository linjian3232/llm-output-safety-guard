"""Stage 1 entry point: download and inspect one Chinese RoBERTa forward pass."""

import json
from dataclasses import asdict

from llm_output_safety_guard.pretrained import inspect_pretrained_model

if __name__ == "__main__":
    inspection = inspect_pretrained_model("该回复拒绝提供违法操作，并建议寻求合法帮助。")
    print(json.dumps(asdict(inspection), ensure_ascii=False, indent=2))
