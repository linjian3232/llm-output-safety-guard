"""Download and inspect the fixed Chinese RoBERTa base model used in later stages."""

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "hfl/chinese-roberta-wwm-ext"
MODEL_REVISION = "5c58d0b8ec1d9014354d691c538661bf00bfdb44"
DEFAULT_CACHE_DIR = Path("artifacts/huggingface")


@dataclass(frozen=True)
class PretrainedModelInspection:
    """The observable inputs and outputs of one encoder forward pass."""

    model_id: str
    model_revision: str
    tokens: list[str]
    input_ids: list[int]
    attention_mask: list[int]
    hidden_state_shape: tuple[int, int, int]


def build_inspection(
    *,
    tokens: list[str],
    input_ids: list[int],
    attention_mask: list[int],
    hidden_state_shape: tuple[int, int, int],
) -> PretrainedModelInspection:
    """Build a serializable observation while keeping the model identity fixed."""
    return PretrainedModelInspection(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokens=tokens,
        input_ids=input_ids,
        attention_mask=attention_mask,
        hidden_state_shape=hidden_state_shape,
    )


def inspect_pretrained_model(
    text: str,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_length: int = 128,
) -> PretrainedModelInspection:
    """Download the trusted fixed revision once and observe its CPU forward pass."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)

    model.eval()
    with torch.inference_mode():
        output = model(**encoded)

    input_ids = encoded["input_ids"][0].tolist()
    return build_inspection(
        tokens=tokenizer.convert_ids_to_tokens(input_ids),
        input_ids=input_ids,
        attention_mask=encoded["attention_mask"][0].tolist(),
        hidden_state_shape=tuple(output.last_hidden_state.shape),
    )
