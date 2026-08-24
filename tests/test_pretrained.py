from llm_output_safety_guard.pretrained import (
    MODEL_ID,
    MODEL_REVISION,
    build_inspection,
)


def test_build_inspection_keeps_tokenization_and_hidden_state_observations() -> None:
    """A model inspection must preserve the facts a learner needs to examine."""
    inspection = build_inspection(
        tokens=["[CLS]", "安", "全", "[SEP]"],
        input_ids=[101, 2128, 1064, 102],
        attention_mask=[1, 1, 1, 1],
        hidden_state_shape=(1, 4, 768),
    )

    assert inspection.model_id == MODEL_ID
    assert inspection.model_revision == MODEL_REVISION
    assert inspection.tokens == ["[CLS]", "安", "全", "[SEP]"]
    assert inspection.input_ids == [101, 2128, 1064, 102]
    assert inspection.attention_mask == [1, 1, 1, 1]
    assert inspection.hidden_state_shape == (1, 4, 768)
