from llm_output_safety_guard.pretrained import (
    MODEL_ID,
    MODEL_REVISION,
    build_inspection,
)


def test_build_inspection_keeps_tokenization_and_hidden_state_observations() -> None:
    """观察报告应原样保留教学所需的 token、掩码和张量形状。

    这是纯函数测试：它不下载模型，也不验证 Transformers 的内部正确性；我们要
    保护的是自己的输出契约，避免未来改报告字段时让学习脚本悄悄失去关键信息。
    """
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
