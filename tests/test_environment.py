from llm_output_safety_guard.environment import check_runtime_dependencies


def test_runtime_dependency_check_reports_all_required_libraries() -> None:
    """A broken environment must be visible before downloading model assets."""
    versions = check_runtime_dependencies()

    assert set(versions) == {"torch", "transformers", "peft", "datasets"}
    assert all(version for version in versions.values())
