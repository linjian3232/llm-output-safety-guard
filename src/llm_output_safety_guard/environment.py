"""Checks for the runtime libraries required before model work begins."""

from importlib import metadata

REQUIRED_PACKAGES = ("torch", "transformers", "peft", "datasets")


def check_runtime_dependencies() -> dict[str, str]:
    """Return installed versions for the libraries used by later stages."""
    return {package: metadata.version(package) for package in REQUIRED_PACKAGES}
