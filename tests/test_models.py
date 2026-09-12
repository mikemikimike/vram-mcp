"""Default registry components must not erase identically named repositories."""

import pytest

from vram_mcp.models import canonical_model


@pytest.mark.parametrize(("name", "expected"), [
    ("library", "library:latest"),
    ("registry.ollama.ai", "registry.ollama.ai:latest"),
    ("library/team/model", "library/team/model:latest"),
    ("registry.ollama.ai/library/LLAMA3", "llama3:latest"),
])
def test_default_components_are_removed_only_in_their_structural_position(name, expected):
    assert canonical_model(name) == expected
