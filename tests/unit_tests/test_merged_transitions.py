#external
import pytest

#local
import alphasynchro.ms.transitions.merged_transitions


@pytest.mark.parametrize(
    "input",
    [
        "MergedFrames",
    ]
)
def test_has_classes(input):
    assert hasattr(alphasynchro.ms.transitions.merged_transitions, input)
