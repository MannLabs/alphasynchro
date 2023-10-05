#external
import pytest

#local
import alphasynchro.ms.peaks.merged_fragments


@pytest.mark.parametrize(
    "input",
    [
        "MergedFragments",
        "FrameIntensitiesCalculator",
        "StatsAggregateCalculator",
    ]
)
def test_has_classes(input):
    assert hasattr(alphasynchro.ms.peaks.merged_fragments, input)
