#external
import pytest

#local
import alphasynchro.algorithms.precursor_slicing


@pytest.mark.parametrize(
    "input",
    [
        "SlicedIMDistribution",
        "SlicedIMDistributionMultithreaded",
    ]
)
def test_has_classes(input):
    assert hasattr(alphasynchro.algorithms.precursor_slicing, input)
