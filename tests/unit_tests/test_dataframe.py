#external
import numpy as np
import pandas as pd
import pytest

#local
import alphasynchro.data.dataframe


@pytest.fixture(scope="module")
def df():
    df = alphasynchro.data.dataframe.DataFrame(
        a=np.arange(10),
        b=np.arange(10)+100,
        c=np.arange(10)*-1,
    )
    return df


def test_len(df):
    output = len(df)
    expected = 10
    assert output == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        ("a", np.arange(10)),
        ("b", np.arange(10)+100),
        ("c", np.arange(10)*-1),
    ]
)
def test_array(df, input, expected):
    output = df.__getattribute__(input)
    assert np.array_equal(output, expected)


@pytest.mark.parametrize(
    "selection, columns",
    [
        (..., ["a", "b"]),
        (slice(None, None, 3), ["a"]),
        ([0, 4, 5], ["a", "c"]),
        ([0, 4, -1], ["a", "b"]),
        ([0, 4, 8], None),
    ]
)
def test_as_dataframe(df, selection, columns):
    expected_df = pd.DataFrame(
        {
            "a": np.arange(10),
            "b": np.arange(10)+100,
            "c": np.arange(10)*-1,
        }
    )
    if columns is not None:
        expected_df = expected_df[columns]
    expected = expected_df.iloc[selection].reset_index(drop=True)
    output = df.as_dataframe(indices=selection, columns=columns)
    assert expected.equals(output)
