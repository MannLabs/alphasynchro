#external
import numpy as np
import pytest
import scipy.signal

#local
import alphasynchro.data.sparse_indices
import alphasynchro.stats.distributions


@pytest.fixture(scope="module")
def cdf_with_offset():
    indptr = np.array([0, 2, 3, 6, 8])
    values = np.array([.5, 1.0, 1.0, .2, .5, 1., .3, 1.0])
    start_offsets = np.array([1, 1, 0, 2], dtype=np.int64)
    cdf_with_offset = alphasynchro.stats.distributions.CDFWithOffset(
        indptr=indptr,
        values=values,
        start_offsets=start_offsets,
    )
    return cdf_with_offset


@pytest.mark.parametrize(
    "input, expected",
    [
        (-1, False),
        (0, True),
        (1, True),
        (2, True),
        (3, True),
        (4, False),
    ]
)
def test_is_valid(cdf_with_offset, input, expected):
    output = cdf_with_offset.is_valid(input)
    assert output is expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (-1, 0),
        (0, 1),
        (1, 1),
        (2, 0),
        (3, 2),
        (4, 0),
    ]
)
def test_get_start_offset(cdf_with_offset, input, expected):
    output = cdf_with_offset.get_start_offset(input)
    assert output == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (-1, 0),
        (0, 3),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
    ]
)
def test_get_end_offset(cdf_with_offset, input, expected):
    output = cdf_with_offset.get_end_offset(input)
    assert output == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (-1, 0),
        (0, 2),
        (1, 1),
        (2, 3),
        (3, 2),
        (4, 0),
    ]
)
def test_get_size(cdf_with_offset, input, expected):
    output = cdf_with_offset.get_size(input)
    assert output == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (-1, np.array([])),
        (0, np.array([.5, 1.])),
        (1, np.array([1.])),
        (2, np.array([.2, .5, 1.])),
        (3, np.array([.3, 1.])),
        (4, np.array([])),
    ]
)
def test_get_cdf(cdf_with_offset, input, expected):
    output = cdf_with_offset.get_cdf(input)
    assert np.array_equal(output, expected)


def test_filter(cdf_with_offset):
    indptr = np.array([0, 1, 4])
    values = np.array([1.0, .2, .5, 1.])
    start_offsets = np.array([1, 0], dtype=np.int64)
    input_data = np.array([1,2], dtype=np.int64)
    output = cdf_with_offset.filter(input_data)
    assert np.array_equal(output.indptr, indptr)
    assert np.array_equal(output.values, values)
    assert np.array_equal(output.start_offsets, start_offsets)


@pytest.mark.parametrize(
    "input, expected",
    [
        (-1, np.array([])),
        (0, np.array([.5, .5])),
        (1, np.array([1.])),
        (2, np.array([.2, .3, .5])),
        (3, np.array([.3, .7])),
        (4, np.array([])),
    ]
)
def test_get_pdf(cdf_with_offset, input, expected):
    output = cdf_with_offset.get_pdf(input)
    assert np.array_equal(output, expected)


@pytest.fixture(scope="module")
def cdf_for_smoothing():
    indptr = np.array([0, 10, 13, 15])
    values = np.arange(15, dtype=np.float64)
    values[:10] /= 10
    values[10] = .5
    values[11] = .5
    values[12] = 1
    values[13:] = 0
    start_offsets = np.array([0, 4, 2], dtype=np.int64)
    summed_values = np.array([1, 2, 0])
    cdf_with_offset_and_summed_values = alphasynchro.stats.distributions.CDFWithOffsetAndSummedValues(
        indptr=indptr,
        values=values,
        start_offsets=start_offsets,
        summed_values=summed_values,
    )
    return cdf_with_offset_and_summed_values


@pytest.mark.parametrize(
    "input",
    [1, 3, 5]
)
def test_smooth_sigma(cdf_for_smoothing, input):
    bins = np.arange(-3 * input, 3 * input + 1)
    normalization_constant = 1 / (input * np.sqrt(2 * np.pi))
    exponents = -bins**2 / (2 * input**2)
    smooth_array = normalization_constant * np.exp(exponents)
    output = cdf_for_smoothing.smooth(0, smooth_array)
    distribution = cdf_for_smoothing.get_pdf(0)
    mu = 0
    sigma = input
    bins = np.arange(-3*sigma, 3*sigma + 1)
    normal = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) )
    expected = scipy.signal.convolve(
        distribution,
        normal,
        mode='same'
    )
    assert np.allclose(output, expected)


@pytest.fixture(scope="module")
def cdf_with_offset_and_summed_values(cdf_with_offset):
    summed_values = np.array([.1, .2, .3, .4])
    cdf_with_offset_and_summed_values = alphasynchro.stats.distributions.CDFWithOffsetAndSummedValues(
        indptr=cdf_with_offset.indptr,
        values=cdf_with_offset.values,
        start_offsets=cdf_with_offset.start_offsets,
        summed_values=summed_values,
    )
    return cdf_with_offset_and_summed_values


@pytest.mark.parametrize(
    "input, expected",
    [
        (-1, 0),
        (0, .1),
        (1, .2),
        (2, .3),
        (3, .4),
        (4, 0),
    ]
)
def test_get_summed_value_of_cdf_with_offsets(cdf_with_offset_and_summed_values, input, expected):
    output = cdf_with_offset_and_summed_values.get_summed_value(input)
    assert output == expected


def test_filter_summed_values_and_offsets(cdf_with_offset_and_summed_values):
    indptr = np.array([0, 1, 4])
    values = np.array([1.0, .2, .5, 1.])
    start_offsets = np.array([1, 0], dtype=np.int64)
    summed_values = np.array([.2, .3,])
    input_data = np.array([1,2], dtype=np.int64)
    output = cdf_with_offset_and_summed_values.filter(input_data)
    assert np.array_equal(output.indptr, indptr)
    assert np.array_equal(output.values, values)
    assert np.array_equal(output.start_offsets, start_offsets)
    assert np.array_equal(output.summed_values, summed_values)


@pytest.fixture(scope="module")
def cdf_with_summed_values(cdf_with_offset):
    summed_values = np.array([.1, .2, .3, .4])
    cdf_with_summed_values = alphasynchro.stats.distributions.CDFWithSummedValues(
        indptr=cdf_with_offset.indptr,
        values=cdf_with_offset.values,
        summed_values=summed_values,
    )
    return cdf_with_summed_values


@pytest.mark.parametrize(
    "input, expected",
    [
        (-1, 0),
        (0, .1),
        (1, .2),
        (2, .3),
        (3, .4),
        (4, 0),
    ]
)
def test_get_summed_value(cdf_with_summed_values, input, expected):
    output = cdf_with_summed_values.get_summed_value(input)
    assert output == expected


def test_filter_summed_values(cdf_with_summed_values):
    indptr = np.array([0, 1, 4])
    values = np.array([1.0, .2, .5, 1.])
    summed_values = np.array([.2, .3,])
    input_data = np.array([1,2], dtype=np.int64)
    output = cdf_with_summed_values.filter(input_data)
    assert np.array_equal(output.indptr, indptr)
    assert np.array_equal(output.values, values)
    assert np.array_equal(output.summed_values, summed_values)


@pytest.fixture(scope="module")
def cdf():
    indptr = np.array([0, 2, 3, 6, 8])
    values = np.array([.5, 1.0, 1.0, .2, .5, 1., .3, 1.0])
    cdf = alphasynchro.stats.distributions.CDF(
        indptr=indptr,
        values=values,
    )
    return cdf


@pytest.fixture(scope="module")
def pdf():
    indptr = np.array([0, 2, 3, 6, 8])
    values = np.array([.5, .5, 1.0, .2, .3, .5, .3, .7])
    pdf = alphasynchro.stats.distributions.PDF(
        indptr=indptr,
        values=values,
    )
    return pdf


def test_cdf_to_pdf(cdf, pdf):
    new_pdf = cdf.to_pdf()
    assert new_pdf == pdf


def test_pdf_to_cdf(cdf, pdf):
    new_cdf = pdf.to_cdf()
    assert new_cdf == cdf
