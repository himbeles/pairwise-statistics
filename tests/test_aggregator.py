from pairwise_statistics import StatisticsAggregator
import numpy as np
import pytest


@pytest.mark.parametrize("ddof", [0, 1])
def test_aggregator(ddof):
    # combine statistics of array a and b
    array_a = np.array([1, 3, 5, 6])
    array_b = np.array([1, 3, 4])
    n_a = len(array_a)
    n_b = len(array_b)
    mean_a = np.mean(array_a)
    mean_b = np.mean(array_b)
    var_a = np.var(array_a, ddof=ddof)
    var_b = np.var(array_b, ddof=ddof)

    sa = StatisticsAggregator(n_a, mean_a, var_a, ddof=ddof)
    sa.add(n_b, mean_b, var_b)

    # result should be variance and mean of concatenated, total array
    array_c = np.hstack([array_a, array_b])
    n_c = len(array_c)
    mean_c = np.mean(array_c)
    var_c = np.var(array_c, ddof=ddof)

    assert sa.n == n_c
    assert np.isclose(sa.mean, mean_c)
    assert np.isclose(sa.var, var_c)
