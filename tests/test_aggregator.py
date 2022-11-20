from pairwise_statistics import StatisticsAggregator
import numpy as np
import pytest


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("use_robust_mean", [True, False])
def test_aggregator(ddof, use_robust_mean):
    # combine statistics of array a, b, c
    array_a = np.array([1, 3, 5, 6])
    array_b = np.array([1, 3, 4])
    array_c = np.array([6, 3])
    n_a = len(array_a)
    n_b = len(array_b)
    n_c = len(array_c)
    mean_a = np.mean(array_a)
    mean_b = np.mean(array_b)
    mean_c = np.mean(array_c)
    var_a = np.var(array_a, ddof=ddof)
    var_b = np.var(array_b, ddof=ddof)
    var_c = np.var(array_c, ddof=ddof)

    sa = StatisticsAggregator(ddof=ddof, use_robust_mean=use_robust_mean)
    sa.add(n_a, mean_a, var_a)
    sa.add(n_b, mean_b, var_b)
    sa.add(n_c, mean_c, var_c)

    # result should be variance and mean of concatenated, total array
    array_d = np.hstack([array_a, array_b, array_c])
    n_d = len(array_d)
    mean_d = np.mean(array_d)
    var_d = np.var(array_d, ddof=ddof)

    assert sa.n == n_d
    assert np.isclose(sa.mean, mean_d)  # type: ignore
    assert np.isclose(sa.var, var_d)  # type: ignore
