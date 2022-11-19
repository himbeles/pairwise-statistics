class StatisticsAggregator:
    def __init__(self, n, mean, var, ddof: int = 0):
        """Aggregator for batch processing of second order statisitics.

        If variance and mean of a large data set are calculated
        or only available in batches,
        `StatisticsAggregator` can be used to combine these into
        the mean and variance of the total data set.

        Arguments:
            n: initial sample count
            mean: initital sample mean
            var: initial sample variance
            ddof: "Delta Degrees of Freedom": the divisor used in
                in the calculation of the variance norm is `n - ddof`,
                where `n` represents the number of elements.
                By default `ddof` is zero (biased estimator).
        """
        self._n = n
        self._mean = mean
        self._ddof = ddof
        self._M2 = var * self._ddof_norm(
            n
        )  # store rescaled variance to prevent numerical inaccuracies

    @property
    def var(self):
        """Current total variance"""
        return self._M2 / self._ddof_norm(self.n)

    @property
    def mean(self):
        """Current total mean"""
        return self._mean

    @property
    def n(self):
        """Current total sample count"""
        return self._n

    @property
    def ddof(self):
        """Delta Degrees of Freedom
        the divisor used in
        in the calculation of the variance norm is `n - ddof`,
        where `n` represents the number of elements.
        By default `ddof` is zero (biased estimator).
        """
        return self._ddof

    def add(self, n_add, mean_add, var_add):
        """Add statistics batch to aggregator.

        Arguments:
            n: sample count to be added
            mean: sample mean to be added
            var: sample variance to be added
        """

        # update n
        n_old = self.n
        self._n += n_add

        # update mean
        delta = mean_add - self.mean
        self._mean += delta * n_add / self.n

        # update variance (M2 / n)
        M2_new = var_add * self._ddof_norm(n_add)
        self._M2 = self._M2 + M2_new + delta**2 * n_old * n_add / self.n

    def _ddof_norm(self, n):
        return n - self.ddof
