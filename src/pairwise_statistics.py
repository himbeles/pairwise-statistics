class StatisticsAggregator:
    def __init__(self, ddof: int = 0, use_robust_mean: bool = True):
        """Aggregator for batch processing of second order statisitics.

        If variance and mean of a large data set are calculated
        or only available in batches,
        `StatisticsAggregator` can be used to combine these into
        the mean and variance of the total data set.

        Arguments:
            ddof: "Delta Degrees of Freedom": the divisor used in
                in the calculation of the variance norm is `n - ddof`,
                where `n` represents the number of elements.
                By default `ddof` is zero (biased estimator).
            use_robust_mean: use numerically robust, but slightly slower
                method for mean aggregation
        """
        self._n = 0
        self._mean = None
        self._M2 = None  # rescaled variance is stored to prevent numerical inaccuracies
        self._ddof = ddof
        self._use_robust_mean = use_robust_mean

    @property
    def var(self):
        """Current total variance"""
        if self._M2 is None:
            return None
        else:
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

        if n_old == 0:
            # initialize mean and M2 (variance * n)
            self._mean = mean_add
            self._M2 = var_add * self._ddof_norm(n_add)
        else:
            # update existing mean
            delta = mean_add - self.mean
            if self._use_robust_mean:
                self._mean = (self.mean * n_old + mean_add * n_add) / (self.n)
            else:
                self._mean += delta * n_add / self.n

            # update existing variance (M2 / n)
            M2_new = var_add * self._ddof_norm(n_add)
            self._M2 = self._M2 + M2_new + delta**2 * n_old * n_add / self.n

    def _ddof_norm(self, n):
        return n - self.ddof
