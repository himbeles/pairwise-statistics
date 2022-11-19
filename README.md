# Pairwise Aggregated Statistics

Python package for batch / parallel processing of second order statisitics.

Follows algorithm by [Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1979), "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances." (PDF), Technical Report STAN-CS-79-773, Department of Computer Science, Stanford University](http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf).

If variance and mean of a large data set are calculated / available in batches,
`pairwise_statistics.StatisticsAggregator` can be used to combine these into the mean and variance of the total data set.

```python
# Start with statistics (sample count, mean and variance) of two data subsets `array_a` and `array_b`
array_a = np.array([1,3,5,6])
array_b = np.array([1,3,4])
n_a = len(array_a)
n_b = len(array_b)
mean_a = np.mean(array_a)
mean_b = np.mean(array_b)
var_a = np.var(array_a)
var_b = np.var(array_b)

# Combine statistics using StatisticsAggregator
sa = StatisticsAggregator(n_a, mean_a, var_a)
sa.add(n_b, mean_b, var_b)

# Check that the StatiticsAggregator `sa` reproduces the sample count (sa.n), mean (sa.mean) and variance (sa.var) of the concatenated, total dataset `array_c`
array_c = np.hstack([array_a,array_b])
n_c = len(array_c)
mean_c = np.mean(array_c)
var_c = np.var(array_c)

assert sa.n == n_c
assert np.isclose(sa.mean, mean_c)
assert np.isclose(sa.var, var_c)
```
