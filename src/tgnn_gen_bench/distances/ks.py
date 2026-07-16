# import packages

from collections.abc import Sequence

import numpy as np

# distance


class KSDistance:
    """Two-sample Kolmogorov-Smirnov distance between two distributions.

    The supremum of the absolute difference between the two empirical CDFs. It
    is bounded in [0, 1] regardless of the units of the underlying metric, which
    is what makes scores from different metrics comparable to one another.

    Implemented directly rather than via scipy.stats.ks_2samp: only the
    statistic is needed, not the p-value, and this keeps scipy out of the
    dependency set.
    """

    name = "ks"

    def compute(self, real: Sequence[float], generated: Sequence[float]) -> float:
        a = np.sort(np.asarray(real, dtype=np.float64))
        b = np.sort(np.asarray(generated, dtype=np.float64))
        if a.size == 0 or b.size == 0:
            return float("nan")

        grid = np.concatenate([a, b])
        cdf_a = np.searchsorted(a, grid, side="right") / a.size
        cdf_b = np.searchsorted(b, grid, side="right") / b.size
        return float(np.max(np.abs(cdf_a - cdf_b)))


__all__ = ["KSDistance"]
