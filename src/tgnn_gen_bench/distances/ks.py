# import packages

from collections.abc import Sequence

import numpy as np

# distance


class KSDistance:
    """Compute the two-sample Kolmogorov-Smirnov distance."""

    name = "ks"

    def compute(self, real: Sequence[float], generated: Sequence[float]) -> float:
        """Return the KS statistic between two value samples."""
        a = np.sort(np.asarray(real, dtype=np.float64))
        b = np.sort(np.asarray(generated, dtype=np.float64))
        if a.size == 0 or b.size == 0:
            return float("nan")

        grid = np.concatenate([a, b])
        cdf_a = np.searchsorted(a, grid, side="right") / a.size
        cdf_b = np.searchsorted(b, grid, side="right") / b.size
        return float(np.max(np.abs(cdf_a - cdf_b)))


__all__ = ["KSDistance"]
