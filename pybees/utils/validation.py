"""
This module gathers a range of utility functions.
"""

# Authors: Joseph Moorhouse <moorhouse@live.co.uk>
#
# License: BSD 3 clause

import numpy as np

from sklearn.utils.validation import assert_all_finite, check_array

# =============================================================================
# Continuous cost functions
# =============================================================================

def check_shape(x, two_dim=False):
    x = x[np.newaxis, :] if x.ndim == 1 else x

    if two_dim:
        if x.shape[1] != 2:
            raise ValueError(f"Bad shape {x.shape}. ``x`` must have "
                            "shape(n_coordinates, 2). Try "
                            f"shape({x.shape[0]}, 2)")

    return x

# =============================================================================
# Input checking
# =============================================================================