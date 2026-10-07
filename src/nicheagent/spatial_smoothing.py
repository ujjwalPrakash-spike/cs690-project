"""Neighborhood-based Spatial Smoothing.

Implements single or multi-round spatial graph majority smoothing
as formulated in NicheAgent.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def spatial_graph_smoothing(
    spot_labels: pd.Series,
    spatial_coords: np.ndarray,
    k_neighbors: int = 6,
    num_rounds: int = 1
) -> pd.Series:
    """Performs majority-vote spatial smoothing over k nearest spatial neighbors.

    Parameters:
    - spot_labels: Initial categorical / string labels per spot.
    - spatial_coords: (N, 2) array of spatial coordinates.
    - k_neighbors: Number of spatial neighbors (Visium grid has 6 immediate neighbors).
    - num_rounds: Number of iterative smoothing rounds (default: 1).
    """
    if num_rounds <= 0:
        return spot_labels.copy()

    # Fit spatial nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm="auto").fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)

    current_labels = np.array(spot_labels.values)

    for r in range(num_rounds):
        new_labels = np.empty_like(current_labels)
        for i in range(len(current_labels)):
            neighbor_indices = indices[i]
            neighbor_votes = current_labels[neighbor_indices]
            # Majority vote
            vals, counts = np.unique(neighbor_votes, return_counts=True)
            new_labels[i] = vals[np.argmax(counts)]
        current_labels = new_labels

    return pd.Series(current_labels, index=spot_labels.index, name=f"smoothed_round_{num_rounds}")
