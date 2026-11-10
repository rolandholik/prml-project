import numpy as np

def flatten(points):
    """Remove dimension with the least variance."""
    variances = np.var(points, axis=0)
    lowest_var = np.argmin(variances)
    reduced = np.delete(points, lowest_var, axis=1)
    return reduced

def arclength(points, n_out=25):
    """Resample the trajectory to n_out points, evenly spaced by arc length."""
    deltas = np.diff(points, axis=0)
    # Euclidian distance
    seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))
    # Cumulative distances for each point
    arc_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = arc_length[-1]

    if total_length == 0:
        return np.repeat(points[:1], n_out, axis=0)

    # Space distanced between points evenly
    target_lengths = np.linspace(0, total_length, n_out)

    # Space the points evenly
    resampled = np.zeros((n_out, points.shape[1]))
    for dim in range(points.shape[1]):
        resampled[:, dim] = np.interp(target_lengths, arc_length, points[:, dim])
    return resampled

def normalize(points):
    """Center on centroid, scale by bounding-box diagonal."""
    centered = points - points.mean(axis=0)
    diag = np.sqrt(((centered.max(axis=0) - centered.min(axis=0)) ** 2).sum())
    if diag < 1e-10:
        diag = 1.0
    return centered / diag