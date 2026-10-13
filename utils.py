import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import Rbf

# You do not have to understand this code.
# You can use these functions to generate plots.

REGRESSION_RMSE_LIMIT = 0.25


def complex_function(x, y):
    return x**3 - 3 * x * y**2


def _plot_surface(ax, x_grid, y_grid, z_grid, z_limits, title, **style):
    surface = ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        cmap="coolwarm",
        vmin=z_limits[0],
        vmax=z_limits[1],
        linewidth=0,
        antialiased=False,
        **style,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_zlim(*z_limits)
    return surface


def plot_complex_fit(X, y_hat, x_range=(-2.0, 2.0), y_range=(-2.0, 2.0), grid_n=100, smooth=0.01):
    """Plot predictions against the true surface.

    Left: a smooth surface through the predictions, plus the predicted points.
    Right: the true function.
    """
    points = np.asarray(X, dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    predictions = np.ravel(y_hat)

    true_values = complex_function(x, y)
    rmse = np.sqrt(np.mean((predictions - true_values) ** 2))

    x_axis = np.linspace(x_range[0], x_range[1], grid_n)
    y_axis = np.linspace(y_range[0], y_range[1], grid_n)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)

    # Turn the scattered predictions into a surface so the two panels can be compared.
    fitted = Rbf(x, y, predictions, smooth=smooth)
    estimated_grid = fitted(x_grid, y_grid)
    true_grid = complex_function(x_grid, y_grid)

    lowest = min(true_grid.min(), estimated_grid.min())
    highest = max(true_grid.max(), estimated_grid.max())
    z_limits = (lowest, highest)

    passed = rmse <= REGRESSION_RMSE_LIMIT
    title = f"RMSE = {rmse:.3f} {'Approved' if passed else 'Not approved'}"
    title_color = "#2ca02c" if passed else "#d62728"

    fig = plt.figure(figsize=(10, 5.2))
    fig.suptitle(title, fontsize=14, color=title_color)
    fit_axis = fig.add_subplot(1, 2, 1, projection="3d")
    true_axis = fig.add_subplot(1, 2, 2, projection="3d")

    _plot_surface(fit_axis, x_grid, y_grid, estimated_grid, z_limits, "Your fit", alpha=0.7)

    # A short line from the smooth surface to each prediction.
    height_on_surface = fitted(x, y)
    segment_start = np.column_stack([x, y, height_on_surface])
    segment_end = np.column_stack([x, y, predictions])
    stems = np.stack([segment_start, segment_end], axis=1)
    fit_axis.add_collection3d(Line3DCollection(stems, colors="0.25", linewidths=0.8, alpha=0.7))
    fit_axis.scatter(x, y, predictions, c="green", s=28, depthshade=False, edgecolors="k", linewidths=0.4)

    _plot_surface(true_axis, x_grid, y_grid, true_grid, z_limits, "True function")

    color_scale = plt.cm.ScalarMappable(norm=plt.Normalize(*z_limits), cmap="coolwarm")
    colorbar_axis = fig.add_axes([0.08, 0.07, 0.84, 0.04])
    fig.colorbar(color_scale, cax=colorbar_axis, orientation="horizontal")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.22)
    return fig, (fit_axis, true_axis), rmse


def plot_decision_boundary(
    predict,
    xlim=(-4, 4),
    ylim=(-4, 4),
    threshold=0.0,
    grid_n=100,
    points=None,
    labels=None,
):
    """Contour plot of a binary classifier in the (x0, x1) plane.

    predict receives an array of shape (n, 2) and returns one score per row.
    Those two columns are the inputs, x0 and x1.

    The black curve is where that score equals threshold.
    Pass probabilities and set threshold=0.5.
    If you pass the value before the sigmoid instead, set threshold=0.
    points and labels, if given, are drawn on top of the contour.
    """
    x_axis = np.linspace(xlim[0], xlim[1], grid_n)
    y_axis = np.linspace(ylim[0], ylim[1], grid_n)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    scores = np.asarray(predict(grid_points), dtype=float).ravel()
    scores = scores.reshape(x_grid.shape)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.contourf(x_grid, y_grid, scores, levels=50, cmap="coolwarm")
    ax.contour(x_grid, y_grid, scores, levels=[threshold], colors="black", linewidths=2)
    if points is not None:
        points = np.asarray(points, dtype=float)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=np.ravel(labels),
            cmap="coolwarm",
            edgecolors="k",
            alpha=0.6,
        )
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title("Decision boundary")
    fig.tight_layout()
    return fig, ax
