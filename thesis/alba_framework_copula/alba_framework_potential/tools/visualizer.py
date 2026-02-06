
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Optional, List, Any

def plot_potential_field_2d(
    optimizer: Any,
    dims: tuple = (0, 1),
    show_points: bool = True,
    title: str = "ALBA-Potential Field (2D Projection)",
    save_path: Optional[str] = None
):
    """
    Visualize the leaf partitioning and potential field projected to 2 dimensions.
    
    Parameters
    ----------
    optimizer : ALBA
        The fitted ALBA optimizer instance.
    dims : tuple
        The two dimensions to visualize (index).
    show_points : bool
        Whether to scatter plot the observed points.
    title : str
        Plot title.
    save_path : Optional[str]
        If provided, save figure to this path.
    """
    fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
    
    d1, d2 = dims
    
    # Get all leaves
    leaves = optimizer.leaves
    coherence_tracker = optimizer._coherence_tracker
    
    # Determine bounds from root
    root_bounds = optimizer.root.bounds
    x_min, x_max = root_bounds[d1]
    y_min, y_max = root_bounds[d2]
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Color map for potential (0=Good, 1=Bad)
    # We use coolwarm: Blue (0) -> Red (1)
    # This is intuitive: Blue/Cool = Minimum (Good), Red/Hot = High Error (Bad)
    cmap = plt.get_cmap("coolwarm") 
    
    # Plot Leaves
    for leaf in leaves:
        # Get bounds for the 2 dims
        b = leaf.bounds
        l1, u1 = b[d1]
        l2, u2 = b[d2]
        
        width = u1 - l1
        height = u2 - l2
        
        # Get potential
        if coherence_tracker and optimizer._use_potential_field:
            pot  = coherence_tracker.get_potential(leaf, leaves)
        else:
            pot = 0.5
            
        color = cmap(pot)
        
        rect = patches.Rectangle(
            (l1, l2), width, height,
            linewidth=1, edgecolor='gray', facecolor=color, alpha=0.5
        )
        ax.add_patch(rect)
        
        # Annotate with potential value if space allows
        if len(leaves) < 50:
            cx, cy = l1 + width/2, l2 + height/2
            ax.text(cx, cy, f"{pot:.1f}", ha='center', va='center', fontsize=8, color='black', fontweight='bold')

    # Plot Points
    if show_points and optimizer.X_all:
        X = np.array(optimizer.X_all)
        # Use simple color for points
        ax.scatter(X[:, d1], X[:, d2], c='black', s=15, alpha=0.6, label='Observations')
        
        # Highlight best
        if optimizer.best_x is not None:
            ax.scatter(
                optimizer.best_x[d1], optimizer.best_x[d2],
                c='gold', s=200, marker='*', edgecolors='black', linewidth=1.5, label='Best (Global Min Estimate)'
            )

    ax.set_xlabel(f"Dim {d1}")
    ax.set_ylabel(f"Dim {d2}")
    ax.set_title(title)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Potential Field Value\n(Blue=0.0 [Target] -> Red=1.0 [Avoid])')
    
    # Add Guide Text Box
    guide_text = (
        "GUIDE:\n"
        "🟦 Blue Regions: Low Potential (Promising)\n"
        "🟥 Red Regions: High Potential (Unpromising)\n"
        "⭐ Gold Star: Best Point Found"
    )
    plt.text(
        0.02, 0.98, guide_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
    )
    
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Test run
    from alba_framework_potential.optimizer import ALBA
    from alba_framework_potential.benchmarks import get_function
    
    print("Running visualizer test on Sphere 2D...")
    func, bounds = get_function("sphere", 2)
    
    alba = ALBA(bounds=bounds, total_budget=100, use_potential_field=True)
    alba.optimize(func, 100)
    
    plot_potential_field_2d(alba, save_path="visualizer_test.png")
