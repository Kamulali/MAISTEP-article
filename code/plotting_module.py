# plotting_module.py
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_predictions(star_id, target, predictions, median, lower_bound, upper_bound, decimals, unit, plot_dir):
    """
    Generates and saves a high-resolution plot for predictions with median and uncertainty.
    """
    # Set up plot dimensions and determine number of bins for histogram
    plt.figure(figsize=(6.5, 6))
    nbins = max(1, int(1 + np.ceil(np.log2(len(predictions)))))  # Handle small dataset case

    # Histogram plot with step line style
    plt.hist(predictions, bins=nbins, histtype='step', lw=3, alpha=0.9,
             weights=np.ones_like(predictions) / len(predictions), color='black')

    # Highlight uncertainty bounds
    ymax = plt.ylim()[1]
    plt.fill_betweenx([0, ymax], lower_bound, upper_bound, color='gray', alpha=0.3)

    # Label with median and uncertainty bounds
    median_text = f"${median:.{decimals}f}^{{+{upper_bound - median:.{decimals}f}}}_{{-{median - lower_bound:.{decimals}f}}}$ {unit}"
    plt.plot([], [], label=median_text)

    # Axis labels and legend
    plt.xlabel(f'{target} ({unit})', fontsize=18)
    plt.legend(frameon=False, fontsize=18, handlelength=0.0, handletextpad=0.0)

    # Plot aesthetics
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([])  # Hide y-axis ticks for cleaner look
    ax.tick_params(axis='both', which='major', direction='out', width=3, length=6, labelsize=16, pad=6)

    # Save the plot
    os.makedirs(plot_dir, exist_ok=True)  # Ensure directory exists
    plot_path = os.path.join(plot_dir, f'{star_id}_{target}_distribution.pdf')
    plt.savefig(plot_path, bbox_inches='tight', dpi=2500)
    plt.close()
    #print(f"Plot saved to {plot_path}")

