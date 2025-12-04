import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
plt.close('all')

# load data
water = loadmat('data/dataset/task_1/water/hist_140.mat')
fat = loadmat('data/dataset/task_1/fat/hist_140.mat')

# extract counts and energies (keV)
water_c = water['x'].flatten()
water_e = (water['y'].flatten() * 1000)[:-1]   # convert to keV, drop last edge
fat_c   = fat['x'].flatten()
fat_e   = (fat['y'].flatten() * 1000)[:-1]

# define detector energy bins: 20-50 keV and 50 keV - inf
mask_20_50_w = (water_e >= 20) & (water_e < 50)
mask_50_inf_w = (water_e >= 50)

mask_20_50_f = (fat_e >= 20) & (fat_e < 50)
mask_50_inf_f = (fat_e >= 50)

# total counts detected in each detector bin
water_counts_bins = np.array([
    np.sum(water_c[mask_20_50_w]),
    np.sum(water_c[mask_50_inf_w])
])

fat_counts_bins = np.array([
    np.sum(fat_c[mask_20_50_f]),
    np.sum(fat_c[mask_50_inf_f])
])

labels = ['20-50 keV', '>=50 keV']

# unified plotting function (creates bar chart, no legend, label under each bar)
def plot_bin_counts(w_bins, f_bins, labels, title, ylabel):
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8,5))
    bars_w = ax.bar(x - width/2, w_bins, width, color='C0', alpha=0.9)
    bars_f = ax.bar(x + width/2, f_bins, width, color='C1', alpha=0.9)

    max_h = max(np.max(w_bins), np.max(f_bins), 1.0)
    ax.set_ylim(bottom=-0.08 * max_h)  # space below for labels

    for bar in bars_w:
        center = bar.get_x() + bar.get_width() / 2
        ax.annotate('Water',
                    xy=(center, 0),
                    xytext=(0, -8),
                    textcoords='offset points',
                    ha='center', va='top', fontsize=9)

    for bar in bars_f:
        center = bar.get_x() + bar.get_width() / 2
        ax.annotate('Fat',
                    xy=(center, 0),
                    xytext=(0, -8),
                    textcoords='offset points',
                    ha='center', va='top', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# plot original counts
plot_bin_counts(
    water_counts_bins,
    fat_counts_bins,
    labels,
    'Detector Counts per Energy Bin (20-50 keV, >=50 keV)',
    'Total Counts'
)

# ======= weighted counts =======
w_low, w_high = 0.8, 0.2
water_weighted_bins = np.array([
    np.sum(water_c[mask_20_50_w] * w_low),
    np.sum(water_c[mask_50_inf_w] * w_high)
])
fat_weighted_bins = np.array([
    np.sum(fat_c[mask_20_50_f] * w_low),
    np.sum(fat_c[mask_50_inf_f] * w_high)
])

# plot weighted counts using same function
plot_bin_counts(
    water_weighted_bins,
    fat_weighted_bins,
    labels,
    'Weighted Detector Counts per Energy Bin (80% for 20-50 keV, 20% for >=50 keV)',
    'Weighted Total Counts'
)

# Calculate contrast ratios normalized to water for both original and weighted counts
def contrast_ratio(tissue_counts, water_counts):
    return (tissue_counts - water_counts) * 100 / water_counts

orig_contrast = contrast_ratio(fat_counts_bins, water_counts_bins)
weighted_contrast = contrast_ratio(fat_weighted_bins, water_weighted_bins)
print("Original Contrast Ratios (Fat vs Water):", orig_contrast)
print("Weighted Contrast Ratios (Fat vs Water):", weighted_contrast)
# The contrast ratios stay the same after weighting since the weights are applied uniformly within each bin.