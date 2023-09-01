import os
import subprocess
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from utils import MASK_DISTANCE_KM, NODAL_WORKING_DIR

FONT_SIZE = 10  # [pt]
plt.rcParams.update({'font.size': FONT_SIZE})

# Read in CSV files for shot Y5
df = pd.read_csv(NODAL_WORKING_DIR / 'shot_gather_measurements' / 'Y5.csv')

# Define values to plot
amplitude = df.sta_lta_amp.values
rms = df.pre_shot_rms.values * 1e6  # [μm/s]
distance = df.dist_m.values / 1000  # [km]

# Mask extreme RMS values
rms_mask = rms <= np.percentile(rms, 95)
amplitude = amplitude[rms_mask]
rms = rms[rms_mask]
distance = distance[rms_mask]

# Define mask for masking distance
distance_mask = distance >= MASK_DISTANCE_KM

# Plot closer markers on top (for those beyond masking distance)
sorted_idx = np.argsort(distance[distance_mask])[::-1]

fig, ax = plt.subplots(figsize=(3.47, 3.3))

# Plot those nodes within the masking distance first (underneath)
ax.scatter(
    x=rms[~distance_mask],
    y=amplitude[~distance_mask],
    s=10,
    clip_on=False,
    lw=0.5,
    facecolor='none',
    edgecolor='tab:gray',
)

# Plot the remaining nodes as colored markers
sm = ax.scatter(
    x=rms[distance_mask][sorted_idx],
    y=amplitude[distance_mask][sorted_idx],
    c=distance[distance_mask][sorted_idx],
    s=15,
    clip_on=False,
    alpha=0.7,
    lw=0,
    cmap=cc.m_rainbow_r,
)

ax.set_xlabel('RMS seismic ground velocity (μm/s),\n20 s window pre-shot Y5')  # WIN_DUR
ax.set_ylabel('STA/LTA amplitude')

ax.set_xlim(0, 1.2)
ax.set_ylim(0.7, 10)

ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(1))

# Make pretty
for side in 'top', 'right':
    ax.spines[side].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['left'].set_bounds(1, ax.get_ylim()[1])

fig.tight_layout(pad=0.2)

# Inset colorbar (after tight_layout) call!
cbar_width = 0.25  # As fraction of total axis width
cbar_aspect = 8
cax = inset_axes(
    ax,
    bbox_to_anchor=(1 - cbar_width, 0.99, cbar_width, cbar_width / cbar_aspect),
    bbox_transform=ax.transAxes,
    width='100%',
    height='100%',
)
fig.colorbar(sm, cax=cax, label='Distance from\nshot Y5 (km)', orientation='horizontal')
cax.xaxis.set_minor_locator(plt.MultipleLocator(5))

fig.show()

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

fig.savefig(
    Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve()
    / 'y5_amp_rms_scatter.pdf'
)
