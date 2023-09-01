import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from utils import NODAL_WORKING_DIR, get_shots

FONT_SIZE = 10  # [pt]
plt.rcParams.update({'font.size': FONT_SIZE})

# Read in CSV files for shot Y5
df = pd.read_csv(NODAL_WORKING_DIR / 'shot_gather_measurements' / 'Y5.csv')

# Define values to plot
amplitude = df.sta_lta_amp
rms = df.pre_shot_rms * 1e6  # [μm/s]
distance = df.dist_m / 1000  # [km]

# Mask extreme RMS values
mask = rms <= np.percentile(rms, 95)
amplitude = amplitude[mask]
rms = rms[mask]
distance = distance[mask]

fig, ax = plt.subplots(figsize=(3.47, 3.3))
sm = ax.scatter(
    x=rms, y=amplitude, c=distance, s=15, clip_on=False, alpha=0.7, lw=0, cmap='turbo_r'
)
ax.set_xlabel('RMS seismic ground velocity (μm/s),\n20 s window pre-shot')  # WIN_DUR
ax.set_ylabel('STA/LTA amplitude')

ax.set_xlim(0, 1.2)
ax.set_ylim(0.6, 10)

ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(1))

# Make pretty
for side in 'top', 'right':
    ax.spines[side].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['left'].set_bounds(1, ax.get_ylim()[1])

fig.tight_layout(pad=0.2)

# Inset colorbar (after tight_layout) call!
cbar_height = 0.25  # As fraction of total axis height
cbar_aspect = 8
cbar_x = 0.745  # In axis coordinates
cax = inset_axes(
    ax,
    bbox_to_anchor=(cbar_x, 1 - cbar_height, cbar_height / cbar_aspect, cbar_height),
    bbox_transform=ax.transAxes,
    width='100%',
    height='100%',
)
fig.colorbar(sm, cax=cax, label='Distance from\nshot (km)')
cax.yaxis.set_minor_locator(plt.MultipleLocator(5))

fig.show()

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

fig.savefig(
    Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve()
    / 'y5_amp_rms_scatter.pdf'
)
