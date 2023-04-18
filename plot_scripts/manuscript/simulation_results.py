import os
import subprocess
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import NODAL_WORKING_DIR

FONT_SIZE = 10  # [pt]
plt.rcParams.update({'font.size': FONT_SIZE})

# Choose 'Y5' or 'X5' — need to run this script once for each, and save figure!
SHOT = 'Y5'

# Some logic to load the remaining transect-specific params correctly
if SHOT == 'Y5':
    RUN = '20_shot_y5_new_stf_hf'
    Z_SRC = 734  # [m]
elif SHOT == 'X5':
    RUN = '22_shot_x5_new_stf_hf'
    Z_SRC = 464  # [m]
else:
    raise ValueError
PRESSURE_SNAPSHOT_DIR = NODAL_WORKING_DIR / 'fdprop' / 'nodal_fdprop_pressure_snapshots'
TIMESTAMPS = np.arange(0, 20000 + 1000, 1000)  # Same for both transects
XLIM = (0, 24)  # [km] Relative to shot x-position
YLIM = (-0.5, 5)  # [km] Relative to shot z-position
DT = 0.004  # [s]
X_SRC = 1500  # [m]

# Constants
M_PER_KM = 1000  # [m/km]


# Helper function to read in pressure matrix for a specified run and timestamp
def _p_matrix_from_timestamp(run, timestamp):
    file_pattern = f'{run}__*_*_*_*_*_{timestamp}.npy'
    files = list(PRESSURE_SNAPSHOT_DIR.glob(file_pattern))
    assert len(files) == 1
    file = files[0]
    p = np.load(file).T
    param = file.stem.replace(RUN + '__', '')  # String of 6 integers
    hoz_min, hoz_max, vert_min, vert_max, dx, _ = np.array(param.split('_')).astype(int)
    # Only include the portion of the matrix that's IN the plotted zone!
    left_trim = ((XLIM[0] * M_PER_KM) + X_SRC) - hoz_min  # [m]
    top_trim = vert_max - ((YLIM[1] * M_PER_KM) + Z_SRC)  # [m]
    right_trim = hoz_max - ((XLIM[1] * M_PER_KM) + X_SRC)  # [m]
    p_trim = p[: -top_trim // dx, left_trim // dx : -right_trim // dx]
    if p_trim.max() > 0:
        p /= p_trim.max()
    return p, hoz_min, hoz_max, vert_min, vert_max, dx


#%% Load in waveform snapshot stuff

# Aggregate pressure matrix
p_agg, hoz_min, hoz_max, vert_min, vert_max, dx = _p_matrix_from_timestamp(
    RUN, TIMESTAMPS[0]
)
for timestamp in tqdm(TIMESTAMPS[1:], initial=1, total=TIMESTAMPS.size):
    p_agg += _p_matrix_from_timestamp(RUN, timestamp)[0]

# Terrain from .dat file for this run
terrain_contour = np.loadtxt(
    NODAL_WORKING_DIR / 'fdprop' / 'Acoustic_2D' / f'imush_{SHOT.lower()}_buffer.dat'
)

#%% Plot

fig, ax = plt.subplots(figsize=(7.17, 1.9))

# Plot pressure
extent = [
    (hoz_min - X_SRC) / M_PER_KM,
    (hoz_max - X_SRC) / M_PER_KM,
    (vert_min - Z_SRC) / M_PER_KM,
    (vert_max - Z_SRC) / M_PER_KM,
]
im = ax.imshow(p_agg, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, extent=extent)

# Plot terrain
ax.fill_between(
    (terrain_contour[:, 0] - X_SRC) / M_PER_KM,
    -1,
    (terrain_contour[:, 1] - Z_SRC) / M_PER_KM,
    lw=0.5,  # Makes pressure–terrain interface a little smoother-looking
    color='tab:gray',
)

# Timestamp labels
text = ax.text(
    0.99,
    0.95,
    ', '.join([f'{timestamp * DT:g}' for timestamp in TIMESTAMPS]) + ' s',
    ha='right',
    va='top',
    transform=ax.transAxes,
)
text.set_path_effects(
    [path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()]
)

# Axis params
ax.set_xlabel(f'Distance from shot {SHOT} (km)')
ax.set_ylabel(f'Elevation relative\nto shot {SHOT} (km)')
ax.set_xlim(XLIM)
ax.set_ylim(YLIM)
major_tick_interval = 2  # [km]
minor_tick_interval = 1  # [km[
ax.xaxis.set_major_locator(plt.MultipleLocator(major_tick_interval))
ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_tick_interval))
ax.yaxis.set_major_locator(plt.MultipleLocator(major_tick_interval))
ax.yaxis.set_minor_locator(plt.MultipleLocator(minor_tick_interval))
ax.set_aspect('equal')
ax.tick_params(top=True, right=True, which='both')

# Layout adjustment (note we're making room for the colorbar here!)
fig.tight_layout(pad=0.2, rect=(0, 0, 0.91, 1))

# Colorbar
cax = fig.add_subplot(111)
ax_pos = ax.get_position()
cax.set_position([ax_pos.xmax + 0.03, ax_pos.ymin, 0.01, ax_pos.height])
fig.colorbar(
    im, cax=cax, ticks=(im.norm.vmin, 0, im.norm.vmax), label='Normalized pressure'
)

fig.show()

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

# fig.savefig(Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve() / f'simulation_results_{SHOT.lower()}.png', dpi=400)
