from pathlib import Path

import imageio
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import NODAL_WORKING_DIR

# Choose 'Y5' or 'X5'
SHOT = 'Y5'

# Some logic to load the remaining transect-specific params correctly
if SHOT == 'Y5':
    RUN = '25_shot_y5_pml_240'
    Z_SRC = 734  # [m]
elif SHOT == 'X5':
    RUN = '24_shot_x5_pml_240'
    Z_SRC = 464  # [m]
else:
    raise ValueError
PRESSURE_SNAPSHOT_DIR = NODAL_WORKING_DIR / 'fdprop' / 'nodal_fdprop_pressure_snapshots'
TIMESTAMPS = np.arange(0, 18000 + 1000, 1000)  # Anything after 1800 is outside!
XLIM = (0, 24)  # [km] Relative to shot x-position
YLIM = (-0.5, 5)  # [km] Relative to shot z-position
X_SRC = 1500  # [m]
DT = 0.004  # [s]

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


# Terrain from .dat file for this run
terrain_contour = np.loadtxt(
    NODAL_WORKING_DIR / 'fdprop' / 'Acoustic_2D' / f'imush_{SHOT.lower()}_buffer.dat'
)

# Loop over each snapshot, making and saving figure
image_dir = NODAL_WORKING_DIR / 'figures' / 'fdprop' / 'simulation_gifs'
image_dir.mkdir(parents=True, exist_ok=True)
for i, timestamp in enumerate(tqdm(TIMESTAMPS)):
    p, hoz_min, hoz_max, vert_min, vert_max, dx = _p_matrix_from_timestamp(
        RUN, timestamp
    )

    fig, ax_im = plt.subplots(figsize=(7.17, 4))

    # Plot pressure
    extent = [
        (hoz_min - X_SRC) / M_PER_KM,
        (hoz_max - X_SRC) / M_PER_KM,
        (vert_min - Z_SRC) / M_PER_KM,
        (vert_max - Z_SRC) / M_PER_KM,
    ]
    im = ax_im.imshow(p, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, extent=extent)

    # Plot terrain
    ax_im.fill_between(
        (terrain_contour[:, 0] - X_SRC) / M_PER_KM,
        -1,
        (terrain_contour[:, 1] - Z_SRC) / M_PER_KM,
        lw=0.5,  # Makes pressure–terrain interface a little smoother-looking
        color='silver',
    )

    # Timestamp labels (note: we don't bother correcting for t0 here since it's so small —
    # in reality the true times are `timestamp * dt + t0`)
    text = ax_im.text(
        0.99,
        0.95,
        f'{timestamp * DT:g} s',
        ha='right',
        va='top',
        transform=ax_im.transAxes,
    )
    text.set_path_effects(
        [path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()]
    )

    # Axis params
    ax_im.set_xlabel(f'Distance from shot {SHOT} (km)')
    ax_im.set_ylabel(f'Elevation relative\nto shot {SHOT} (km)')
    ax_im.set_xlim(XLIM)
    ax_im.set_ylim(YLIM)
    major_tick_interval = 2  # [km]
    minor_tick_interval = 1  # [km[
    ax_im.xaxis.set_major_locator(plt.MultipleLocator(major_tick_interval))
    ax_im.xaxis.set_minor_locator(plt.MultipleLocator(minor_tick_interval))
    ax_im.yaxis.set_major_locator(plt.MultipleLocator(major_tick_interval))
    ax_im.yaxis.set_minor_locator(plt.MultipleLocator(minor_tick_interval))
    ax_im.set_aspect('equal')
    ax_im.tick_params(top=True, right=True, which='both')

    # Layout adjustment (note we're making room for the colorbar here!)
    fig.tight_layout(pad=0.2, rect=(0.07, 0, 0.84, 1))

    # Colorbar
    cax = fig.add_subplot(111)
    ax_im_pos = ax_im.get_position()
    cax.set_position([ax_im_pos.xmax + 0.03, ax_im_pos.ymin, 0.01, ax_im_pos.height])
    fig.colorbar(
        im, cax=cax, ticks=(im.norm.vmin, 0, im.norm.vmax), label='Normalized pressure'
    )

    # Save
    fig.savefig(
        image_dir / f'{i:02}.png', dpi=400, bbox_inches='tight', pad_inches=0.05
    )
    plt.close(fig)

# %% Convert PNG files to GIF

imageio.mimwrite(
    image_dir / f'{RUN}.gif',
    [imageio.imread(file) for file in sorted(image_dir.glob('*.png'))],
    duration=0.2,  # Duration [s] of each frame, larger values for "slower" GIF
    subrectangles=True,  # Works VERY well for saving space in this use case!
)
