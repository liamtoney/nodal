import pandas as pd

from utils import NODAL_WORKING_DIR, station_map

SHOT = 'Y5'  # Shot to plot

# ----------------------------
# PLOTTING PARAMETERS TO TWEAK
# ----------------------------
SAVE = False  # Toggle saving PNG files
DPI = 600  # PNG DPI
# ----------------------------

# Read in all the measurements
df = pd.read_csv(NODAL_WORKING_DIR / 'shot_gather_measurements' / f'{SHOT}.csv')

#%% Plot STA/LTA amplitudes

fig = station_map(
    df.lon,
    df.lat,
    df.sta_lta_amp,
    cbar_label='STA/LTA amplitude',
    plot_shot=SHOT,
)
if SAVE:
    fig.savefig(
        NODAL_WORKING_DIR / 'figures' / 'amplitude_maps' / f'shot_{SHOT}.png', dpi=DPI
    )

#%% Plot path differences

fig = station_map(
    df.lon,
    df.lat,
    df.path_length_diff_m,
    cbar_label='Difference between shortest diffracted path and direct path (m)',
    plot_shot=SHOT,
    reverse_cmap=True,
    vmax=60,  # [m] Making this smaller highlights the differences better!
)
if SAVE:
    fig.savefig(
        NODAL_WORKING_DIR / 'figures' / 'path_diff_maps' / f'shot_{SHOT}.png', dpi=DPI
    )
