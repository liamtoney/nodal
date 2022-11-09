import pandas as pd

from utils import MASK_DISTANCE_KM, NODAL_WORKING_DIR, station_map

SHOT = 'Y5'

# Read in all the measurements
df = pd.read_csv(NODAL_WORKING_DIR / 'shot_gather_measurements' / f'{SHOT}.csv')

# Plot
fig = station_map(
    df.lon,
    df.lat,
    df.sta_lta_amp,
    sta_dists=df.dist_m,
    cbar_label=f'Shot {SHOT} STA/LTA amplitude',
    plot_inset=True,
    cbar_tick_ints='a1f0.5',
    vmin=1,
    vmax=9,
    mask_distance=MASK_DISTANCE_KM,
)

# fig.savefig(NODAL_WORKING_DIR / 'figures' / 'imush_station_map_with_y5.png', dpi=600)
