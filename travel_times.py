"""
Travel time analysis for observed/infresnel-modeled/FDTD-modeled results...
"""

import matplotlib.pyplot as plt
import pandas as pd
from obspy import UTCDateTime

from utils import NODAL_WORKING_DIR, get_shots

SHOT = 'Y5'  # Shot to analyze

# Get shot data
shot = get_shots().loc[SHOT]

# Get measurements for this shot
df = pd.read_csv(NODAL_WORKING_DIR / 'shot_gather_measurements' / f'{shot.name}.csv')


# Define helper function to return a reasonable static sound speed (as estimated from
# Fig. 1b) for an input shot (ROUGH ESTIMATE!)
def celerity_estimate(shot):
    if shot.time < UTCDateTime(2014, 7, 26):
        return 336  # [m/s]
    else:
        return 342  # [m/s]


#%% infresnel stuff

# Compute time delays owing to propagation over (rather than thru) topography
celerity = celerity_estimate(shot)
delays = df.path_length_diff_m / celerity  # [s]

# Shared plotting stuff
shot_kw = dict(color='lightgray', marker='s', s=50, ec='black', zorder=3)
title = f'Shot {shot.name}, {celerity} m/s'
vmin = 0
vmax = delays.max()
cmap = 'cividis'
delay_label = '$infresnel\,$-sourced travel time delay (s)'

# Plot delay vs. distance
fig, ax = plt.subplots(figsize=(12, 4))
ax.scatter(
    df.direct_path_length,
    delays,
    c=delays,
    clip_on=False,
    vmin=vmin,
    vmax=vmax,
    cmap=cmap,
    zorder=3,
)
ax.grid(axis='y', linestyle=':', color='gray', alpha=0.5)
ax.scatter(0, 0, clip_on=False, **shot_kw)
ax.set_xlim(left=0)
ax.set_ylim(vmin, vmax)
ax.set_xlabel('3D slant distance from shot (m)')
ax.set_ylabel(delay_label)
ax.set_title(title, loc='left', weight='bold')
for side in 'top', 'right':
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.show()

# Plot map view
fig, ax = plt.subplots()
sm = ax.scatter(df.lon, df.lat, c=delays, vmin=vmin, vmax=vmax, cmap=cmap)
ax.scatter(shot.lon, shot.lat, **shot_kw)
fig.colorbar(sm, label=delay_label)
ax.set_title(title, weight='bold')
fig.tight_layout()
fig.show()
