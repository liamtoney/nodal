import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from infresnel import calculate_paths
from matplotlib.colors import Normalize

from utils import NODAL_WORKING_DIR, get_shots, get_stations

M_PER_KM = 1000  # [m/km] CONSTANT

SHOT = 'Y5'  # Shot to calculate paths for

# Get shot and station info
df = get_shots()
inv = get_stations()

# Calculate paths
ds_list, _ = calculate_paths(
    src_lat=df.loc[SHOT].lat,
    src_lon=df.loc[SHOT].lon,
    rec_lat=[sta.latitude for sta in inv[0]],
    rec_lon=[sta.longitude for sta in inv[0]],
    full_output=True,
)

#%% Plot all profiles + paths (optional)

CMAP = 'inferno_r'  # Colormap for path length differences
EQUAL_ASPECT = False  # Toggle for equal aspect ratio

# Set up axes
fig, axes = plt.subplot_mosaic(
    [['a', 'cax'], ['b', 'cax'], ['c', 'cax']],
    gridspec_kw=dict(width_ratios=(60, 1)),
    figsize=(16.5, 7.5),
)
axes['a'].sharex(axes['b'])
axes['b'].sharex(axes['c'])
axes['a'].sharey(axes['b'])
axes['b'].sharey(axes['c'])
axes['a'].tick_params(labelbottom=False)
axes['b'].tick_params(labelbottom=False)

# Grab path length differences to set up colormap
path_length_diffs = np.array([ds.path_length_difference for ds in ds_list])
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.get_cmap(CMAP),
    norm=Normalize(vmin=path_length_diffs.min(), vmax=path_length_diffs.max()),
)

# Plot all profiles + paths, sorted from smallest to largest path difference
for i in path_length_diffs.argsort():

    # Get the Dataset
    ds = ds_list[i]

    # Plot (onverting horizontal distance to km here)
    color = sm.to_rgba(ds.path_length_difference)
    common_kwargs = dict(linewidth=0.5, solid_capstyle='round')
    axes['a'].plot(ds.distance / M_PER_KM, ds.elevation, c=color, **common_kwargs)
    axes['b'].plot(ds.distance / M_PER_KM, ds.direct_path, c=color, **common_kwargs)
    axes['c'].plot(ds.distance / M_PER_KM, ds.diffracted_path, c=color, **common_kwargs)

# Plot and label shot on each panel; using coords of last profile here (all same)
shot_coords = (ds.distance[0] / M_PER_KM, ds.elevation[0])
for ax in axes['a'], axes['b'], axes['c']:
    ax.scatter(*shot_coords, s=250, marker='s', color='black', zorder=5, ec='white')
    ax.text(*shot_coords, SHOT, color='white', va='center', ha='center', zorder=6)

# Label panels
loc = (0.993, 0.9)
axes['a'].text(*loc, 'Elevation profiles', ha='right', transform=axes['a'].transAxes)
axes['b'].text(*loc, 'Direct paths', ha='right', transform=axes['b'].transAxes)
axes['c'].text(
    *loc, 'Shortest diffracted paths', ha='right', transform=axes['c'].transAxes
)

# Final axis tweaks
if EQUAL_ASPECT:
    for ax in axes['a'], axes['b'], axes['c']:
        ax.set_aspect(1 / M_PER_KM)
axes['b'].set_ylabel('Elevation (m)')  # Since this is the middle panel
axes['c'].set_xlabel('Horizontal distance (km)')

# Add overall colorbar
fig.colorbar(
    sm,
    cax=axes['cax'],
    label='Difference between shortest diffracted path and direct path (m)',
)

fig.tight_layout()
fig.show()

#%% Export path differences as CSV

data_dict = dict(
    station=[sta.code for sta in inv[0]],
    path_length_diff_m=[ds.path_length_difference for ds in ds_list],
)
df = pd.DataFrame(data=data_dict)
df.to_csv(NODAL_WORKING_DIR / 'fresnel' / f'{SHOT}_path_differences.csv', index=False)
