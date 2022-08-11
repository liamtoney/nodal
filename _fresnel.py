#%% Get DEM and project to UTM; define functions

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utm
import xarray as xr
from matplotlib.colors import Normalize
from pygmt.datasets import load_earth_relief
from pyproj import CRS

from utils import get_shots, get_stations

M_PER_KM = 1000  # [m/km] CONSTANT

MAIN_REGION = (-122.42, -121.98, 46.06, 46.36)  # From station map

dem = load_earth_relief(resolution='01s', region=MAIN_REGION, use_srtm=True)
dem.rio.write_crs(dem.horizontal_datum, inplace=True)
dem_utm = dem.rio.reproject(dem.rio.estimate_utm_crs())


# Helper function to calculate horizontal difference vector for a profile DataArray
def _horiz_dist(profile):
    return np.hstack(
        [0, np.cumsum(np.linalg.norm([np.diff(profile.x), np.diff(profile.y)], axis=0))]
    )


# Function for computing shortest diffracted path
def diffracted_path(x, y):

    # Compute the direct path (subtract x[0] from x to make intercept work!)
    direct_path = (y[-1] - y[0]) / (x[-1] - x[0]) * (x - x[0]) + y[0]

    # If y is everywhere "on" or "underneath" the direct path, we're done (first
    # we mask values that are ~equal; then we check for "less than")
    isclose = np.isclose(y, direct_path)
    if (y[~isclose] < direct_path[~isclose]).all():
        return direct_path

    # Location of maximum of profile (detrended using direct_path)
    max_ind = np.argmax(y - direct_path)

    # Split profile into two pieces here (including common midpoint in both)
    left = slice(None, max_ind + 1)
    right = slice(max_ind, None)

    # Recursively call this function
    path_left = diffracted_path(x[left], y[left])
    path_right = diffracted_path(x[right], y[right])

    # Join at common midpoint, removing duplicate
    return np.hstack([path_left[:-1], path_right])


#%% Make profiles through DEM for a specified shot -> all nodes

SHOT = 'Y5'

# [m] Target horizontal spacing for profile (determines # points) - does not seem to
# slow down code much if this is decreased
TARGET_SPACING = 10

# Get UTM coords for shot
df = get_shots()
utm_zone_number = int(CRS(dem_utm.rio.crs).utm_zone[:-1])  # Ensure same UTM zone; hacky
shot_x, shot_y = utm.from_latlon(
    df.loc[SHOT].lat, df.loc[SHOT].lon, force_zone_number=utm_zone_number
)[:2]

# Get UTM coords for all nodes
sta_x_list, sta_y_list = [], []
inv = get_stations()
for sta in inv[0]:
    sta_x, sta_y = utm.from_latlon(
        sta.latitude, sta.longitude, force_zone_number=utm_zone_number
    )[:2]
    sta_x_list.append(sta_x)
    sta_y_list.append(sta_y)

# Iterate over all nodes, grabbing profile
profiles = []
total_its = len(sta_x_list)
counter = 0
for sta_x, sta_y in zip(sta_x_list, sta_y_list):

    # Determine # of points in profile
    dist = np.linalg.norm([shot_x - sta_x, shot_y - sta_y])
    n = int(np.ceil(dist / TARGET_SPACING))

    # Make profile, clean up, and add to list
    profile = dem_utm.interp(
        x=xr.DataArray(np.linspace(shot_x, sta_x, n)),
        y=xr.DataArray(np.linspace(shot_y, sta_y, n)),
        method='linear',
    )
    profile = profile.assign_coords(dim_0=_horiz_dist(profile))
    profile = profile.rename(dim_0='distance').drop('spatial_ref')
    profile.attrs = {}

    profiles.append(profile)

    # Print progress
    counter += 1
    print('{:.1f}%'.format((counter / total_its) * 100), end='\r')

print('Done')

#%% Calculate and plot all profiles + paths

CMAP = 'inferno_r'  # Colormap for path length differences
EQUAL_ASPECT = False  # Toggle for equal aspect ratio

# Iterate over all profiles, calculating paths
ds_list = []
for profile in profiles:

    # Ensure numpy.ndarray type for function input
    x = profile.distance.values
    y = profile.values

    # Compute DIRECT path
    direct_path = (y[-1] - y[0]) / (x[-1] - x[0]) * (x - x[0]) + y[0]
    direct_path_len = np.linalg.norm([np.diff(x), np.diff(direct_path)], axis=0).sum()

    # Compute SHORTEST DIFFRACTED path
    diff_path = diffracted_path(x, y)
    diff_path_len = np.linalg.norm([np.diff(x), np.diff(diff_path)], axis=0).sum()

    # Make nice Dataset of all info
    ds = xr.Dataset(
        {
            profile.name: profile,
            'direct_path': ('distance', direct_path, dict(length=direct_path_len)),
            'diffracted_path': ('distance', diff_path, dict(length=diff_path_len)),
        },
        attrs=dict(path_length_difference=diff_path_len - direct_path_len, units='m'),
    )
    ds_list.append(ds)

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
df.to_csv(
    Path(os.environ['NODAL_WORKING_DIR'])
    / 'fresnel'
    / f'{SHOT.lower()}_path_differences.csv',
    index=False,
)
