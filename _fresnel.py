#%% Get DEM and project to UTM; define functions

import matplotlib.pyplot as plt
import numpy as np
import utm
import xarray as xr
from pygmt.datasets import load_earth_relief
from pyproj import CRS

from utils import get_shots, get_stations

M_PER_KM = 1000  # [m/km]

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

    # Make profile and add to list
    profile = dem_utm.interp(
        x=xr.DataArray(np.linspace(shot_x, sta_x, n)),
        y=xr.DataArray(np.linspace(shot_y, sta_y, n)),
        method='linear',
    )
    profiles.append(profile)

    # Print progress
    counter += 1
    print('{:.1f}%'.format((counter / total_its) * 100), end='\r')

print('Done')

#%% Plot all profiles + paths

fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=3, sharex=True, sharey=True, figsize=(16.5, 7.5)
)

# Iterate over all profiles, calculating diffracted path and plotting profiles + paths
for profile in profiles:

    # Get horizontal distance; ensure numpy.ndarray type for function input
    x = _horiz_dist(profile)
    y = profile.values

    # Compute DIRECT path
    direct_path = (y[-1] - y[0]) / (x[-1] - x[0]) * (x - x[0]) + y[0]
    direct_path_len = np.linalg.norm([np.diff(x), np.diff(direct_path)], axis=0).sum()

    # Compute SHORTEST DIFFRACTED path
    diff_path = diffracted_path(x, y)
    diff_path_len = np.linalg.norm([np.diff(x), np.diff(diff_path)], axis=0).sum()

    # Plot profiles + paths, converting horizontal distance to km
    common_kwargs = dict(linewidth=0.5, solid_capstyle='round')
    ax1.plot(x / M_PER_KM, y, **common_kwargs)
    ax2.plot(x / M_PER_KM, direct_path, **common_kwargs)
    ax3.plot(x / M_PER_KM, diff_path, **common_kwargs)

# Plot and label shot on each panel
shot_coords = (x[0] / M_PER_KM, y[0])  # Using coords of last profile here (all same)
for ax in ax1, ax2, ax3:
    ax.scatter(
        *shot_coords, s=250, marker='s', color='black', zorder=5, edgecolor='white'
    )
    ax.text(*shot_coords, SHOT, color='white', va='center', ha='center', zorder=6)

# Label panels
loc = (0.993, 0.9)
ax1.text(*loc, 'Elevation profiles', ha='right', transform=ax1.transAxes)
ax2.text(*loc, 'Direct paths', ha='right', transform=ax2.transAxes)
ax3.text(*loc, 'Shortest diffracted paths', ha='right', transform=ax3.transAxes)

# Final axis tweaks
if False:
    for ax in ax1, ax2, ax3:
        ax.set_aspect(1 / M_PER_KM)
ax2.set_ylabel('Elevation (m)')  # Since this is the middle panel
ax3.set_xlabel('Horizontal distance (km)')

fig.tight_layout()
fig.show()
