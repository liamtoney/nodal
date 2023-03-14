import json

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from infresnel import calculate_paths
from pyproj import CRS, Geod, Transformer

from utils import NODAL_WORKING_DIR, get_shots, get_stations

M_PER_KM = 1000  # [m/km]

# KEY: Select which transect we are running! Currently only shot Y5 and X5 supported
TRANSECT = 'Y5'

# Use shot location as start of profile
if TRANSECT == 'Y5':
    shot = get_shots().loc['Y5']
elif TRANSECT == 'X5':
    shot = get_shots().loc['X5']
else:
    raise ValueError()

# (latitude, longitude) coordinates of profile endpoints
profile_start = (shot.lat, shot.lon)
if TRANSECT == 'Y5':
    profile_end = (46.224, -122.031)
elif TRANSECT == 'X5':
    profile_end = (46.138, -122.297)
else:
    raise ValueError()

# Find (latitude, longitude) of extended line (this is to pad the domain!)
EXTEND = 500  # [m]
g = Geod(ellps='WGS84')
az_end_to_start = g.inv(*profile_end[::-1], *profile_start[::-1])[0]
profile_start = g.fwd(*profile_start[::-1], az_end_to_start, EXTEND)[:2][::-1]

# Get elevation profile
ds_list, dem = calculate_paths(
    src_lat=profile_start[0],
    src_lon=profile_start[1],
    rec_lat=profile_end[0],
    rec_lon=profile_end[1],
    dem_file=None,  # Use 30-m SRTM data
    full_output=True,
)
profile = ds_list[0].elevation

# Convert station coordinates to UTM
dem_crs = CRS(dem.rio.crs)
proj = Transformer.from_crs(dem_crs.geodetic_crs, dem_crs)
sta_x, sta_y, sta_elev, sta_code = [], [], [], []
for sta in get_stations()[0]:
    x, y = proj.transform(sta.latitude, sta.longitude)
    sta_x.append(x)
    sta_y.append(y)
    # sta_elev.append(sta.elevation)  # Use metadata elevation (ISSUES!)
    sta_elev.append(dem.sel(x=x, y=y, method='nearest').values)  # Use DEM elevation
    sta_code.append(sta.code)
sta_x = np.array(sta_x)
sta_y = np.array(sta_y)
sta_elev = np.array(sta_elev)
sta_code = np.array(sta_code)

# Find m and b in y = mx + b for profile
m = (profile.y[-1] - profile.y[0]) / (profile.x[-1] - profile.x[0])
b = profile.y[0] - m * profile.x[0]

# Compute closest distance to profile and distance along profile for each station [m]
out_of_plane_dists = []
along_profile_dists = []
for x, y in zip(sta_x, sta_y):
    u2, v2 = x, (m * x) + b
    u3, v3 = (y - b) / m, y
    theta = np.arctan((v3 - v2) / (u2 - u3))
    d = np.sin(theta) * (u2 - u3)
    delta = np.sin(theta) * (v3 - v2)
    dd = np.linalg.norm(np.array([u2, v2]) - np.array([profile.x[0], profile.y[0]]))
    out_of_plane_dists.append(d)
    along_profile_dists.append(dd - delta)
out_of_plane_dists = np.array(out_of_plane_dists)
along_profile_dists = np.array(along_profile_dists)

# [m] Outside this distance from the profile, we discard stations (if None, then don't
# discard any stations)
MASK_DIST = 500

if not MASK_DIST:
    MASK_DIST = np.abs(out_of_plane_dists).max()
outside = np.abs(out_of_plane_dists) > MASK_DIST

# Plot map view
fig, ax = plt.subplots(figsize=(8, 7))
dem.plot.imshow(ax=ax, cmap=cc.m_gray)
ax.plot(
    [profile.x[0], profile.x[-1]], [profile.y[0], profile.y[-1]], color='tab:orange'
)
# Plot shot location
ax.scatter(*proj.transform(shot.lat, shot.lon), marker='*', color='black', zorder=10)
xlim, ylim = ax.get_xlim(), ax.get_ylim()
sm = ax.scatter(
    sta_x[~outside],
    sta_y[~outside],
    s=20,
    c=out_of_plane_dists[~outside],
    edgecolor='black',
    linewidths=0.5,
    vmin=-MASK_DIST,
    vmax=MASK_DIST,
    cmap=cc.m_CET_D13,
    zorder=10,
)
ax.scatter(
    sta_x[outside],
    sta_y[outside],
    s=20,
    facecolor='none',
    edgecolor='black',
    linewidths=0.5,
    alpha=0.3,
)
ax.ticklabel_format(style='plain')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
cbar = fig.colorbar(sm, label='Distance from profile (m)', orientation='horizontal')
cbar.ax.set_xticklabels([f'{x:g}' for x in np.abs(cbar.ax.get_xticks())])
fig.tight_layout()
fig.show()

# Plot profile view
fig, ax = plt.subplots(figsize=(20, 2.5))
profile.plot(x='distance', ax=ax, color='tab:orange')
# Plot shot location
ax.scatter(
    EXTEND,
    profile.sel(distance=EXTEND, method='nearest'),
    marker='*',
    color='black',
    zorder=10,
)
sm = ax.scatter(
    along_profile_dists[~outside],
    sta_elev[~outside],
    s=20,
    c=out_of_plane_dists[~outside],
    edgecolor='black',
    linewidths=0.5,
    vmin=-MASK_DIST,
    vmax=MASK_DIST,
    cmap=cc.m_CET_D13,
    zorder=10,
)
ax.scatter(
    along_profile_dists[outside],
    sta_elev[outside],
    s=20,
    facecolor='none',
    edgecolor='black',
    linewidths=0.5,
    alpha=0.3,
)
ax.set_aspect('equal')
cbar = fig.colorbar(sm, label='Distance from profile (m)')
cbar.ax.set_yticklabels([f'{y:g}' for y in np.abs(cbar.ax.get_yticks())])
fig.tight_layout()
fig.show()

# Write file with information about "inside" stations
sta_info = dict(
    zip(
        sta_code[~outside],
        [
            [apd, oopd]
            for apd, oopd in zip(
                along_profile_dists[~outside], np.abs(out_of_plane_dists[~outside])
            )
        ],
    )
)
if TRANSECT == 'Y5':
    json_filename = 'imush_y5_transect_stations.json'
elif TRANSECT == 'X5':
    json_filename = 'imush_x5_transect_stations.json'
else:
    raise ValueError()
with open(NODAL_WORKING_DIR / 'metadata' / json_filename, 'w') as f:
    json.dump(sta_info, f, indent='\t')

# Print info about the profile
print(f'\nx-extent: {profile.distance[-1] / M_PER_KM:.1f} km')
print(f'Profile minimum: {profile.min() / M_PER_KM:.4f} km')

#%% Write .dat file

x = profile.distance.values
z = (profile - profile.min()).values

if TRANSECT == 'Y5':
    dat_filename = 'imush_y5.dat'
elif TRANSECT == 'X5':
    dat_filename = 'imush_x5.dat'
else:
    raise ValueError()
dat_file = NODAL_WORKING_DIR / 'fdprop' / 'Acoustic_2D' / dat_filename
np.savetxt(dat_file, np.transpose([x, z]), fmt='%.2f')
print(f'Wrote {dat_file}')
