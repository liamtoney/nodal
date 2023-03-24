import json

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from infresnel import calculate_paths
from matplotlib.patches import Rectangle
from pyproj import CRS, Geod, Transformer

from utils import NODAL_WORKING_DIR, get_shots, get_stations

M_PER_KM = 1000  # [m/km]

# KEY: Select which transect we are running! Currently only shot Y5 and X5 supported
TRANSECT = 'X5'

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

# Find (latitude, longitude) of buffered / extended terrain
EXTEND = 500  # [m] Distance of shot from domain boundary
BUFFER = 1000  # [m] Additional amount of topography to include outside of domain
g = Geod(ellps='WGS84')
az_end_to_start, az_start_to_end, _ = g.inv(*profile_end[::-1], *profile_start[::-1])
profile_start = g.fwd(*profile_start[::-1], az_end_to_start, EXTEND + BUFFER)[:2][::-1]
profile_end = g.fwd(*profile_end[::-1], az_start_to_end, BUFFER)[:2][::-1]

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

# Compute closest distance to profile and distance along profile for each station [m]
# (see http://jpfoss.blogspot.com/2012/04/distance-between-point-and-line.html)

# Find m and b in y = mx + b for profile
m = (profile.y[-1] - profile.y[0]) / (profile.x[-1] - profile.x[0])
b = profile.y[0] - m * profile.x[0]

# Iterate through each station, finding distances
out_of_plane_dists = []
along_profile_dists = []
for p, q in zip(sta_x, sta_y):
    # Find intersection point (x, y)
    x, y = (p + q * m - b * m) / (m**2 + 1), (b + m * (p + q * m)) / (m**2 + 1)
    # Use distance formula to find distance, h, between (p, q) and intersection point (x, y)
    h = np.sqrt((b + m * p - q) ** 2 / (m**2 + 1))  # Unsigned distance!
    # Find distance from start of profile to intersection point (x, y)
    d = np.linalg.norm(np.array([x, y]) - np.array([profile.x[0], profile.y[0]]))
    # Append these measurements
    out_of_plane_dists.append(h)
    along_profile_dists.append(d)
out_of_plane_dists = np.array(out_of_plane_dists)
along_profile_dists = np.array(along_profile_dists)

# [m] Outside this distance from the profile, we discard stations (if None, then don't
# discard any stations)
MASK_DIST = 500

if not MASK_DIST:
    MASK_DIST = out_of_plane_dists.max()
outside = out_of_plane_dists > MASK_DIST

# Plot map view
fig, ax = plt.subplots(figsize=(8, 7))
dem.plot.imshow(ax=ax, cmap=cc.m_gray)
ax.plot([profile.x[0], profile.x[-1]], [profile.y[0], profile.y[-1]], color='tab:blue')
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
    vmin=0,
    vmax=MASK_DIST,
    cmap=cc.m_fire_r,
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
fig.colorbar(sm, label='Distance from profile (m)', orientation='horizontal')
fig.tight_layout()
fig.show()

# Plot profile view
fig, ax = plt.subplots(figsize=(20, 2.5))
profile.plot(x='distance', ax=ax, color='tab:blue')
# Plot shot location
ax.scatter(
    EXTEND + BUFFER,
    profile.sel(distance=EXTEND + BUFFER, method='nearest'),
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
    vmin=0,
    vmax=MASK_DIST,
    cmap=cc.m_fire_r,
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
fig.colorbar(sm, label='Distance from profile (m)')
fig.tight_layout()
fig.show()

# Write file with information about "inside" stations
sta_info = dict(
    zip(
        sta_code[~outside],
        [
            [apd, oopd]
            for apd, oopd in zip(
                along_profile_dists[~outside], out_of_plane_dists[~outside]
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

Z_BUFFER = 100  # [m]

x = profile.distance.values
z = (profile - profile.min()).values + Z_BUFFER

if TRANSECT == 'Y5':
    dat_filename = 'imush_y5_buffer.dat'
elif TRANSECT == 'X5':
    dat_filename = 'imush_x5_buffer.dat'
else:
    raise ValueError()
dat_file = NODAL_WORKING_DIR / 'fdprop' / 'Acoustic_2D' / dat_filename
np.savetxt(dat_file, np.transpose([x, z]), fmt='%.2f')
print(f'Wrote {dat_file}')

#%% Plot what the FDTD simulation is "seeing" in the domain coordinate system

# TODO: These all must be copied from main.cpp
x_bnds_g = (1000, 26000)  # [m]
z_bnds_g = (0, 6000)  # [m]
x_src = 1500  # [m]
z_src = 464  # [m] X5
# z_src = 734  # [m] Y5

# This plot is using the domain coordinate system!
fig, ax = plt.subplots(figsize=(15, 5))
ax.add_patch(  # Domain
    Rectangle(
        (x_bnds_g[0], z_bnds_g[0]),
        np.diff(x_bnds_g)[0],
        np.diff(z_bnds_g)[0],
        color='lightgray',
        lw=0,
        zorder=-1,
    )
)
minor_int = 500  # [m]
ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_int))
ax.yaxis.set_minor_locator(plt.MultipleLocator(minor_int))
ax.grid(which='both', linestyle=':', color='gray', zorder=0)
ax.plot(x, z, color='tab:brown', zorder=3)  # Terrain
ax.scatter(x_src, z_src, color='tab:orange', zorder=4)  # Source
# for issue_coords in (26059, 413), (941, 703):  # Issue areas
#     ax.axvline(issue_coords[0], color='red', zorder=5)
#     ax.scatter(*issue_coords, edgecolor='red', facecolor='none', zorder=5)
ax.set_aspect('equal')
ax.set_xlabel('$x$ (m)')
ax.set_ylabel('$z$ (m)')
fig.tight_layout()
fig.show()
