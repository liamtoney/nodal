import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from infresnel import calculate_paths
from pyproj import CRS, Transformer

from utils import get_shots, get_stations

# Use shot location as start of profile
shot = get_shots().loc['Y5']

# (latitude, longitude) coordinates of profile endpoints
profile_start = (shot.lat, shot.lon)
profile_end = (46.224, -122.031)
profile_end = (46.122, -122.032)

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
sta_x, sta_y, sta_elev = [], [], []
for sta in get_stations()[0]:
    x, y = proj.transform(sta.latitude, sta.longitude)
    sta_x.append(x)
    sta_y.append(y)
    sta_elev.append(sta.elevation)

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
dist_lim = np.min([np.abs(np.min(out_of_plane_dists)), np.max(out_of_plane_dists)])

# Plot map view
fig, ax = plt.subplots(figsize=(8, 7))
dem.plot.imshow(ax=ax, cmap=cc.m_gray)
ax.plot(
    [profile.x[0], profile.x[-1]], [profile.y[0], profile.y[-1]], color='tab:orange'
)
xlim, ylim = ax.get_xlim(), ax.get_ylim()
sm = ax.scatter(
    sta_x,
    sta_y,
    s=20,
    c=out_of_plane_dists,
    edgecolor='black',
    linewidths=0.5,
    vmin=-dist_lim,
    vmax=dist_lim,
    cmap=cc.m_CET_D13,
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
sm = ax.scatter(
    along_profile_dists,
    sta_elev,
    s=20,
    c=out_of_plane_dists,
    edgecolor='black',
    linewidths=0.5,
    vmin=-dist_lim,
    vmax=dist_lim,
    cmap=cc.m_CET_D13,
)
ax.set_aspect('equal')
cbar = fig.colorbar(sm, label='Distance from profile (m)')
cbar.ax.set_yticklabels([f'{y:g}' for y in np.abs(cbar.ax.get_yticks())])
fig.tight_layout()
fig.show()

# Print info about the profile
print(f'\nx-extent: {profile.distance[-1] / 1000:.1f} km')
