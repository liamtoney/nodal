import matplotlib.pyplot as plt
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

# Plot map view
fig, ax = plt.subplots()
dem.plot.imshow(ax=ax)
ax.plot(
    [profile.x[0], profile.x[-1]], [profile.y[0], profile.y[-1]], color='tab:orange'
)
xlim, ylim = ax.get_xlim(), ax.get_ylim()
dem_crs = CRS(dem.rio.crs)
proj = Transformer.from_crs(dem_crs.geodetic_crs, dem_crs)
for sta in get_stations()[0]:
    ax.scatter(
        *proj.transform(sta.latitude, sta.longitude),
        s=10,
        color='white',
        edgecolor='black',
    )
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
fig.show()

# Plot profile view
fig, ax = plt.subplots(figsize=(20, 2.5))
profile.plot(x='distance', ax=ax, color='tab:orange')
ax.set_aspect('equal')
fig.show()

# Print info about the profile
print(f'\nx-extent: {profile.distance[-1] / 1000:.1f} km')
