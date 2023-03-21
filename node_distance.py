import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from infresnel._georeference import _estimate_utm_crs
from pyproj import Transformer
from scipy.spatial.distance import cdist

from utils import get_stations

# Get node coordinates
net = get_stations()[0]
lats = [sta.latitude for sta in net]
lons = [sta.longitude for sta in net]

# Convert coordinates to UTM
utm_crs = _estimate_utm_crs(np.mean(lats), np.mean(lons))
proj = Transformer.from_crs(utm_crs.geodetic_crs, utm_crs)
x, y = proj.transform(lats, lons)

# Plot UTM coordinates
fig1, ax1 = plt.subplots()
msz = 10
ax1.scatter(x, y, color='lightgray', s=msz)
ax1.set_aspect('equal')
ax1.ticklabel_format(style='plain')
ax1.tick_params(top=True, right=True, which='both')
ax1.set_xlabel('UTM easting (m)')
ax1.set_ylabel('UTM northing (m)')
fig1.show()

# Compute distance matrix
pts = np.array([x, y]).T
distance = cdist(pts, pts)  # [m]

# Plot distance matrix
fig2, ax2 = plt.subplots()
mask = np.tri(distance.shape[0]).astype(bool)  # Makes the zero diagonal NaN!
distance[~mask] = np.nan
im = ax2.imshow(distance, cmap=cc.m_fire, vmin=0, vmax=1000)  # Clipping here
for side in 'top', 'right':
    ax2.spines[side].set_visible(False)
ax2.set_xlabel('Node $j$')
ax2.set_ylabel('Node $i$')
fig2.colorbar(im, label='Distance between nodes $i$ and $j$ (m)')
fig2.show()

# Find smallest distance node pair and plot on map
distance_nonzero = distance.copy()
distance_nonzero[distance_nonzero == 0] = np.nan
inds_min = np.where(distance_nonzero == np.nanmin(distance_nonzero))
for i, ind_min in enumerate(inds_min):
    ax1.scatter(
        x[ind_min[0]],
        y[ind_min[0]],
        color='red',
        s=msz,
        label=f'{distance_nonzero[inds_min][0]:.1f} m' if i else None,  # Fun hack ;)
    )
ax1.legend(frameon=False)
fig1.show()
