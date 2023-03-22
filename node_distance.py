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
codes = [sta.code for sta in net]

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

# Plot a few of the most closely-spaced node pairs on the previously-created map
distance_sorted = distance.copy()
distance_sorted[distance_sorted == 0] = np.nan  # Nodes can't pair with themselves!
distance_sorted = distance_sorted.flatten()
distance_sorted = distance_sorted[~np.isnan(distance_sorted)]
distance_sorted.sort()  # Size is "898 choose 2"
for distance_to_plot in distance_sorted[:6]:
    inds = np.where(distance == distance_to_plot)
    ax1.plot(
        [x[inds[0][0]], x[inds[1][0]]],
        [y[inds[0][0]], y[inds[1][0]]],
        marker='o',
        markersize=np.sqrt(msz),
        label=f'{distance[inds][0]:.1f} m ({codes[inds[1][0]]}–{codes[inds[0][0]]})',
    )
fig1.subplots_adjust(top=0.8)
fig1.legend(frameon=False, loc='upper center', ncol=2, numpoints=2)
fig1.show()
