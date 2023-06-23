#!/usr/bin/env python

from infresnel import calculate_paths_grid

from utils import get_shots
from utils.utils import M_PER_KM

SHOT = 'Y5'
RADIUS = 25 * M_PER_KM  # [m]
SPACING = 100  # [m]

shot = get_shots().loc[SHOT]

path_length_differences, dem = calculate_paths_grid(
    src_lat=shot.lat, src_lon=shot.lon, radius=RADIUS, spacing=SPACING
)

# %% Plot

import colorcet as cc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS, Transformer

from utils import get_stations
from utils.utils import INNER_RING_REGION

net = get_stations()[0]

if True:
    pld_crop = path_length_differences.rio.clip_box(
        minx=INNER_RING_REGION[0],
        miny=INNER_RING_REGION[2],
        maxx=INNER_RING_REGION[1],
        maxy=INNER_RING_REGION[3],
        crs='EPSG:4326',
    )
else:
    # Get from interactive figure
    xlim = (552697.4404622609, 565240.2832041964)
    ylim = (5123241.659916841, 5131127.24459426)
    pld_crop = path_length_differences.rio.clip_box(
        minx=xlim[0], miny=ylim[0], maxx=xlim[1], maxy=ylim[1]
    )

# Define projection
utm_crs = CRS(pld_crop.rio.crs)
proj = Transformer.from_crs(utm_crs.geodetic_crs, utm_crs)

# Convert from pixel registration back to gridline registration for plotting
xvec, yvec = pld_crop.x, pld_crop.y
spacing = pld_crop.spacing
xvec_plot = np.hstack([xvec, xvec[-1] + spacing]) - spacing / 2
yvec_plot = np.hstack([yvec, yvec[-1] - spacing]) + spacing / 2  # Note sign change!

# Plot
fig, ax = plt.subplots()
hs = dem.copy()
hs.data = matplotlib.colors.LightSource().hillshade(
    dem.data,
    dx=abs(dem.x.diff('x').mean().values),
    dy=abs(dem.y.diff('y').mean().values),
)
im = hs.plot.imshow(ax=ax, cmap='Greys_r', add_colorbar=False, add_labels=False)
qm = ax.pcolormesh(
    xvec_plot,
    yvec_plot,
    pld_crop,
    cmap=cc.m_fire_r,
    shading='flat',
    alpha=0.6,
)
ax.scatter(*proj.transform(shot.lat, shot.lon), marker='*', s=80, color='black', lw=0)
ax.scatter(
    *proj.transform([sta.latitude for sta in net], [sta.longitude for sta in net]),
    s=3,
    color='black',
    lw=0,
)
ax.set_aspect('equal')
ax.set_xlim(xvec_plot.min(), xvec_plot.max())
ax.set_ylim(yvec_plot.min(), yvec_plot.max())
ax.tick_params(top=True, right=True, which='both')
ax.set_xlabel(f'UTM zone {utm_crs.utm_zone} easting (m)')
ax.set_ylabel(f'UTM zone {utm_crs.utm_zone} northing (m)')
ax.ticklabel_format(style='plain')
fig.colorbar(qm, label='Path length difference (m)')
fig.autofmt_xdate()
plt.show()

# fig.savefig('grid.png', bbox_inches='tight', dpi=300)
