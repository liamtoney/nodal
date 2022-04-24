import os
from pathlib import Path

import numpy as np
import pygmt

from utils import get_shots, get_stations

# Set PyGMT defaults
pygmt.config(MAP_FRAME_TYPE='plain', FORMAT_GEO_MAP='D', FONT='10p')

# Read in info for plotting
df = get_shots()
net = get_stations()[0]

# Create station info arrays
lons = [sta.longitude for sta in net]
lats = [sta.latitude for sta in net]
codes = [int(sta.code) for sta in net]

# Nice regions
MAIN_REGION = [-122.42, -121.98, 46.06, 46.36]
INSET_REGION = [-123.1, -121.3, 45.6, 46.8]

# Plot
fig = pygmt.Figure()
shaded_relief = pygmt.grdgradient(
    '@earth_relief_01s', region=MAIN_REGION, azimuth=-45.0, normalize='t1+a0'
)
pygmt.makecpt(cmap='gray', series=[-2, shaded_relief.values.max()])
fig.grdimage(
    shaded_relief, cmap=True, projection='M4i', region=MAIN_REGION, frame=False
)
pygmt.makecpt(series=[np.min(codes), np.max(codes)], cmap='turbo', reverse=True)
fig.plot(x=lons, y=lats, color=codes, style='c0.05i', cmap=True, pen='black')  # Nodes
fig.plot(x=df.lon, y=df.lat, style='s0.2i', color='black', pen='white')  # Shots
fig.text(x=df.lon, y=df.lat, text=df.shot, font='6p,white', justify='CM')  # Shot labels
fig.basemap(map_scale='g-122.04/46.09+w5+f+l', frame=['WESN', 'a0.1f0.02'])
fig.colorbar(frame='a200f100+l"Station code"')
with fig.inset(position='JTR+w1.5i+o-0.5i/-1i', box='+gwhite+p1p'):
    fig.plot(
        x=lons,
        y=lats,
        color='black',
        style='c0.01i',
        region=INSET_REGION,
        projection='M?',
    )
    in_main_map = (
        (df.lon > MAIN_REGION[0])
        & (df.lon < MAIN_REGION[1])
        & (df.lat > MAIN_REGION[2])
        & (df.lat < MAIN_REGION[3])
    )
    kwargs = dict(style='s0.07i', pen='black')
    fig.plot(x=df[in_main_map].lon, y=df[in_main_map].lat, color='black', **kwargs)
    fig.plot(x=df[~in_main_map].lon, y=df[~in_main_map].lat, color='white', **kwargs)
    fig.basemap(map_scale='g-122.2/45.8+w50')

fig.show()

# fig.savefig(Path(os.environ['NODAL_WORKING_DIR']) / 'figures' / 'imush_station_map.png', dpi=600)
