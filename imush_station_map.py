#!/usr/bin/env python

import os
from pathlib import Path

import numpy as np
import pygmt
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# Set PyGMT defaults
pygmt.config(MAP_FRAME_TYPE='plain', FORMAT_GEO_MAP='D', FONT='10p')

# Grab "MSH Node Array"
# http://ds.iris.edu/mda/1D/?starttime=2014-01-01T00:00:00&endtime=2014-12-31T23:59:59
net = Client('IRIS').get_stations(
    network='1D', starttime=UTCDateTime(2014, 1, 1), endtime=UTCDateTime(2014, 12, 31)
)[0]

# Create station info arrays
lons = [sta.longitude for sta in net]
lats = [sta.latitude for sta in net]
codes = [int(sta.code) for sta in net]

# A nice region
REGION = [-122.34, -122.02, 46.08, 46.32]

# Plot
fig = pygmt.Figure()
shaded_relief = pygmt.grdgradient(
    '@earth_relief_01s', region=REGION, azimuth=-45.0, normalize='t1+a0'
)
pygmt.makecpt(cmap='gray', series=[-2, shaded_relief.values.max()])
fig.grdimage(shaded_relief, cmap=True, projection='M4i', region=REGION, frame=False)
pygmt.makecpt(series=[np.min(codes), np.max(codes)], cmap='turbo', reverse=True)
fig.plot(x=lons, y=lats, color=codes, style='c0.05i', cmap=True, pen='black')
fig.basemap(map_scale='g-122.07/46.1+w5+f+l', frame=['WESN', 'a0.1f0.02'])
fig.colorbar(frame='a200f100+l"Station code"')
fig.show()

# fig.savefig(Path(os.environ['NODAL_WORKING_DIR']) / 'figures' / 'imush_station_map.png', dpi=600)
