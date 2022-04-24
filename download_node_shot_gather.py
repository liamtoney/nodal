#!/usr/bin/env python

import os
from pathlib import Path

import numpy as np
from matplotlib.dates import SEC_PER_MIN
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics.base import gps2dist_azimuth

from utils import get_shots

SHOT = 'Y5'  # Shot to gather data for

MIN_CELERITY = 280  # [m/s] For calculating propagation times
PRE_ROLL = 30  # [s] Duration of data to download prior to shot time

client = Client('IRIS')

# Read in shot info
df = get_shots()

# Grab "MSH Node Array" coordinates
# http://ds.iris.edu/mda/1D/?starttime=2014-01-01T00:00:00&endtime=2014-12-31T23:59:59
net = client.get_stations(
    network='1D', starttime=UTCDateTime(2014, 1, 1), endtime=UTCDateTime(2014, 12, 31)
)[0]
station_distances = []
for lat, lon in zip([sta.latitude for sta in net], [sta.longitude for sta in net]):
    dist_m = gps2dist_azimuth(lat, lon, df.loc[SHOT].lat, df.loc[SHOT].lon)[0]
    station_distances.append(dist_m)

# Calculate required data duration
max_prop_time = np.max(station_distances) / MIN_CELERITY  # [s]
max_prop_time_min_ceil = np.ceil(max_prop_time / SEC_PER_MIN) * SEC_PER_MIN

# Gather data
st = client.get_waveforms(
    network='1D',
    station='*',
    location='--',
    channel='DPZ',
    starttime=df.loc[SHOT].time - PRE_ROLL,
    endtime=df.loc[SHOT].time + max_prop_time_min_ceil,
)

# Save data
st.write(Path(os.environ['NODAL_WORKING_DIR']) / 'data' / f'{SHOT}.mseed')
