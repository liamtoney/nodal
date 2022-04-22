#!/usr/bin/env python

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.dates import SEC_PER_MIN
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics.base import gps2dist_azimuth

SHOT = 'Y5'  # Shot to gather data for

MIN_CELERITY = 280  # [m/s] For calculating propagation times
PRE_ROLL = 30  # [s] Duration of data to download prior to shot time

client = Client('IRIS')

# Read in shot info
df = pd.read_excel(Path(os.environ['NODAL_WORKING_DIR']) / 'iMUSH_shot_metadata.xlsx')
df.dropna(inplace=True)
df.set_index('Shot', inplace=True)  # So we can easily look up shots by name

# Get shot origin time
frac_sec, whole_sec = math.modf(df.loc[SHOT].Sec)
shot_time = UTCDateTime(
    year=2014,
    julday=int(df.loc[SHOT].Julian),
    hour=int(df.loc[SHOT]['UTC Hour']),
    minute=int(df.loc[SHOT].Min),
    second=int(whole_sec),
    microsecond=int(frac_sec * 1e6),
)

# Grab "MSH Node Array" coordinates
# http://ds.iris.edu/mda/1D/?starttime=2014-01-01T00:00:00&endtime=2014-12-31T23:59:59
net = client.get_stations(
    network='1D', starttime=UTCDateTime(2014, 1, 1), endtime=UTCDateTime(2014, 12, 31)
)[0]
station_distances = []
for lat, lon in zip([sta.latitude for sta in net], [sta.longitude for sta in net]):
    dist_m = gps2dist_azimuth(lat, lon, df.loc[SHOT].Lat, df.loc[SHOT].Lon)[0]
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
    starttime=shot_time - PRE_ROLL,
    endtime=shot_time + max_prop_time_min_ceil,
)

# Save data
st.write(Path(os.environ['NODAL_WORKING_DIR']) / 'data' / f'{SHOT}.mseed')
