#!/usr/bin/env python

"""Downloads iMUSH nodal data for a specified shot.

Cut shot gathers are saved as miniSEED files. Usage: ./download_waveforms_shot.py SHOT
where SHOT is a valid iMUSH shot name.
"""

import os
import sys
from pathlib import Path

import numpy as np
from matplotlib.dates import SEC_PER_MIN
from obspy.clients.fdsn import Client
from obspy.geodetics.base import gps2dist_azimuth

from utils import get_shots, get_stations

# Input checks
assert len(sys.argv) == 2, 'Must provide exactly one argument!'
shot = sys.argv[1]
assert shot in get_shots().index, 'Argument must be a valid shot name!'

MIN_CELERITY = 280  # [m/s] For calculating propagation times
PRE_ROLL = 30  # [s] Duration of data to download prior to shot time

# Read in shot info
df = get_shots()

# Calculate required data duration
net = get_stations()[0]
station_distances = []
for lat, lon in zip([sta.latitude for sta in net], [sta.longitude for sta in net]):
    dist_m = gps2dist_azimuth(lat, lon, df.loc[shot].lat, df.loc[shot].lon)[0]
    station_distances.append(dist_m)
max_prop_time = np.max(station_distances) / MIN_CELERITY  # [s]
max_prop_time_min_ceil = np.ceil(max_prop_time / SEC_PER_MIN) * SEC_PER_MIN

# Gather data
print(f'Downloading waveforms for shot {shot}...')
st = Client('IRIS').get_waveforms(
    network='1D',
    station='*',
    location='--',
    channel='DPZ',
    starttime=df.loc[shot].time - PRE_ROLL,
    endtime=df.loc[shot].time + max_prop_time_min_ceil,
)
print('Done')

# Save as miniSEED file
st.write(Path(os.environ['NODAL_WORKING_DIR']) / 'data' / 'mseed' / f'{shot}.mseed')
