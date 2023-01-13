#!/usr/bin/env python

"""Creates CSV file of measured parameters for a given shot.

Usage: ./make_shot_gather_measurements.py SHOT (where SHOT is a valid iMUSH shot name)

Alternate IPython usage (for bulk processing!):

from utils import get_shots
for shot in get_shots().index:
    %run make_shot_gather_measurements.py {shot}
"""

import sys

import numpy as np
import pandas as pd
from infresnel import calculate_paths
from obspy.geodetics.base import gps2dist_azimuth

from utils import NODAL_WORKING_DIR, get_shots, get_stations, get_waveforms_shot

# Read in shot info
df = get_shots()

# Input checks
assert len(sys.argv) == 2, 'Must provide exactly one argument!'
SHOT = sys.argv[1]
assert SHOT in df.index, 'Argument must be a valid shot name!'

# -------------------------------
# MEASUREMENT PARAMETERS TO TWEAK
# -------------------------------
FREQMIN = 1  # [Hz]
FREQMAX = 50  # [Hz]
STA = 0.2  # [s]
LTA = 2  # [s]
CELERITY_LIMITS = (330, 350)  # [m/s] For defining acoustic arrival window
WIN_DUR = 20  # [s] Seconds before shot time to include in RMS velocity calculation
# -------------------------------

# Read in station info and shot data
inv = get_stations()
st = get_waveforms_shot(SHOT)

# Assign coordinates and distances
for tr in st:
    # Shot Y4 data are from Brandon, so they don't match IRIS inv
    try:
        coords = inv.get_coordinates(tr.id)
    except Exception:
        print(f'{tr.id} not found in inventory. Removing.')
        st.remove(tr)
        continue
    tr.stats.latitude = coords['latitude']
    tr.stats.longitude = coords['longitude']
    tr.stats.distance = gps2dist_azimuth(
        tr.stats.latitude, tr.stats.longitude, df.loc[SHOT].lat, df.loc[SHOT].lon
    )[0]

# Remove sensitivity — st.remove_response() is SLOW; I think just sensitivity removal is
# OK here? Units are m/s after this step
if SHOT == 'Y4':
    for tr in st:
        fudge_factor = 87921  # TODO: See _plot_node_shot_gather.py
        tr.data *= fudge_factor
st.remove_sensitivity(inv)

# Make copy of unprocessed Stream to use for RMS window calculation
st_rms = st.copy()

# Detrend, taper, filter
st.detrend('demean')
st.taper(0.05)
st.filter('bandpass', freqmin=FREQMIN, freqmax=FREQMAX)

# Apply STA/LTA
st.trigger('classicstalta', sta=STA, lta=LTA)  # TODO: Try other trigger algs?

# Merge, as some stations have multiple Traces (doing this as late as possible)
st.merge(fill_value=np.nan)

#%% Calculate amplitudes (STA/LTA maxima)

amps = []
for tr in st.copy():  # Copying since we're destructively trimming here
    tlim = [df.loc[SHOT].time + (tr.stats.distance / c) for c in CELERITY_LIMITS[::-1]]
    tr.trim(*tlim)
    amps.append(tr.max())  # Simply taking the maximum of the STA/LTA function...

#%% Calculate RMS velocity in pre-shot windows

# Merge, if needed (IMPORTANT since AO4 gaps are in the pre-shot RMS window!)
if SHOT == 'AO4':
    st_rms.merge()  # Can't use fill_value=np.nan here!

# Now that we've merged, check that st matches st_rms in terms of Traces!
assert [tr.id for tr in st] == [tr.id for tr in st_rms]

# Trim to pre-shot window
st_rms.trim(df.loc[SHOT].time - WIN_DUR, df.loc[SHOT].time)

# Compute RMS (within WIN_DUR) for each Trace
rms_vals = []
for tr in st_rms:
    rms_vals.append(np.sqrt(np.mean(tr.data**2)))

#%% Calculate path differences

# Grab coordinates as a list
lats = [tr.stats.latitude for tr in st]
lons = [tr.stats.longitude for tr in st]

# Just using ~30 m resolution SRTM data here
path_diffs = calculate_paths(
    src_lat=df.loc[SHOT].lat, src_lon=df.loc[SHOT].lon, rec_lat=lats, rec_lon=lons
)

#%% Export everything as CSV (we only have rows for stations with data!)

data_dict = dict(
    station=[int(tr.stats.station) for tr in st],  # int() since station names are #s
    lat=lats,
    lon=lons,
    dist_m=[tr.stats.distance for tr in st],  # [m]
    path_length_diff_m=path_diffs,  # [m]
    sta_lta_amp=amps,
    pre_shot_rms=rms_vals,  # [m/s]
)
data_df = pd.DataFrame(data=data_dict)
data_df.to_csv(
    NODAL_WORKING_DIR / 'shot_gather_measurements' / f'{SHOT}.csv', index=False
)
