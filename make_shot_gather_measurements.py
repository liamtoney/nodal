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
from scipy.signal import welch

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
FREQMIN = 5  # [Hz]
FREQMAX = 50  # [Hz]
STA = 0.2  # [s]
LTA = 2  # [s]
CELERITY_LIMITS = (330, 350)  # [m/s] For defining acoustic arrival window
FREQ_WIN_DUR = 10  # [s] Duration of window (centered on arrival) for peak freq. calc.
RMS_WIN_DUR = 20  # [s] Seconds before shot time to include in RMS velocity calculation
# -------------------------------

# Read in station info and shot data
inv = get_stations()
st = get_waveforms_shot(SHOT, processed=True)  # Response already removed!

# Assign coordinates and distances
for tr in st:
    coords = inv.get_coordinates(tr.id)
    tr.stats.latitude = coords['latitude']
    tr.stats.longitude = coords['longitude']
    tr.stats.distance = gps2dist_azimuth(
        tr.stats.latitude, tr.stats.longitude, df.loc[SHOT].lat, df.loc[SHOT].lon
    )[0]

# Detrend, taper, filter
st.detrend('demean')
st.taper(0.05)
st.filter('bandpass', freqmin=FREQMIN, freqmax=FREQMAX, zerophase=True)

# Make copy of Stream, pre-STA/LTA, to use for peak frequency calculation
st_freq = st.copy()

# Make copy of Stream, pre-STA/LTA, to use for RMS window calculation
st_rms = st.copy()

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

#%% Calculate peak frequencies

# Merge, if needed (otherwise we get an error for trying to compute spectra for slices)
if SHOT == 'AO4':
    st_freq.merge()

# Now that we've merged, check that st matches st_freq in terms of Traces!
assert sorted([tr.id for tr in st]) == sorted([tr.id for tr in st_freq])

peak_freqs = []
for tr in st_freq.copy():  # Copying since we're destructively trimming here
    arr_time = df.loc[SHOT].time + (tr.stats.distance / np.mean(CELERITY_LIMITS))
    tlim = [arr_time - (FREQ_WIN_DUR / 2), arr_time + (FREQ_WIN_DUR / 2)]
    tr.trim(*tlim)

    welch_win_dur = 4  # [s]
    nperseg = int(welch_win_dur * tr.stats.sampling_rate)  # Samples
    nfft = np.power(2, int(np.ceil(np.log2(nperseg))) + 3)  # Pad FFT (extra padding!)
    f, pxx = welch(tr.data, tr.stats.sampling_rate, nperseg=nperseg, nfft=nfft)

    peak_freqs.append(f[pxx.argmax()])

#%% Calculate RMS velocity in pre-shot windows

# Merge, if needed (IMPORTANT since AO4 gaps are in the pre-shot RMS window!)
if SHOT == 'AO4':
    st_rms.merge()  # Can't use fill_value=np.nan here!

# Now that we've merged, check that st matches st_rms in terms of Traces!
assert sorted([tr.id for tr in st]) == sorted([tr.id for tr in st_rms])

# Trim to pre-shot window
st_rms.trim(df.loc[SHOT].time - RMS_WIN_DUR, df.loc[SHOT].time)

# Compute RMS (within RMS_WIN_DUR) for each Trace
rms_vals = []
for tr in st_rms:
    rms_vals.append(np.sqrt(np.mean(tr.data**2)))

#%% Calculate path differences

# Grab coordinates as a list
lats = [tr.stats.latitude for tr in st]
lons = [tr.stats.longitude for tr in st]

# Just using ~30 m resolution SRTM data here
ds_list, _ = calculate_paths(
    src_lat=df.loc[SHOT].lat,
    src_lon=df.loc[SHOT].lon,
    rec_lat=lats,
    rec_lon=lons,
    full_output=True,
)
path_diffs = [ds.path_length_difference for ds in ds_list]
direct_path_length = [ds.direct_path.length for ds in ds_list]
diffracted_path_length = [ds.diffracted_path.length for ds in ds_list]

#%% Export everything as CSV (we only have rows for stations with data!)

data_dict = dict(
    station=[int(tr.stats.station) for tr in st],  # int() since station names are #s
    lat=lats,
    lon=lons,
    dist_m=[tr.stats.distance for tr in st],  # [m]
    path_length_diff_m=path_diffs,  # [m]
    direct_path_length=direct_path_length,  # [m]
    diffracted_path_length=diffracted_path_length,  # [m]
    sta_lta_amp=amps,
    peak_freq=peak_freqs,  # [Hz]
    pre_shot_rms=rms_vals,  # [m/s]
)
data_df = pd.DataFrame(data=data_dict)
data_df.to_csv(
    NODAL_WORKING_DIR / 'shot_gather_measurements' / f'{SHOT}.csv', index=False
)
