#!/usr/bin/env python

"""Ingest iMUSH nodal data for shot Y4 from Brandon's provided MAT file.

Cut shot gathers are saved as miniSEED files. Usage: ./ingest_shot_y4_waveforms.py
"""

import os
from pathlib import Path

from obspy import Stream, Trace
from scipy.io import loadmat

from utils import get_shots

SHOT = 'Y4'  # Hard-coded for this special case
PRE_ROLL = 20  # [s] Duration of data prior to shot time (from Brandon)
SAMPLING_RATE = 250  # [Hz] Native sampling rate

# Path to Y4 MAT file
Y4_PATH = Path(os.environ['NODAL_WORKING_DIR']) / 'data' / 'shot_y4.mat'

# Read in data
print(f'Reading waveforms for shot {SHOT}...')
y4_data = loadmat(Y4_PATH)
print('Done')

# Convert to Stream
df = get_shots()  # Read in shot info to get timing information for the shot
starttime = df.loc[SHOT].time - PRE_ROLL
st = Stream()
for data, metadata in zip(y4_data['seis'].T, y4_data['S'].squeeze()):
    header = dict(
        network='1D',
        station=str(metadata[0][0][0]),
        location='',
        channel='DPZ',
        starttime=starttime,
        sampling_rate=SAMPLING_RATE,
    )
    tr = Trace(data=data.astype('float'), header=header)  # Float conversion for saving
    st += tr

# Save as miniSEED file
st.write(Path(os.environ['NODAL_WORKING_DIR']) / 'data' / 'mseed' / f'{SHOT}.mseed')
