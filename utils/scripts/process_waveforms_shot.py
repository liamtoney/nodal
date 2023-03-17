#!/usr/bin/env python

"""Pre-process iMUSH nodal data for a specified shot.

Carefully removes the full instrument response, applies an anti-aliasing lowpass filter,
and downsamples. Usage: process_waveforms_shot.py SHOT
where SHOT is a valid iMUSH shot name.

NOTE: Also "corrects" the shot Y4 data scaling issue!
"""

import sys

from utils import NODAL_WORKING_DIR, get_shots, get_stations, get_waveforms_shot

# Read in shot info
df = get_shots()

# Input checks
assert len(sys.argv) == 2, 'Must provide exactly one argument!'
shot = sys.argv[1]
assert shot in df.index, 'Argument must be a valid shot name!'

PRE_FILT = (0.5, 1, 50, 100)  # [Hz] Corners of filter to apply in frequency domain
WATER_LEVEL = 44  # [dB] Chosen to place the "corner" at around 1 Hz
LOWPASS_FREQ = 50  # [Hz] Anti-aliasing lowpass filter cutoff frequency
NEW_FS = 125  # [Hz] The new sampling rate to downsample to

# Read in raw data
st = get_waveforms_shot(shot)

# Read in station info
inv = get_stations()

# A couple steps which only apply to shot Y4
if shot == 'Y4':
    # Shot Y4 data are from Brandon, so they don't match IRIS inv
    for tr in st:
        try:
            _ = inv.get_coordinates(tr.id)  # Serves to check if tr.id is in inv
        except Exception:
            print(f'[Shot Y4] {tr.id} not found in IRIS inventory. Removing.')
            st.remove(tr)
            continue
    # TODO: Figure out Y4 data scaling issue... see "acoustic waves on nodes?" email chain
    fudge_factor = 87921  # Chosen to make max amp of closest station match shot Y5
    for tr in st:
        tr.data *= fudge_factor

# Carefully remove the full instrument response
print('Removing response...')
st.remove_response(pre_filt=PRE_FILT, water_level=WATER_LEVEL, inventory=inv)

# Apply an additional anti-aliasing lowpass filter before downsampling to be safe
st.filter('lowpass', freq=LOWPASS_FREQ, corners=10, zerophase=True)

# Downsample
print('Interpolating...')
st.interpolate(sampling_rate=NEW_FS, method='lanczos', a=20)

# Save processed data as miniSEED file
st.write(
    NODAL_WORKING_DIR / 'data' / 'mseed' / 'processed' / f'{shot}_processed.mseed',
    encoding='FLOAT64',  # Needed since we're not in counts anymore (makes bigger file!)
)
print('Done')
