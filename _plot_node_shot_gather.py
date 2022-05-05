import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy.geodetics.base import gps2dist_azimuth

from utils import get_shots, get_stations, get_waveforms_shot

SHOT = 'Y5'  # Shot to plot

# Read in shot info
df = get_shots()

# Read in station info
inv = get_stations()

# Read in data
st = get_waveforms_shot(SHOT)

# Assign coordinates and distances
for tr in st:
    coords = inv.get_coordinates(tr.id)
    tr.stats.latitude = coords['latitude']
    tr.stats.longitude = coords['longitude']
    tr.stats.distance = gps2dist_azimuth(
        tr.stats.latitude, tr.stats.longitude, df.loc[SHOT].lat, df.loc[SHOT].lon
    )[0]

# Remove sensitivity (fast but NOT accurate!)
st.remove_sensitivity(inv)

# Detrend, taper, filter
st.detrend('demean')
st.taper(0.05)
FREQMIN = 1  # [Hz]
FREQMAX = 50  # [Hz]
st.filter('bandpass', freqmin=FREQMIN, freqmax=FREQMAX)

# Apply STA/LTA
STA = 0.2  # [s]
LTA = 2  # [s]
st.trigger('classicstalta', sta=STA, lta=LTA)

# Ensure data are all same length (UGLY)
vc = pd.Series([tr.stats.npts for tr in st]).value_counts()
most_common_npts = vc[vc == vc.max()].index.values[0]
st = st.select(npts=most_common_npts)
print(f'Removed {vc[vc.index != most_common_npts].sum()} Trace(s)')

# Plot
dist_km = np.array([tr.stats.distance / 1000 for tr in st])
dist_idx = np.argsort(dist_km)
fig, ax = plt.subplots()
ax.pcolormesh(
    st[0].times(reftime=df.loc[SHOT].time),
    dist_km[dist_idx],  # Converting to km
    np.array([tr.data for tr in st])[dist_idx, :],
)
ax.set_xlabel('Time from shot (s)')
ax.set_ylabel('Distance from shot (km)')
ax.set_title(
    f'Shot {SHOT}, {FREQMIN}–{FREQMAX} Hz bandpass, STA = {STA} s, LTA = {LTA} s'
)
fig.show()

# fig.savefig(Path(os.environ['NODAL_WORKING_DIR']) / 'figures' / f'shot_{SHOT}.png', dpi=300, bbox_inches='tight')
