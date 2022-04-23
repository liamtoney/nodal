import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client
from obspy.geodetics.base import gps2dist_azimuth

SHOT = 'Y5'  # Shot to plot

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

# Read in data
st = read(str(Path(os.environ['NODAL_WORKING_DIR']) / 'data' / f'{SHOT}.mseed'))

# Grab "MSH Node Array"
# http://ds.iris.edu/mda/1D/?starttime=2014-01-01T00:00:00&endtime=2014-12-31T23:59:59
print('Downloading coordinate and response information...')
inv = Client('IRIS').get_stations(
    network='1D',
    starttime=UTCDateTime(2014, 1, 1),
    endtime=UTCDateTime(2014, 12, 31),
    level='response',
)
print('Done')

# Assign coordinates and distances
for tr in st:
    coords = inv.get_coordinates(tr.id)
    tr.stats.latitude = coords['latitude']
    tr.stats.longitude = coords['longitude']
    tr.stats.distance = gps2dist_azimuth(
        tr.stats.latitude, tr.stats.longitude, df.loc[SHOT].Lat, df.loc[SHOT].Lon
    )[0]

# Remove sensitivity (fast but NOT accurate!)
st.remove_sensitivity(inv)

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
    st[0].times(reftime=shot_time),
    dist_km[dist_idx],  # Converting to km
    np.array([tr.data for tr in st])[dist_idx, :],
    vmin=-1e-5,
    vmax=1e-5,
    cmap='seismic',
)
ax.set_xlabel('Time from shot (s)')
ax.set_ylabel('Distance from shot (km)')
ax.set_title(f'Shot {SHOT}')
fig.show()
