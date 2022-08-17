import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy.geodetics.base import gps2dist_azimuth

from utils import NODAL_WORKING_DIR, get_shots, get_stations, get_waveforms_shot

SHOT = 'Y5'  # Shot to plot

# Read in shot info
df = get_shots()

# Read in station info
inv = get_stations()

# Read in data
st = get_waveforms_shot(SHOT)

# Assign coordinates and distances
for tr in st:
    # Need the "try" statement here for the shot Y4 data from Brandon
    try:
        coords = inv.get_coordinates(tr.id)
    except Exception:
        print(f'{tr.id} not found on IRIS. Removing.')
        st.remove(tr)
        continue
    tr.stats.latitude = coords['latitude']
    tr.stats.longitude = coords['longitude']
    tr.stats.distance = gps2dist_azimuth(
        tr.stats.latitude, tr.stats.longitude, df.loc[SHOT].lat, df.loc[SHOT].lon
    )[0]

# Remove sensitivity (fast but NOT accurate!)
if SHOT != 'Y4':
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

# Merge, as some stations have multiple Traces (doing this as late as possible)
st.merge(fill_value=np.nan)

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
if SHOT == 'Y4':
    ax.set_xlim(-30, 120)  # To match other plots
fig.show()

# fig.savefig(NODAL_WORKING_DIR / 'figures' / 'processed_gathers' / f'shot_{SHOT}.png', dpi=300, bbox_inches='tight')

#%% (Plot min/max celerities on top of shot gather created above)

celerity_limits = (330, 350)

xlim = ax.get_xlim()
ylim = ax.get_ylim()

for celerity in celerity_limits:
    ax.plot(
        [0, xlim[1]],
        [0, (celerity / 1000) * xlim[1]],
        color='white',
        alpha=0.5,
        linewidth=1,
    )

ax.set_xlim(xlim)
ax.set_ylim(ylim)

fig.show()

#%% Align acoustic arrivals using a best-choice celerity (trial and error for each shot)

# [m/s] Celerity to use for aligning arrivals
celerity = 343

# [s] Padding before and after estimated arrival (defines trimmed Trace duration)
pre = 4
post = 10

st_window = st.copy()  # Since we're destructively trimming here

for tr in st_window:
    start = df.loc[SHOT].time + (tr.stats.distance / celerity) - pre
    end = start + post
    tr.trim(start, end)

dist_km = np.array([tr.stats.distance / 1000 for tr in st_window])
dist_idx = np.argsort(dist_km)
fig, ax = plt.subplots()
ax.pcolormesh(
    st_window[0].times() - pre,
    dist_km[dist_idx],  # Converting to km
    np.array([tr.data for tr in st_window])[dist_idx, :],
)
ax.set_xlabel('Time (s) relative to estimated arrival time')
ax.set_ylabel('Distance from shot (km)')
ax.set_title(f'Shot {SHOT}, celerity = {celerity} m/s')
fig.show()
