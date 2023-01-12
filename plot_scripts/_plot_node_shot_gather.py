# TODO: Make this code available to make_shot_gather_measurements.py for troubleshooting

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy.geodetics.base import gps2dist_azimuth

from utils import NODAL_WORKING_DIR, get_shots, get_stations, get_waveforms_shot

SHOT = 'Y5'  # Shot to plot

RAW = False  # If True, then produce a "raw" gather — we just remove the sensitivity

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

# TODO: Figure out Y4 data scaling issue... see "acoustic waves on nodes?" email chain
if SHOT == 'Y4':
    fudge_factor = 87921  # Chosen to make max amp of closest station match shot Y5
    for tr in st:
        tr.data *= fudge_factor

# Remove sensitivity (fast but NOT accurate!) to get m/s units
st.remove_sensitivity(inv)

# If we're not making a raw gather, then process
if not RAW:

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

    # Define title
    title = (
        f'Shot {SHOT}, {FREQMIN}–{FREQMAX} Hz bandpass, STA = {STA} s, LTA = {LTA} s'
    )

    # Define plotting options
    vmin = None
    vmax = None
    cmap = None

    # Define subdirectory
    subdir = 'processed_gathers'

else:

    # Define title
    title = f'Shot {SHOT}'

    # Define plotting options
    vmin = -1e-5  # [m/s]
    vmax = 1e-5  # [m/s]
    cmap = 'seismic'

    # Define subdirectory
    subdir = 'raw_gathers'

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
    vmin=vmin,
    vmax=vmax,
    cmap=cmap,
)
ax.set_xlabel('Time from shot (s)')
ax.set_ylabel('Distance from shot (km)')
ax.set_title(title)
if SHOT == 'Y4':
    ax.set_xlim(-30, 120)  # To match other plots
    ax.text(
        0.99,
        0.985,
        f'{fudge_factor = }',
        transform=ax.transAxes,
        ha='right',
        va='top',
        color='black' if RAW else 'white',
        fontname=['JetBrains Mono', 'monospace'],
    )

# For AGU poster (shots X5 and Y5)
# ax.set_xlim(-10, 90)
# ax.set_ylim(5, 26)
# fig.savefig(
#     f'/Users/ldtoney/work/meetings/agu_22/poster/graphics/figures/{SHOT}_section.png',
#     dpi=300,
#     bbox_inches='tight',
# )

fig.show()

# fig.savefig(NODAL_WORKING_DIR / 'figures' / subdir / f'shot_{SHOT}.png', dpi=300, bbox_inches='tight')

#%% (Plot min/max celerities on top of shot gather created above)

celerity_limits = (330, 350)

xlim = ax.get_xlim()
ylim = ax.get_ylim()

for celerity in celerity_limits:
    ax.plot(
        [0, xlim[1]],
        [0, (celerity / 1000) * xlim[1]],
        color='black' if RAW else 'white',
        linewidth=1,
    )

ax.set_xlim(xlim)
ax.set_ylim(ylim)

fig.show()

#%% (Plot mask distance params on top of shot gather created above)

from utils import MASK_DISTANCE_KM
from utils.utils import T_SEP, V_P, C

xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Plot the two wavespeeds
for speed in C, V_P:
    ax.plot(
        [0, xlim[1]],
        [0, (speed / 1000) * xlim[1]],
        color='black' if RAW else 'white',
        linewidth=1,
    )
# Plot t_sep
ax.plot(
    [T_SEP, xlim[1] + T_SEP],
    [0, (speed / 1000) * xlim[1]],
    color='black' if RAW else 'white',
    linewidth=1,
    linestyle='--',
)
# Plot masking distance
ax.fill_between(
    xlim, [MASK_DISTANCE_KM, MASK_DISTANCE_KM], linewidth=0, color='white', alpha=0.8
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
    vmin=vmin,
    vmax=vmax,
    cmap=cmap,
)
ax.set_xlabel('Time (s) relative to estimated arrival time')
ax.set_ylabel('Distance from shot (km)')
ax.set_title(f'Shot {SHOT}, celerity = {celerity} m/s')
if SHOT == 'Y4':
    ax.text(
        0.99,
        0.985,
        f'{fudge_factor = }',
        transform=ax.transAxes,
        ha='right',
        va='top',
        color='black' if RAW else 'white',
        fontname=['JetBrains Mono', 'monospace'],
    )
fig.show()
