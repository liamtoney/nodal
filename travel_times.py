"""
Travel time analysis for observed/infresnel-modeled/FDTD-modeled results...
"""

import matplotlib.pyplot as plt
import pandas as pd
from obspy import UTCDateTime

from utils import NODAL_WORKING_DIR, get_shots, get_waveforms_shot

SHOT = 'Y5'  # Shot to analyze

# Get shot data
shot = get_shots().loc[SHOT]

# Get measurements for this shot
df = pd.read_csv(NODAL_WORKING_DIR / 'shot_gather_measurements' / f'{shot.name}.csv')


# Define helper function to return a reasonable static sound speed (as estimated from
# Fig. 1b) for an input shot (ROUGH ESTIMATE!)
def celerity_estimate(shot):
    if shot.time < UTCDateTime(2014, 7, 26):
        return 336  # [m/s]
    else:
        return 342  # [m/s]


#%% infresnel stuff

# Compute time delays owing to propagation over (rather than thru) topography
celerity = celerity_estimate(shot)
delays = df.path_length_diff_m / celerity  # [s]

# Shared plotting stuff
shot_kw = dict(color='lightgray', marker='s', s=50, ec='black', zorder=3)
title = f'Shot {shot.name}, {celerity} m/s'
vmin = 0
vmax = delays.max()
cmap = 'cividis'
delay_label = '$infresnel\,$-sourced travel time delay (s)'

# Plot delay vs. distance
fig, ax = plt.subplots(figsize=(12, 4))
ax.scatter(
    df.direct_path_length,
    delays,
    c=delays,
    clip_on=False,
    vmin=vmin,
    vmax=vmax,
    cmap=cmap,
    zorder=3,
)
ax.grid(axis='y', linestyle=':', color='gray', alpha=0.5)
ax.scatter(0, 0, clip_on=False, **shot_kw)
ax.set_xlim(left=0)
ax.set_ylim(vmin, vmax)
ax.set_xlabel('3D slant distance from shot (m)')
ax.set_ylabel(delay_label)
ax.set_title(title, loc='left', weight='bold')
for side in 'top', 'right':
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.show()

# Plot map view
fig, ax = plt.subplots()
sm = ax.scatter(df.lon, df.lat, c=delays, vmin=vmin, vmax=vmax, cmap=cmap)
ax.scatter(shot.lon, shot.lat, **shot_kw)
fig.colorbar(sm, label=delay_label)
ax.set_title(title, weight='bold')
fig.tight_layout()
fig.show()

#%% Sandbox for comparing STA/LTA time picks with waveform arrivals

# These are the params used in make_shot_gather_measurements.py, or...
# TODO new parameters to experiment with!!!
FREQMIN = 5  # [Hz]
FREQMAX = 50  # [Hz]
STA = 0.2  # [s]
LTA = 2  # [s]
CELERITY_LIMITS = (330, 350)  # [m/s] For defining acoustic arrival window

st = get_waveforms_shot(shot.name, processed=True)
station = '4106'
tr = st.select(station=station)[0]
distance = df[df.station == int(station)].dist_m.values[0]

# Process as in make_shot_gather_measurements.py
tr.detrend('demean')
tr.taper(0.05)
tr.filter('bandpass', freqmin=FREQMIN, freqmax=FREQMAX, zerophase=True)
tr_sta_lta = tr.copy()
tr_sta_lta.trigger('classicstalta', sta=STA, lta=LTA)

# Plot
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(14, 5))
ax1.plot(tr.times(reftime=shot.time), tr.data)
ax2.fill_between(tr_sta_lta.times(reftime=shot.time), tr_sta_lta.data, lw=0)
ax2.set_xlim(0, 80)
ax2.set_ylim(bottom=0)
ax1.set_ylabel('Velocity (m/s)')
ax2.set_ylabel('STA/LTA amplitude')
ax2.set_xlabel('Time from shot (s)')
for ax in ax1, ax2:
    ax.axvspan(
        *[(distance / c) for c in CELERITY_LIMITS[::-1]],
        zorder=-1,
        color='gray',
        alpha=0.3,
        lw=0,
    )
fig.tight_layout()
fig.show()
