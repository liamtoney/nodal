"""
Travel time analysis for observed/infresnel-modeled/FDTD-modeled results...
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime

from utils import NODAL_WORKING_DIR, get_shots, get_waveforms_shot

SHOT = 'Y5'  # Shot to analyze

CELERITY_LIMITS = (330, 350)  # [m/s] For defining acoustic arrival window

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


#%% (A) infresnel stuff

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

#%% (B) Sandbox for comparing STA/LTA time picks with waveform arrivals

# These are the params used in make_shot_gather_measurements.py, or...
# TODO new parameters to experiment with!!!
FREQMIN = 5  # [Hz]
FREQMAX = 50  # [Hz]
STA = 0.2  # [s]
LTA = 2  # [s]

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

# Window around acoustic and get peak
rel_win = [(distance / c) for c in CELERITY_LIMITS[::-1]]
tr_sta_lta_win = tr_sta_lta.copy().trim(*[shot.time + t for t in rel_win])
max_val = tr_sta_lta_win.max()
argmax = tr_sta_lta_win.data.argmax()
rel_t_max = tr_sta_lta_win.times(reftime=shot.time)[argmax]

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
    ax.axvspan(*rel_win, zorder=-1, color='gray', alpha=0.3, lw=0)
    ax.axvline(rel_t_max, color='red', zorder=3)
ax2.scatter(rel_t_max, max_val, color='red', zorder=3)
fig.tight_layout()
fig.show()

#%% (C) Observed arrival time plotting and analysis

removal_celerity = np.max(CELERITY_LIMITS)  # [m/s]

# "Confidence check" scatter plot
fig, ax = plt.subplots()
df_sorted = df.sort_values(by='sta_lta_amp')  # Plot highest STA/LTA values on top
sm = ax.scatter(
    df_sorted.arr_time - (df_sorted.dist_m / removal_celerity),
    df_sorted.dist_m,
    c=df_sorted.sta_lta_amp,
)
dmin, dmax = df_sorted.dist_m.min(), df_sorted.dist_m.max()
ax.fill_betweenx(
    [dmin, dmax],
    *[
        [(dmin / c) - (dmin / removal_celerity), (dmax / c) - (dmax / removal_celerity)]
        for c in CELERITY_LIMITS
    ],
    zorder=-1,
    color='gray',
    alpha=0.3,
    lw=0,
    label='{}–{} m/s\ncelerity range'.format(*CELERITY_LIMITS),
)
ax.set_xlabel(f'Time from shot (s) removed by {removal_celerity} m/s')
ax.set_ylabel('Great circle distance (m)')
ax.set_title(f'Shot {shot.name}')
ax.legend(frameon=False)
fig.colorbar(sm, label='STA/LTA amplitude')
fig.tight_layout()
fig.show()
