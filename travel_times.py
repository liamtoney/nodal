"""
Travel time analysis for observed/infresnel-modeled/FDTD-modeled results...
"""

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from obspy import UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth

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

#%% (C1) Observed arrival time plotting and analysis

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
ax.set_xlabel(f'Time from shot (s), reduced by {removal_celerity} m/s')
ax.set_ylabel('Great circle distance (m)')
ax.set_title(f'Shot {shot.name}')
ax.legend(frameon=False)
fig.colorbar(sm, label='STA/LTA amplitude')
fig.tight_layout()
fig.show()

#%% (C2) Encode STA/LTA to transparency so we can color markers by azimuth

USE_DIFF_PATH = False  # Toggle using diffracted path length or great circle distance
GAMMA = 2  # Exponent for accentuating higher STA/LTA values

# Compute the shot–node azimuth and add to df_sorted
df_sorted['azimuth'] = [
    gps2dist_azimuth(shot.lat, shot.lon, lat, lon)[1]
    for (lat, lon) in zip(df_sorted.lat, df_sorted.lon)
]

# Select the distance metric to plot with
d = df_sorted.diffracted_path_length if USE_DIFF_PATH else df_sorted.dist_m

# Define normalization for STA/LTA (transparency mapping)
norm = PowerNorm(
    gamma=GAMMA, vmin=df_sorted.sta_lta_amp.min(), vmax=df_sorted.sta_lta_amp.max()
)

# Plot
fig, (ax, _, cax1, cax2) = plt.subplots(
    ncols=4, gridspec_kw=dict(width_ratios=[150, 20, 5, 5])
)
_.remove()
sm = ax.scatter(
    df_sorted.arr_time - (d / removal_celerity),
    d,
    c=df_sorted.azimuth,
    cmap=cc.m_CET_I1,
    alpha=norm(df_sorted.sta_lta_amp),
    lw=0,
)
ax.set_xlabel(f'Time from shot (s), reduced by {removal_celerity} m/s')
if USE_DIFF_PATH:
    ylabel = 'Diffracted path length (m)'
else:
    ylabel = 'Great circle distance (m)'
ax.set_ylabel(ylabel)
ax.set_title(f'Shot {shot.name}', loc='left', weight='bold')

# Plot celerity guides
xlim, ylim = ax.get_xlim(), ax.get_ylim()
celerities_to_plot = (338, 340, 342)
guide_color = 'tab:gray'
for celerity in celerities_to_plot:
    ax.plot(
        [
            (ylim[0] / celerity) - (ylim[0] / removal_celerity),
            (ylim[1] / celerity) - (ylim[1] / removal_celerity),
        ],
        [ylim[0], ylim[1]],
        zorder=-1,
        lw=plt.rcParams['axes.linewidth'],
        color=guide_color,
        linestyle='--',
        dash_capstyle='round',
        clip_on=False,
    )
    ax.text(
        (ylim[1] / celerity) - (ylim[1] / removal_celerity),
        ylim[1],
        f'{celerity} m/s',
        va='bottom',
        ha='right',
        rotation_mode='anchor',
        transform_rotates_text=True,
        color=guide_color,
        rotation=np.rad2deg(np.arctan(1 / ((1 / celerity) - (1 / removal_celerity)))),
    )
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Colorbar 1: STA/LTA value (transparency)
ylim = norm.vmin, norm.vmax
npts = 1000
cax1.pcolormesh(
    [0, 1],  # Just to stretch the image
    np.linspace(*ylim, npts),  # Linear y-axis
    np.power(np.expand_dims(np.linspace(*ylim, npts - 1), 1), norm.gamma),
    cmap=cc.m_gray_r,
    rasterized=True,
)
cax1.set_xticks([])
cax1.set_ylabel('STA/LTA amplitude')

# Colorbar 2: Shot–node azimuth
fig.colorbar(sm, cax=cax2, label='Shot–node azimuth (°)')

for side in 'top', 'right':
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.subplots_adjust(wspace=0)
fig.show()
