"""
Travel time analysis for observed/infresnel-modeled/FDTD-modeled results...
"""

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, PowerNorm
from obspy import UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.trigger import trigger_onset
from scipy.interpolate import griddata

from get_and_plot_nam_hrrr import build_url, ingest_grib_nam
from utils import INNER_RING_REGION, NODAL_WORKING_DIR, get_shots, get_waveforms_shot

SHOT = 'Y5'  # Shot to analyze

CELERITY_LIMITS = (320, 350)  # [m/s] For defining acoustic arrival window

M_PER_KM = 1000  # [m/km] CONSTANT

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

# This is *just* for the travel time picks
TRIGGER_ON = 4
TRIGGER_OFF = 2

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
tr_sta_lta.taper(0.05)

# Window around acoustic and get peak
rel_win = [(distance / c) for c in CELERITY_LIMITS[::-1]]
tr_sta_lta_win = tr_sta_lta.copy().trim(*[shot.time + t for t in rel_win])
max_val = tr_sta_lta_win.max()
argmax = tr_sta_lta_win.data.argmax()
rel_t_max = tr_sta_lta_win.times(reftime=shot.time)[argmax]

# Formally trigger the STA/LTA
trig_wins = trigger_onset(tr_sta_lta_win.data, TRIGGER_ON, TRIGGER_OFF)
print(f'{len(trig_wins)} trigger(s) within celerity window')

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
    for trig_win in trig_wins:
        ax.axvline(
            tr_sta_lta_win.times(reftime=shot.time)[trig_win[0]],  # Line at ONSET!
            color='limegreen',
            zorder=3,
        )
ax2.scatter(rel_t_max, max_val, color='red', zorder=3)
ax1.set_title(
    f'Station {station} — STA/LTA (on, off) = {TRIGGER_ON, TRIGGER_OFF} — {len(trig_wins)} trigger(s) within celerity window'
)
fig.tight_layout()
fig.show()

# Investigate how "pick error" affects calculated celerities
delta_c = lambda c_ref, d, delta_t: -(c_ref * delta_t) / ((d / c_ref) + delta_t)
npts = 500
d_vec = np.linspace(7.5, 25, npts) * M_PER_KM  # [m]
delta_t_vec = np.linspace(0, 0.3, npts)  # [s]
C_REF = 341.5  # [m/s]

args = d_vec, delta_t_vec, delta_c(C_REF, *np.meshgrid(d_vec, delta_t_vec))

fig, ax = plt.subplots()
pcm = ax.pcolormesh(*args, cmap=cc.m_fire)
cs = ax.contour(
    *args, levels=np.arange(-3, 0, 0.5), colors='black', linewidths=0.5, linestyles='-'
)
fig.colorbar(pcm, label='$\Delta c$ (m/s)')
ax.set_xlabel('$d$ (m)')
ax.set_ylabel('$\Delta t$ (s)')
left_title = ax.set_title('$c_\mathrm{ref}$ = ' + f'{C_REF} m/s', loc='left', y=1.02)
ax.set_title(
    r'$\Delta c~=~-\frac{\Delta t c_\mathrm{ref}}{d~/~c_\mathrm{ref}~+~\Delta t}$',
    loc='right',
    y=left_title._y,
)
ax.clabel(cs, cs.levels, fmt=lambda x: f'${x:g}$ m/s', inline=True)
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

USE_DIFF_PATH = True  # Toggle using diffracted path length or great circle distance
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
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x / M_PER_KM:g}'))
if USE_DIFF_PATH:
    ylabel = 'Diffracted path length (km)'
else:
    ylabel = 'Great circle distance (km)'
ax.set_ylabel(ylabel)
ax.set_title(f'Shot {shot.name}', loc='left', weight='bold')

# Plot celerity guides
xlim, ylim = ax.get_xlim(), ax.get_ylim()
est_celerity = celerity_estimate(shot)
celerities_to_plot = np.array([-4, -2, 0, 2, 4]) + est_celerity
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

# Colorbar 1: STA/LTA value
ylim = norm.vmin, norm.vmax
npts = 1000
pcm = cax1.pcolormesh(
    [0, 1],  # Just to stretch the image
    np.linspace(*ylim, npts),  # Linear y-axis
    np.ones((npts - 1, 1)),  # Solid black
    alpha=norm(np.expand_dims(np.linspace(*ylim, npts - 1), 1)),
    cmap=cc.m_gray,
    rasterized=True,  # Avoids unsightly horizontal stripes
)
cax1.set_xticks([])
cax1.set_ylabel('STA/LTA amplitude')

# Colorbar 2: Shot–node azimuth
fig.colorbar(sm, cax=cax2, label='Shot–node azimuth (°)')

for side in 'top', 'right':
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.subplots_adjust(wspace=0)

# Colorbar 1: Checkerboard pattern (after everything else has been adjusted!)
def callback(event=None):
    if len(cax1.collections) > 1:
        cax1.collections[1].remove()
    width = 4  # [squares] Width of checkerboard pattern
    cax1.set_aspect('equal', adjustable='datalim')
    fig.canvas.draw()  # KEY: So Matplotlib can register that we've changed the aspect!
    height = round(width * np.diff(cax1.get_ylim())[0] / np.diff(cax1.get_xlim())[0])
    cax1.set_aspect('auto', adjustable='datalim')
    check_shape = (height, width)
    cax1.pcolormesh(
        np.linspace(0, 1, check_shape[1] + 1),
        np.linspace(*ylim, check_shape[0] + 1),
        np.indices(check_shape).sum(axis=0) % 2,  # The checker pattern
        cmap=cc.m_gray_r,
        vmax=8,  # Effectively controls how gray the checkerboard is (`vmax=1` is black)
        shading='flat',
        zorder=pcm.zorder - 1,  # Ensure we're behind the transparency layer
    )


callback()
fig.canvas.mpl_connect('resize_event', callback)  # Extra fun: If we resize, we re-draw
fig.show()

#%% (C3) Map view plots of quantities w/ STA/LTA shading

shot_kw = dict(color='black', marker='s', s=50, zorder=10)

# Azimuth
fig, ax = plt.subplots()
sm = ax.scatter(
    df_sorted.lon,
    df_sorted.lat,
    c=df_sorted.azimuth,
    cmap=cc.m_CET_I1,
    alpha=norm(df_sorted.sta_lta_amp),
    lw=0,
)
ax.scatter(shot.lon, shot.lat, **shot_kw)
ax.set_title(f'Shot {shot.name}', weight='bold')
fig.colorbar(sm, label='Shot–node azimuth (°)')
fig.tight_layout()
fig.show()

# Delays — like in (A) but with STA/LTA transparency added in [this shows how the delays
# due to topography are not measurable as they are small!)
cmap = cc.m_CET_I1
inc = 0.05
bnorm = BoundaryNorm(np.arange(0, 0.15 + inc, inc), cmap.N)
celerity = celerity_estimate(shot)
fig, ax = plt.subplots()
sm = ax.scatter(
    df_sorted.lon,
    df_sorted.lat,
    c=df_sorted.path_length_diff_m / celerity,  # [s],
    norm=bnorm,
    cmap=cmap,
    alpha=norm(df_sorted.sta_lta_amp),
    lw=0,
)
ax.scatter(shot.lon, shot.lat, **shot_kw)
ax.set_title(f'Shot {shot.name}, {celerity} m/s', weight='bold')
fig.colorbar(
    sm,
    label='$infresnel\,$-sourced travel time delay (s)',
    ticks=plt.MultipleLocator(inc),
)
fig.tight_layout()
fig.show()

#%% (C4) Celerity map view w/ wind arrows overlain

# Celerity [measured] accounting for topography via infresnel!
measured_celerity = df_sorted.diffracted_path_length / df_sorted.arr_time

# Boundary norm setup
CEL_MIN = 339  # [m/s]
CEL_MAX = 341.5  # [m/s]
CEL_INC = 0.5  # [m/s]
cmap = cc.m_CET_D11
bnorm = BoundaryNorm(np.arange(CEL_MIN, CEL_MAX + CEL_INC, CEL_INC), cmap.N)

# Plot
fig, ax = plt.subplots()
sm = ax.scatter(
    df_sorted.lon,
    df_sorted.lat,
    c=measured_celerity,
    cmap=cmap,
    norm=bnorm,
    alpha=norm(df_sorted.sta_lta_amp),
    lw=0,
)
ax.scatter(shot.lon, shot.lat, **shot_kw)
fig.colorbar(sm, label='Measured, $infresnel\,$-adjusted celerity (m/s)')

# Get wind data!
time = pd.Timestamp(shot.time.datetime).round('1h').to_pydatetime()  # Nearest hour!
u = ingest_grib_nam(
    build_url(time.year, time.month, time.day, time.hour, measurement='UGRD')
)
v = ingest_grib_nam(
    build_url(time.year, time.month, time.day, time.hour, measurement='VGRD')
)

# Crop each DataArray for speed
minx, maxx, miny, maxy = INNER_RING_REGION
mask_lon = (u.longitude >= minx) & (u.longitude <= maxx)
mask_lat = (u.latitude >= miny) & (u.latitude <= maxy)
u = u.where(mask_lon & mask_lat, drop=True)
mask_lon = (v.longitude >= minx) & (v.longitude <= maxx)
mask_lat = (v.latitude >= miny) & (v.latitude <= maxy)
v = v.where(mask_lon & mask_lat, drop=True)

# Plot arrows
xlim = ax.get_xlim()
ylim = ax.get_ylim()
sm = ax.quiver(u.longitude, u.latitude, u, v, zorder=4, width=0.004)
reference_speed = 5  # [m/s]
ax.quiverkey(
    sm, 0.05, 1.02, reference_speed, label=f'{reference_speed} m/s', coordinates='axes'
)
ax.set_title(f'Shot {shot.name}', weight='bold')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
fig.tight_layout()
fig.show()

#%% (C5) Interpolate winds along path from shot to each node and plot like in C4

interpolation_method = 'linear'

dot_product_medians = []
for lon, lat in zip(df_sorted.lon, df_sorted.lat):

    # Define path to interpolate along
    npts = 100  # TODO should this vary with the shot–node distance?
    x = np.linspace(shot.lon, lon, npts)
    y = np.linspace(shot.lat, lat, npts)

    # Interpolate the wind grids
    u_profile = griddata(
        (u.longitude.values.flatten(), u.latitude.values.flatten()),
        u.values.flatten(),
        (x, y),
        method=interpolation_method,
    )
    v_profile = griddata(
        (v.longitude.values.flatten(), v.latitude.values.flatten()),
        v.values.flatten(),
        (x, y),
        method=interpolation_method,
    )
    waz = (90 - np.rad2deg(np.arctan2(v_profile, u_profile))) % 360  # [° from N]
    baz = gps2dist_azimuth(shot.lat, shot.lon, lat, lon)[1]  # [° shot-node az]
    wmag = np.sqrt(u_profile**2 + v_profile**2)
    angle_diff = waz - baz  # Sign does not matter!
    dot_product = wmag * np.cos(np.deg2rad(angle_diff))  # Treating baz as unit vector
    # dot_product = np.cos(np.deg2rad(angle_diff))  # Treating BOTH as unit vectors

    dot_product_medians.append(np.median(dot_product))
dot_product_medians = np.array(dot_product_medians)

# Boundary norm setup
MIN = -3.5  # [m/s]
MAX = -2  # [m/s]
INC = 0.3  # [m/s]
cmap = cc.m_CET_D11
bnorm = BoundaryNorm(np.arange(MIN, MAX + INC, INC), cmap.N)

# Plot
fig, ax = plt.subplots()
sm = ax.scatter(
    df_sorted.lon,
    df_sorted.lat,
    c=dot_product_medians,
    cmap=cmap,
    norm=bnorm,
    alpha=norm(df_sorted.sta_lta_amp),
    lw=0,
)
ax.scatter(shot.lon, shot.lat, **shot_kw)
fig.colorbar(sm, label='Median prop. dir. wind comp. (m/s)')

# Plot arrows
xlim = ax.get_xlim()
ylim = ax.get_ylim()
sm = ax.quiver(u.longitude, u.latitude, u, v, zorder=4, width=0.004)
reference_speed = 5  # [m/s]
ax.quiverkey(
    sm, 0.05, 1.02, reference_speed, label=f'{reference_speed} m/s', coordinates='axes'
)
ax.set_title(f'Shot {shot.name}', weight='bold')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
fig.tight_layout()
fig.show()

#%% (C6) Plot measured vs. modeled celerities against each other

static_sound_speed = 342  # [m/s]

fig, ax = plt.subplots()
sm = ax.scatter(
    measured_celerity,
    dot_product_medians + static_sound_speed,
    alpha=norm(df_sorted.sta_lta_amp),
    c=df_sorted.azimuth,
    cmap=cc.m_CET_I1,
    lw=0,
)
fig.colorbar(sm, label='Shot–node azimuth (°)')
ax.set_xlabel('Measured, $infresnel\,$-adjusted celerity (m/s)')
ax.set_ylabel(f'Median prop. dir. wind comp. + {static_sound_speed} m/s (m/s)')
ax.set_xlim(338, 343)
ax.set_ylim(338, 343)
ax.set_aspect('equal')
c_equal = np.linspace(330, 350, 2)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.plot(c_equal, c_equal, linestyle='--', zorder=-5, color='lightgray')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
fig.show()
