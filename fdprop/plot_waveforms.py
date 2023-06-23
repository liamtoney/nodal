import json
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from obspy import Stream, Trace

from utils import NODAL_WORKING_DIR, get_shots, get_waveforms_shot

# CHANGE ME!
SHOT = 'Y5'

# Some logic to load the remaining transect-specific params correctly
if SHOT == 'Y5':
    RUN = '20_shot_y5_new_stf_hf'
    SYN_SCALE = 5
    OBS_SCALE = 300
elif SHOT == 'X5':
    RUN = '22_shot_x5_new_stf_hf'
    SYN_SCALE = 10
    OBS_SCALE = 1200
else:
    raise ValueError
WAVEFORM_SNAPSHOT_INTERVAL = 5  # TODO from main.cpp
X_SRC = 1500  # [m] TODO from main.cpp

M_PER_KM = 1000  # [m/km] CONSTANT

# READ IN "FAKE" DATA
st_syn = Stream()
for file in (
    NODAL_WORKING_DIR / 'fdprop' / 'nodal_fdprop_runs' / RUN / 'OUTPUT_FILES'
).glob('process*_waveforms_pressure.txt'):
    # Read in params from file
    dt, t0 = np.loadtxt(file, max_rows=2, usecols=2)  # [s]
    x_locs = np.loadtxt(file, skiprows=2, max_rows=1) / M_PER_KM  # [km]
    # Row 4 is elevation, which we skip
    traces = np.loadtxt(file, skiprows=4).T  # [Pa] Each row is a waveform!
    interval = dt * WAVEFORM_SNAPSHOT_INTERVAL  # [s] True sampling interval of data
    # Add to Stream
    for trace, x in zip(traces, x_locs):
        tr = Trace(data=trace, header=dict(sampling_rate=1 / interval))
        tr.stats.starttime += t0  # Shift for t0
        tr.stats.t0 = t0  # [s]
        tr.stats.x = x  # [km]
        st_syn += tr
st_syn.sort(keys=['x'])  # Sort by increasing x distance
st_syn = st_syn[::2]  # IMPORTANT: Keep only EVEN indices (0, 2, 4, ...)
st_syn.filter(type='lowpass', freq=4, zerophase=False, corners=4)  # KEY!

# READ IN "REAL" DATA
with open(
    NODAL_WORKING_DIR / 'metadata' / f'imush_{SHOT.lower()}_transect_stations.json'
) as f:
    sta_info = json.load(f)
st = get_waveforms_shot(SHOT, processed=True)
# Detrend, taper, filter
st.detrend('demean')
st.taper(0.05)
FREQMIN = 5  # [Hz]
FREQMAX = 50  # [Hz]
st.filter('bandpass', freqmin=FREQMIN, freqmax=FREQMAX, zerophase=True)
include = np.array([tr.stats.station in sta_info.keys() for tr in st])
st = Stream(compress(st, include))
for tr in st:
    x, out_of_plane_dist = sta_info[tr.stats.station]  # [m]
    tr.stats.x = x / M_PER_KM  # [km]
    tr.stats.out_of_plane_dist = out_of_plane_dist  # [m]
st.sort(keys=['x'])  # Sort by increasing x distance
for tr in st:
    tr.data *= 1e6  # Convert to μm/s

# %% Plot simple record sections / images for synthetic waveforms

MAX_RANGE = 10  # [km] How far out to go for record section / image

d = np.array([tr.stats.x - X_SRC / M_PER_KM for tr in st_syn])
source_ind = np.where(d == 0)[0][0]
end_ind = np.where(d == MAX_RANGE)[0][0]

d_plot = d[source_ind : end_ind + 1]
st_syn_plot = st_syn[source_ind : end_ind + 1]

# Waveform plot
SKIP = 50  # TODO: Plot every SKIP stations
SCALE = 1  # TODO: Divide each waveform by this SCALE to make it fit nicely
fig, ax = plt.subplots(figsize=(12, 9))
for tr, d in zip(st_syn_plot[::SKIP], d_plot[::SKIP]):
    ax.plot(tr.times(), (tr.data / SCALE) + d)
ax.set_xlim(0, 35)
ax.set_ylim(-0.5, MAX_RANGE)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance (km)')
fig.tight_layout()
fig.show()

# Image plot
fig, ax = plt.subplots(figsize=(12, 9))
ax.imshow(
    np.flipud(np.array([tr.data / tr.data.max() for tr in st_syn_plot])),
    extent=(
        st_syn_plot[0].times()[0],
        st_syn_plot[0].times()[-1],
        d_plot[0],
        d_plot[-1],
    ),
    aspect='auto',
    cmap='seismic',
    vmin=-1,
    vmax=1,
    interpolation='none',
)
ax.set_xlim(0, 35)
ax.set_ylim(-0.5, MAX_RANGE)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance (km)')
fig.tight_layout()
fig.show()

# %% Plot transmission loss for synthetics

d = np.array([tr.stats.x - X_SRC / M_PER_KM for tr in st_syn])  # [km] Dist. from source
peak_amp = np.array([tr.data.max() for tr in st_syn])  # [Pa] Peak amplitude

d_ref = 24 / M_PER_KM  # [km] TODO: Reference distance

tl = 20 * np.log10(peak_amp / peak_amp[np.isclose(d, d_ref)])
cyl_tl = 20 * np.log10(np.sqrt(d_ref / d))
sph_tl = 20 * np.log10(d_ref / d)

fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(d, tl)
ax.plot(
    d,
    cyl_tl,
    color='gray',
    linestyle='--',
    zorder=-2,
    label='Cylindrical',
    dash_capstyle='round',
)
ax.plot(
    d,
    sph_tl,
    color='gray',
    linestyle=':',
    zorder=-1,
    dash_capstyle='round',
    label='Spherical',
)
ax.set_xlabel('Distance from shot (km)')
ax.set_ylabel(f'Ground transmission loss\n(dB rel. {d_ref * M_PER_KM:g} m)')
ax.set_xlim(0, 24)
ax.set_ylim(-65, 0)
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
for side in 'top', 'right':
    ax.spines[side].set_visible(False)
ax.legend(title=r'${\bf Geometric\ spreading}$', alignment='left', frameon=False)
fig.tight_layout()
fig.show()

# %% Plot comparison between synthetics and observed GCAs

# Plotting config params
SKIP = 75  # Plot every SKIP stations
SELF_NORMALIZE = True
MIN_TIME, MAX_TIME = 0, 80  # [s]
MIN_DIST, MAX_DIST = 0, 24  # [km]
POST_ROLL = 10  # [s]
TOPO_FILE = (
    NODAL_WORKING_DIR / 'fdprop' / 'Acoustic_2D' / f'imush_{SHOT.lower()}_buffer.dat'
)  # TODO from main.cpp
# Shot-dependent celerity for reduced time!
if SHOT == 'X5':
    removal_celerity = 0.336  # [km/s]
elif SHOT == 'Y5':
    removal_celerity = 0.342  # [km/s]


# Hacky params
PRE_ROLL = [
    0.96,
    0.94,
]  # [s] TODO must manually set this so that it doesn't go beyond topo_ax

# Form [subsetted] plotting Stream for FAKE data
starttime = st_syn[0].stats.starttime - st_syn[0].stats.t0  # Start at t = 0
st_syn_plot = st_syn.copy().trim(starttime + MIN_TIME, starttime + MAX_TIME)[::SKIP]

# Form plotting Stream for REAL data
starttime = get_shots().loc[SHOT].time
st_plot = st.copy().trim(starttime + MIN_TIME, starttime + MAX_TIME)


# Helper function to get the onset time for a [synthetic] waveform
def _get_onset_time(tr):
    dist_km = tr.stats.x - X_SRC / M_PER_KM
    if dist_km >= 0:
        return tr.stats.starttime + dist_km / removal_celerity
    else:
        print(f'Removed station with x = {tr.stats.x - X_SRC / M_PER_KM:.2} km')
        return None  # We're on the wrong side of the source


# BIG helper function for plotting wfs
def process_and_plot(st, ax, scale, pre_roll, log=True):
    # Make measurements on the windowed traces
    maxes = []
    p2p_all = []
    for tr in st:
        tr_measure = tr.copy()
        onset_time = _get_onset_time(tr_measure)
        if onset_time:
            tr_measure.trim(onset_time - pre_roll, onset_time + POST_ROLL)
            maxes.append(tr_measure.data.max())
            p2p_all.append(tr_measure.data.max() - tr_measure.data.min())  # [Pa]
        else:  # No break!
            st.remove(tr)
    maxes = np.array(maxes)
    p2p_all = np.array(p2p_all)

    # Further subset Stream
    xs = np.array([tr.stats.x for tr in st]) - X_SRC / M_PER_KM  # Set source at x = 0
    include = (xs >= MIN_DIST) & (xs <= MAX_DIST)

    # Configure colormap limits from p2p measurements of windowed traces
    cmap = plt.cm.viridis
    p2p_all = p2p_all[include]
    if log:
        norm = LogNorm(vmin=np.min(p2p_all), vmax=np.max(p2p_all))
    else:
        norm = plt.Normalize(vmin=np.min(p2p_all), vmax=np.percentile(p2p_all, 80))

    st = Stream(compress(st, include))
    for tr in st[::-1]:  # Plot the closest waveforms on top!
        tr_plot = tr.copy()
        onset_time = _get_onset_time(tr_plot)
        tr_plot.trim(
            onset_time - pre_roll, onset_time + POST_ROLL, pad=True, fill_value=0
        )
        p2p = tr_plot.data.max() - tr_plot.data.min()  # [Pa]
        if SELF_NORMALIZE:
            data_scaled = tr_plot.copy().normalize().data / (scale / maxes.max())
        else:
            data_scaled = tr_plot.data / scale
        ax.plot(
            tr_plot.times() - pre_roll,
            data_scaled + tr_plot.stats.x - X_SRC / M_PER_KM,  # Source at x = 0
            color=cmap(norm(p2p)),
            clip_on=False,
            solid_capstyle='round',
            lw=0.5,
        )
    return norm, cmap


# Load topography
topo_x, topo_z = np.loadtxt(TOPO_FILE).T / M_PER_KM  # [km]
topo_x -= X_SRC / M_PER_KM
mask = (topo_x >= MIN_DIST) & (topo_x <= MAX_DIST)  # Since we use clip_on = False later
topo_x = topo_x[mask]
topo_z = topo_z[mask]

# Make plot
fig, (topo_ax1, ax1, topo_ax2, ax2) = plt.subplots(
    ncols=4, figsize=(4, 10), gridspec_kw=dict(width_ratios=(1, 10, 1, 10))
)

topo_ax1.sharey(ax1)
topo_ax2.sharey(ax2)
ax1.sharey(ax2)

for topo_ax in topo_ax1, topo_ax2:
    topo_ax.fill_betweenx(
        topo_x, topo_z.min(), topo_z, lw=0, color='lightgray', clip_on=False
    )
    topo_ax.set_xlim(
        topo_z.min(), topo_z[0]
    )  # Axis technically ends at elevation of shot
    # topo_ax.set_aspect('equal')  # WHY DOES THIS NOT WORK!??
    topo_ax.set_zorder(-5)
    topo_ax.tick_params(bottom=False, labelbottom=False)
    for side in 'top', 'right', 'bottom':
        topo_ax.spines[side].set_visible(False)
topo_ax2.axis('off')
topo_ax1.set_ylabel(f'Distance from shot {SHOT} (km)', labelpad=20, rotation=-90)

norms = []
for ax, st, scale, pre_roll, log in zip(
    [ax1, ax2], [st_syn_plot, st_plot], [SYN_SCALE, OBS_SCALE], PRE_ROLL, [False, False]
):
    norm, cmap = process_and_plot(st, ax, scale, pre_roll, log=log)
    norms.append(norm)
    ax.set_xlim(0, POST_ROLL)
    ax.set_ylim(MIN_DIST, MAX_DIST)
    for side in 'top', 'right', 'left':
        ax.spines[side].set_visible(False)

    ax.tick_params(left=False, labelleft=False)
    ax.patch.set_alpha(0)
    ax.spines['bottom'].set_position(('outward', 10))
    fig.subplots_adjust(hspace=0.03, wspace=0.1)

for ax, topo_ax in zip([ax1, ax2], [topo_ax1, topo_ax2]):
    # Kind of hacky, but nifty!
    ax_pos = ax.get_position()
    topo_ax_pos = topo_ax.get_position()
    topo_ax.set_position(
        [
            ax_pos.x0 - topo_ax_pos.width,  # Elevation of shot is at t = 0!
            topo_ax_pos.y0,
            topo_ax_pos.width,
            topo_ax_pos.height,
        ]
    )

# Add spine to cover wf ends
_spine = fig.add_subplot(111)
_spine.set_position(topo_ax1.get_position())
for side in 'top', 'bottom', 'right':
    _spine.spines[side].set_visible(False)
_spine.patch.set_alpha(0)
_spine.set_xticks([])
_spine.set_yticks([])

cax = fig.add_subplot(111)
ax1_pos = ax1.get_position()
ax2_pos = ax2.get_position()
cax.set_position(
    [
        ax1_pos.xmin,
        ax1_pos.ymax * 1.06,
        ax2_pos.xmax - ax1_pos.xmin,
        ax1_pos.height / 70,
    ]
)
cax.set_xticks([])
cax.set_yticks([])

for norm in norms:
    _cax = fig.add_subplot(111)
    _cax.set_position(cax.get_position())
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=_cax, orientation='horizontal'
    )
    if norm == norms[0]:
        _cax.xaxis.set_ticks_position('top')
        _cax.xaxis.set_label_position('top')
        _cax.set_xlabel('Peak-to-peak pressure (Pa)', labelpad=7)
    else:
        _cax.set_xlabel('Peak-to-peak velocity (μm/s)', labelpad=5)

cax.set_xlabel(
    f'Time (s), reduced by {removal_celerity * M_PER_KM:g} m/s', labelpad=625
)

fig.show()

# fig.savefig(NODAL_WORKING_DIR / 'figures' / 'fdprop' / 'final' / f'fdprop_imush_{RUN[:2]}_waveforms.png', dpi=300, bbox_inches='tight')
