from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from obspy import Stream, Trace

from utils import NODAL_WORKING_DIR

# CHANGE ME!
run = '06_shot_y5'
dir0 = NODAL_WORKING_DIR / 'fdprop' / 'nodal_fdprop_runs' / run / 'OUTPUT_FILES'
WAVEFORM_SNAPSHOT_INTERVAL = 5  # TODO from make_main.py

M_PER_KM = 1000  # [m/km] CONSTANT

# Read in files to an ObsPy Stream
st_syn = Stream()
for file in dir0.glob('process*_waveforms_pressure.txt'):

    # Read in params from file
    dt, t0 = np.loadtxt(file, max_rows=2, usecols=2)  # [s]
    x_locs = np.loadtxt(file, skiprows=2, max_rows=1) / M_PER_KM  # [km]
    traces = np.loadtxt(file, skiprows=3).T  # [Pa] Each row is a waveform!

    interval = dt * WAVEFORM_SNAPSHOT_INTERVAL  # [s] True sampling interval of data

    # Add to Stream
    for trace, x in zip(traces, x_locs):
        tr = Trace(data=trace, header=dict(sampling_rate=1 / interval))
        tr.stats.starttime += t0  # Shift for t0
        tr.stats.t0 = t0  # [s]
        tr.stats.x = x  # [km]
        st_syn += tr

st_syn.sort(keys=['x'])  # Sort by increasing x distance

#%% Plot

# Plotting config params
SKIP = 100  # Plot every SKIP stations
SCALE = 0.03  # [Pa] Single scale factor
SELF_NORMALIZE = True
MIN_TIME, MAX_TIME = 0, 80  # [s]
MIN_DIST, MAX_DIST = 0, 25  # [km]
POST_ROLL = 10  # [s]
TOPO_FILE = (
    NODAL_WORKING_DIR / 'fdprop' / 'Acoustic_2D' / 'imush_y5.dat'
)  # TODO from make_main.py
REMOVAL_CELERITY = 0.343  # [km/s] For reduced time

# Hacky params
PRE_ROLL = 0.55  # [s] TODO must manually set this so that it doesn't go beyond topo_ax
X_SRC = 500  # [m] TODO from make_main.py

# Form subsetted plotting Stream
starttime = st_syn[0].stats.starttime - st_syn[0].stats.t0  # Start at t = 0
st_syn_plot = st_syn.copy().trim(starttime + MIN_TIME, starttime + MAX_TIME)[::SKIP]


# Helper function to get the onset time for a [synthetic] waveform
def _get_onset_time(tr):
    dist_km = tr.stats.x - X_SRC / M_PER_KM
    if dist_km >= 0:
        return tr.stats.starttime + dist_km / REMOVAL_CELERITY
    else:
        print(f'Removed station with x = {tr.stats.x - X_SRC / M_PER_KM:.2} km')
        return None  # We're on the wrong side of the source


# Make measurements on the windowed traces
maxes = []
p2p_all = []
for tr in st_syn_plot:
    tr_measure = tr.copy()
    onset_time = _get_onset_time(tr_measure)
    if onset_time:
        tr_measure.trim(onset_time - PRE_ROLL, onset_time + POST_ROLL)
        maxes.append(tr_measure.data.max())
        p2p_all.append(tr_measure.data.max() - tr_measure.data.min())  # [Pa]
    else:  # No break!
        st_syn_plot.remove(tr)
maxes = np.array(maxes)
p2p_all = np.array(p2p_all)

# Further subset Stream
xs = (
    np.array([tr.stats.x for tr in st_syn_plot]) - X_SRC / M_PER_KM
)  # Set source at x = 0
include = (xs >= MIN_DIST) & (xs <= MAX_DIST)
st_syn_plot = Stream(compress(st_syn_plot, include))

# Configure colormap limits from p2p measurements of windowed traces
cmap = plt.cm.viridis
p2p_all = p2p_all[include]
norm = LogNorm(vmin=np.min(p2p_all), vmax=np.max(p2p_all))

# Load topography
topo_x, topo_z = np.loadtxt(TOPO_FILE).T / M_PER_KM  # [km]
topo_x -= X_SRC / M_PER_KM
mask = (topo_x >= MIN_DIST) & (topo_x <= MAX_DIST)  # Since we use clip_on = False later
topo_x = topo_x[mask]
topo_z = topo_z[mask]

# Make plot
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(4, 10),
    gridspec_kw=dict(width_ratios=(1, 10), height_ratios=(1, 70)),
)
_, cax, topo_ax, ax = axes.flatten()
_.axis('off')
topo_ax.sharey(ax)
for tr in st_syn_plot[::-1]:  # Plot the closest waveforms on top!
    tr_plot = tr.copy()
    onset_time = _get_onset_time(tr_plot)
    tr_plot.trim(onset_time - PRE_ROLL, onset_time + POST_ROLL, pad=True, fill_value=0)
    p2p = tr_plot.data.max() - tr_plot.data.min()  # [Pa]
    if SELF_NORMALIZE:
        data_scaled = tr_plot.copy().normalize().data / (SCALE / maxes.max())
    else:
        data_scaled = tr_plot.data / SCALE
    ax.plot(
        tr_plot.times() - PRE_ROLL,
        data_scaled + tr_plot.stats.x - X_SRC / M_PER_KM,  # Source at x = 0
        color=cmap(norm(p2p)),
        clip_on=False,
        solid_capstyle='round',
        lw=0.5,
    )
topo_ax.fill_betweenx(
    topo_x, topo_z.min(), topo_z, lw=0, color='tab:gray', clip_on=False
)
topo_ax.set_xlim(topo_z.min(), topo_z[0])  # Axis technically ends at elevation of shot
topo_ax.set_aspect('equal')
topo_ax.set_zorder(5)
ax.set_xlim(0, POST_ROLL)
ax.set_ylim(MIN_DIST, MAX_DIST)
ax.set_xlabel(f'Time (s), reduced by {REMOVAL_CELERITY * M_PER_KM:g} m/s', labelpad=10)
topo_ax.set_ylabel('Distance from shot Y5 (km)', labelpad=20, rotation=-90)
fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal'
)
cax.xaxis.set_ticks_position('top')
cax.set_xlabel('Peak-to-peak pressure (Pa)', labelpad=10)
cax.xaxis.set_label_position('top')
for side in 'top', 'right', 'left':
    ax.spines[side].set_visible(False)
for side in 'top', 'right', 'bottom':
    topo_ax.spines[side].set_visible(False)
ax.tick_params(left=False, labelleft=False)
topo_ax.tick_params(bottom=False, labelbottom=False)
ax.patch.set_alpha(0)
topo_ax.patch.set_alpha(0)
ax.spines['bottom'].set_position(('outward', 10))
fig.subplots_adjust(hspace=0.03)

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

fig.show()
