import json
import os
import subprocess
from itertools import compress
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox
from obspy import Stream, Trace
from tqdm import tqdm

from utils import NODAL_WORKING_DIR, get_shots, get_waveforms_shot

FONT_SIZE = 10  # [pt]
plt.rcParams.update({'font.size': FONT_SIZE})

# Choose 'Y5' or 'X5' — need to run this script once for each, and save figure!
SHOT = 'Y5'

# Some logic to load the remaining transect-specific params correctly
if SHOT == 'Y5':
    RUN = '20_shot_y5_new_stf_hf'
    Z_SRC = 734  # [m]
    SYN_SCALE = 12
    OBS_SCALE = 300
    REMOVAL_CELERITY = 0.342  # [km/s]
elif SHOT == 'X5':
    RUN = '22_shot_x5_new_stf_hf'
    Z_SRC = 464  # [m]
    SYN_SCALE = 14
    OBS_SCALE = 1800
    REMOVAL_CELERITY = 0.336  # [km/s]
else:
    raise ValueError
PRESSURE_SNAPSHOT_DIR = NODAL_WORKING_DIR / 'fdprop' / 'nodal_fdprop_pressure_snapshots'
TIMESTAMPS = np.arange(1000, 15000 + 1000, 1000)  # Same for both transects
XLIM = (0, 24)  # [km] Relative to shot x-position
YLIM = (-0.5, 5)  # [km] Relative to shot z-position
DT = 0.004  # [s]
X_SRC = 1500  # [m]
WAVEFORM_SNAPSHOT_INTERVAL = 5

# Constants
M_PER_KM = 1000  # [m/km]


# Helper function to read in pressure matrix for a specified run and timestamp
def _p_matrix_from_timestamp(run, timestamp):
    file_pattern = f'{run}__*_*_*_*_*_{timestamp}.npy'
    files = list(PRESSURE_SNAPSHOT_DIR.glob(file_pattern))
    assert len(files) == 1
    file = files[0]
    p = np.load(file).T
    param = file.stem.replace(RUN + '__', '')  # String of 6 integers
    hoz_min, hoz_max, vert_min, vert_max, dx, _ = np.array(param.split('_')).astype(int)
    # Only include the portion of the matrix that's IN the plotted zone!
    left_trim = ((XLIM[0] * M_PER_KM) + X_SRC) - hoz_min  # [m]
    top_trim = vert_max - ((YLIM[1] * M_PER_KM) + Z_SRC)  # [m]
    right_trim = hoz_max - ((XLIM[1] * M_PER_KM) + X_SRC)  # [m]
    p_trim = p[: -top_trim // dx, left_trim // dx : -right_trim // dx]
    if p_trim.max() > 0:
        p /= p_trim.max()
    return p, hoz_min, hoz_max, vert_min, vert_max, dx


#%% Load in waveform snapshot stuff

# Aggregate pressure matrix
p_agg, hoz_min, hoz_max, vert_min, vert_max, dx = _p_matrix_from_timestamp(
    RUN, TIMESTAMPS[0]
)
for timestamp in tqdm(TIMESTAMPS[1:], initial=1, total=TIMESTAMPS.size):
    p_agg += _p_matrix_from_timestamp(RUN, timestamp)[0]

# Terrain from .dat file for this run
terrain_contour = np.loadtxt(
    NODAL_WORKING_DIR / 'fdprop' / 'Acoustic_2D' / f'imush_{SHOT.lower()}_buffer.dat'
)

#%% Load in synthetic and observed data

# Synthetic data
st_syn = Stream()
files = list(
    (NODAL_WORKING_DIR / 'fdprop' / 'nodal_fdprop_runs' / RUN / 'OUTPUT_FILES').glob(
        'process*_waveforms_pressure.txt'
    )
)
for file in tqdm(files):
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

# Observed data
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

#%% Plot

# Waveform plotting config params
SKIP = 50  # Plot every SKIP stations
POST_ROLL = 8  # [s]
PRE_ROLL = 0.78  # [s] TODO must manually set this so that it doesn't go below topo_ax

TOPO_COLOR = 'silver'

FIGSIZE = (7.17, 10)  # [in.] Figure height is more than we need; we only save a portion
fig, (ax0, ax1, topo_ax1, ax2, topo_ax2) = plt.subplots(
    nrows=5, figsize=FIGSIZE, sharex=True
)

# Plot pressure
extent = [
    (hoz_min - X_SRC) / M_PER_KM,
    (hoz_max - X_SRC) / M_PER_KM,
    (vert_min - Z_SRC) / M_PER_KM,
    (vert_max - Z_SRC) / M_PER_KM,
]
im = ax0.imshow(p_agg, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, extent=extent)

# Plot terrain
ax0.fill_between(
    (terrain_contour[:, 0] - X_SRC) / M_PER_KM,
    -1,
    (terrain_contour[:, 1] - Z_SRC) / M_PER_KM,
    lw=0.5,  # Makes pressure–terrain interface a little smoother-looking
    color=TOPO_COLOR,
)

# Timestamp labels
text = ax0.text(
    0.99,
    0.95,
    ', '.join([f'{timestamp * DT:g}' for timestamp in TIMESTAMPS]) + ' s',
    ha='right',
    va='top',
    transform=ax0.transAxes,
)
text.set_path_effects(
    [path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()]
)

# Axis params
ax0.set_ylabel(f'Elevation relative\nto shot {SHOT} (km)')
ax0.set_xlim(XLIM)
ax0.set_ylim(YLIM)
major_tick_interval = 2  # [km]
minor_tick_interval = 1  # [km[
ax0.xaxis.set_major_locator(plt.MultipleLocator(major_tick_interval))
ax0.xaxis.set_minor_locator(plt.MultipleLocator(minor_tick_interval))
ax0.yaxis.set_major_locator(plt.MultipleLocator(major_tick_interval))
ax0.yaxis.set_minor_locator(plt.MultipleLocator(minor_tick_interval))
ax0.set_aspect('equal')
ax0.tick_params(top=True, right=True, which='both')

# Layout adjustment (note we're making room for the colorbar here!)
y_offset = 0.03
left_pad = 0.04
right_pad = 0.12
fig.tight_layout(
    pad=0.2, rect=(left_pad, y_offset, 1 - left_pad - right_pad, 1 + y_offset)
)

# Colorbar
cax = fig.add_subplot(111)
ax0_pos = ax0.get_position()
cax.set_position([ax0_pos.xmax + 0.1, ax0_pos.ymin, 0.01, ax0_pos.height])
fig.colorbar(
    im, cax=cax, ticks=(im.norm.vmin, 0, im.norm.vmax), label='Normalized pressure'
)

# Form [subsetted] plotting Stream for FAKE data
starttime = st_syn[0].stats.starttime - st_syn[0].stats.t0  # Start at t = 0
st_syn_plot = st_syn.copy().trim(starttime, starttime + 80)[::SKIP]

# Form plotting Stream for REAL data
starttime = get_shots().loc[SHOT].time
st_plot = st.copy().trim(starttime, starttime + 80)


# Helper function to get the onset time for a [synthetic] waveform
def _get_onset_time(tr):
    dist_km = tr.stats.x - X_SRC / M_PER_KM
    if dist_km >= 0:
        return tr.stats.starttime + dist_km / REMOVAL_CELERITY
    else:
        print(f'Removed station with x = {tr.stats.x - X_SRC / M_PER_KM:.2} km')
        return None  # We're on the wrong side of the source


# BIG helper function for plotting wfs
def process_and_plot(st, ax, scale, pre_roll):

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
    include = (xs >= XLIM[0]) & (xs <= XLIM[1])

    # Configure colormap limits from p2p measurements of windowed traces
    cmap = plt.cm.viridis
    p2p_all = p2p_all[include]
    norm = plt.Normalize(vmin=np.min(p2p_all), vmax=np.percentile(p2p_all, 80))

    st = Stream(compress(st, include))
    for tr in st[::-1]:  # Plot the closest waveforms on top!
        tr_plot = tr.copy()
        onset_time = _get_onset_time(tr_plot)
        tr_plot.trim(
            onset_time - pre_roll, onset_time + POST_ROLL, pad=True, fill_value=0
        )
        p2p = tr_plot.data.max() - tr_plot.data.min()  # [Pa]
        data_scaled = tr_plot.copy().normalize().data / (scale / maxes.max())
        ax.plot(
            -1 * data_scaled + tr_plot.stats.x - X_SRC / M_PER_KM,  # Source at x = 0
            tr_plot.times() - pre_roll,
            color=cmap(norm(p2p)),
            clip_on=False,
            solid_capstyle='round',
            lw=0.4,
        )
    return norm, cmap


# Load topography
topo_x, topo_z = terrain_contour.T / M_PER_KM  # [km]
topo_x -= X_SRC / M_PER_KM
topo_z -= Z_SRC / M_PER_KM
mask = (topo_x >= XLIM[0]) & (topo_x <= XLIM[1])  # Since we use clip_on = False later
topo_x = topo_x[mask]
topo_z = topo_z[mask]

for topo_ax in topo_ax1, topo_ax2:
    topo_ax.fill_between(
        topo_x, YLIM[0], topo_z, lw=0.5, color=TOPO_COLOR, clip_on=False
    )
    topo_ax.set_ylim(YLIM[0], 0)  # Axis technically ends at elevation of shot
    topo_ax.set_aspect('equal')
    topo_ax.set_zorder(-5)
    topo_ax.tick_params(left=False, labelleft=False)
    for side in 'right', 'top':
        topo_ax.spines[side].set_visible(False)
topo_ax2.set_xlabel(f'Distance from shot {SHOT} (km)')

norms = []
for ax, st, scale in zip([ax1, ax2], [st_syn_plot, st_plot], [SYN_SCALE, OBS_SCALE]):
    norm, cmap = process_and_plot(st, ax, scale, PRE_ROLL)
    norms.append(norm)
    ax.set_ylim(0, POST_ROLL)
    ax.set_xlim(XLIM)
    for side in 'top', 'right', 'bottom':
        ax.spines[side].set_visible(False)
    ax.tick_params(bottom=False, labelbottom=False, which='both')
    ax.patch.set_alpha(0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))


def _reposition():

    ax0_pos = ax0.get_position()
    y_height = (YLIM[1] / np.diff(YLIM)[0]) * ax0_pos.height
    spacing = 0.025

    ax1_pos = ax1.get_position()
    ax1.set_position(
        [ax0_pos.xmin, ax0_pos.ymin - y_height - spacing, ax1_pos.width, y_height]
    )
    topo_ax1_pos = topo_ax1.get_position()
    topo_ax1.set_position(
        [
            ax0_pos.xmin,
            ax1_pos.ymin - topo_ax1_pos.height,
            topo_ax1_pos.width,
            topo_ax1_pos.height,
        ]
    )
    topo_ax1_pos = topo_ax1.get_position()
    ax2_pos = ax2.get_position()
    ax2.set_position(
        [ax0_pos.xmin, topo_ax1_pos.ymin - y_height - spacing, ax2_pos.width, y_height]
    )
    topo_ax2_pos = topo_ax2.get_position()
    ax2_pos = ax2.get_position()
    topo_ax2.set_position(
        [
            ax0_pos.xmin,
            ax2_pos.ymin - topo_ax2_pos.height,
            topo_ax2_pos.width,
            topo_ax2_pos.height,
        ]
    )


# Needs to be called twice, I guess?
_reposition()
_reposition()

# Colorbar
ax1_pos = ax1.get_position()
topo_ax2_pos = topo_ax2.get_position()
cax_pos = cax.get_position()
position = [
    cax_pos.xmin,
    topo_ax2_pos.ymin,
    cax_pos.width,
    ax1_pos.ymax - topo_ax2_pos.ymin,
]
for norm in norms:
    _cax = fig.add_subplot(111)
    _cax.set_position(position)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=_cax)
    if norm == norms[0]:
        _cax.yaxis.set_ticks_position('left')
        _cax.yaxis.set_label_position('left')
        _cax.set_ylabel('Peak-to-peak pressure (Pa)')
    else:
        _cax.set_ylabel('Peak-to-peak velocity (μm/s)')

# Shared y-axis label
label_ax = fig.add_subplot(111)
label_ax.set_position(
    [topo_ax2_pos.xmin, topo_ax2_pos.ymin, topo_ax2_pos.width, position[-1]]
)
label_ax.patch.set_alpha(0)
for spine in label_ax.spines.values():
    spine.set_visible(False)
label_ax.set_xticks([])
label_ax.set_yticks([])
label_ax.set_ylabel(
    f'Time (s), reduced by {REMOVAL_CELERITY * M_PER_KM:g} m/s', labelpad=15
)

# Add spine to cover wf ends
for topo_ax in topo_ax1, topo_ax2:
    _spine = fig.add_subplot(111)
    _spine.set_position(topo_ax.get_position())
    for side in 'top', 'left', 'right':
        _spine.spines[side].set_visible(False)
    _spine.patch.set_alpha(0)
    _spine.set_xticks([])
    _spine.set_yticks([])

# Plot (a), (b), and (c) tags
for ax, label in zip([ax0, ax1, ax2], ['(a)', '(b)', '(c)']):
    ax.text(
        -0.115,
        1,
        s=label,
        transform=ax.transAxes,
        ha='right',
        va='top',
        weight='bold',
        fontsize=12,
    )

fig.show()

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

if False:
    portion_to_save = 0.45  # Vertical fraction of figure to actually save
    fig.savefig(
        Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve()
        / f'simulation_results_{SHOT.lower()}.png',
        dpi=400,
        bbox_inches=Bbox.from_bounds(
            0,
            (1 - portion_to_save) * FIGSIZE[1],
            FIGSIZE[0],
            FIGSIZE[1] * portion_to_save,
        ),
    )
