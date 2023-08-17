import json
import os
import subprocess
from itertools import compress
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
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
    RUN = '25_shot_y5_pml_240'
    Z_SRC = 734  # [m]
    REMOVAL_CELERITY = 0.342  # [km/s]
elif SHOT == 'X5':
    RUN = '24_shot_x5_pml_240'
    Z_SRC = 464  # [m]
    REMOVAL_CELERITY = 0.336  # [km/s]
else:
    raise ValueError
PRESSURE_SNAPSHOT_DIR = NODAL_WORKING_DIR / 'fdprop' / 'nodal_fdprop_pressure_snapshots'
TIMESTAMPS = np.arange(1000, 18000 + 1000, 1000)  # Anything after 1800 is outside!
SYN_SCALE = 6  # Divide normalized synthetic waveforms by this factor for plotting
OBS_SCALE = 10  # Divide normalized observed waveforms by this factor for plotting
XLIM = (0, 24)  # [km] Relative to shot x-position
YLIM = (-0.5, 5)  # [km] Relative to shot z-position
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


# %% Load in waveform snapshot stuff

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

# %% Load in synthetic and observed data

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

# %% Plot

# Waveform plotting config params
SKIP = 50  # Plot every SKIP stations
START = 45  # Start at this station index (for shifting which set of traces is plotted)
DUR = 80  # [s] Duration of waveforms to cut, intially
PRE_ROLL = -0.9  # [s] Chosen to place t = 0 at the shot elevation for topo!
POST_ROLL = 6  # [s]

TOPO_COLOR = 'silver'

FIGSIZE = (7.17, 10)  # [in.] Figure height is more than we need; we only save a portion
fig, (ax_im, ax0, topo_ax0, ax1, topo_ax1, ax2, topo_ax2) = plt.subplots(
    nrows=7, figsize=FIGSIZE, sharex=True
)

# Plot pressure
extent = [
    (hoz_min - X_SRC) / M_PER_KM,
    (hoz_max - X_SRC) / M_PER_KM,
    (vert_min - Z_SRC) / M_PER_KM,
    (vert_max - Z_SRC) / M_PER_KM,
]
im = ax_im.imshow(p_agg, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, extent=extent)

# Plot terrain
ax_im.fill_between(
    (terrain_contour[:, 0] - X_SRC) / M_PER_KM,
    -1,
    (terrain_contour[:, 1] - Z_SRC) / M_PER_KM,
    lw=0.5,  # Makes pressure–terrain interface a little smoother-looking
    color=TOPO_COLOR,
)

# Timestamp labels (note: we don't bother correcting for t0 here since it's so small —
# in reality the true times are `timestamp * dt + t0`)
text = ax_im.text(
    0.99,
    0.95,
    ', '.join([f'{timestamp * dt:g}' for timestamp in TIMESTAMPS]) + ' s',
    ha='right',
    va='top',
    transform=ax_im.transAxes,
)
text.set_path_effects(
    [path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()]
)

# Axis params
ax_im.set_ylabel(f'Elevation relative\nto shot {SHOT} (km)')
ax_im.set_xlim(XLIM)
ax_im.set_ylim(YLIM)
major_tick_interval = 2  # [km]
minor_tick_interval = 1  # [km[
ax_im.xaxis.set_major_locator(plt.MultipleLocator(major_tick_interval))
ax_im.xaxis.set_minor_locator(plt.MultipleLocator(minor_tick_interval))
ax_im.yaxis.set_major_locator(plt.MultipleLocator(major_tick_interval))
ax_im.yaxis.set_minor_locator(plt.MultipleLocator(minor_tick_interval))
ax_im.set_aspect('equal')
ax_im.tick_params(top=True, right=True, which='both')

# Layout adjustment (note we're making room for the colorbar here!)
fig.tight_layout(pad=0.2, rect=(0.07, 0, 0.84, 1))

# Colorbar
cax = fig.add_subplot(111)
ax_im_pos = ax_im.get_position()
cax.set_position([ax_im_pos.xmax + 0.09, ax_im_pos.ymin, 0.01, ax_im_pos.height])
fig.colorbar(
    im, cax=cax, ticks=(im.norm.vmin, 0, im.norm.vmax), label='Normalized pressure'
)

# Plot transmission loss
d = np.array([tr.stats.x - X_SRC / M_PER_KM for tr in st_syn])  # [km] Dist. from source
mask = (d > XLIM[0] + np.diff(d)[0]) & (d <= XLIM[1])  # So we can use `clip_on=False`
d = d[mask]
peak_amp = np.array([tr.data.max() for tr in st_syn])[mask]  # [Pa] Peak amplitude
d_ref = 24 / M_PER_KM  # [km] TODO: Reference distance
tl = 20 * np.log10(peak_amp / peak_amp[np.isclose(d, d_ref)])
cyl_tl = 20 * np.log10(np.sqrt(d_ref / d))
sph_tl = 20 * np.log10(d_ref / d)
tl[tl > 0] = np.nan
cyl_tl[cyl_tl > 0] = np.nan
sph_tl[sph_tl > 0] = np.nan
line_kw = dict(clip_on=False, lw=1, solid_capstyle='round', dash_capstyle='round')
ref_kw = dict(color='black', alpha=0.5)
ax0.plot(d, tl, color='black', **line_kw)
ax0.plot(d, cyl_tl, linestyle='--', zorder=-2, **line_kw, **ref_kw)
ax0.plot(d, sph_tl, linestyle=':', zorder=-1, **line_kw, **ref_kw)
for label, geo in zip(['Cylindrical', 'Spherical'], [cyl_tl, sph_tl]):
    ax0.text(
        XLIM[1] + 0.2,
        geo[-1],
        label,
        ha='left',
        va='center',
        weight='bold',
        fontsize=8,
        **ref_kw,
    )
ax0.set_ylabel('TL (dB)', labelpad=3)
ax0.set_ylim(-65, 0)
ax0.yaxis.set_minor_locator(plt.MultipleLocator(10))
for side in 'top', 'right':
    ax0.spines[side].set_visible(False)
ax0.patch.set_alpha(0)

# Form [subsetted] plotting Stream for FAKE data
starttime = st_syn[0].stats.starttime - st_syn[0].stats.t0  # Start at t = 0
st_syn_plot = st_syn.copy().trim(starttime, starttime + DUR)[START::SKIP]
assert X_SRC / M_PER_KM in [tr.stats.x for tr in st_syn_plot]  # Check we have wf at src

# Form plotting Stream for REAL data
starttime = get_shots().loc[SHOT].time
st_plot = st.copy().trim(starttime, starttime + DUR)


# Helper function to get the onset time for a waveform
def _get_onset_time(tr):
    dist_km = tr.stats.x - X_SRC / M_PER_KM  # Get distance from SOURCE
    if dist_km > 0:
        return tr.stats.starttime + dist_km / REMOVAL_CELERITY
    else:
        print(f'Removed station with x = {tr.stats.x - X_SRC / M_PER_KM:.2} km')
        return None  # We're on the wrong side of the source, or AT the source


# BIG helper function for plotting wfs
def process_and_plot(st, ax, scale):
    # Make measurements on the windowed traces
    maxes_all = []
    for tr in st:
        tr_measure = tr.copy()
        onset_time = _get_onset_time(tr_measure)
        if onset_time:
            tr_measure.trim(onset_time, onset_time + POST_ROLL)
            maxes_all.append(tr_measure.data.max())
        else:  # No break!
            st.remove(tr)
    maxes_all = np.array(maxes_all)

    # Further subset Stream
    xs = np.array([tr.stats.x for tr in st]) - X_SRC / M_PER_KM  # Set source at x = 0
    include = (xs >= XLIM[0]) & (xs <= XLIM[1])

    # Configure colormap limits from p2p measurements of windowed traces
    cmap = plt.cm.viridis
    maxes_all = maxes_all[include]
    norm = LogNorm(vmin=np.percentile(maxes_all, 20), vmax=np.percentile(maxes_all, 95))

    st = Stream(compress(st, include))
    for tr, mx in zip(st[::-1], maxes_all[::-1]):  # Plot the closest waveforms on top!
        tr_plot = tr.copy()
        onset_time = _get_onset_time(tr_plot)
        tr_plot.trim(
            onset_time + PRE_ROLL, onset_time + POST_ROLL, pad=True, fill_value=0
        )
        data_scaled = tr_plot.copy().normalize().data / scale
        ax.plot(
            -1 * data_scaled + tr_plot.stats.x - X_SRC / M_PER_KM,  # Source at x = 0
            tr_plot.times() + PRE_ROLL,
            color=cmap(norm(mx)),
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

for topo_ax in topo_ax0, topo_ax1, topo_ax2:
    topo_ax.fill_between(
        topo_x, YLIM[0], topo_z, lw=0.5, color=TOPO_COLOR, clip_on=False
    )
    topo_ax.set_ylim(YLIM[0], 0)  # Axis technically ends at elevation of shot
    topo_ax.set_aspect('equal')
    topo_ax.set_zorder(-5)
    topo_ax.axis('off')

norms = []
for ax, st, scale in zip([ax1, ax2], [st_syn_plot, st_plot], [SYN_SCALE, OBS_SCALE]):
    norm, cmap = process_and_plot(st, ax, scale)
    norms.append(norm)
    ax.set_ylim(PRE_ROLL, POST_ROLL)
    ax.set_xlim(XLIM)
    for side in 'top', 'right':
        ax.spines[side].set_visible(False)
    ax.patch.set_alpha(0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))


def _position_ax_below(ax_above, ax_below, height=None, spacing=None):
    ax_above_pos = ax_above.get_position()
    ax_below_pos = ax_below.get_position()
    if height is None:
        height = ax_below_pos.height  # Just use the original height of `ax_below`
    if spacing is None:
        spacing = -height  # Place `ax_below` at same ymin as `ax_above`
    ax_below.set_position(
        [
            ax_above_pos.xmin,  # Horizontally align with `ax_above`
            ax_above_pos.ymin - height - spacing,
            ax_below_pos.width,
            height,
        ]
    )


# Set height and vertical spacing for subplots
height = ax_im.get_position().height  # Panel (a) height
spacing = 0.025

# Panel (b) — TL
_position_ax_below(ax_im, ax0, height=0.5 * height, spacing=spacing)
_position_ax_below(ax0, topo_ax0)
# Panel (c) — Synthetic waveforms
_position_ax_below(topo_ax0, ax1, height=0.7 * height, spacing=spacing)
_position_ax_below(ax1, topo_ax1)
# Panel (d) — Observed waveforms
_position_ax_below(topo_ax1, ax2, height=0.7 * height, spacing=spacing)
_position_ax_below(ax2, topo_ax2)

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
    extend_frac = 0.02
    _cax = fig.add_subplot(111)
    _cax.set_position(position)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=_cax,
        extend='both',
        extendfrac=extend_frac,
    )
    if norm == norms[0]:
        _cax.yaxis.set_ticks_position('left')
        _cax.yaxis.set_label_position('left')
        _cax.set_ylabel('Peak pressure (Pa)')
    else:
        _cax.set_ylabel('Peak velocity (μm/s)')
    pos = _cax.get_position()
    triangle_height = extend_frac * pos.height
    ymin = pos.ymin
    height = pos.height
    ymin -= triangle_height
    height += 2 * triangle_height
    _cax.set_position([pos.xmin, ymin, pos.width, height])

ax2.tick_params(labelbottom=True)  # Hmm...
ax2.set_xlabel(f'Distance from shot {SHOT} (km)')

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
    f'Time (s), reduced by {REMOVAL_CELERITY * M_PER_KM:g} m/s', labelpad=20
)

# Plot subpanel labels
for ax, label in zip([ax_im, ax0, ax1, ax2], ['(a)', '(b)', '(c)', '(d)']):
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
    portion_to_save = 0.47  # Vertical fraction of figure to actually save
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
