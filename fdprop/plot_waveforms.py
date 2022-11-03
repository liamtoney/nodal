from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, Trace

from utils import NODAL_WORKING_DIR

# CHANGE ME!
run = '04_back_to_lf'
dir0 = NODAL_WORKING_DIR / 'fdprop' / 'nodal_fdprop_runs' / run / 'OUTPUT_FILES'
WAVEFORM_SNAPSHOT_INTERVAL = 5  # TODO from make_main.py

M_PER_KM = 1000  # [m/km] CONSTANT

# Read in files to an ObsPy Stream
st = Stream()
for file in dir0.glob('process*_waveforms_pressure.txt'):

    # Read in params from file
    dt, t0 = np.loadtxt(file, max_rows=2, delimiter=' = ', usecols=1)  # [s]
    x_locs = np.loadtxt(file, skiprows=2, max_rows=1) / M_PER_KM  # [km]
    traces = np.loadtxt(file, skiprows=3).T  # [Pa] Each row is a waveform!

    interval = dt * WAVEFORM_SNAPSHOT_INTERVAL  # [s] True sampling interval of data

    # Add to Stream
    for trace, x in zip(traces, x_locs):
        tr = Trace(data=trace, header=dict(sampling_rate=1 / interval))
        tr.stats.starttime += t0  # Shift for t0
        tr.stats.t0 = t0  # [s]
        tr.stats.x = x  # [km]
        st += tr

st.sort(keys=['x'])  # Sort by increasing x distance

#%% Plot

# Plotting config params
SKIP = 100  # Plot every SKIP stations
SCALE = 0.005  # [Pa] Single scale factor
MIN_TIME, MAX_TIME = 0, 70  # [s]
MIN_DIST, MAX_DIST = 0, 25  # [km]
PRE_ROLL = 2  # [s]
POST_ROLL = 18  # [s]

# Hacky params
MIN_PEAK_PRESSURE = 1e-5  # [Pa] Don't plot signals w/ peak pressures less than this
X_SRC = 500  # [m] TODO from make_main.py

# Form subsetted plotting Stream
starttime = st[0].stats.starttime - st[0].stats.t0  # Start at t = 0
st_plot = st.copy().trim(starttime + MIN_TIME, starttime + MAX_TIME)[::SKIP]

# Make measurements on the windowed traces
maxes = []
p2p_all = []
for tr in st_plot:
    tr_measure = tr.copy()
    first_ind = np.argwhere(tr_measure.data)[0][0]  # Find index of first non-zero value
    onset_time = tr_measure.times('UTCDateTime')[first_ind]
    tr_measure.trim(onset_time - PRE_ROLL, onset_time + POST_ROLL)
    maxes.append(tr_measure.data.max())
    p2p_all.append(tr_measure.data.max() - tr_measure.data.min() * 1e6)  # [μPa]
maxes = np.array(maxes)
p2p_all = np.array(p2p_all)

# Further subset Stream
xs = np.array([tr.stats.x for tr in st_plot]) - X_SRC / M_PER_KM  # Set source at x = 0
include = (xs >= MIN_DIST) & (xs <= MAX_DIST) & (maxes >= MIN_PEAK_PRESSURE)
st_plot = Stream(compress(st_plot, include))

# Configure colormap limits from p2p measurements of windowed traces
cmap = plt.cm.viridis
p2p_all = p2p_all[include]
norm = plt.Normalize(vmin=np.min(p2p_all), vmax=np.max(p2p_all))

# Make plot
fig, ax = plt.subplots(figsize=(7, 10))
for tr in st_plot[::-1]:  # Plot the closest waveforms on top!
    tr_plot = tr.copy()
    first_ind = np.argwhere(tr_plot.data)[0][0]  # Find index of first non-zero value
    onset_time = tr_plot.times('UTCDateTime')[first_ind]
    starttime = np.max([onset_time - PRE_ROLL, tr_plot.stats.starttime])  # For nearest
    tr_plot.trim(starttime, onset_time + POST_ROLL)
    p2p = (tr_plot.data.max() - tr_plot.data.min()) * 1e6  # [μPa]
    ax.plot(
        tr_plot.times() + MIN_TIME + (starttime - tr.stats.starttime),  # CAREFUL!
        (tr_plot.data / SCALE) + tr_plot.stats.x - X_SRC / M_PER_KM,  # Source at x = 0
        color=cmap(norm(p2p)),
        clip_on=False,
        solid_capstyle='round',
    )
ax.set_xlim(MIN_TIME, MAX_TIME)
ax.set_ylim(MIN_DIST, MAX_DIST)
ax.set_xlabel('Time from "shot" (s)')
ax.set_ylabel('Distance from "shot" (km)')
cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap), location='top', aspect=40, pad=0.02
)
cbar.set_label('Peak-to-peak pressure (μPa)', labelpad=10)
for side in 'top', 'right':
    ax.spines[side].set_visible(False)
ax.spines['left'].set_position(('outward', 15))
ax.spines['bottom'].set_position(('outward', 20))
fig.show()
