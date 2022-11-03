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
SKIP = 50  # Plot every SKIP stations
SCALE = 0.005  # [Pa] Single scale factor
MIN_TIME = 0  # [s]
MAX_TIME = 25  # [s]
MIN_DIST = 1  # [km]
MAX_DIST = 7  # [km]

# Hacky params
MIN_PEAK_PRESSURE = 0.5e-4  # [Pa] Don't plot signals w/ peak pressures less than this
X_SRC = 500  # [m] TODO from make_main.py

# Form plotting Stream
starttime = st[0].stats.starttime - st[0].stats.t0  # Start at t = 0
st_plot = st.copy().trim(starttime + MIN_TIME, starttime + MAX_TIME)

# Edit and remove traces not meeting criteria
xs = np.array([tr.stats.x for tr in st_plot]) - X_SRC / M_PER_KM  # Set source at x = 0
maxes = np.array([tr.data.max() for tr in st_plot])
include = (xs >= MIN_DIST) & (xs <= MAX_DIST) & (maxes >= MIN_PEAK_PRESSURE)
st_plot = Stream(compress(st_plot, include))[::SKIP]

# Define colormap normalized to waveform peak-to-peak amplitudes
cmap = plt.cm.viridis
p2p_all = np.array([tr.data.max() - tr.data.min() for tr in st_plot]) * 1e6  # [μPa]
norm = plt.Normalize(vmin=np.min(p2p_all), vmax=np.max(p2p_all))

# Make plot
fig, ax = plt.subplots(figsize=(7, 10))
for tr in st_plot[::-1]:  # Plot the closest waveforms on top!
    p2p = (tr.data.max() - tr.data.min()) * 1e6  # [μPa]
    ax.plot(
        tr.times() + MIN_TIME,
        (tr.data / SCALE) + tr.stats.x - X_SRC / M_PER_KM,  # Set source at x = 0
        color=cmap(norm(p2p)),
        clip_on=False,
        solid_capstyle='round',
    )
ax.set_xlim(MIN_TIME, MAX_TIME)
ax.set_ylim(MIN_DIST, MAX_DIST)
ax.set_xlabel('Time from "shot" (s)')
ax.set_ylabel('Distance from "shot" (km)')
cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap), location='top', aspect=40
)
cbar.set_label('Peak-to-peak pressure (μPa)', labelpad=10)
for side in 'top', 'right':
    ax.spines[side].set_visible(False)
ax.spines['left'].set_position(('outward', 20))
ax.spines['bottom'].set_position(('outward', 30))
fig.show()
