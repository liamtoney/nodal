import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, Trace

from utils import NODAL_WORKING_DIR

SNAPSHOT_INTERVAL = 5  # TODO this should be stored + read from somewhere!

run = '04_back_to_lf'
dir0 = NODAL_WORKING_DIR / 'fdprop' / 'nodal_fdprop_runs' / run / 'OUTPUT_FILES'
process = 0  # Which process domain to read waveforms from

file = dir0 / f'process{process}_waveforms_pressure.txt'

# Read in params from file
dt, t0 = np.loadtxt(file, max_rows=2, delimiter=' = ', usecols=1)  # [s]
x_locs = np.loadtxt(file, skiprows=2, max_rows=1)  # [m]
traces = np.loadtxt(file, skiprows=3).T  # [Pa] Each row is a waveform!

interval = dt * SNAPSHOT_INTERVAL  # [s] True sampling interval of waveforms

# Make ObsPy Stream
st = Stream()
for trace in traces:
    tr = Trace(data=trace, header=dict(sampling_rate=1 / interval))
    tr.stats.starttime += t0  # Shift for t0
    st += tr

#%% Plot

# Plotting config params
SKIP = 5  # Plot every SKIP stations
SCALE = 0.0001  # [Pa] Single scale factor
MAX_TIME = 8  # [s]
MIN_DIST = 100  # [m]
MAX_DIST = 700  # [m]

# Hacky params
MIN_PEAK_PRESSURE = 1e-4  # [Pa] Don't plot waveforms w/ peak pressures less than this
X_SRC = 500  # [m] TODO this should be stored + read from somewhere!

# Form plotting Stream
starttime = st[0].stats.starttime - t0  # Start at t = 0
st_plot = st.copy().trim(starttime, starttime + MAX_TIME)  # Since it blows up later

x_locs_src = x_locs - X_SRC  # [m] Adjust for source location!

# Define colormap normalized to waveforms within the space, time, and min pressure
cmap = plt.cm.viridis
peak_p = np.array([tr.data.max() for tr in st_plot])[
    (x_locs_src >= MIN_DIST) & (x_locs_src <= MAX_DIST)
]
peak_p = peak_p[peak_p > MIN_PEAK_PRESSURE]
norm = plt.Normalize(vmin=np.min(peak_p), vmax=np.max(peak_p))

# Make plot
fig, ax = plt.subplots()
for tr, xloc in zip(st_plot[::SKIP], x_locs_src[::SKIP]):
    if (tr.data.max() > MIN_PEAK_PRESSURE) & (xloc >= MIN_DIST) & (xloc <= MAX_DIST):
        max_p = abs(tr.data.max())
        ax.plot(tr.times(), (tr.data / SCALE) + xloc, color=cmap(norm(max_p)))
ax.set_xlim(0, MAX_TIME)
ax.set_ylim(MIN_DIST, MAX_DIST)
ax.set_xlabel('Time from "shot" (s)')
ax.set_ylabel('Distance from "shot" (m)')
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Peak pressure (Pa)')
fig.show()
