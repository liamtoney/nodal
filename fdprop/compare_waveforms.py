# DRAFT CODE!

import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, Trace

from utils import NODAL_WORKING_DIR

# Read in synthetics for several runs
st_dict = {}
for run, waveform_snapshot_interval, x_src in zip(
    ['11_shot_y5_hf', '13_shot_y5_hf_smaller_dt', '14_shot_y5_smaller_dx'],
    [5, 5, 5],
    [500, 500, 500],
):
    st = Stream()
    for file in (
        NODAL_WORKING_DIR / 'fdprop' / 'nodal_fdprop_runs' / run / 'OUTPUT_FILES'
    ).glob('process*_waveforms_pressure.txt'):
        # Read in params from file
        dt, t0 = np.loadtxt(file, max_rows=2, usecols=2)  # [s]
        x_locs = (
            np.loadtxt(file, skiprows=2, max_rows=1) - x_src
        )  # [m] Distance from SOURCE!
        # Row 4 is elevation, which we skip
        traces = np.loadtxt(file, skiprows=4).T  # [Pa] Each row is a waveform!
        delta = dt * waveform_snapshot_interval  # [s] True sampling interval of data
        # Add to Stream
        for trace, x in zip(traces, x_locs):
            tr = Trace(data=trace, header=dict(sampling_rate=1 / delta))
            tr.stats.starttime += t0  # Shift for t0
            tr.stats.t0 = t0  # [s]
            tr.stats.x = x  # [m]
            st += tr
    st.sort(keys=['x'])  # Sort by increasing x distance
    st = st[::2]  # IMPORTANT: Keep only EVEN indices (0, 2, 4, ...)
    st_dict[run] = st

#%% Plot time domain comparison

TARGET_DISTANCE = 4000  # [m]

CELERITY = 343  # [m/s] For estimating first arrival time

fig, ax = plt.subplots(figsize=(11, 4))
for run, st in st_dict.items():
    ind = np.argwhere(np.array([tr.stats.x for tr in st]) == TARGET_DISTANCE)[0][0]
    tr = st[ind]
    assert tr.stats.x == TARGET_DISTANCE
    arrival_time = tr.stats.starttime + tr.stats.x / CELERITY
    # [s] Trim to time window to compute amp for normalization
    tr_first_arrival = tr.copy().trim(arrival_time - 0.5, arrival_time + 0.5)
    ax.plot(tr.times(), tr.data / tr_first_arrival.data.max(), label=run)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude normalized to first peak')  # MANUALLY VERIFY THIS ON PLOT!
ax.set_title(f'{TARGET_DISTANCE} m')
ax.legend()
fig.tight_layout()
fig.show()

#%% Plot spectra comparison

from scipy.fft import fft, fftfreq

fig, ax = plt.subplots()
for run, st in st_dict.items():
    ind = np.argwhere(np.array([tr.stats.x for tr in st]) == TARGET_DISTANCE)[0][0]
    tr = st[ind]
    assert tr.stats.x == TARGET_DISTANCE
    arrival_time = tr.stats.starttime + tr.stats.x / CELERITY
    # [s] Trim to time window to compute amp for normalization
    tr_first_arrival = tr.copy().trim(arrival_time - 0.5, arrival_time + 0.5)
    tr_spec = tr.copy().trim(tr.stats.starttime + 10, tr.stats.starttime + 20)
    yf = fft(tr_spec.data / tr_first_arrival.data.max())
    xf = fftfreq(len(tr_spec.times()), tr_spec.stats.delta)[: len(tr_spec.times()) // 2]
    amp = 2.0 / len(tr_spec.times()) * np.abs(yf[0 : len(tr_spec.times()) // 2])
    ax.plot(xf, amp, label=run)
ax.set_xlim(0, 20)
ax.set_ylim(bottom=0)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Normalized amplitude')
ax.set_title(f'{TARGET_DISTANCE} m')
ax.legend()
fig.tight_layout()
fig.show()
