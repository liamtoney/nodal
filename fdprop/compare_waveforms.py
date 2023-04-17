import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, Trace
from scipy.fft import fft, fftfreq

from utils import NODAL_WORKING_DIR

# Read in synthetics for several runs (takes a bit!)
st_dict = {}
for run, waveform_snapshot_interval, x_src in zip(
    ['20_shot_y5_new_stf_hf', '22_shot_x5_new_stf_hf'], [5, 5], [1500, 1500]
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

#%% Plot a series of waveforms at different distances

for run in '20_shot_y5_new_stf_hf', '22_shot_x5_new_stf_hf':
    fig, ax = plt.subplots()
    for tr in st_dict[run][2155:2290:10]:
        ax.plot(tr.times(), tr.data, label=f'{tr.stats.x:g} m')
    ax.set_xlim(22, 28)
    ax.set_ylim(-0.25, 0.25)
    ax.legend(loc='upper left')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pressure (Pa)')
    ax.set_title(run)
    ax.grid(linestyle=':')
    ax.set_axisbelow(True)
    fig.show()

#%% Plot time and frequency domain comparison between runs

TARGET_DISTANCE = 4000  # [m]

CELERITY = 338  # [m/s] For estimating first arrival time

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(11, 7))

for run, st in st_dict.items():

    # Select the waveform at this distance
    ind = np.argwhere(np.array([tr.stats.x for tr in st]) == TARGET_DISTANCE)[0][0]
    tr = st[ind]
    assert tr.stats.x == TARGET_DISTANCE

    # Obtain slice of waveform [ostensibly] containing the first peak
    arrival_time = tr.stats.starttime + tr.stats.x / CELERITY
    first_peak_win = arrival_time - 0.5, arrival_time + 1
    tr_first_peak = tr.copy().trim(*first_peak_win)

    # Plot waveforms normalized to the amplitude of their first peak
    ax1.plot(tr.times(), tr.data / tr_first_peak.data.max())

    # Compute and plot spectra (code adapted from Figure0_Source.py)
    spec_win = arrival_time - 5, arrival_time + 10
    tr_spec = tr.copy().trim(*spec_win)  # Avoid waveform "edges"
    yf = fft(tr_spec.data / tr_first_peak.data.max())
    xf = fftfreq(len(tr_spec.times()), tr_spec.stats.delta)[: len(tr_spec.times()) // 2]
    amp = 2.0 / len(tr_spec.times()) * np.abs(yf[0 : len(tr_spec.times()) // 2])
    ax2.plot(xf, amp, label=run)

    # Plot the time windows used for selecting first peak and waveform for spectrum
    ax1.axvspan(
        *[t - tr.stats.starttime for t in first_peak_win],
        color='tab:brown',
        alpha=0.1,
        zorder=-1,
        lw=0,
    )
    ax1.axvspan(
        *[t - tr.stats.starttime for t in spec_win],
        color='gray',
        alpha=0.05,
        zorder=-2,
        lw=0,
    )

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude\n(waveform normalized\nto first peak)')
ax1.set_xlim([t - tr.stats.starttime for t in (spec_win[0] - 2, spec_win[1] + 2)])
ax1.set_title(f'{TARGET_DISTANCE} m')
ax2.set_xlim(0, 20)
ax2.set_ylim(bottom=0)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Amplitude\n(waveform normalized\nto first peak)')
ax2.legend(frameon=False)
fig.tight_layout()
fig.show()
