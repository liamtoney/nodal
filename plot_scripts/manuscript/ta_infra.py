import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from multitaper import MTSpec
from obspy import Stream
from obspy.clients.fdsn import Client
from obspy.geodetics.base import gps2dist_azimuth
from scipy.fftpack import next_fast_len

from utils import get_shots

FONT_SIZE = 10  # [pt]
plt.rcParams.update({'font.size': FONT_SIZE})

M_PER_KM = 1000  # [m/km]

# Ordered by increasing distance from shot
SHOT_STATION_PAIRS = (['Y1', 'F05D'], ['Y7', 'E04D'], ['Y6', 'E04D'], ['Y8', 'E04D'])

CELERITY = 340  # [m/s] For windowing acoustic arrival
DATA_WIN = 60  # [s] How much data to compute the PSD for

# Where data and metadata come from
client = Client('IRIS')

# Load in shot info
df = get_shots()

# Get station locations
unique_stations = np.unique([pair[1] for pair in SHOT_STATION_PAIRS])
inv = client.get_stations(network='TA', station=','.join(unique_stations))
station_coords = {}
for net in inv:
    for sta in net:
        station_coords[sta.code] = (sta.latitude, sta.longitude)
assert sorted(station_coords.keys()) == sorted(unique_stations)  # Easy check

# Get data
st_signal = Stream()
st_noise = Stream()
for shot_name, station in SHOT_STATION_PAIRS:
    shot = df.loc[shot_name]
    dist_m = gps2dist_azimuth(*station_coords[station], shot.lat, shot.lon)[0]
    arr_time = shot.time + dist_m / CELERITY
    signal_win = dict(
        starttime=arr_time - DATA_WIN / 2, endtime=arr_time + DATA_WIN / 2
    )
    noise_win = dict(
        starttime=signal_win['starttime'] - DATA_WIN, endtime=signal_win['starttime']
    )
    kwargs = dict(
        network='TA', station=station, channel='BDF', location='*', attach_response=True
    )
    st_tmp_signal = client.get_waveforms(**signal_win, **kwargs)
    st_tmp_noise = client.get_waveforms(**noise_win, **kwargs)
    for tr in st_tmp_signal + st_tmp_noise:
        tr.stats.shot = shot_name
        tr.stats.distance = dist_m
    st_signal += st_tmp_signal
    st_noise += st_tmp_noise
for st in st_signal, st_noise:
    assert st.count() == len(SHOT_STATION_PAIRS)  # Easy check
    st.detrend('linear')
    st.taper(0.05)
    st.remove_response()

#%% Calculate multitapers and plot

P_REF = 20e-6  # [Pa]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][: len(SHOT_STATION_PAIRS)]
signal_plot_kw = dict(linestyle=None, alpha=1)
noise_plot_kw = dict(linestyle=':', alpha=0.5)

# Axis setup
fig, axs = plt.subplots(
    ncols=2,
    nrows=len(SHOT_STATION_PAIRS),
    sharex=True,
    sharey=True,
    figsize=(7.17, 3.2),
)
gs = axs[0, -1].get_gridspec()  # Top-right corner
spec_ax = fig.add_subplot(gs[:, -1])  # Right-most column
for ax in axs[:, -1]:
    ax.remove()  # Remove the subplots in the right-most column

for st, plot_kwargs in zip([st_signal, st_noise], [signal_plot_kw, noise_plot_kw]):

    # Calculate multitapers and plot lines
    for tr, color in zip(st, colors):
        mtspec = MTSpec(
            tr.data,
            nw=4,  # Time-bandwidth product
            kspec=10,  # Number of tapers (after a certain point this saturates)
            dt=tr.stats.delta,
            nfft=next_fast_len(tr.stats.npts),
        )
        f, pxx = mtspec.rspec()
        f = f.squeeze()
        pxx = pxx.squeeze()

        # Remove DC component to avoid divide by zero errors later
        f = f[1:]
        pxx = pxx[1:]

        # Convert to dB
        pxx_db = 10 * np.log10(pxx / (P_REF**2))

        # Plot
        spec_ax.semilogx(f, pxx_db, color=color, **plot_kwargs)

for tr, ax, color in zip(st_signal, axs[:, 0], colors):
    reduced_time = (
        tr.times(reftime=df.loc[tr.stats.shot].time) - tr.stats.distance / CELERITY
    )
    ax.plot(reduced_time, tr.data, color=color)
    ax.text(
        0.01,
        0.94,
        rf'$\bf{{{tr.stats.shot}}}$–{tr.stats.station}',
        transform=ax.transAxes,
        ha='left',
        va='top',
    )
    ax.text(
        0.01,
        0.02,
        f'{tr.stats.distance / M_PER_KM:.1f} km',
        transform=ax.transAxes,
        ha='left',
        va='bottom',
    )
    ax.set_ylim(-1, 1)

spec_ax.set_xlim(0.5, 10)
spec_ax.set_ylim(20, 70)

spec_ax.set_xlabel('Frequency (Hz)')
spec_ax.set_ylabel(f'Power (dB rel. [{P_REF * 1e6:g} μPa]$^2$ Hz$^{{-1}}$)')
spec_ax.yaxis.set_label_position(position='right')
spec_ax.yaxis.tick_right()

axs[-1, 0].set_xlabel(f'Time from shot (s), reduced by {CELERITY} m/s')
axs[-1, 0].set_xlim(-5, 5)
axs[-1, 0].xaxis.set_minor_locator(plt.MultipleLocator(1))
label_ax = fig.add_subplot(gs[:, 0])
label_ax.patch.set_alpha(0)
label_ax.set_xticks([])
label_ax.set_yticks([])
for spine in label_ax.spines.values():
    spine.set_visible(False)
label_ax.set_ylabel('Pressure (Pa)', labelpad=20)

# Make dummy legend
spec_ax.plot(np.nan, np.nan, color='black', label='Signal window', **signal_plot_kw)
spec_ax.plot(np.nan, np.nan, color='black', label='Noise window', **noise_plot_kw)
spec_ax.legend(frameon=False)

fig.tight_layout(pad=0.2, rect=(-0.035, 0, 1, 1))
fig.subplots_adjust(wspace=0.05)

fig.show()

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

# fig.savefig(Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve() / 'ta_infra.png', dpi=600)
