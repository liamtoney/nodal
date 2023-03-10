import webbrowser

import matplotlib.pyplot as plt
import numpy as np
from obspy.clients.fdsn import Client
from obspy.geodetics.base import gps2dist_azimuth

from utils import get_shots

client = Client('IRIS')

df = get_shots()
# df.sort_values(by='time', inplace=True)  # Sort in chronological order... helpful?

CHANNEL = 'BDF'
RADIUS_DEG = 2  # [deg.] Radius to search around MSH for stations

#%% Make matrix of shot-station distances

# From https://volcano.si.edu/volcano.cfm?vn=321050
MSH_LAT = 46.2
MSH_LON = -122.18

# Open map in browser
url = f'https://ds.iris.edu/gmap/#channel={CHANNEL}&starttime={df.time.min()}&endtime={df.time.max()}&latitude={MSH_LAT}&longitude={MSH_LON}&maxradius={RADIUS_DEG}'
webbrowser.open(url)

inventory = client.get_stations(
    latitude=MSH_LAT,
    longitude=MSH_LON,
    maxradius=RADIUS_DEG,
    channel=CHANNEL,
    level='channel',
    starttime=df.time.min(),  # Start of shots
    endtime=df.time.max(),  # End of shots
)

# Add distances to each shot
dist_all = []
station_names = []
station_MSH_dists = []
for net in inventory:
    for sta in net:
        station_names.append(sta.code)
        station_MSH_dists.append(
            gps2dist_azimuth(sta.latitude, sta.longitude, MSH_LAT, MSH_LON)[0]
        )
        shot_dists = []
        for shot, row in df.iterrows():
            dist_km = (
                gps2dist_azimuth(sta.latitude, sta.longitude, row.lat, row.lon)[0]
                / 1000
            )
            shot_dists.append(dist_km)
        dist_all.append(shot_dists)
dist_all = np.array(dist_all)
station_names = np.array(station_names)

#%% Plot matrix

sorted_idx = np.argsort(station_MSH_dists)  # Sort by distance from MSH

fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(dist_all[sorted_idx, :], cmap='cividis_r', vmax=150)
ax.set_xticks(range(dist_all.shape[1]))
ax.set_yticks(range(dist_all.shape[0]))
ax.set_xticklabels(df.index)
ax.set_yticklabels(station_names[sorted_idx])
ax.set_xlabel('Shot')
ax.set_ylabel(f'Station code')
cbar = fig.colorbar(im, label='Shot-to-station distance (km)')
cbar.ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
nw_code_string = ', '.join([nw.code for nw in inventory])
ax.set_title(
    f'Proximity of {nw_code_string} infrasound stations [within {RADIUS_DEG}° of MSH] to iMUSH shots'
)

fig.tight_layout()
fig.show()

#%% Flatten into DataFrame

import pandas as pd

shot_matrix, station_names_matrix = np.meshgrid(df.index, station_names)

ta_df = pd.DataFrame(
    dict(
        shot=shot_matrix.flatten(),
        station=station_names_matrix.flatten(),
        dist_km=dist_all.flatten(),
    )
)

#%% Unified record section

import matplotlib.patches as patches

MAX_DIST = 40  # [km]

# [Hz]
FREQMIN = 0.5
FREQMAX = 5

# [m/s]
MIN_C = 320
MAX_C = 345

fig, ax = plt.subplots()

for _, row in ta_df[ta_df.dist_km < MAX_DIST].sort_values(by='dist_km').iterrows():

    print(f'{row.shot}–{row.station} ({row.dist_km:.2f} km)')

    # Download waveform
    st = client.get_waveforms(
        network='*',
        station=row.station,
        location='*',
        channel=CHANNEL,
        starttime=df.loc[row.shot].time
        + row.dist_km / (MAX_C / 1000)
        - 10,  # Extra for tapering
        endtime=df.loc[row.shot].time
        + row.dist_km / (MIN_C / 1000)
        + 10,  # Covers spatial extent
        attach_response=True,
    )
    assert st.count() == 1

    # Process waveform
    st.detrend('linear')
    st.remove_response()
    st.taper(0.05)
    st.filter('bandpass', freqmin=FREQMIN, freqmax=FREQMAX, zerophase=True)
    st.normalize()

    t = st[0].times(reftime=df.loc[row.shot].time)
    ax.plot(t, st[0].data + row.dist_km, lw=0.5, color='black', solid_capstyle='round')
    ax.text(
        t[-1],
        row.dist_km,
        rf' $\bf{{{row.shot}}}$–{row.station}',
        ha='left',
        va='center',
    )

# ax.set_xlim(left=0)
# ax.set_ylim(bottom=0)

# Hard-coded for the 40 km max dist. case
ax.set_xlim(0, 140)
ax.set_ylim(0, 40)

# Add celerity colors
inc = 0.5  # [m/s]
celerity_list = np.arange(MIN_C, MAX_C + inc, inc)
cmap = plt.cm.get_cmap('cet_rainbow', len(celerity_list))
colors = [cmap(i) for i in range(cmap.N)]
xlim = np.array(ax.get_xlim())
ylim = np.array(ax.get_ylim())
for celerity, color in zip(celerity_list, colors):
    ax.plot(
        xlim,
        xlim * celerity / 1000,
        label=f'{celerity:g}',
        color=color,
        zorder=-2,
        lw=0.5,
    )
ax.set_xlim(xlim)
ax.set_ylim(ylim)  # Scale y-axis to pre-plotting extent
cel_alpha = 0.5
rect = patches.Rectangle(
    (xlim[0], ylim[0]),
    np.diff(xlim)[0],
    np.diff(ylim)[0],
    edgecolor='none',
    facecolor='white',
    alpha=cel_alpha,
    zorder=-1,
)
ax.add_patch(rect)
mapper = plt.cm.ScalarMappable(cmap=cmap)
mapper.set_array(celerity_list)
cbar = fig.colorbar(mapper, ax=ax, label='Celerity (m/s)', aspect=30, pad=0.07)
cbar.solids.set_alpha(1 - cel_alpha)

ax.set_xlabel('Time from shot (s)')
ax.set_ylabel('Distance from shot (km)')

for side in 'top', 'right':
    ax.spines[side].set_visible(False)

fig.show()

#%% Investigate the nice arrivals

from multitaper import MTSpec  # pip install multitaper
from obspy import Stream
from python_util import plot
from scipy.fftpack import next_fast_len
from scipy.signal import welch, windows

CELERITY = 343  # [m/s]

PLOT_NOISE = False

noise = -120 if PLOT_NOISE else 0  # Noise window offset to before signal arrival

# These are the nice 5 arrivals
SHOT_STATION_PAIRS = [
    ['Y1', 'F05D'],
    ['Y7', 'E04D'],
    ['Y6', 'E04D'],
    # ['X2', 'F04D'],  # NOISIER THAN OTHER ONES!
    ['Y8', 'E04D'],
]

good_st = Stream()
bad_st = Stream()

for (shot, station) in SHOT_STATION_PAIRS:

    dist_km = ta_df[(ta_df.station == station) & (ta_df.shot == shot)].dist_km.values[0]

    # Download waveform
    kwargs = dict(
        network='*',
        station=station,
        location='*',
        channel=CHANNEL,
        attach_response=True,
    )
    arr_time = df.loc[shot].time + dist_km / (CELERITY / 1000)
    sig_win = (arr_time - 60, arr_time + 60 + 5)
    st_sig = client.get_waveforms(starttime=sig_win[0], endtime=sig_win[1], **kwargs)
    st_noise = client.get_waveforms(
        starttime=sig_win[0] - 120, endtime=sig_win[1] - 120, **kwargs
    )
    assert st_sig.count() == 1
    assert st_noise.count() == 1

    st_sig[0].stats.network = shot  # For record-keeping
    st_noise[0].stats.network = shot  # For record-keeping

    # Process waveform
    for st in st_sig, st_noise:
        st.detrend('linear')
        st.remove_response()

    good_st += st_sig[0]
    bad_st += st_noise[0]

    # Plot
    fig = plot.spec(good_st[0])
    fig.axes[0].set_title(f'{shot}–{station} ({dist_km:.2f} km)')

#%% Multitapers

pxx_dict = dict(signal=[], noise=[])
for st, k in zip([good_st, bad_st], pxx_dict.keys()):

    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    for tr in st:

        if False:
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
        else:
            win_dur = 50  # [s]
            nperseg = int(win_dur * tr.stats.sampling_rate)  # Samples
            nfft = np.power(2, int(np.ceil(np.log2(nperseg))) + 1)  # Pad FFT

            f, pxx = welch(tr.data, tr.stats.sampling_rate, nperseg=nperseg, nfft=nfft)

        # Remove DC component to avoid divide by zero errors later
        f = f[1:]
        pxx = pxx[1:]

        ref_val = 20e-6
        pxx_db = 10 * np.log10(
            pxx / (ref_val**2)
        )  # [dB rel. (ref_val <ref_val_unit>)^2 Hz^-1]

        if True:
            win = windows.hann(20)  # Use Hann window
            pxx_db = np.convolve(pxx_db, win, mode='same') / sum(win)

        shot = tr.stats.network
        station = tr.stats.station
        ax.semilogx(f, pxx_db, label=rf'$\bf{{{shot}}}$–{station}')

        pxx_dict[k].append(pxx_db)

    ax.set_xlim(0.5, 10)
    ax.set_ylim(20, 80)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dB)')

    ax.legend()

    fig.show()

#%% SNR

fig, ax = plt.subplots(figsize=(8.6, 4.8))

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

ymax = 30

for i, (shot, station) in enumerate(SHOT_STATION_PAIRS):

    color = colors[i]

    snr = pxx_dict['signal'][i] - pxx_dict['noise'][i]

    ax.semilogx(f, snr, color=color, label=rf'$\bf{{{shot}}}$–{station}')

    peak_freq = f[np.argmax(snr)]

    ax.axvline(peak_freq, color=color, linestyle=':')

    ax.text(
        peak_freq,
        ymax,
        f'  {peak_freq:.1f} Hz',
        ha='center',
        va='bottom',
        rotation=90,
        color=color,
    )

    ax.set_xlim(0.5, 10)
    ax.set_ylim(top=ymax)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('SNR (dB)')

ax.legend()

fig.show()

#%% Examine coupling on TA stations

from urllib.request import urlopen

# Load xcorr_coh() function from David's GitHub repo
URL = 'https://raw.githubusercontent.com/davidfee5/seismoacoustic/master/xcorr_coh.py?token=GHSAT0AAAAAABXVS5XXMGIC2JCEUG6RU6ROZAKNRUQ'
exec(urlopen(URL).read())

# xcorr_coh() parameters
WINLEN = 10  # [s] Data window length
OVERLAP = 0.9  # Data window overlap
NPER = 64  # Number of points in FFT segment
NPEROVER = 0.8  # FFT overlap(?)
FILT = [2, 15]  # [Hz] Bandpass corners
SHIFTSEC = 0.2  # [s] Amount xcorr can shift

for (shot, station) in SHOT_STATION_PAIRS:

    dist_km = ta_df[(ta_df.station == station) & (ta_df.shot == shot)].dist_km.values[0]

    # Download waveform
    kwargs = dict(
        network='*',
        station=station,
        location='*',
        channel='BDF,BHZ',
        attach_response=True,
    )
    arr_time = df.loc[shot].time + dist_km / (CELERITY / 1000)
    sig_win = (arr_time - 60, arr_time + 60 + 5)
    st = client.get_waveforms(starttime=sig_win[0], endtime=sig_win[1], **kwargs)
    assert st.count() == 2  # Infra and seismic!
    st.sort(keys=['channel'])  # For proper ordered input to xcorr_coh()

    # Process waveforms
    for tr in st:
        tr.detrend('linear')
        tr.remove_response()

    # Run xcorr_coh()
    fig = xcorr_coh(st, WINLEN, OVERLAP, NPER, NPEROVER, FILT, SHIFTSEC, 1)
    fig.axes[0].set_title(f'{shot}–{station} ({dist_km:.2f} km)')
    fig.savefig(f'/Users/ldtoney/Downloads/{shot}-{station}.png', bbox_inches='tight')

#%% Plot close-up of best coupled arrival and obtain "scale factor"

# See Novoselov (2020), §4.3

import matplotlib.dates as mdates
from scipy.signal import hilbert

SHOT = 'Y6'
STATION = 'E04D'

dist_km = ta_df[(ta_df.station == STATION) & (ta_df.shot == SHOT)].dist_km.values[0]

# Download waveform
kwargs = dict(
    network='*',
    station=STATION,
    location='*',
    channel='BDF,BHZ',
    attach_response=True,
)
arr_time = df.loc[shot].time + dist_km / (CELERITY / 1000)
pad = 4
sig_win = (arr_time - pad, arr_time + pad + 5)
st = client.get_waveforms(starttime=sig_win[0], endtime=sig_win[1], **kwargs)
assert st.count() == 2  # Infra and seismic!

# Process waveforms
st.detrend('linear')
st.remove_response()
stf = st.copy()
stf.taper(0.05)
fmin = 8
fmax = 10
stf.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True)  # Narrow-band here!

# Plot
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
for tr, ax in zip(stf, (ax1, ax2)):
    ax.plot(tr.times('matplotlib'), tr.data, color='black', lw=0.5)
ax1.set_ylabel('Velocity (m/s)')
ax2.set_ylabel('Pressure (Pa)')
loc = ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
for ax in (ax1, ax2):
    ax.autoscale(enable=True, axis='x', tight=True)
fig.show()

# Get transfer coeff
stfe = stf.copy()
for tr in stfe:
    npts = tr.count()
    # The below line is much faster than using obspy.signal.envelope()
    # See https://github.com/scipy/scipy/issues/6324#issuecomment-425752155
    tr.data = np.abs(hilbert(tr.data, N=next_fast_len(npts))[:npts])
for tr, ax in zip(stfe, (ax1, ax2)):
    ax.plot(tr.times('matplotlib'), tr.data, color='red')
fig.show()

# The coeff!
cas = stfe[0].data.max() / stfe[1].data.max()  # [(m/s) / Pa]
print(f'C_AS = {cas:.10f} m s^-1 Pa^-1 @ {np.mean([fmin, fmax]):g} Hz')

ax2.plot(stf[0].times('matplotlib'), stf[0].data / cas, color='red', lw=0.5)
fig.show()
