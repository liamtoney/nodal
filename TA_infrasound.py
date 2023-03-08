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

from obspy import Stream
from python_util import plot

CELERITY = 343  # [m/s]

PLOT_NOISE = False

noise = -120 if PLOT_NOISE else 0  # Noise window offset to before signal arrival

# These are the nice 5 arrivals
SHOT_NODE_PAIRS = [
    ['Y1', 'F05D'],
    ['Y7', 'E04D'],
    ['Y6', 'E04D'],
    # ['X2', 'F04D'],  # NOISIER THAN OTHER ONES!
    ['Y8', 'E04D'],
]

good_st = Stream()

for (shot, station) in SHOT_NODE_PAIRS:

    dist_km = ta_df[(ta_df.station == station) & (ta_df.shot == shot)].dist_km.values[0]

    # Download waveform
    st = client.get_waveforms(
        network='*',
        station=station,
        location='*',
        channel=CHANNEL,
        starttime=df.loc[shot].time + dist_km / (CELERITY / 1000) - 60 + noise,
        endtime=df.loc[shot].time + dist_km / (CELERITY / 1000) + 60 + 5 + noise,
        attach_response=True,
    )
    assert st.count() == 1

    st[0].stats.network = shot  # For record-keeping

    # Process waveform
    st.detrend('linear')
    st.remove_response()

    good_st += st[0]

    # Plot
    fig = plot.spec(st[0])
    fig.axes[0].set_title(f'{shot}–{station} ({dist_km:.2f} km)')

fig = plot.psd(good_st, win_dur=20, db_lim=(20, 80))
plt.gcf().axes[0].set_xlim(0.5, 10)
