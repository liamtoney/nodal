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

#%% Plot waveform for a shot

SHOT = 'Y6'
STATION = 'E04D'

FREQMIN = 0.5
FREQMAX = 5

CELERITY = 340  # [m/s] For marking predicted arrival

# Download waveform
st = client.get_waveforms(
    network='*',
    station=STATION,
    location='*',
    channel=CHANNEL,
    starttime=df.loc[SHOT].time - 100,  # Extra for tapering
    endtime=df.loc[SHOT].time + 500,  # Covers spatial extent
    attach_response=True,
)

# Calculate distance from shot to station
coords = inventory.get_coordinates(st[0].id)
dist_m = gps2dist_azimuth(
    df.loc[SHOT].lat, df.loc[SHOT].lon, coords['latitude'], coords['longitude']
)[0]

# Process waveform
st.detrend('linear')
st.remove_response()
stf = st.copy()
stf.taper(0.05)
stf.filter('bandpass', freqmin=FREQMIN, freqmax=FREQMAX, zerophase=True)

# Make waveform plot w/ predicted arrival marked
fig = plt.figure()
stf.plot(fig=fig, method='full')
ax = fig.axes[0]
pred_arr_time = df.loc[SHOT].time + (dist_m / CELERITY)
ax.axvline(
    pred_arr_time.matplotlib_date,
    color='red',
    zorder=-5,
    linewidth=2,
    label=f'Predicted arrival time\n({CELERITY} m/s celerity)',
)
ax.set_title(f'Shot {SHOT}, {FREQMIN}–{FREQMAX} Hz bandpass')
ax.set_ylabel('Pressure (Pa)')
ax.set_xlim(df.loc[SHOT].time.matplotlib_date, (pred_arr_time + 30).matplotlib_date)
ax.legend(loc='lower left')
fig.show()

# fig.savefig(f'/Users/ldtoney/Downloads/{SHOT}-{STATION}.png', dpi=300, bbox_inches='tight')
