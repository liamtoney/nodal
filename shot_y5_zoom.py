import webbrowser
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import pygmt
from matplotlib.colors import to_hex
from obspy import Stream
from obspy.geodetics.base import gps2dist_azimuth

from utils import get_shots, get_stations, get_waveforms_shot

M_PER_KM = 1000  # [m/km] CONSTANT

REMOVAL_CELERITY = 0.343  # [km/s] For reduced time

REGION = [-122.254, -122.2, 46.285, 46.31]  # [xmin, xmax, ymin, ymax] Ridge/valley area

# Hard-coded lists of sensors in the above REGION (ordered from W to E)
RIDGE_SENSORS = [4960, 4961, 4962, 4963, 4964, 4965, 4966, 4967]
VALLEY_SENSORS = [4704, 4981, 4994, 4980, 4995, 4979, 4996, 4978, 4997]

#%% Open IRIS gmap station map

URL = f'https://ds.iris.edu/gmap/#net=1D&minlon={REGION[0]}&maxlon={REGION[1]}&minlat={REGION[2]}&maxlat={REGION[3]}&drawingmode=box'
_ = webbrowser.open(URL)

#%% PyGMT map

# Get stations, data, shots
inv = get_stations()
st = get_waveforms_shot('Y5')
shot = get_shots().loc['Y5']

# Assign coordinates and distance [km] to traces
for tr in st:
    coords = inv.get_coordinates(tr.id)
    tr.stats.latitude = coords['latitude']
    tr.stats.longitude = coords['longitude']
    tr.stats.distance = (
        gps2dist_azimuth(shot.lat, shot.lon, tr.stats.latitude, tr.stats.longitude)[0]
        / M_PER_KM
    )

# Make region mask
lons = np.array([tr.stats.longitude for tr in st])
lats = np.array([tr.stats.latitude for tr in st])
in_region = (
    (lons > REGION[0]) & (lons < REGION[1]) & (lats > REGION[2]) & (lats < REGION[3])
)

# Plot
PLOT_REGION = [-122.36, -122.16, 46.26, 46.32]
fig = pygmt.Figure()
shaded_relief = pygmt.grdgradient(
    '@earth_relief_01s_g', region=PLOT_REGION, azimuth=-45.0, normalize='t1+a0'
)
pygmt.makecpt(cmap='gray', series=[-2, shaded_relief.values.max()])  # -2 is nice(?)
fig.grdimage(
    shaded_relief,
    cmap=True,
    projection='M6i',
    region=PLOT_REGION,
    transparency=30,
)
with pygmt.config(MAP_FRAME_TYPE='plain', FORMAT_GEO_MAP='D'):
    fig.basemap(frame=['a0.1f0.02', 'WESN'], map_scale='g-122.33/46.27+w2+f+l')
node_style = 'c0.1i'
fig.plot(
    x=lons[~in_region], y=lats[~in_region], style=node_style, pen=True, transparency=50
)
colors_rgb = plt.get_cmap('tab20').colors
for lon, lat, color_rgb in zip(lons[in_region], lats[in_region], colors_rgb):
    fig.plot(x=lon, y=lat, style=node_style, pen=True, color=to_hex(color_rgb))
fig.plot(x=shot.lon, y=shot.lat, style='s0.2i', color='black')
fig.text(x=shot.lon, y=shot.lat, text=shot.name, font='5p,white', justify='CM')
fig.show()
# fig.savefig('/Users/ldtoney/Downloads/y5_zoom_map.png', dpi=600)

#%% Corresponding waveform plot

TIME_LIM = (-1, 4)
EQUAL_SCALE = True

# Subset stream, assign colors to traces early on
st_region = Stream(compress(st, in_region)).copy()
for tr, color_rgb in zip(st_region, colors_rgb):
    tr.stats.color = to_hex(color_rgb)

# Process!
st_region.remove_response(inventory=inv)  # [m/s]

# Plot
fig, axes = plt.subplots(
    nrows=in_region.sum(), sharex=True, sharey=EQUAL_SCALE, figsize=(8, 13)
)
for station, ax in zip(RIDGE_SENSORS + VALLEY_SENSORS, axes):
    tr = st_region.select(station=str(station))[0]
    t = tr.times(reftime=shot.time) - tr.stats.distance / REMOVAL_CELERITY
    t_win = t[(t >= TIME_LIM[0]) & (t <= TIME_LIM[1])]
    data_win = tr.data[(t >= TIME_LIM[0]) & (t <= TIME_LIM[1])]
    ax.plot(
        t_win,
        data_win,
        color=tr.stats.color,
    )
    if tr.stats.station == str(RIDGE_SENSORS[0]):
        ax.set_title('Ridge sensors, W to E:')
    if tr.stats.station == str(VALLEY_SENSORS[0]):
        ax.set_title('Valley sensors, W to E:')
axes[0].set_xlim(TIME_LIM)  # [m/s] Sets for all
axes[-1].set_xlabel(f'Time (s), reduced by {REMOVAL_CELERITY * M_PER_KM:g} m/s')
fig.tight_layout()
fig.show()
# fig.savefig(f"/Users/ldtoney/Downloads/y5_zoom_wfs{'_equal_scale' if EQUAL_SCALE else ''}.png", dpi=300)
