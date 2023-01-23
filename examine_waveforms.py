import webbrowser
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import pygmt
from matplotlib.colors import to_hex
from obspy import Stream
from obspy.geodetics.base import degrees2kilometers, gps2dist_azimuth

from utils import get_shots, get_stations, get_waveforms_shot

M_PER_KM = 1000  # [m/km] CONSTANT

SHOT = 'X4'

# [xmin, xmax, ymin, ymax] Region to examine
# ------------------------------------------
# Shot Y5
# ------------------------------------------
# REGION = [-122.254, -122.2, 46.285, 46.31]  # Ridge/valley area
# REGION = [-122.2791, -122.2629, 46.2772, 46.2887]  # Tight cluster in hummocks
# REGION = [-122.2549, -122.2276, 46.2723, 46.2791]  # Loowit(?) ridge
# ------------------------------------------
# Shot AI1
# ------------------------------------------
# REGION = [-122.2411, -122.2267, 46.1486, 46.161]  # Near Butte Camp TH
# ------------------------------------------
# Shots X4, Y4
# ------------------------------------------
# REGION = [-122.2584, -122.2526, 46.1651, 46.1693]  # Near Blue Lake TH (DENSE!)
# ------------------------------------------
# Shot X3
# ------------------------------------------
# REGION = [-122.2958, -122.2755, 46.2291, 46.2412]  # Near Castle Lake
# ------------------------------------------
# Shot X4
# ------------------------------------------
REGION = [-122.1399, -122.1297, 46.1462, 46.1547]  # Y-shaped dense area

#%% Open IRIS gmap station map

# All stations: https://ds.iris.edu/gmap/#net=1D&starttime=2014-07-01&endtime=2014-09-01
url = f'https://ds.iris.edu/gmap/#net=1D&minlon={REGION[0]}&maxlon={REGION[1]}&minlat={REGION[2]}&maxlat={REGION[3]}&drawingmode=box'
_ = webbrowser.open(url)

#%% PyGMT map

# Get stations, data, shots
inv = get_stations()
st = get_waveforms_shot(SHOT)
shot = get_shots().loc[SHOT]

# Assign coordinates and distance [km] to traces
for tr in st:
    try:
        coords = inv.get_coordinates(tr.id)
    except Exception:
        print(f'Removing {tr.id}')
        st.remove(tr)
    tr.stats.latitude = coords['latitude']
    tr.stats.longitude = coords['longitude']
    tr.stats.distance = (
        gps2dist_azimuth(shot.lat, shot.lon, tr.stats.latitude, tr.stats.longitude)[0]
        / M_PER_KM
    )

# Sort by distance to source
st.sort(keys=['distance'])

# Make region mask
lons = np.array([tr.stats.longitude for tr in st])
lats = np.array([tr.stats.latitude for tr in st])
in_region = (
    (lons > REGION[0]) & (lons < REGION[1]) & (lats > REGION[2]) & (lats < REGION[3])
)

# Plot
BUFFER = 10  # [%] Buffer (percent of extent to pad on each side)
width = REGION[1] - REGION[0]  # [deg.]
height = REGION[3] - REGION[2]  # [deg.]
x_buf = (BUFFER / 100) * width  # [deg.]
y_buf = (BUFFER / 100) * height  # [deg.]
plot_region = [
    REGION[0] - x_buf,
    REGION[1] + x_buf,
    REGION[2] - y_buf,
    REGION[3] + y_buf,
]
fig = pygmt.Figure()
shaded_relief = pygmt.grdgradient(
    '@earth_relief_01s_g', region=plot_region, azimuth=-45.0, normalize='t1+a0'
)
pygmt.makecpt(cmap='gray', series=[-2, shaded_relief.values.max()])  # -2 is nice(?)
fig.grdimage(
    shaded_relief,
    cmap=True,
    projection='M6i',
    region=plot_region,
    transparency=30,
)
SCALE_FRAC = 0.2  # How wide the scalebar should be as fraction of map width
scale_width = np.ceil(degrees2kilometers(width * SCALE_FRAC))  # [km] Nearest whole
with pygmt.config(MAP_FRAME_TYPE='plain', FORMAT_GEO_MAP='D'):
    fig.basemap(
        frame=['a0.1f0.01', 'WESN'],
        map_scale=f'g{np.mean(REGION[:2])}/{REGION[2]}+w{scale_width}+f+l',
    )
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

#%% Corresponding waveform plot

TIME_LIM = (-1, 4)
EQUAL_SCALE = False

# [km/s] For reduced time (use this to select which type of arrival!)
# -------------------------------------------------------------------
REMOVAL_CELERITY = 0.34
# REMOVAL_CELERITY = 4.8

# Subset stream, assign colors to traces early on
st_region = Stream(compress(st, in_region)).copy()
for tr, color_rgb in zip(st_region, colors_rgb):
    tr.stats.color = to_hex(color_rgb)

# Process!
if SHOT == 'Y4':
    for tr in st_region:
        fudge_factor = 87921  # TODO: See _plot_node_shot_gather.py
        tr.data *= fudge_factor
st_region.remove_sensitivity(inventory=inv)  # [m/s] Full response removal is trickier!

# Plot
fig, axes = plt.subplots(
    nrows=in_region.sum(), sharex=True, sharey=EQUAL_SCALE, figsize=(8, 13)
)
for tr, ax in zip(st_region, axes):
    t = tr.times(reftime=shot.time) - tr.stats.distance / REMOVAL_CELERITY
    t_win = t[(t >= TIME_LIM[0]) & (t <= TIME_LIM[1])]
    data_win = tr.data[(t >= TIME_LIM[0]) & (t <= TIME_LIM[1])]
    ax.plot(t_win, data_win * 1e6, color=tr.stats.color)
    ax.text(
        1.01,
        0.5,
        f'{tr.stats.station}\n{tr.stats.distance:.2f} km',
        ha='left',
        va='center',
        transform=ax.transAxes,
    )
axes[0].set_xlim(TIME_LIM)  # [m/s] Sets for all
axes[-1].set_xlabel(
    rf'Time from shot {shot.name} (s), reduced by $\bf{{{REMOVAL_CELERITY * M_PER_KM:g}~m/s}}$'
)
axes[-1].set_ylabel('μm/s')
fig.tight_layout()
fig.show()
