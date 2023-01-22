import matplotlib.pyplot as plt
import numpy as np
from obspy.geodetics.base import gps2dist_azimuth
from rtm import define_grid, grid_search, plot_time_slice, produce_dem

from utils import NODAL_WORKING_DIR, get_shots, get_stations, get_waveforms_shot

y5 = get_shots().loc['Y5']
inv = get_stations()

RADIUS = 2000  # [m]

# Define search grid
grid = define_grid(
    lon_0=y5.lon,
    lat_0=y5.lat,
    x_radius=RADIUS,  # [m]
    y_radius=RADIUS,  # [m]
    spacing=50,  # [m]
    projected=True,
    plot_preview=True,
)

# Separate grid that covers all of the nodes
grid_dem = define_grid(
    lon_0=np.mean([sta.longitude for sta in inv[0]]),
    lat_0=np.mean([sta.latitude for sta in inv[0]]),
    x_radius=15 * 1000,  # [m]
    y_radius=15 * 1000,  # [m]
    spacing=100,  # [m]
    projected=True,
)
dem = produce_dem(grid_dem, plot_output=True)

#%% Get and process waveforms

# Read in data
st = get_waveforms_shot(y5.name)

# Assign coordinates and distances
for tr in st:
    coords = inv.get_coordinates(tr.id)
    tr.stats.latitude = coords['latitude']
    tr.stats.longitude = coords['longitude']
    tr.stats.distance = gps2dist_azimuth(
        tr.stats.latitude, tr.stats.longitude, y5.lat, y5.lon
    )[0]

# Sort by increasing distance from shot
st.sort(keys=['distance'])

# Detrend, taper, filter
st.detrend('demean')
st.taper(0.05)
FREQMIN = 1  # [Hz]
FREQMAX = 50  # [Hz]
st.filter('bandpass', freqmin=FREQMIN, freqmax=FREQMAX)

# Apply STA/LTA
STA = 0.2  # [s]
LTA = 2  # [s]
st.trigger('classicstalta', sta=STA, lta=LTA)

# Decimate for speed
st.interpolate(sampling_rate=50, method='lanczos', a=20)

#%% Test waveform plot

st.select(station='4106').plot(fig=plt.figure())

#%% Perform grid search

st_subset = st[:400]  # Subsetting stations here

S = grid_search(
    processed_st=st_subset,
    grid=grid,
    time_method='celerity',
    starttime=y5.time - 20,
    endtime=y5.time + 20,
    stack_method='sum',
    celerity=343,  # Could refine this...
    dem=dem,
)

#%% Plot results

fig = plot_time_slice(
    S,
    st_subset,
    label_stations=False,
    dem=produce_dem(grid, plot_output=False),  # Somehow have to provide this?
    plot_peak=True,
    xy_grid=RADIUS,
)
fig.axes[0].get_legend().get_texts()[0].set_text(f'Shot {y5.name}\nlocation')
fig.axes[1].axvline(
    y5.time.matplotlib_date, color='limegreen', lw=2, zorder=-5, label='Shot Y5\ntime'
)
fig.axes[1].legend()

# fig.savefig(NODAL_WORKING_DIR / 'figures' / 'y5_rtm.png', dpi=300, bbox_inches='tight')
