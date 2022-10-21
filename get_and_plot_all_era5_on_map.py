import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from infresnel._georeference import _estimate_utm_crs
from pyproj import Transformer

from get_and_plot_era5 import get_era5_u_v_z
from utils import (
    ERA5_PRESSURE_LEVELS,
    FULL_REGION,
    NODAL_WORKING_DIR,
    get_shots,
    get_stations,
)

SAVE = False  # Toggle saving PNG file

# Read in shot info
df = get_shots()

# Get shot info (takes a bit... would be better to save these as files locally!)
ds_list = []
for _, shot in df.iterrows():
    time = pd.Timestamp(shot.time.datetime).round('1h').to_pydatetime()  # Nearest hour!

    # Get region surrounding shot so we can find nearest neighbor
    # TODO: Interpolate to exact shot time and location?
    ds = get_era5_u_v_z(
        time.year,
        time.month,
        time.day,
        time.hour,
        level=ERA5_PRESSURE_LEVELS,  # Up to around 50 km?
        region=FULL_REGION,  # To encompass all shots
    )

    # Select nearest 1D profile
    ds_shot = ds.sel(latitude=shot.lat, longitude=shot.lon, method='nearest')
    ds_shot['alt_km'] = (ds_shot.z / 9.8) / 1000  # [km] Convert to geopotential height

    ds_list.append(ds_shot)

#%% Plot the map

MAX_ALT = 12  # [km]

# Define coordinate transformation
crs = _estimate_utm_crs(df.lat.mean(), df.lon.mean())
proj = Transformer.from_crs(crs.geodetic_crs, crs)

# Form colormap
cmap = cc.m_rainbow_r.copy()
under_color = 'tab:gray'
cmap.set_under(under_color)

fig, ax = plt.subplots(figsize=(12, 7))

ax.scatter(
    *np.array(
        [proj.transform(sta.latitude, sta.longitude) for sta in get_stations()[0]]
    ).T,
    s=2,
    color='black',
    lw=0,
)

# Plot shots w/ wind arrows
for (_, shot), ds_shot in zip(df.iterrows(), ds_list):

    shot_x, shot_y = proj.transform(shot.lat, shot.lon)

    ds_plot = ds_shot.where(ds_shot.alt_km < MAX_ALT, drop=True)

    # Wind
    sm = ax.quiver(
        shot_x * np.ones(ds_plot.alt_km.size),
        shot_y * np.ones(ds_plot.alt_km.size),
        ds_plot.u,
        ds_plot.v,
        ds_plot.alt_km,  # We're coloring the arrows by altitude
        cmap=cmap,
        angles='xy',
        scale_units='xy',
        scale=0.002,
        width=0.002,
        clim=(df.elev_m.min() / 1000, MAX_ALT),  # Start colormap at lowest shot elev
    )

    # Shot w/ text
    ax.scatter(
        shot_x,
        shot_y,
        s=shot.weight_lb * 0.05,
        color='black' if shot.gcas_on_nodes else 'white',
        marker='s',
        ec='black',
        lw=0.5,
    )
    ax.text(
        shot_x,
        shot_y,
        s=shot.name,
        fontsize=3.5,
        color='white' if shot.gcas_on_nodes else 'black',
        ha='center',
        va='center',
        clip_on=True,
    )


@ticker.FuncFormatter
def major_formatter(x, pos):
    return f'{x / 1000:g}'  # Just converting from m to km here


for axis in ax.xaxis, ax.yaxis:
    axis.set_major_formatter(major_formatter)
ax.tick_params(top=True, right=True, which='both')
ax.set_aspect('equal')
ax.set_xlabel('UTM easting (km)')
ax.set_ylabel('UTM northing (km)')

reference_speed = 10  # [m/s]  # Ideally matches the reference speed for grid plots
ax.quiverkey(
    sm,
    0.93,
    0.05,
    reference_speed,
    label=f'{reference_speed} m/s',
    coordinates='axes',
)
cbar = fig.colorbar(sm, label='Geopotential height (km)', aspect=35)
cbar.ax.axhline(
    df.elev_m.min() / 1000,  # Horizontal line at lowest shot elev
    linestyle='--',
    color='black',
    linewidth=plt.rcParams['axes.linewidth'],
)
cbar.ax.set_ylim(bottom=0)
cbar.ax.set_facecolor(under_color)

fig.tight_layout()
fig.show()

if SAVE:
    fig.savefig(
        NODAL_WORKING_DIR / 'figures' / 'all_era5_on_map.png',
        dpi=600,
        bbox_inches='tight',
    )
