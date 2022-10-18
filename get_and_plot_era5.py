#!/usr/bin/env python

import sys

import cdsapi
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pooch
import xarray as xr
from obspy import UTCDateTime

from utils import ERA5_PRESSURE_LEVELS, FULL_REGION, NODAL_WORKING_DIR, get_shots

SAVE = True  # Toggle saving PNG files

# Read in shot info
df = get_shots()

# Input checks
assert len(sys.argv) == 2, 'Must provide exactly one argument!'
SHOT = sys.argv[1]
assert SHOT in df.index, 'Argument must be a valid shot name!'


def get_era5_u_v_z(year, month, day, hour, level, region=None):

    c = cdsapi.Client()

    params = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'u_component_of_wind',
            'v_component_of_wind',
            'geopotential',
        ],
        'year': [str(y) for y in np.atleast_1d(year)],
        'month': [f'{m:02}' for m in np.atleast_1d(month)],
        'day': [f'{d:02}' for d in np.atleast_1d(day)],
        'time': [f'{h:02}' for h in np.atleast_1d(hour)],
        'pressure_level': [str(l) for l in np.atleast_1d(level)],  # [hPa]
    }
    if region:  # [xmin, xmax, ymin, ymax]
        params['area'] = [region[3], region[0], region[2], region[1]]

    ds = xr.open_dataset(
        pooch.retrieve(
            c.retrieve('reanalysis-era5-pressure-levels', params).location,
            known_hash=None,
        )
    ).squeeze()

    return ds


#%% Get ERA5 profile for a given shot

# Get shot info
shot = df.loc[SHOT]
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

#%% Plot the profile (vertical view)

MAX_ALT = 12  # [km]

ds_plot = ds_shot.where(ds_shot.alt_km < MAX_ALT, drop=True)

fig, ax = plt.subplots(figsize=(5, 9))
for wind, dir in zip([ds_plot.u, ds_plot.v], [270, 180]):
    ax.plot(
        wind,
        ds_plot.alt_km,
        color=cc.m_CET_C6(plt.Normalize(0, 360)(dir)),  # TODO: Must match grid code!
        label=wind.standard_name.replace('_', ' ').capitalize(),
    )
ax.axvline(
    0,  # Show 0 m/s wind as vertical line
    linestyle=':',
    color=plt.rcParams['grid.color'],
    linewidth=plt.rcParams['grid.linewidth'],
    zorder=-5,
)
ax.axhline(
    shot.elev_m / 1000,  # Horizontal line at shot elevation
    linestyle='--',
    color='black',
    linewidth=plt.rcParams['axes.linewidth'],
    zorder=-5,
    label='Shot elevation',
)
ax.set_xlabel('Wind speed (m/s)')
ax.set_ylabel('Altitude (km)')
title_str = (
    rf'$\bf{{Shot~{shot.name}}}$'
    + '\nModel time: '
    + UTCDateTime(str(ds_plot.time.values)).strftime('%Y-%m-%d %H:%M')
    + f'\nProfile location: ({ds_plot.latitude:.2f}°, {ds_plot.longitude:.2f}°)'.replace(
        '-', '–'
    )
)
ax.set_title(title_str)
ax.set_ylim(0, MAX_ALT)
ax.legend()
fig.show()

#%% Plot the profile (map view) [ABOVE CELL MUST BE RUN FIRST]

fig, (ax, cax) = plt.subplots(
    ncols=2, gridspec_kw=dict(width_ratios=[40, 1]), figsize=(8, 8)
)

# Form colormap
cmap = cc.m_rainbow_r.copy()
under_color = 'tab:gray'
cmap.set_under(under_color)

sm = ax.quiver(
    np.zeros(ds_plot.alt_km.size),
    np.zeros(ds_plot.alt_km.size),
    ds_plot.u,
    ds_plot.v,
    ds_plot.alt_km,  # We're coloring the arrows by altitude
    cmap=cmap,
    angles='xy',
    scale_units='xy',
    scale=1,
    width=0.007,
    clim=(shot.elev_m / 1000, MAX_ALT),  # Start colormap at shot elevation
    clip_on=False,
)
ax.set_xlim(min([ds_plot.u.min(), 0]), max([ds_plot.u.max(), 0]))
ax.set_ylim(min([ds_plot.v.min(), 0]), max([ds_plot.v.max(), 0]))
ax.set_aspect('equal')
reference_speed = 5  # [m/s]
ax.quiverkey(
    sm,
    0,
    ds_plot.v.max(),
    reference_speed,
    label=f'{reference_speed} m/s',
    coordinates='data',
)
ax.axis('off')
fig.colorbar(sm, cax=cax, label='Altitude (km)')
cax.axhline(
    shot.elev_m / 1000,  # Horizontal line at shot elevation
    linestyle='--',
    color='black',
    linewidth=plt.rcParams['axes.linewidth'],
)
cax.set_ylim(bottom=0)
cax.set_facecolor(under_color)
ax_pos = ax.get_position()
cax_pos = cax.get_position()
cax.set_position([cax_pos.x0, ax_pos.y0, cax_pos.width, ax_pos.height])
ax.set_title(title_str, pad=60)

fig.show()

if SAVE:
    fig.savefig(
        NODAL_WORKING_DIR / 'figures' / 'era5_profiles' / f'shot_{SHOT}.png',
        dpi=300,
        bbox_inches='tight',
    )
