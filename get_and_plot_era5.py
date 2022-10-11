#%% Define function(s)

import cdsapi
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pooch
import xarray as xr
from obspy import UTCDateTime

from utils import ERA5_PRESSURE_LEVELS, FULL_REGION, get_shots


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

SHOT = 'X4'

# Get shot info
shot = get_shots().loc[SHOT]
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

#%% Plot the profile

cmap = cc.m_CET_C6  # Must match grid plotting code!
norm = plt.Normalize(0, 360)

fig, ax = plt.subplots(figsize=(5, 9))
altitude = (ds_shot.z / 9.8) / 1000  # [km] Converting to geopotential here
for wind, dir in zip([ds_shot.u, ds_shot.v], [270, 180]):
    ax.plot(
        wind,
        altitude,
        color=cmap(norm(dir)),
        label=wind.standard_name.replace('_', ' ').capitalize(),
    )
ax.axvline(0, linestyle=':', color='black', zorder=-5)
ax.set_xlabel('Wind speed (m/s)')
ax.set_ylabel('Altitude (km)')
ax.set_title(
    rf'$\bf{{Shot~{shot.name}}}$'
    + '\nModel time: '
    + UTCDateTime(str(ds_shot.time.values)).strftime('%Y-%m-%d %H:%M')
    + f'\nProfile location: ({ds_shot.latitude:.2f}°, {ds_shot.longitude:.2f}°)'.replace(
        '-', '–'
    )
)
ax.set_ylim(0, 50)
ax.legend()
fig.show()
