#!/usr/bin/env python

"""Run the following in an IPython console to iterate over all shots:

from utils import get_shots
for shot in get_shots().index:
    %run get_and_plot_nam_hrrr.py {shot}
"""

import sys

import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pooch
import xarray as xr
from obspy import UTCDateTime

from utils import (
    FULL_REGION,
    INNER_RING_REGION,
    NODAL_WORKING_DIR,
    get_shots,
    get_stations,
)
from utils.utils import _outside_arrow

SAVE = True  # Toggle saving PNG files

# Read in shot info
df = get_shots()

# Input checks
assert len(sys.argv) == 2, 'Must provide exactly one argument!'
SHOT = sys.argv[1]
assert SHOT in df.index, 'Argument must be a valid shot name!'


def build_url(year, month, day, hour, measurement):
    month = f'{month:02}'
    day = f'{day:02}'
    hour = f'{hour:02}'
    return f'https://mtarchive.geol.iastate.edu/{year}/{month}/{day}/grib2/ncep/RTMA/{year}{month}{day}{hour}00_{measurement}.grib2'


def ingest_grib_nam(url):

    file_path = pooch.retrieve(url=url, known_hash=None, progressbar=True)
    ds = xr.open_dataset(
        file_path,
        indexpath='',
    )

    # Pre-process
    data_vars = list(ds.data_vars)
    assert len(data_vars) == 1, 'More than one data variable in file!'
    da = ds[data_vars[0]]
    da['longitude'] = (da.longitude + 180) % 360 - 180  # Convert to -180, 180 range

    return da


def ingest_grib_hrrr(url):

    file_path = pooch.retrieve(url=url, known_hash=None, progressbar=True)
    ds = xr.open_dataset(
        file_path,
        indexpath='',
        filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 10},
    )

    # Pre-process
    ds['longitude'] = (ds.longitude + 180) % 360 - 180  # Convert to -180, 180 range

    return ds.u10, ds.v10


def plot_wind_speed_direction(
    u, v, grid_type, shot, region=FULL_REGION, combo_plot=False
):

    # Crop each DataArray
    minx, maxx, miny, maxy = region
    mask_lon = (u.longitude >= minx) & (u.longitude <= maxx)
    mask_lat = (u.latitude >= miny) & (u.latitude <= maxy)
    u = u.where(mask_lon & mask_lat, drop=True)
    mask_lon = (v.longitude >= minx) & (v.longitude <= maxx)
    mask_lat = (v.latitude >= miny) & (v.latitude <= maxy)
    v = v.where(mask_lon & mask_lat, drop=True)

    # Get times and check them!
    dt1 = u.time.values
    dt2 = v.time.values
    assert dt1 == dt2, 'Files not from same time!'

    # Get shots and stations for plotting
    net = get_stations()[0]

    wind_speed = 'Wind speed (m/s)'
    wind_direction = 'Wind direction (°)'

    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return f'{x:.1f}°'.replace('-', '–')

    if combo_plot:
        cmaps = cc.m_gray_r, cc.m_CET_C10  # ISOLUMINANT
    else:
        cmaps = cc.m_gray_r, cc.m_CET_C6

    # Convert U and V to wind speed [m/s] and wind direction [° from north]
    wind = np.sqrt(u**2 + v**2)
    wdir = (270 - np.rad2deg(np.arctan2(v, u))) % 360

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(16.8, 4.5))

    for ax, da, cmap, type in zip(
        axes, (wind, wdir), cmaps, (wind_speed, wind_direction)
    ):
        if type == wind_direction:
            vmin, vmax = 0, 360
        else:
            vmin, vmax = None, None
        if combo_plot and type == wind_direction:
            wind_alpha = wind.copy() ** 2  # Highlight windy areas!
            wind_alpha = (wind_alpha - wind_alpha.min()) / (
                wind_alpha.max() - wind_alpha.min()
            )
            alpha = wind_alpha.fillna(0).data
        else:
            alpha = None
        sm = ax.pcolormesh(
            da.longitude,
            da.latitude,
            da.data,
            alpha=alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if type == wind_speed and combo_plot:
            sm.remove()
            sm = ax.quiver(
                da.longitude,
                da.latitude,
                wind * np.cos(np.deg2rad(270 - wdir)),  # U
                wind * np.sin(np.deg2rad(270 - wdir)),  # V
                wdir,
                cmap=cmaps[1],
                clim=(0, 360),
                scale=100,
                scale_units='width',
            )
            reference_speed = 10  # [m/s]
            ax.quiverkey(
                sm,
                0.05,
                1.05,
                reference_speed,
                label=f'{reference_speed} m/s',
                coordinates='axes',
            )
        ax.set_xlim(region[:2])
        ax.set_ylim(region[2:])
        ax.set_aspect('equal')

        # Get station coordinates
        sta_lons = np.array([sta.longitude for sta in net])
        sta_lats = np.array([sta.latitude for sta in net])

        # Plot shot
        if not (
            (shot.lon > region[0])
            & (shot.lon < region[1])
            & (shot.lat > region[2])
            & (shot.lat < region[3])
        ):
            # If shot is outside the region
            tail_coords, head_coords, tail_shot_dist_km = _outside_arrow(
                region,
                sta_lons.mean(),
                sta_lats.mean(),
                shot.lon,
                shot.lat,
                0.5,  # [km] Padding from boundary
                8,  # [km] Length of arrow
            )
            # Plot arrow
            ax.annotate(
                xy=head_coords,
                xytext=tail_coords,
                text='',
                arrowprops=dict(arrowstyle='->'),
            )
            # Plot arrow distance text
            angle = np.rad2deg(
                np.arctan(
                    (tail_coords[1] - head_coords[1])
                    / (tail_coords[0] - head_coords[0])
                )
            )
            offset_angle = np.deg2rad(angle + 90)
            offset_amt = 0.01  # [deg.]
            xoff = offset_amt * np.cos(offset_angle)
            yoff = offset_amt * np.sin(offset_angle)
            ax.text(
                x=np.mean([tail_coords[0], head_coords[0]]) + xoff,
                y=np.mean([tail_coords[1], head_coords[1]]) + yoff,
                rotation=angle,
                s=f'{tail_shot_dist_km:.1f} km',
                va='center',
                ha='center',
                weight='bold',
            )
            shot_x = tail_coords[0]
            shot_y = tail_coords[1]
        else:
            shot_x = shot.lon
            shot_y = shot.lat
        ax.scatter(
            shot_x,
            shot_y,
            s=shot.weight_lb * 0.04,  # Arbitrary scale factor here
            color='black' if shot.gcas_on_nodes else 'white',
            marker='s',
            ec='black',
            zorder=5,
        )

        ax.scatter(sta_lons, sta_lats, s=2, color='black', lw=0)
        for axis in ax.xaxis, ax.yaxis:
            axis.set_major_locator(plt.MultipleLocator(0.1))
            axis.set_major_formatter(major_formatter)
        ax.tick_params(top=True, right=True, which='both')
        if not (combo_plot and type == wind_speed):
            cbar = fig.colorbar(sm, ax=ax, label=type)
            if type == wind_direction:
                cbar.ax.yaxis.set_major_locator(plt.MultipleLocator(45))
                cbar.ax.yaxis.set_minor_locator(plt.MultipleLocator(15))
    fig.suptitle(
        rf'$\bf{{Shot~{shot.name}}}$'
        + '\nModel time: '
        + UTCDateTime(str(dt1)).strftime('%Y-%m-%d %H:%M')
        + f'\nGrid type: {grid_type}, 10 m above surface'
    )
    fig.autofmt_xdate()
    fig.tight_layout()

    fig.set_size_inches((12, 4.5))  # Hacky but makes plot more compact!

    fig.show()

    return fig


#%% Get 10-m wind grid for a given shot

TYPE = 'nam'  # 'nam' or 'hrrr'

# Get shot info
shot = df.loc[SHOT]
time = pd.Timestamp(shot.time.datetime).round('1h').to_pydatetime()  # Nearest hour!

# TODO: Interpolate to exact shot time?
if TYPE == 'hrrr':
    url = f'https://storage.googleapis.com/high-resolution-rapid-refresh/hrrr.{time.year}{time.month:02}{time.day:02}/conus/hrrr.t{time.hour:02}z.wrfprsf00.grib2'
    u, v = ingest_grib_hrrr(url)
    grid_name = 'HRRR'
elif TYPE == 'nam':
    u = ingest_grib_nam(
        build_url(time.year, time.month, time.day, time.hour, measurement='UGRD')
    )
    v = ingest_grib_nam(
        build_url(time.year, time.month, time.day, time.hour, measurement='VGRD')
    )
    grid_name = 'NAM NEST CONUS'
else:
    raise ValueError("TYPE must be either 'hrrr' or 'nam'")

#%% Plot grid

fig = plot_wind_speed_direction(
    u, v, grid_name, shot, region=INNER_RING_REGION, combo_plot=True
)

if SAVE:
    fig.savefig(
        NODAL_WORKING_DIR / 'figures' / '10_m_wind_grids' / f'shot_{SHOT}.png',
        dpi=300,
        bbox_inches='tight',
    )
