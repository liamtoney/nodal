import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pygmt
from obspy import UTCDateTime, read, read_inventory

# Define working directory here so that it can be exposed for easy import
NODAL_WORKING_DIR = Path(os.environ['NODAL_WORKING_DIR'])

# Nicely-rounded regions
FULL_REGION = (-123.1, -121.3, 45.6, 46.8)  # All 23 shots
INNER_RING_REGION = (-122.42, -121.98, 46.06, 46.36)  # Inner ring of 8 shots

# Read in and process shot metadata
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=UserWarning)  # Ignore "extension" warning
    df = pd.read_excel(NODAL_WORKING_DIR / 'metadata' / 'iMUSH_shot_metadata.xlsx')
df.dropna(inplace=True)
df.rename(
    columns={'Shot': 'shot', 'Lat': 'lat', 'Lon': 'lon', 'Weight (lb)': 'weight_lb'},
    inplace=True,
)
df.set_index('shot', inplace=True)  # So we can easily look up shots by name
# fmt: off
def _form_time(row):
    """Convert iMUSH shot metadata spreadsheet times to UTCDateTime format."""
    frac_sec, whole_sec = np.modf(row.Sec)
    time = UTCDateTime(
        year=2014,  # Not included in spreadsheet
        julday=row.Julian,
        hour=int(row['UTC Hour']),
        minute=int(row.Min),
        second=int(whole_sec),
        microsecond=int(frac_sec * 1e6),  # Converting from seconds to microseconds
    )
    return time
# fmt: on
df['time'] = df.apply(_form_time, axis='columns')
df.drop(
    columns=[
        'Julian',
        'UTC Hour',
        'Min',
        'Sec',
        'Elev',
        'Est Mag (PNSN)',
        'Unnamed: 9',
    ],
    inplace=True,
)

# Read in station information
inv = read_inventory(str(NODAL_WORKING_DIR / 'metadata' / '1D.xml'))


def get_stations():
    """Return ObsPy Inventory containing 1D network information."""
    return inv


def get_waveforms_shot(shot):
    """Return ObsPy Stream containing waveforms for a given shot."""
    st = read(str(NODAL_WORKING_DIR / 'data' / 'mseed' / f'{shot}.mseed'))
    return st


def get_shots():
    """Return pandas DataFrame containing iMUSH shot metadata."""
    return df


def station_map(
    sta_lons,
    sta_lats,
    sta_values,
    cbar_label,
    cbar_tick_ints='',  # GMT formatting; use empty string for automatic
    region=INNER_RING_REGION,
    cmap='viridis',
    reverse_cmap=False,
    plot_inset=False,
):
    """Plot nodes and shots with nodes colored by provided values."""

    # Set PyGMT defaults (inside function since we might want to make FONT an input arg)
    pygmt.config(MAP_FRAME_TYPE='plain', FORMAT_GEO_MAP='D', FONT='10p')

    # Plot
    fig = pygmt.Figure()
    shaded_relief = pygmt.grdgradient(
        '@earth_relief_01s', region=region, azimuth=-45.0, normalize='t1+a0'
    )
    pygmt.makecpt(cmap='gray', series=[-2, shaded_relief.values.max()])  # -2 is nice(?)
    fig.grdimage(shaded_relief, cmap=True, projection='M4i', region=region, frame=False)
    pygmt.makecpt(
        series=[np.min(sta_values), np.max(sta_values)],
        cmap=cmap,
        reverse=reverse_cmap,
        background=True,
    )
    fig.plot(
        x=sta_lons, y=sta_lats, color=sta_values, style='c0.05i', cmap=True, pen='black'
    )  # Nodes
    fig.plot(x=df.lon, y=df.lat, style='s0.2i', color='black', pen='white')  # Shots
    fig.text(
        x=df.lon, y=df.lat, text=df.index, font='6p,white', justify='CM'
    )  # Shot names
    # TODO: below fig.basemap() values optimized for the default region only!
    fig.basemap(map_scale='g-122.04/46.09+w5+f+l', frame=['WESN', 'a0.1f0.02'])
    fig.colorbar(frame=f'{cbar_tick_ints}+l"{cbar_label}"')
    if plot_inset:
        with fig.inset(position='JTR+w1.5i+o-0.5i/-1i', box='+gwhite+p1p'):
            fig.plot(
                x=sta_lons,
                y=sta_lats,
                color='black',
                style='c0.01i',
                region=FULL_REGION,
                projection='M?',
            )
            in_main_map = (
                (df.lon > region[0])
                & (df.lon < region[1])
                & (df.lat > region[2])
                & (df.lat < region[3])
            )
            kwargs = dict(style='s0.07i', pen='black')
            fig.plot(
                x=df[in_main_map].lon, y=df[in_main_map].lat, color='black', **kwargs
            )
            fig.plot(
                x=df[~in_main_map].lon, y=df[~in_main_map].lat, color='white', **kwargs
            )
            fig.basemap(map_scale='g-122.2/45.8+w50')

    fig.show()

    return fig
