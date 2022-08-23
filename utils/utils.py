import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pygmt
from obspy import UTCDateTime, read, read_inventory

# Define working directory here so that it can be exposed for easy import
NODAL_WORKING_DIR = Path(os.environ['NODAL_WORKING_DIR'])

# Parameters for mask distance calculation
T_SEP = 20  # [s] Coda length
C = 340  # [m/s] Sound speed
V_P = 5000  # [m/s] P-wave speed

# Nicely-rounded regions
FULL_REGION = (-123.1, -121.3, 45.6, 46.8)  # All 23 shots
INNER_RING_REGION = (-122.42, -121.98, 46.06, 46.36)  # Inner ring of 8 shots

# "Outside arrow" parameters
BOUNDARY_PAD = 1  # [km] Padding between map boundary and arrow head
ARROW_LENGTH = 3  # [km] Length of arrow

# Code constants
M_PER_KM = 1000  # [m/km] CONSTANT

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
    sta_dists=None,  # [m] (Optional) station distances, needed for mask_distance > 0
    cbar_tick_ints='',  # GMT formatting; use empty string for automatic
    region=INNER_RING_REGION,
    vmin=None,  # If None, then uses minimum of sta_values
    vmax=None,  # If None, then uses maximum of sta_values
    cmap='viridis',
    reverse_cmap=False,
    plot_shot='all',  # Or a shot name or list of shot names
    plot_inset=False,
    mask_distance=0,  # [km] Plot markers within this range differently
):
    """Plot nodes and shots with nodes colored by provided values."""

    # Convert lons/lats/values to arrays
    sta_lons = np.array(sta_lons)
    sta_lats = np.array(sta_lats)
    sta_values = np.array(sta_values)

    # Set PyGMT defaults (inside function since we might want to make FONT an input arg)
    pygmt.config(MAP_FRAME_TYPE='plain', FORMAT_GEO_MAP='D', FONT='10p')

    # Determine which shots to plot in main map
    df_plot = df.loc[df.index if plot_shot == 'all' else np.atleast_1d(plot_shot)]

    # Determine which nodes to mask
    if mask_distance > 0:
        if sta_dists is None:
            raise ValueError('sta_dists must be provided to use mask_distance!')
        else:
            is_masked = sta_dists < mask_distance * M_PER_KM
    else:  # Nothing should be masked, as mask_distance is 0
        is_masked = np.full(len(sta_lons), False)

    # Plot
    fig = pygmt.Figure()
    shaded_relief = pygmt.grdgradient(
        '@earth_relief_01s_g', region=region, azimuth=-45.0, normalize='t1+a0'
    )
    pygmt.makecpt(cmap='gray', series=[-2, shaded_relief.values.max()])  # -2 is nice(?)
    fig.grdimage(
        shaded_relief,
        cmap=True,
        projection='M4i',
        region=region,
        frame=False,
        transparency=30,
    )
    pygmt.makecpt(
        series=[
            sta_values.min() if vmin is None else vmin,
            sta_values.max() if vmax is None else vmax,
        ],
        cmap=cmap,
        reverse=reverse_cmap,
        background=True,
    )
    node_style = 'c0.05i'
    # Plot nodes INSIDE mask (if any!)
    if is_masked.any():
        fig.plot(
            x=sta_lons[is_masked],
            y=sta_lats[is_masked],
            color=sta_values[is_masked],
            style=node_style,
            pen='gray31',
        )
    # Plot nodes OUTSIDE mask
    fig.plot(
        x=sta_lons[~is_masked],
        y=sta_lats[~is_masked],
        color=sta_values[~is_masked],
        style=node_style,
        cmap=True,
        pen='black',
    )

    # Plot shots
    if (
        df_plot.shape[0] == 1
        and not (
            (df_plot.lon > region[0])
            & (df_plot.lon < region[1])
            & (df_plot.lat > region[2])
            & (df_plot.lat < region[3])
        ).to_numpy()
    ):
        # If only one shot is requested AND it's outside the map
        tail_coords, head_coords, tail_shot_dist_km = _outside_arrow(
            region,
            sta_lons.mean(),
            sta_lats.mean(),
            df_plot.lon.squeeze(),
            df_plot.lat.squeeze(),
            BOUNDARY_PAD,
            ARROW_LENGTH,
        )
        # Plot arrow
        fig.plot(
            x=tail_coords[0],
            y=tail_coords[1],
            style='v0.1i+e+gblack+a45+s',
            pen='1p',
            direction=[[head_coords[0]], [head_coords[1]]],
        )
        # Plot arrow distance text
        fig.text(
            x=tail_coords[0],
            y=tail_coords[1],
            text=f'{tail_shot_dist_km:.1f} km',
            font='6p,Helvetica-Bold',
            justify='LM',
            offset='0.12i/0',
            no_clip=True,
        )
        shot_x = tail_coords[0]
        shot_y = tail_coords[1]
    else:
        shot_x = df_plot.lon
        shot_y = df_plot.lat
    # Plot shots
    fig.plot(x=shot_x, y=shot_y, style='s0.2i', color='black', pen='white')
    # Plot shot names
    fig.text(x=shot_x, y=shot_y, text=df_plot.index, font='6p,white', justify='CM')

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


def _outside_arrow(
    region, lon1, lat1, lon2, lat2, boundary_pad_km, arrow_length_km, interval_km=0.1
):
    """Determine optimal arrow tail and head locations subject to placement constraints.

    The user request an arrow length and the amount of padding between the arrow head
    and the region boundary. The arrow points from (lon1, lat1), inside the map, to
    (lon2, lat2), outside the map.

    Args:
        region (list): Region, in PyGMT format
        lon1 (int or float): Longitude of inside point
        lat1 (int or float): Latitude of inside point
        lon2 (int or float): Longitude of outside point
        lat2 (int or float): Latitude of outside point
        boundary_pad_km (int or float): [km] Padding between arrow head and region
            boundary (measured along the arrow's axis!)
        arrow_length_km (int or float): [km] Length of arrow from tail to head
        interval_km (int or float): [km] Spacing of intermediate points, used to
            determine optimal arrow location - may need to decrease this for smaller
            regions

    Returns:
        tuple: Tuple containing:
        * tuple: (tail_lon, tail_lat)
        * tuple: (head_lon, head_lat)
        * float: [km] Distance from arrow tail to the outside point (lon2, lat2)
    """

    points = _get_points(lon1, lat1, lon2, lat2, interval_km)
    in_region = (
        (points.r > region[0])
        & (points.r < region[1])
        & (points.s > region[2])
        & (points.s < region[3])
    )
    end_ind = min(-1, -int(round(boundary_pad_km / interval_km)))
    start_ind = end_ind - int(round(arrow_length_km / interval_km))

    return (
        (points[in_region].iloc[start_ind].r, points[in_region].iloc[start_ind].s),
        (points[in_region].iloc[end_ind].r, points[in_region].iloc[end_ind].s),
        points.iloc[-1].p - points[in_region].iloc[start_ind].p,
    )


def _get_points(lon1, lat1, lon2, lat2, interval_km):
    """Calculate intermediate longitudes, latitudes, and distances between two points.

    Uses pyproj (WGS 84 ellipsoid) if it's installed, otherwise uses PyGMT (spherical).

    Args:
        lon1 (int or float): Longitude of first point
        lat1 (int or float): Latitude of first point
        lon2 (int or float): Longitude of second point
        lat2 (int or float): Latitude of second point
        interval_km (int or float): Desired spacing between intermediate points

    Returns:
        pandas.DataFrame: DataFrame of longitudes, latitudes, and distances in km
    """

    try:
        from pyproj import Geod  # fmt: skip
        print('pyproj found, using pyproj to calculate points')

        geod = Geod(ellps='WGS84')
        rs = geod.inv_intermediate(
            lon1,
            lat1,
            lon2,
            lat2,
            del_s=interval_km * M_PER_KM,  # [m]
            initial_idx=0,
            terminus_idx=0,
        )
        p = geod.inv(
            np.full(rs.npts, rs.lons[0]), np.full(rs.npts, rs.lats[0]), rs.lons, rs.lats
        )[2]
        points = pd.DataFrame(dict(r=rs.lons, s=rs.lats, p=np.array(p) / M_PER_KM))

    except ImportError:
        print('pyproj not found, using PyGMT to calculate points')

        points = pygmt.project(
            center=(lon1, lat1), endpoint=(lon2, lat2), generate=interval_km, unit=True
        )

    return points


def _estimate_mask_distance(t_sep, c, v_p):
    """Estimate distance within which we expect acoustic arrivals to be masked by coda."""
    return t_sep * ((1 / c) - (1 / v_p)) ** -1  # [m] (if inputs are all in SI!)


# Call function to create masking distance, which is then exposed
MASK_DISTANCE_KM = _estimate_mask_distance(T_SEP, C, V_P) / M_PER_KM  # [km]
