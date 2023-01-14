import os
import warnings
from pathlib import Path

import geopandas  # pip install geopandas
import numpy as np
import pandas as pd
import pygmt
from infresnel._georeference import _estimate_utm_crs
from obspy import UTCDateTime, read, read_inventory
from obspy.geodetics.base import gps2dist_azimuth
from pyproj import Transformer
from shapely.geometry import LineString

# Define working directory here so that it can be exposed for easy import
NODAL_WORKING_DIR = Path(os.environ['NODAL_WORKING_DIR'])

# The 37 ERA5 pressure levels [hPa]
# fmt: off
ERA5_PRESSURE_LEVELS = (
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500,
     450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100,  70,  50,  30,  20,  10,
       7,   5,   3,   2,   1,
)
# fmt: on

# Nicely-rounded regions
FULL_REGION = (-123.2, -121.2, 45.6, 46.8)  # All 23 shots
INNER_RING_REGION = (-122.42, -121.98, 46.06, 46.36)  # Inner ring of 8 shots

# Parameters for mask distance calculation
T_SEP = 20  # [s] Coda length
C = 340  # [m/s] Sound speed
V_P = 5000  # [m/s] P-wave speed

# "Outside arrow" parameters
BOUNDARY_PAD = 0.5  # [km] Pad between map boundary and arrow head (along arrow axis!)
ARROW_LENGTH = 4.5  # [km] Length of arrow
OFFSET = 0.05  # [in] Distance text offset (perpendicular to arrow axis)

# Code constants
M_PER_KM = 1000  # [m/km] CONSTANT

# Read in and process shot metadata
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=UserWarning)  # Ignore "extension" warning
    df = pd.read_excel(NODAL_WORKING_DIR / 'metadata' / 'iMUSH_shot_metadata.xlsx')
df.dropna(inplace=True)
df.rename(
    columns={
        'Shot': 'shot',
        'Lat': 'lat',
        'Lon': 'lon',
        'Elev': 'elev_m',
        'Weight (lb)': 'weight_lb',
    },
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
        'Est Mag (PNSN)',
        'Unnamed: 9',
    ],
    inplace=True,
)
df['gcas_on_nodes'] = pd.read_json(
    NODAL_WORKING_DIR / 'metadata' / 'shot_gcas_on_nodes.json', typ='series'
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

    # For AGU poster (profile endpoints from make_topography.py!)
    # profile_start = (df.loc['Y5'].lat, df.loc['Y5'].lon)
    # profile_end = (46.224, -122.031)
    # crs = _estimate_utm_crs(*profile_start)
    # proj = Transformer.from_crs(crs.geodetic_crs, crs)
    # s = geopandas.GeoSeries(
    #     LineString([proj.transform(*profile_start), proj.transform(*profile_end)])
    # )
    # buffer = s.buffer(500, cap_style=2)  # [m] TODO must match MASK_DIST!
    # lats, lons = proj.transform(*buffer[0].exterior.coords.xy, direction='INVERSE')
    # fig.plot(x=lons, y=lats, close=True, color='black', transparency=60)
    # fig.plot(
    #     x=[profile_start[1], profile_end[1]],
    #     y=[profile_start[0], profile_end[0]],
    #     pen='0.5p',
    # )

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
        az = gps2dist_azimuth(*tail_coords[::-1], *head_coords[::-1])[1]
        angle = 90 - az % 180  # Degrees CCW from horizontal (make always upright!)
        offset_angle = np.deg2rad(angle + 90)
        fig.text(
            x=np.mean([tail_coords[0], head_coords[0]]),
            y=np.mean([tail_coords[1], head_coords[1]]),
            angle=angle,
            text=f'{tail_shot_dist_km:.1f} km',
            font='6p,Helvetica-Bold',
            justify='CB',
            offset=f'{OFFSET * np.cos(offset_angle)}i/{OFFSET * np.sin(offset_angle)}i',
        )
        shot_x = tail_coords[0]
        shot_y = tail_coords[1]
    else:
        shot_x = df_plot.lon
        shot_y = df_plot.lat

    # Ensure we can index these variables (i.e., if they are scalars)
    shot_x = np.atleast_1d(shot_x)
    shot_y = np.atleast_1d(shot_y)

    # Plot shots
    size_1000_lb = 0.2  # [in] Marker size for the smaller, 1000-lb shots
    kwargs = dict(style='si', pen=True)
    scale = size_1000_lb / 1000  # [in/lb] Scale shot weights to marker sizes
    fig.plot(
        x=list(shot_x[df_plot.gcas_on_nodes]) + [np.nan],
        y=list(shot_y[df_plot.gcas_on_nodes]) + [np.nan],
        size=list(df_plot[df_plot.gcas_on_nodes].weight_lb * scale) + [np.nan],
        color='black',
        label=f'GCAs observed+S{size_1000_lb}i',
        **kwargs,
    )
    fig.plot(
        x=list(shot_x[~df_plot.gcas_on_nodes]) + [np.nan],
        y=list(shot_y[~df_plot.gcas_on_nodes]) + [np.nan],
        size=list(df_plot[~df_plot.gcas_on_nodes].weight_lb * scale) + [np.nan],
        color='white',
        label=f'GCAs not observed+S{size_1000_lb}i',
        **kwargs,
    )
    # Plot shot names
    justify = 'CM'
    fontsize = 5  # [pts]
    fig.text(
        x=list(shot_x[df_plot.gcas_on_nodes]) + [np.nan],
        y=list(shot_y[df_plot.gcas_on_nodes]) + [np.nan],
        text=list(df_plot[df_plot.gcas_on_nodes].index) + [np.nan],
        font=f'{fontsize}p,white',
        justify=justify,
    )
    fig.text(
        x=list(shot_x[~df_plot.gcas_on_nodes]) + [np.nan],
        y=list(shot_y[~df_plot.gcas_on_nodes]) + [np.nan],
        text=list(df_plot[~df_plot.gcas_on_nodes].index) + [np.nan],
        font=f'{fontsize}p',
        justify=justify,
    )

    # TODO: below fig.basemap() values optimized for the default region only!
    fig.basemap(map_scale='g-122.04/46.09+w5+f+l', frame=['WESN', 'a0.1f0.02'])
    plot_legend = not (df_plot.shape[0] == 1 and not plot_inset)
    if plot_legend:
        position = 'JBL+jML'  # Shift colorbar to make room for legend
    else:
        position = 'JBC+jMC'
    fig.colorbar(
        frame=f'{cbar_tick_ints}+l"{cbar_label}"', position=position + '+o0/-0.5i+h'
    )
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
            kwargs = dict(style='si', pen=True)
            scale = 0.00007  # [in/lb] Scale shot weights to marker sizes
            fig.plot(
                x=df[df.gcas_on_nodes].lon,
                y=df[df.gcas_on_nodes].lat,
                size=df[df.gcas_on_nodes].weight_lb * scale,
                color='black',
                **kwargs,
            )
            fig.plot(
                x=df[~df.gcas_on_nodes].lon,
                y=df[~df.gcas_on_nodes].lat,
                size=df[~df.gcas_on_nodes].weight_lb * scale,
                color='white',
                **kwargs,
            )
            fig.basemap(map_scale=f'g{np.mean(FULL_REGION[:2])}/45.75+w50')

    # Make legend if appropriate
    if plot_legend:
        fig.legend(position='JBR+jML+o-0.6i/-0.5i+l1.5')  # +l controls line spacing!

    fig.show()

    return fig


def _outside_arrow(
    region, lon1, lat1, lon2, lat2, boundary_pad_km, arrow_length_km, interval_km=0.01
):
    """Determine optimal arrow tail and head locations subject to placement constraints.

    The user requests an arrow length and the amount of padding between the arrow head
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
