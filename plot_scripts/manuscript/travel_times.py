import os
import subprocess
import tempfile
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmt
from matplotlib.colors import PowerNorm
from obspy.geodetics.base import gps2dist_azimuth
from scipy.interpolate import griddata

from get_and_plot_nam_hrrr import build_url, ingest_grib_nam
from utils import INNER_RING_REGION, MASK_DISTANCE_KM, NODAL_WORKING_DIR, get_shots

# Get Y5 info
shot = get_shots().loc['Y5']

# Set PyGMT defaults
pygmt.config(
    MAP_FRAME_TYPE='plain', FORMAT_GEO_MAP='D', FONT='10p', PROJ_LENGTH_UNIT='i'
)

# Shared region for maps (buffer around nodes + shot Y5, to nearest 0.02°)
REGION = (-122.36, -122.02, 46.08, 46.32)

MAP_WIDTH = 2.98  # [in]

# CONSTANT
M_PER_KM = 1000

# Get wind data and crop for speed
time = pd.Timestamp(shot.time.datetime).round('1h').to_pydatetime()  # Nearest hour!
u = ingest_grib_nam(
    build_url(time.year, time.month, time.day, time.hour, measurement='UGRD')
)
v = ingest_grib_nam(
    build_url(time.year, time.month, time.day, time.hour, measurement='VGRD')
)
minx, maxx, miny, maxy = INNER_RING_REGION
mask_lon = (u.longitude >= minx) & (u.longitude <= maxx)
mask_lat = (u.latitude >= miny) & (u.latitude <= maxy)
u = u.where(mask_lon & mask_lat, drop=True)
mask_lon = (v.longitude >= minx) & (v.longitude <= maxx)
mask_lat = (v.latitude >= miny) & (v.latitude <= maxy)
v = v.where(mask_lon & mask_lat, drop=True)

# Read in all the measurements
df = pd.read_csv(NODAL_WORKING_DIR / 'shot_gather_measurements' / f'{shot.name}.csv')

# Define normalization for STA/LTA (transparency mapping)
GAMMA = 3  # Exponent for accentuating higher STA/LTA values
norm = PowerNorm(gamma=GAMMA, vmin=df.sta_lta_amp.min(), vmax=df.sta_lta_amp.max())

# Calculate w_p
interpolation_method = 'linear'
dot_product_medians = []
for lon, lat in zip(df.lon, df.lat):

    # Define path to interpolate along
    npts = 100  # TODO should this vary with the shot–node distance?
    x = np.linspace(shot.lon, lon, npts)
    y = np.linspace(shot.lat, lat, npts)

    # Interpolate the wind grids
    u_profile = griddata(
        (u.longitude.values.flatten(), u.latitude.values.flatten()),
        u.values.flatten(),
        (x, y),
        method=interpolation_method,
    )
    v_profile = griddata(
        (v.longitude.values.flatten(), v.latitude.values.flatten()),
        v.values.flatten(),
        (x, y),
        method=interpolation_method,
    )
    waz = (90 - np.rad2deg(np.arctan2(v_profile, u_profile))) % 360  # [° from N]
    baz = gps2dist_azimuth(shot.lat, shot.lon, lat, lon)[1]  # [° shot-node az]
    wmag = np.sqrt(u_profile**2 + v_profile**2)
    angle_diff = waz - baz  # Sign does not matter!
    dot_product = wmag * np.cos(np.deg2rad(angle_diff))  # Treating baz as unit vector

    dot_product_medians.append(np.median(dot_product))
dot_product_medians = np.array(dot_product_medians)

#%% Plot

# Define function to plot node values on an EXISTING map
def plot_node_values(
    fig,  # Figure to plot into
    sta_values,
    cbar_label,
    sta_dists=None,  # [m] (Optional) station distances, needed for mask_distance > 0
    cbar_tick_ints='',  # GMT formatting; use empty string for automatic
    vmin=None,  # If None, then uses minimum of sta_values
    vmax=None,  # If None, then uses maximum of sta_values
    mask_distance=0,  # [km] Plot markers within this range differently
    frame='WESN',  # Which sides of frame to tick vs. annotate
    ref_arrow_xshift=None,
    cbar_pos='left',  # Choose 'left' or 'right' alignment
    sta_lta_transparent=False,
):

    # Determine which nodes to mask
    if mask_distance > 0:
        if sta_dists is None:
            raise ValueError('sta_dists must be provided to use mask_distance!')
        else:
            is_masked = sta_dists < mask_distance * M_PER_KM
    else:  # Nothing should be masked, as mask_distance is 0
        is_masked = np.full(df.lon.size, False)

    # Plot
    fig.basemap(
        projection=f'M{MAP_WIDTH}i',
        region=REGION,
        frame='+n',
    )

    series = [
        sta_values.min() if vmin is None else vmin,
        sta_values.max() if vmax is None else vmax,
    ]
    CMAP = '/Users/ldtoney/Documents/CETperceptual_GMT/CET-D11.cpt'
    pygmt.makecpt(series=series, cmap=CMAP, background=True)
    node_style = 'c0.05i'
    # Plot nodes INSIDE mask (if any!)
    if is_masked.any():
        fig.plot(
            x=df.lon[is_masked],
            y=df.lat[is_masked],
            color=sta_values[is_masked],
            style=node_style,
            pen='gray31',
        )
    # Plot nodes OUTSIDE mask
    if sta_lta_transparent:
        transparency = (1 - norm(df.sta_lta_amp[~is_masked])) * 100  # [% transparency]
    else:
        transparency = None
    fig.plot(
        x=df.lon[~is_masked],
        y=df.lat[~is_masked],
        color=sta_values[~is_masked],
        style=node_style,
        cmap=True,
        transparency=transparency,
        pen='gray31',
    )
    clip_low = series[0] > sta_values.min()
    clip_high = series[1] < sta_values.max()
    if clip_low and clip_high:
        tri = '+ebf'
    elif clip_low and not clip_high:
        tri = '+eb'
    elif not clip_low and clip_high:
        tri = '+ef'
    else:
        tri = ''
    cbar_height = 0.1  # [in]
    cbar_width = 2.6  # [in]
    if sta_lta_transparent:
        cbar_yoff = 0.7  # [in]
    else:
        cbar_yoff = 0.35 - cbar_height  # [in]
    if cbar_pos == 'left':
        dir = -1
    elif cbar_pos == 'right':
        dir = 1
    else:
        raise ValueError
    fig.colorbar(
        frame=f'{cbar_tick_ints}+l"{cbar_label}"',
        position=f'JBC+o{dir * (MAP_WIDTH - cbar_width) / 2}i/{cbar_height + cbar_yoff}i+w{cbar_width}i/{cbar_height}i'
        + tri,
    )
    if sta_lta_transparent:
        height = 5  # [squares] Height of checkerboard pattern
        width = round(height * (cbar_width / cbar_height))
        check_shape = (height, width)
        fig_cbar, ax = plt.subplots()
        ax.pcolormesh(
            np.linspace(0, cbar_width, check_shape[1] + 1),
            np.linspace(0, cbar_height, check_shape[0] + 1),
            np.indices(check_shape).sum(axis=0) % 2,  # The checker pattern
            cmap=cc.m_gray_r,
            vmax=8,  # Effectively controls how gray checkerboard is (`vmax=1` is black)
        )
        npts = 1000
        ax.pcolormesh(
            np.linspace(0, cbar_width, npts + 1),
            [0, cbar_height],
            np.ones((1, npts)),  # Solid black
            alpha=norm(np.expand_dims(np.linspace(norm.vmin, norm.vmax, npts), 1).T),
            cmap=cc.m_gray,
            rasterized=True,  # Avoids unsightly horizontal stripes
        )
        ax.set_aspect('equal')
        ax.axis('off')
        transp_cbar_pos = f'JBC+o{dir * (MAP_WIDTH - cbar_width) / 2}i/{cbar_yoff}i+w{cbar_width}i/{cbar_height}i'
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            fig_cbar.savefig(f.name, dpi=1000, bbox_inches='tight', pad_inches=0)
            fig.image(f.name, position=transp_cbar_pos)
        plt.close(fig_cbar)
        pygmt.makecpt(series=[norm.vmin, norm.vmax], transparency=100)  # For frame
        fig.colorbar(frame='+l"STA/LTA amplitude"', position=transp_cbar_pos + '+m')

    # Plot winds
    in_per_ms = 0.06  # [in/(m/s)]
    vector_style = 'V4.5p+e+a45+n'  # +n means that vector heads are always drawn
    vector_pen = '1p'
    assert np.all(u.longitude.values == v.longitude.values)
    assert np.all(u.latitude.values == v.latitude.values)
    wind = np.sqrt(u.values.flatten() ** 2 + v.values.flatten() ** 2)
    wdir = (270 - np.rad2deg(np.arctan2(v.values.flatten(), u.values.flatten()))) % 360
    fig.plot(
        data=np.vstack(
            [
                u.longitude.values.flatten(),  # x
                u.latitude.values.flatten(),  # y
                (wdir - 180) % 360,  # azimuth (flip vector direction here for plotting)
                wind * in_per_ms,  # length
            ]
        ).T,
        color='black',
        style=vector_style,
        pen=vector_pen,
    )
    if ref_arrow_xshift:
        reference_speed = 5  # [m/s]
        arrow_yoff = -0.4  # [in]
        fig.plot(
            data=np.vstack(
                [
                    REGION[0],  # x
                    REGION[2],  # y
                    90,  # azimuth
                    reference_speed * in_per_ms,  # length
                ]
            ).T,
            style=vector_style + '+jc',
            pen=vector_pen,
            color='black',
            no_clip=True,
            offset=f'{ref_arrow_xshift}/{arrow_yoff}i',
        )
        fig.text(
            x=REGION[0],
            y=REGION[2],
            text=f'{reference_speed} m/s',
            no_clip=True,
            offset=f'{ref_arrow_xshift}/{arrow_yoff - 0.135}i',
            justify='TC',
        )

    # Plot and label shot Y5
    size_1000_lb = 0.2  # [in] Marker size for the smaller, 1000-lb shots
    scale = size_1000_lb / 1000  # [in/lb] Scale shot weights to marker sizes
    fig.plot(
        x=[shot.lon],
        y=[shot.lat],
        size=[shot.weight_lb * scale],
        color='black',
        style='si',
        pen=True,
    )
    fig.text(x=shot.lon, y=shot.lat, text=shot.name, font='5p,white', justify='CM')

    # Map scale
    fig.basemap(map_scale='g-122.1/46.1+w5+f+l', frame=[frame, 'a0.1f0.02'])


fig = pygmt.Figure()
plot_node_values(
    fig,
    df.diffracted_path_length / df.arr_time,
    cbar_label='Modified celerity (m/s)',
    cbar_tick_ints='a0.5',
    vmin=339,
    vmax=341.5,
    frame='WeSN',
    cbar_pos='left',
    sta_dists=df.dist_m,
    mask_distance=MASK_DISTANCE_KM,
    sta_lta_transparent=True,
)
xshift = 3.32  # [in]
fig.shift_origin(xshift=f'{xshift}i')
plot_node_values(
    fig,
    dot_product_medians,
    cbar_label='Median @%Helvetica-Oblique%w@-p@-@%% (m/s)',  # LOL
    cbar_tick_ints='a0.3-0.2',  # Making a "phase" adjustment here
    vmin=-3.5,
    vmax=-2,
    frame='wESN',
    cbar_pos='right',
    ref_arrow_xshift=-(xshift - MAP_WIDTH) / 2,
    sta_lta_transparent=False,
)

# Plot (a) and (b) tags
tag_kwargs = dict(position='TL', no_clip=True, justify='TR', font='12p,Helvetica-Bold')
x_offset = -0.05  # [in]
fig.text(text='(a)', offset=f'{x_offset - xshift}i/0', **tag_kwargs)
fig.text(text='(b)', offset=f'{x_offset}i/0', **tag_kwargs)

fig.show()

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

# fig.savefig(Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve() / 'travel_times.png', dpi=600, resize='+m2p')
