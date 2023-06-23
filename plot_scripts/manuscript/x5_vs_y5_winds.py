import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pygmt

from get_and_plot_nam_hrrr import build_url, ingest_grib_nam
from utils import get_shots, get_stations

# Get shot info
df = get_shots()

# Set PyGMT defaults
pygmt.config(
    MAP_FRAME_TYPE='plain', FORMAT_GEO_MAP='D', FONT='10p', PROJ_LENGTH_UNIT='i'
)

# Shared region for maps (same as `y5_amps_vs_diffs.py`)
REGION = (-122.36, -122.02, 46.08, 46.32)

# CONSTANT
M_PER_KM = 1000

# Get node lat/lon/elevation
net = get_stations()[0]
sta_lons = [sta.longitude for sta in net]
sta_lats = [sta.latitude for sta in net]


# Define function to plot wind arrows on an EXISTING map
def plot_winds(
    fig,  # Figure to plot into
    shot_name,
    frame='WESN',  # Which sides of frame to tick vs. annotate
    cbar_xpos=None,
    ref_arrow=False,
):
    # Get shot
    shot = df.loc[shot_name]

    # Get wind data
    time = pd.Timestamp(shot.time.datetime).round('1h').to_pydatetime()  # Nearest hour!
    u = ingest_grib_nam(
        build_url(time.year, time.month, time.day, time.hour, measurement='UGRD')
    )
    v = ingest_grib_nam(
        build_url(time.year, time.month, time.day, time.hour, measurement='VGRD')
    )

    # Plot
    fig.basemap(
        projection='M2.98i',
        region=REGION,
        frame='+n',
    )
    # Plot nodes
    fig.plot(x=sta_lons, y=sta_lats, color='black', style='c0.02i')

    # Plot winds
    in_per_ms = 0.04  # [in/(m/s)]
    vector_style = 'V4.5p+e+a45+n'  # +n means that vector heads are always drawn
    vector_pen = '1p'
    assert np.all(u.longitude.values == v.longitude.values)
    assert np.all(u.latitude.values == v.latitude.values)
    wind = np.sqrt(u.values.flatten() ** 2 + v.values.flatten() ** 2)
    wdir = (270 - np.rad2deg(np.arctan2(v.values.flatten(), u.values.flatten()))) % 360
    CMAP = '/Users/ldtoney/Documents/CETperceptual_GMT/CET-C6.cpt'
    pygmt.makecpt(cmap=CMAP, series=[0, 360])
    fig.plot(
        data=np.vstack(
            [
                u.longitude.values.flatten(),  # x
                u.latitude.values.flatten(),  # y
                wdir,  # color
                (wdir - 180) % 360,  # azimuth (flip vector direction here for plotting)
                wind * in_per_ms,  # length
            ]
        ).T,
        style=vector_style,
        pen=f'{vector_pen}+c',
        cmap=True,
    )
    if cbar_xpos:
        fig.colorbar(
            frame='a45f15+l"Wind direction (\\260)"',
            position=f'JBC+o{cbar_xpos}i/0.35i+w4i/0.1i',
        )
    if ref_arrow:
        reference_speed = 10  # [m/s]
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
            style=vector_style,
            pen=vector_pen,
            color='black',
            no_clip=True,
            offset=f'0/{arrow_yoff}i',
        )
        fig.text(
            x=REGION[0],
            y=REGION[2],
            text=f'{reference_speed} m/s',
            no_clip=True,
            offset=f'0/{arrow_yoff - 0.135}i',
            justify='TL',
        )

    # Plot and label shot
    size_1000_lb = 0.2  # [in] Marker size for the smaller, 1000-lb shots
    scale = size_1000_lb / 1000  # [in/lb] Scale shot weights to marker sizes
    fig.plot(
        x=[shot.lon],
        y=[shot.lat],
        size=[shot.weight_lb * scale],
        color='black' if shot.gcas_on_nodes else 'white',
        style='si',
        pen=True,
    )
    tcolor = 'white' if shot.gcas_on_nodes else 'black'
    fig.text(x=shot.lon, y=shot.lat, text=shot.name, font=f'5p,{tcolor}', justify='CM')

    # Map scale
    fig.basemap(map_scale='g-122.1/46.1+w5+f+l', frame=[frame, 'a0.1f0.02'])


fig = pygmt.Figure()
plot_winds(fig, 'X5', frame='WeSN', ref_arrow=True)
xshift = 3.32  # [in]
fig.shift_origin(xshift=f'{xshift}i')
plot_winds(fig, 'Y5', frame='wESN', cbar_xpos=-xshift / 2)

# Plot (a) and (b) tags
tag_kwargs = dict(position='TL', no_clip=True, justify='TR', font='12p,Helvetica-Bold')
x_offset = -0.05  # [in]
fig.text(text='(a)', offset=f'{x_offset - xshift}i/0', **tag_kwargs)
fig.text(text='(b)', offset=f'{x_offset}i/0', **tag_kwargs)

fig.show()

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

# fig.savefig(Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve() / 'x5_vs_y5_winds.png', dpi=600, resize='+m2p')
