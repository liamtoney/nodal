import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pygmt

from utils import MASK_DISTANCE_KM, NODAL_WORKING_DIR, get_shots

# Get Y5 info
shot = get_shots().loc['Y5']

# Set PyGMT defaults
pygmt.config(MAP_FRAME_TYPE='plain', FORMAT_GEO_MAP='D', FONT='10p')

# Shared region for maps (buffer around nodes + shot Y5, to nearest 0.02°)
REGION = (-122.36, -122.02, 46.08, 46.32)

# CONSTANT
M_PER_KM = 1000

# Read in all the measurements
df = pd.read_csv(NODAL_WORKING_DIR / 'shot_gather_measurements' / f'{shot.name}.csv')

# Define function to plot node values on an EXISTING map
def plot_node_values(
    fig,  # Figure to plot into
    sta_values,
    cbar_label,
    sta_dists=None,  # [m] (Optional) station distances, needed for mask_distance > 0
    cbar_tick_ints='',  # GMT formatting; use empty string for automatic
    vmin=None,  # If None, then uses minimum of sta_values
    vmax=None,  # If None, then uses maximum of sta_values
    cmap='viridis',
    reverse_cmap=False,
    mask_distance=0,  # [km] Plot markers within this range differently
    frame='WESN',  # Which sides of frame to tick vs. annotate
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
    shaded_relief = pygmt.grdgradient(
        '@earth_relief_01s_g',
        region=REGION,
        azimuth=-45.0,
        normalize='t1+a0',
    )
    pygmt.makecpt(cmap='gray', series=[-2, shaded_relief.values.max()])  # -2 is nice(?)
    fig.grdimage(
        shaded_relief,
        cmap=True,
        projection='M2.98i',
        region=REGION,
        frame=False,
        transparency=30,
    )

    series = [
        sta_values.min() if vmin is None else vmin,
        sta_values.max() if vmax is None else vmax,
    ]
    pygmt.makecpt(series=series, cmap=cmap, reverse=reverse_cmap, background=True)
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
    fig.plot(
        x=df.lon[~is_masked],
        y=df.lat[~is_masked],
        color=sta_values[~is_masked],
        style=node_style,
        cmap=True,
        pen='black',
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
    fig.basemap(map_scale='g-122.1/46.1+w5+f+l', frame=[frame, 'a0.1f0.02'])
    clip_low = series[0] > sta_values.min()
    clip_high = series[1] < sta_values.max()
    if clip_low and clip_high:
        position = '+ebf'
    elif clip_low and not clip_high:
        position = '+eb'
    elif not clip_low and clip_high:
        position = '+ef'
    else:
        position = None
    fig.colorbar(frame=f'{cbar_tick_ints}+l"{cbar_label}"', position=position)


fig = pygmt.Figure()
plot_node_values(
    fig,
    df.sta_lta_amp,
    sta_dists=df.dist_m,
    cbar_label='STA/LTA amplitude',
    vmin=2,
    mask_distance=MASK_DISTANCE_KM,
    frame='WeSN',
)
xshift = 3.32  # [in]
fig.shift_origin(xshift=f'{xshift}i')
plot_node_values(
    fig,
    df.path_length_diff_m,
    cbar_label='Path length difference (m)',
    vmax=35,  # [m]
    reverse_cmap=True,
    frame='wESN',
)

# Plot (a) and (b) tags
tag_kwargs = dict(position='TL', no_clip=True, justify='TR', font='12p,Helvetica-Bold')
x_offset = -0.05  # [in]
fig.text(text='(a)', offset=f'{x_offset - xshift}i/0', **tag_kwargs)
fig.text(text='(b)', offset=f'{x_offset}i/0', **tag_kwargs)

fig.show()

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

# fig.savefig(Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve() / 'y5_amps_vs_diffs.png', dpi=600, resize='+m2p')
