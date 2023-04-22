import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from obspy import UTCDateTime

from utils import NODAL_WORKING_DIR, get_shots

FONT_SIZE = 10  # [pt]
plt.rcParams.update({'font.size': FONT_SIZE})

# Sort shots in time
df = get_shots()
df.time = [pd.Timestamp(t.isoformat()) for t in df.time]
df = df.sort_values(by=['gcas_on_nodes', 'time'])
df.time = [UTCDateTime(t) for t in df.time]

# Read in individual CSV files (one per shot)
shot_data = {}
for shot in df.index:
    shot_data[shot] = pd.read_csv(
        NODAL_WORKING_DIR / 'shot_gather_measurements' / f'{shot}.csv'
    )

# Define values to plot
rms = [data.pre_shot_rms.median() * 1e6 for data in shot_data.values()]  # [μm/s]
distance = [data.dist_m.median() / 1000 for data in shot_data.values()]  # [km]

boundaries = [
    UTCDateTime(2014, 7, 24),  # 1st day of shots!
    UTCDateTime(2014, 7, 25),  # 2nd day of shots!
    UTCDateTime(2014, 8, 1),  # 3rd day of shots!
    UTCDateTime(2014, 8, 2),
]
boundaries_num = [t.matplotlib_date for t in boundaries]
cmap = ListedColormap(['#4e79a7', '#f28e2b', '#59a14f'])  # TABLEAU 10
bnorm = BoundaryNorm(boundaries_num, cmap.N)

size_1000_lb = 90  # Marker size for the smaller, 1000-lb shots
scale = size_1000_lb / 1000  # [1/lb] Scale shot weights to marker sizes

fig, ax = plt.subplots(figsize=(3.47, 3.3))
sm = ax.scatter(
    x=rms,
    y=distance,
    c=[t.matplotlib_date for t in df.time],
    s=df.weight_lb * scale * 1.6,
    marker='s',
    alpha=0.6,
    lw=0,
    norm=bnorm,
    cmap=cmap,
    clip_on=False,
    zorder=-5,
)
ax.scatter(
    x=rms,
    y=distance,
    c=df.gcas_on_nodes,
    s=df.weight_lb * scale,
    marker='s',
    cmap='Greys',
    edgecolors='black',
    clip_on=False,
    lw=0.5,
)
x_offsets = dict(AO4=0.035, Y8=0.045, Y2=-0.045)
for x, y, shot in zip(rms, distance, list(df.index)):
    ha = 'center'
    shot_str = shot
    if shot in x_offsets.keys():
        xoff = x_offsets[shot]
        ax.plot([x, x + xoff], [y, y], lw=0.5, color='black', zorder=-5)
        x += xoff
        if xoff > 0:
            ha = 'left'
            shot_str = ' ' + shot
        else:
            ha = 'right'
            shot_str = shot + ' '
    ax.text(
        x=x,
        y=y,
        s=shot_str,
        color='white' if df.loc[shot].gcas_on_nodes else 'black',
        va='center',
        ha=ha,
        fontsize=4.5,
        clip_on=False,
    )
ax.set_xlabel('Median RMS velocity (μm/s),\n20 s window pre-shot')  # WIN_DUR
ax.set_ylabel('Median shot–node distance (km)')

ax.set_xlim(right=0.8)
ax.set_ylim(top=80)

ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

# Add grid
xticks_all = sorted(ax.get_xticks().tolist() + ax.xaxis.get_minorticklocs().tolist())
for ticks, func in zip([xticks_all[:-1], ax.get_yticks()], [ax.axvline, ax.axhline]):
    for loc in ticks[:-1]:
        func(
            loc,
            linestyle=':',
            zorder=-10,
            color='lightgray',
            linewidth=plt.rcParams['grid.linewidth'],
        )

# Make pretty
for side in 'top', 'right':
    ax.spines[side].set_visible(False)

fig.tight_layout(pad=0.2)

# Inset colorbar (after tight_layout) call!
cbar_height = 0.25  # As fraction of total axis height
cbar_aspect = 8
cbar_x = 0.745  # In axis coordinates
cax = inset_axes(
    ax,
    bbox_to_anchor=(cbar_x, 1.02 - cbar_height, cbar_height / cbar_aspect, cbar_height),
    bbox_transform=ax.transAxes,
    width='100%',
    height='100%',
)
fmt = '%-d %B'
tick_labels = [f'{day.strftime(fmt)}' for day in boundaries[:-1]]
cbar = fig.colorbar(
    sm, cax=cax, ticks=boundaries_num[:-1] + np.diff(boundaries_num) / 2
)
for v in boundaries_num[1:-1]:
    cax.axhline(v, lw=plt.rcParams['axes.linewidth'], color='black')
cax.invert_yaxis()
cax.tick_params(which='minor', right=False)
cax.set_yticklabels(tick_labels)

fig.show()

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

# fig.savefig(Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve() / 'distance_rms_scatter.png', dpi=600)
