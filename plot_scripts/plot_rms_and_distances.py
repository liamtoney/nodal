#%% Load in data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import NODAL_WORKING_DIR, get_shots

# Sort shots in time
df = get_shots().sort_values(by='time')

# Read in individual CSV files (one per shot)
shot_data = {}
for shot in df.index:
    shot_data[shot] = pd.read_csv(
        NODAL_WORKING_DIR / 'shot_gather_measurements' / f'{shot}.csv'
    )

# Define values and y-axis labels
values_list = [
    [data.pre_shot_rms.median() * 1e6 for data in shot_data.values()],  # [μm/s]
    [data.dist_m.median() / 1000 for data in shot_data.values()],  # [km]
]
ylabel_list = [
    'Median RMS velocity (μm/s),\n20 s window pre-shot',  # WIN_DUR
    'Median shot–node distance (km)',
]

#%% 1. Make two bar plots

for values, ylabel in zip(values_list, ylabel_list):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(
        x=range(len(shot_data)),
        height=values,
        tick_label=list(shot_data.keys()),
        color=['black' if detect else 'lightgray' for detect in df.gcas_on_nodes],
        edgecolor='black',
        width=0.6,
    )
    ax.set_ylabel(ylabel)

    # Make legend using dummy entries
    kwargs = dict(x=np.nan, y=np.nan, edgecolor='black', marker='s', s=130)
    ax.scatter(color='black', label='GCAs observed', **kwargs)
    ax.scatter(color='lightgray', label='GCAs not observed', **kwargs)
    ax.legend(frameon=False)

    # Make pretty
    for side in 'top', 'right':
        ax.spines[side].set_visible(False)
    ax.set_xlim(-0.7, 22.5)

    fig.tight_layout()
    fig.show()

#%% 2. Make one scatter plot

size_1000_lb = 100  # Marker size for the smaller, 1000-lb shots
scale = size_1000_lb / 1000  # [1/lb] Scale shot weights to marker sizes

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(
    x=values_list[0],
    y=values_list[1],
    c=df.gcas_on_nodes,
    s=df.weight_lb * scale,
    marker='s',
    cmap='Greys',
    edgecolors='black',
)
for x, y, shot in zip(values_list[0], values_list[1], list(df.index)):
    ax.text(
        x=x,
        y=y,
        s=shot,
        color='white' if df.loc[shot].gcas_on_nodes else 'black',
        va='center',
        ha='center',
        fontsize=5,
    )
ax.set_xlabel(ylabel_list[0])
ax.set_ylabel(ylabel_list[1])

# Make pretty
for side in 'top', 'right':
    ax.spines[side].set_visible(False)

fig.show()
