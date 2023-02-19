import os
import subprocess
from pathlib import Path

from utils import get_stations, station_map

# Read in station info for plotting
net = get_stations()[0]

# Plot
fig = station_map(
    [sta.longitude for sta in net],
    [sta.latitude for sta in net],
    [int(sta.code) for sta in net],
    cbar_label='Station code',
    cbar_tick_ints='a200f100',
    cmap='turbo',
    reverse_cmap=True,
    plot_inset=True,
)

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

# fig.savefig(Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve() / 'overview_map.png', dpi=600)
