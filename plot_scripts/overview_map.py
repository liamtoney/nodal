import os
import subprocess
from pathlib import Path

from pygmt.datasets import load_earth_relief

from utils import INNER_RING_REGION, get_stations, station_map

# Read in station info for plotting
net = get_stations()[0]

# Get DEM to use for obtaining real station elevations
dem = load_earth_relief(
    resolution='01s', registration='gridline', region=INNER_RING_REGION
)

# Plot
fig = station_map(
    [sta.longitude for sta in net],
    [sta.latitude for sta in net],
    [
        dem.sel(lon=sta.longitude, lat=sta.latitude, method='nearest').values
        for sta in net
    ],
    cbar_label='Node elevation (m)',
    cmap='dem1',
    plot_inset=True,
)

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

# fig.savefig(Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve() / 'overview_map.png', dpi=600)
