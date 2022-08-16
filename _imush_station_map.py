from utils import NODAL_WORKING_DIR, get_stations, station_map

# Read in station info for plotting
net = get_stations()[0]

# Plot
fig = station_map(
    [sta.longitude for sta in net],
    [sta.latitude for sta in net],
    [int(sta.code) for sta in net],
    cbar_label='Station code',
    cmap='turbo',
    reverse_cmap=True,
    plot_inset=True,
)

# fig.savefig(NODAL_WORKING_DIR / 'figures' / 'imush_station_map.png', dpi=600)
