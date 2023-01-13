import matplotlib.pyplot as plt
import numpy as np
from obspy.geodetics.base import gps2dist_azimuth

from utils import get_shots, get_stations

M_PER_KM = 1000  # [m/km] CONSTANT

df = get_shots()
inv = get_stations()

# Sort shots by time
df = df.sort_values(by='time')

med_distance_dict = {}
for shot in df.index:
    distances = []
    for station in inv[0]:
        distance = (
            gps2dist_azimuth(
                df.loc[shot].lat,
                df.loc[shot].lon,
                station.latitude,
                station.longitude,
            )[0]
            / M_PER_KM
        )  # [km]
        distances.append(distance)
    med_distance_dict[shot] = np.median(distances)

#%% Make bar chart

no_detect_color = 'lightgray'

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.bar(
    x=range(len(med_distance_dict)),
    height=med_distance_dict.values(),
    tick_label=list(med_distance_dict.keys()),
    color=['black' if detect else no_detect_color for detect in df.gcas_on_nodes],
    edgecolor='black',
    width=0.6,
)
ax.set_ylabel('Median shot–node distance (km)')

# Make legend using dummy entries
kwargs = dict(edgecolor='black', marker='s', s=130)
ax.scatter(np.nan, np.nan, color='black', label='GCAs observed', **kwargs)
ax.scatter(np.nan, np.nan, color=no_detect_color, label='GCAs not observed', **kwargs)
ax.legend(frameon=False)

# Make pretty
for side in 'top', 'right':
    ax.spines[side].set_visible(False)
ax.set_xlim(-0.7, 22.5)

fig.tight_layout()
fig.show()
