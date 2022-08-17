import matplotlib.pyplot as plt
import numpy as np
from obspy.geodetics.base import gps2dist_azimuth

from utils import get_shots, get_stations, get_waveforms_shot

# Read in shot info
df = get_shots()

# Read in station info
inv = get_stations()

# iterate over shots
shot_list = ['AI1', 'AI2', 'AI3', 'AI4', 'X4', 'X5', 'Y4', 'Y5']

fig, axes = plt.subplots(
    nrows=len(shot_list), sharey=False, sharex=True, figsize=(6, 13)
)

for SHOT, ax in zip(shot_list, axes):

    # Read in data
    st = get_waveforms_shot(SHOT)

    # Assign coordinates and distances
    for tr in st:
        # Need the "try" statement here for the shot Y4 data from Brandon
        try:
            coords = inv.get_coordinates(tr.id)
        except Exception:
            print(f'{tr.id} not found on IRIS. Removing.')
            st.remove(tr)
            continue
        tr.stats.latitude = coords['latitude']
        tr.stats.longitude = coords['longitude']
        tr.stats.distance = gps2dist_azimuth(
            tr.stats.latitude, tr.stats.longitude, df.loc[SHOT].lat, df.loc[SHOT].lon
        )[0]

    # Remove sensitivity (fast but NOT accurate!)
    st.remove_sensitivity(inv)

    target_dist = 8
    ind = np.argmin(
        np.abs(np.array([tr.stats.distance for tr in st]) - (target_dist * 1000))
    )
    tr = st[ind]
    ax.plot(tr.times(reftime=df.loc[SHOT].time), tr.data)

    ax.set_title(f'{SHOT}, {tr.stats.distance / 1000:.3f} km')

fig.tight_layout()
fig.show()
