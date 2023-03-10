import os
import subprocess
from pathlib import Path

import colorcet as cc
import datashader as ds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import xarray as xr
from obspy.geodetics.base import gps2dist_azimuth

from utils import get_shots, get_stations, get_waveforms_shot

M_PER_KM = 1000  # [m/km]

df = get_shots().sort_values(by='time')  # Shot info, sorted by time
inv = get_stations()  # Station (node) info

# Processing params (TODO: Presumably these must match the other measurement scripts?)
FREQMIN = 1  # [Hz]
FREQMAX = 50  # [Hz]
STA = 0.2  # [s]
LTA = 2  # [s]

# Process the data once, before moving on to the plotting stage
da_list = []  # Initalize list to hold 23 DataArrays (one for each shot)
print('Processing shot gather data...')
for shot in tqdm.tqdm(df.index):

    # Read in data
    st = get_waveforms_shot(shot)

    # Assign coordinates and distances
    for tr in st:
        # Need the "try" statement here for the shot Y4 data from Brandon
        try:
            coords = inv.get_coordinates(tr.id)
        except Exception:
            # print(f'{tr.id} not found on IRIS. Removing.')
            st.remove(tr)
            continue
        dist_m = gps2dist_azimuth(
            coords['latitude'], coords['longitude'], df.loc[shot].lat, df.loc[shot].lon
        )[0]
        tr.stats.dist_km = dist_m / M_PER_KM

    # TODO: Figure out Y4 data scaling issue... see "acoustic waves on nodes?" email chain
    if shot == 'Y4':
        fudge_factor = 87921  # Chosen to make max amp of closest station match shot Y5
        for tr in st:
            tr.data *= fudge_factor

    # Remove sensitivity (fast but NOT accurate!) to get m/s units
    st.remove_sensitivity(inv)

    # Detrend, taper, filter
    st.detrend('demean')
    st.taper(0.05)
    st.filter('bandpass', freqmin=FREQMIN, freqmax=FREQMAX, zerophase=True)

    # Apply STA/LTA
    st.trigger('classicstalta', sta=STA, lta=LTA)

    # Merge, as some stations have multiple Traces (doing this as late as possible)
    st.merge(fill_value=np.nan)

    # Ensure data are all same length (UGLY but needed for image plotting!)
    vc = pd.Series([tr.stats.npts for tr in st]).value_counts()
    most_common_npts = vc[vc == vc.max()].index.values[0]
    st = st.select(npts=most_common_npts)
    # print(f'Removed {vc[vc.index != most_common_npts].sum()} Trace(s)')

    # Form DataArray
    dist_km = np.array([tr.stats.dist_km for tr in st])
    sort_idx = np.argsort(dist_km)
    da = xr.DataArray(
        data=np.array([tr.data for tr in st])[sort_idx, :],
        coords=dict(
            distance=dist_km[sort_idx],  # [km]
            time=st[0].times(reftime=df.loc[shot].time),  # [s]
        ),
        name=shot,
    )
    da_list.append(da)

#%% Make quadmeshes

# Create canvas
x_range = (
    np.min([da.time.min() for da in da_list]),
    np.max([da.time.max() for da in da_list]),
)
y_range = (
    np.min([da.distance.min() for da in da_list]),
    np.max([da.distance.max() for da in da_list]),
)
cvs = ds.Canvas(plot_height=500, plot_width=2000, x_range=x_range, y_range=y_range)

# Iterate over each shot DataArray and plot
qm_list = []  # Initalize list to hold 23 quadmesh DataArrays
print('Making quadmeshes...')
for da in tqdm.tqdm(da_list):
    qm = cvs.quadmesh(da, agg=ds.mean(da.name))
    qm.name = da.name
    qm_list.append(qm)

#%% Make plot

# Set up figure and axes
fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(7, 9), sharex=True, sharey=True)
axs = axs.flatten()
axs[-1].remove()  # Only 23 shots!
axs[19].tick_params(labelbottom=True)  # Must label the plot above since bottom removed

# Iterate over each shot DataArray and plot
for qm, ax in zip(qm_list, axs):
    qm.plot.imshow(
        ax=ax,
        add_labels=False,
        add_colorbar=False,
        # vmin=np.percentile(qm, 90),
        # vmax=np.percentile(qm, 99),
        # cmap=cc.m_fire_r,
    )
    ax.text(
        0.9, 0.9, qm.name, ha='right', va='top', transform=ax.transAxes, weight='bold'
    )

# Show ALL data
# xlim = x_range
# ylim = y_range

# Show ALL GCAs
# xlim = (-5, 200)
# ylim = (5, 55)

# Show ALL GCAs except for X2 and Y6
xlim = (-5, 100)
ylim = (5, 40)

# Set limits
axs[0].set_xlim(xlim)
axs[0].set_ylim(ylim)

# Add overall axis labels
label_ax = fig.add_subplot(111)
label_ax.patch.set_alpha(0)
for spine in label_ax.spines.values():
    spine.set_visible(False)
label_ax.set_xticks([])
label_ax.set_yticks([])
label_ax.set_xlabel('Time from shot (s)', labelpad=25)
label_ax.set_ylabel('Distance from shot (km)', labelpad=25)

# Finalize
fig.tight_layout()
fig.show()

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

# fig.savefig(Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve() / 'all_gathers.new.png', dpi=300, bbox_inches='tight')
