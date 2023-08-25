import os
import subprocess
from pathlib import Path

import datashader as ds
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import xarray as xr
from matplotlib.collections import LineCollection
from obspy.geodetics.base import gps2dist_azimuth

from utils import get_shots, get_stations, get_waveforms_shot
from utils.utils import V_P, C

FONT_SIZE = 10  # [pt]
plt.rcParams.update({'font.size': FONT_SIZE})

M_PER_KM = 1000  # [m/km]

df = get_shots()  # Shot info
inv = get_stations()  # Station (node) info

# Processing params (TODO: Presumably these must match the other measurement scripts?)
FREQMIN = 5  # [Hz]
FREQMAX = 50  # [Hz]
STA = 0.2  # [s]
LTA = 2  # [s]

# Process the data once, before moving on to the plotting stage
da_list = []  # Initalize list to hold 23 DataArrays (one for each shot)
print('Processing shot gather data...')
for shot in tqdm.tqdm(['X5', 'Y5']):
    # Read in data
    st = get_waveforms_shot(shot, processed=True)

    # Assign coordinates and distances
    for tr in st:
        coords = inv.get_coordinates(tr.id)
        dist_m = gps2dist_azimuth(
            coords['latitude'], coords['longitude'], df.loc[shot].lat, df.loc[shot].lon
        )[0]
        tr.stats.dist_km = dist_m / M_PER_KM

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

# %% Make quadmeshes

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
qm_list = []  # Initalize list to hold quadmesh DataArrays
print('Making quadmeshes...')
for da in tqdm.tqdm(da_list):
    qm = cvs.quadmesh(da, agg=ds.mean(da.name))
    qm.name = da.name
    qm_list.append(qm)

# %% Make plot

# Set up figure and axes
fig, axs = plt.subplots(ncols=2, figsize=(7.17, 3), sharex=True, sharey=True)
for ax in axs:
    ax.tick_params(which='both', top=True)
axs[-1].tick_params(which='both', left=False)
axs[-1].tick_params(which='both', right=True)

# These limits show ALL GCAs
xlim = (0, 80)
ylim = (5, 26)

# Iterate over each shot DataArray and plot
for da, qm, ax in zip(da_list, qm_list, axs.flatten()):
    qm.plot.imshow(
        ax=ax,
        add_labels=False,
        add_colorbar=False,
        vmin=np.nanmedian(qm),
        vmax=np.nanpercentile(qm, 98),
    )
    ax.text(
        0.97,
        0.03,
        qm.name,
        ha='right',
        va='bottom',
        transform=ax.transAxes,
        color='white',
        weight='bold',
    )

# Set limits
axs[0].set_xlim(xlim)
axs[0].set_ylim(ylim)

# Major ticks
axs[0].xaxis.set_major_locator(plt.MultipleLocator(20))
axs[0].yaxis.set_major_locator(plt.MultipleLocator(5))

# Minor ticks
axs[0].xaxis.set_minor_locator(plt.MultipleLocator(5))
axs[0].yaxis.set_minor_locator(plt.MultipleLocator(1))

# Add overall axis labels
label_ax = fig.add_subplot(111)
label_ax.patch.set_alpha(0)
for spine in label_ax.spines.values():
    spine.set_visible(False)
label_ax.set_xticks([])
label_ax.set_yticks([])
label_ax.set_xlabel('Time from shot (s)', labelpad=25)
axs[0].set_ylabel('Distance from shot (km)')

# Finalize layout
fig.tight_layout(pad=0.2, rect=(0, -0.075, 1, 1))
fig.subplots_adjust(wspace=0.1)

# Plot and label moveout lines
time_shift = 5  # [s] Aesthetic (so we can see the arrivals!)
npts = 500  # Just make this high
solid_fraction = 0.2  # Make this part of the line (after `ygap` adjustment) opaque
for moveout_velocity, label, text_gap, y_gap in zip(
    [C, V_P],
    [f'$c$ = {C} m/s', f'$v_\mathrm{{P}}$ = {V_P / M_PER_KM:g} km/s'],
    [0.14, 0.17],  # How much of a gap to make in the line for the text label
    [1.5, 0],  # [km] Space between end of line and axis boundary
):
    for ax in axs:
        # Plot line
        yvec = np.linspace(ylim[0] + y_gap, ylim[1] - y_gap, npts)
        xvec = (yvec / (moveout_velocity / M_PER_KM)) + time_shift
        solid_size = int(solid_fraction * npts)  # Size of solid portion
        alpha_size = int(((1 - solid_fraction) / 2) * npts)  # Size of gradient
        assert solid_size + 2 * alpha_size == npts
        # Form the transparency gradient
        alpha = np.concatenate(
            [
                np.linspace(0, 1, alpha_size),
                np.ones(solid_size),
                np.linspace(1, 0, alpha_size),
            ]
        )
        # Make gap for the text
        xvec[
            npts // 2 - int(text_gap * npts) : npts // 2 + int(text_gap * npts)
        ] = np.nan
        points = np.vstack((xvec, yvec)).T.reshape(-1, 1, 2)
        segments = np.hstack((points[:-1], points[1:]))
        colors = np.ones((npts, 4))
        colors[:, -1] = alpha
        lc = LineCollection(
            segments,
            colors=colors,
            lw=1,
            path_effects=[path_effects.Stroke(capstyle='projecting')],  # KEY
        )
        ax.add_collection(lc)

        # Plot angled text (based on https://stackoverflow.com/a/38414616)
        p1 = ax.transData.transform_point((xvec[0], yvec[0]))
        p2 = ax.transData.transform_point((xvec[-1], yvec[-1]))
        ax.text(
            (yvec.mean() / (moveout_velocity / M_PER_KM)) + time_shift,
            yvec.mean(),  # Vertically center in line!
            label,
            rotation=np.rad2deg(np.arctan((p2[1] - p1[1]) / (p2[0] - p1[0]))),
            va='center',
            ha='center',
            color='white',
        )

fig.show()

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

# fig.savefig(Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve() / 'x5_y5_gathers.png', dpi=300)
