import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mat73 import loadmat
from obspy import Stream, Trace

# Load data
MSH_NODE_SEIS = loadmat(
    Path(os.environ['NODAL_WORKING_DIR']) / 'data' / 'MSH_node_seis.mat'
)

# From Brandon's code
SAMPLING_RATE = 80  # [Hz]

# Corners for bandpass filter
FREQMIN = 1  # [Hz]
FREQMAX = 10  # [Hz]

# For STA/LTA processing
NSTA = 20  # [samples]

# Load data into an ObsPy Stream, make a copy for processing
st = Stream()
for seis, dist in zip(MSH_NODE_SEIS.all_seis.T, MSH_NODE_SEIS.all_dist):
    if not np.all(np.isnan(seis)):  # Skip purely NaN traces
        st += Trace(data=seis, header=dict(sampling_rate=SAMPLING_RATE, distance=dist))
st.sort(keys=['distance'])  # Sort by distance
stp = st.copy()

# Taper
r1 = 0.1 / FREQMIN
r = r1 / (st[0].stats.delta * 0.5 * st[0].stats.npts)
stp.taper(type='cosine', max_percentage=r / 2)

# Filter
stp.filter(type='bandpass', freqmin=FREQMIN, freqmax=FREQMAX, corners=1, zerophase=True)

# Plot section
stp_mat = np.array(stp)
vmax = np.nanpercentile(
    np.abs(stp_mat), q=65
)  # Use a suitable percentile to avoid outliers
fig, ax = plt.subplots()
ax.imshow(stp_mat.T, cmap='seismic', vmin=-vmax, vmax=vmax)
ax.axis('off')
fig.show()

# STA/LTA
stp.trigger(type='classicstalta', nsta=NSTA, nlta=6 * NSTA)
if False:  # Toggle smoothing using something similar to MATLAB's smooth()
    window_size = int(np.rint(0.5 * NSTA))
    for tr in stp:
        tr.data = np.convolve(tr.data, np.ones(window_size), mode='same') / window_size

# Plot section, again
stp_mat = np.array(stp)
vmax = np.nanpercentile(stp_mat, q=95)  # Use a suitable percentile to avoid outliers
fig, ax = plt.subplots()
ax.imshow(stp_mat.T, cmap='Greys', vmin=0, vmax=vmax)
ax.axis('off')
fig.show()

# Distance stacking
inc = 0.05
dist_centers = np.arange(4, 22 + inc, inc)
bin_width = 0.15
all_dists = np.array([tr.stats.distance for tr in stp])
stp_bins = np.empty((stp[0].stats.npts, dist_centers.size))
for i, dist in enumerate(dist_centers):
    dist_ind = np.where(
        (all_dists > dist - bin_width / 2) & (all_dists < dist + bin_width / 2)
    )[0]
    stp_bins[:, i] = np.mean(stp_mat[dist_ind, :], axis=0)  # Also can try sum, product

# Plot section, again (again)
stp_bins[stp_bins == 0] = np.nan  # Set 0 to NaN
PERCENTILE_RANGE = 96  # [%] Central range of data to show
vmin = np.nanpercentile(stp_bins, q=(100 - PERCENTILE_RANGE) / 2)
vmax = np.nanpercentile(stp_bins, q=PERCENTILE_RANGE + (100 - PERCENTILE_RANGE) / 2)
fig, ax = plt.subplots()
ax.pcolormesh(
    dist_centers,
    stp[0].times() - 20,
    stp_bins,
    vmin=vmin,
    vmax=vmax,
    cmap='inferno',
    shading='nearest',
)
ax.set_xlim(4, 20)
ax.set_ylim(55, -15)
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Time (s)')
ax.set_title(f'{FREQMIN}–{FREQMAX} Hz STA/LTA iMUSH shots')
fig.show()

# fig.savefig(Path(os.environ['NODAL_WORKING_DIR']) / 'figures' / 'iMUSH_shot_acoustic_python.png', dpi=300, bbox_inches='tight')
