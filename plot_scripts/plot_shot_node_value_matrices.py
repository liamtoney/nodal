import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm

from utils import NODAL_WORKING_DIR, get_shots

df_shot = get_shots().sort_values(by='time')  # Sorted by increasing time!

# Get array of station numbers present for ALL shots
station_list = []
for shot in df_shot.index:
    df = pd.read_csv(NODAL_WORKING_DIR / 'shot_gather_measurements' / f'{shot}.csv')
    station_list += df.station.tolist()
vc = pd.value_counts(station_list)
consistent_stations = vc[vc == df_shot.shape[0]].index.values  # Appearing in all shots!
consistent_stations.sort()

# Form matrices of peak frequencies and RMS velocities
freq = np.empty((df_shot.shape[0], consistent_stations.size))
rms = freq.copy()
for i, shot in enumerate(df_shot.index):
    df = pd.read_csv(NODAL_WORKING_DIR / 'shot_gather_measurements' / f'{shot}.csv')
    df_consistent = df[df.station.isin(consistent_stations)]
    df_consistent = df_consistent.sort_values(by='station')
    freq[i, :] = df_consistent.peak_freq
    rms[i, :] = df_consistent.pre_shot_rms * 1e6  # [µm/s]

# %% Plot matrices

FIGSIZE = (23, 5)  # [in.] Size for both plots

# Plot frequency matrix
cmap = plt.get_cmap('turbo')
inc = 5
norm = BoundaryNorm(np.arange(5, 45 + inc, inc), cmap.N)
fig1, ax1 = plt.subplots(figsize=FIGSIZE)
im1 = ax1.imshow(freq, norm=norm, cmap=cmap, interpolation='none')

# Plot RMS matrix
fig2, ax2 = plt.subplots(figsize=FIGSIZE)
im2 = ax2.imshow(rms, vmax=np.percentile(rms, 95), cmap='inferno', interpolation='none')

# Tweaks for both plots
for fig, im, label in zip(
    [fig1, fig2],
    [im1, im2],
    ['Peak frequency (Hz)', 'RMS velocity (μm/s), 20 s window pre-shot'],
):
    cbar = fig.colorbar(
        im,
        orientation='horizontal',
        location='top',
        shrink=0.15,
        aspect=30,
    )
    cbar.set_label(label, labelpad=15)
for ax in ax1, ax2:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r'increasing station number $\rightarrow$')
    ax.set_ylabel(r'$\leftarrow$ increasing time')
for fig in fig1, fig2:
    fig.tight_layout()
    fig.show()
