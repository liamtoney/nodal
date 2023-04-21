import os
import subprocess
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from utils import get_shots, get_stations, get_waveforms_shot

FONT_SIZE = 10  # [pt]
plt.rcParams.update({'font.size': FONT_SIZE})

SHOT = 'X4'  # This shot has the best *spatially clustered* coupled arrivals

# These are the stations whose data we will plot
STATIONS = (
    '4302',
    '4303',
    '4304',
    '4305',
    '4306',
    '4307',
    '4308',
    '4309',
    '4310',
    '4311',
    '4312',
    '4319',
)

# Get stations and data
inv = get_stations()
st = get_waveforms_shot(SHOT, processed=True)  # Response is already removed!

# Reduce Stream to just the stations we want, and trim
for tr in st:
    if tr.stats.station not in STATIONS:
        st.remove(tr)
for tr in st:
    coords = inv.get_coordinates(tr.id)
    tr.stats.latitude = coords['latitude']
    tr.stats.longitude = coords['longitude']
st.trim(starttime=get_shots().loc[SHOT].time)  # Stream starts at shot time

WF_LIM = (-3, 3)  # [μm/s]
SPEC_LIM = (10, 50)  # [Hz]
DB_LIM = (-160, -130)  # [dB]
WIN_DUR = 0.5  # [s]
TIME_LIM = (40, 55)  # [s]


# Define function to plot waveform and spectrogram into existing axes
def spec_existing(tr, spec_ax, wf_ax):

    # Compute spectrogram
    fs = tr.stats.sampling_rate
    nperseg = int(WIN_DUR * fs)  # Samples
    nfft = np.power(2, int(np.ceil(np.log2(nperseg))) + 1)  # Pad FFT with zeroes
    f, t, sxx = spectrogram(
        tr.data, fs, window='hann', nperseg=nperseg, noverlap=nperseg // 2, nfft=nfft
    )
    # Remove DC component to avoid divide by zero errors later
    f = f[1:]
    sxx = sxx[1:, :]
    ref_val = 1  # [m/s]
    # [dB rel. (ref_val <ref_val_unit>)^2 Hz^-1]
    sxx_db = 10 * np.log10(sxx / (ref_val**2))

    # Plot waveform and spectrogram
    wf_ax.plot(tr.times(), tr.data * 1e6, 'black', linewidth=0.5)  # Converting to μm/s
    spec_ax.pcolormesh(
        t,
        f,
        sxx_db,
        cmap='inferno',
        rasterized=True,
        shading='nearest',
        vmin=DB_LIM[0],
        vmax=DB_LIM[1],
    )

    # Axis adjustments
    for ax in wf_ax, spec_ax:
        ax.grid(linestyle=':', which='both')
    wf_ax.set_ylim(WF_LIM)
    spec_ax.set_ylim(SPEC_LIM)
    wf_ax.set_xlim(TIME_LIM)
    wf_ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    wf_ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
    wf_ax.yaxis.set_major_locator(plt.MultipleLocator(WF_LIM[1]))
    spec_ax.yaxis.set_minor_locator(plt.MultipleLocator(10))

    # Label each panel with the station name
    spec_ax.text(
        0.025,
        0.95,
        tr.stats.station,
        transform=spec_ax.transAxes,
        color='white',
        va='top',
        ha='left',
        fontsize=8,
    )


# Axis setup
spec_ratios = (6, 2, 0.6)  # `spec_ax` height, `wf_ax` height, spacer height
spec_rows = 3
spec_cols = 4
assert spec_rows * spec_cols == st.count()  # Basic check that we have enough subplots
fig, axs = plt.subplots(
    ncols=spec_cols + 1,  # Extra column for `map_ax`
    nrows=(2 * spec_rows) + (spec_rows - 1),  # Two axes for each spec, plus spacer axes
    figsize=(10, 4.8),
    gridspec_kw=dict(
        height_ratios=(spec_ratios * spec_rows)[:-1], width_ratios=[1] * spec_cols + [2]
    ),
    sharex=True,  # All share the same time axis
    sharey='row',  # Can't use `True` here since we have spec and wf axes both
)
gs = axs[0, -1].get_gridspec()  # Top-right corner
map_ax = fig.add_subplot(gs[:, -1])  # Right-most column
for ax in axs[:, -1]:
    ax.remove()  # Remove the subplots in the right-most column
for ax in axs[2::3, :-1].flatten():
    ax.remove()  # Remove the "spacer" subplots
for spec_ind in np.arange(0, spec_rows * 3, 3)[:-1]:  # Share y-axes across rows
    wf_ind = spec_ind + 1
    axs[spec_ind, 0].sharey(axs[spec_ind + 3, 0])
    axs[wf_ind, 0].sharey(axs[wf_ind + 3, 0])

# Plot spectrograms
spec_axs = axs[0::3, :-1].flatten()
wf_axs = axs[1::3, :-1].flatten()
for tr, spec_ax, wf_ax in zip(st, spec_axs, wf_axs):
    spec_existing(tr, spec_ax, wf_ax)

# Plot map
map_ax.imshow(mpimg.imread('/Users/ldtoney/Documents/image.png'), aspect='auto')
map_ax.set_xticks([])
map_ax.set_yticks([])

# Label axes
for ax in axs[0::3, 0]:
    ax.set_ylabel('$f$ (Hz)')
for ax in axs[1::3, 0]:
    ax.set_ylabel('$v$ (μm/s)')
gs = axs[0, 0].get_gridspec()  # Top-left corner
label_ax = fig.add_subplot(gs[:, :-1])  # All BUT the right-most column
label_ax.patch.set_alpha(0)
label_ax.set_xticks([])
label_ax.set_yticks([])
for spine in label_ax.spines.values():
    spine.set_visible(False)
label_ax.set_xlabel(f'Time from shot {SHOT} (s)', labelpad=20)

# Final figure tweaks to achieve proper borders
fig.tight_layout(pad=0.2, rect=(0, -0.05, 1, 1))
fig.subplots_adjust(hspace=0)

fig.show()

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

# fig.savefig(Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve() / 'spectrogram_comparison.png', dpi=400)
