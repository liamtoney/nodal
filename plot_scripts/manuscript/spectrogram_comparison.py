import os
import subprocess
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import simplekml
from matplotlib.colors import to_hex
from matplotlib.transforms import Bbox
from scipy.signal import spectrogram

from utils import NODAL_WORKING_DIR, get_shots, get_stations, get_waveforms_shot

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

# Print the centroid of the stations
print(
    f'({np.mean([tr.stats.latitude for tr in st]):.4f}, {np.mean([tr.stats.longitude for tr in st]):.4f})'
)

# Export KML file of station locations
kml = simplekml.Kml()
color = 'tab:orange'
for tr in st:
    pnt = kml.newpoint(name=tr.stats.station)
    pnt.coords = [(tr.stats.longitude, tr.stats.latitude)]
    pnt.style.iconstyle.scale = 2
    pnt.style.iconstyle.icon.href = (
        'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    )
    pnt.style.iconstyle.color = simplekml.Color.hex(to_hex(color).lstrip('#'))
    pnt.style.labelstyle.scale = 2
kml.save(NODAL_WORKING_DIR / 'metadata' / 'x4_spectrogram_stations.kml')

WF_LIM = (-3, 3)  # [μm/s]
SPEC_LIM = (10, 50)  # [Hz]
DB_LIM = (-160, -130)  # [dB]
WIN_DUR = 0.5  # [s]
TIME_LIM = (40, 55)  # [s]
V_REF = 1  # [m/s]


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
    sxx_db = 10 * np.log10(sxx / (V_REF**2))

    # Plot waveform and spectrogram
    wf_ax.plot(tr.times(), tr.data * 1e6, 'black', linewidth=0.5)  # Converting to μm/s
    pcm = spec_ax.pcolormesh(
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
        ax.grid(linestyle=':', which='both', color='lightgray')
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
        weight='bold',
    )

    return pcm


# Axis setup
FIGSIZE = (7.17, 4)  # [in]
spec_ratios = (6, 2, 0.5)  # `spec_ax` height, `wf_ax` height, spacer height
spec_rows = 3
spec_cols = 4
assert spec_rows * spec_cols == st.count()  # Basic check that we have enough subplots
fig, axs = plt.subplots(
    ncols=spec_cols + 1,  # Extra column for `map_ax`
    nrows=(2 * spec_rows) + (spec_rows - 1),  # Two axes for each spec, plus spacer axes
    figsize=FIGSIZE,
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
    pcm = spec_existing(tr, spec_ax, wf_ax)

# Plot map, instructions for making `x4_spectrogram_stations.png` are:
# (1) Open `x4_spectrogram_stations.kml` in Google Earth (keep default view)
# (2) Go to historical imagery from 7/2014
# (3) File -> Save -> Save Image...
# (4) Include only a "Scale" and "Compass" and set "Scaling" to 350%
# (5) Drag north arrow and scalebar to a reasonable location
# (6) Save image as a PNG to the filepath and name given by `image` below
# (7) Crop image to a reasonable extent with a 5:3 aspect ratio (e.g., 1,560 x 2,600)
image = NODAL_WORKING_DIR / 'data' / 'imagery' / 'x4_spectrogram_stations.png'
map_ax.set_aspect('equal', adjustable='datalim')  # For aspect ratio calculation later
map_ax.imshow(mpimg.imread(image), aspect='auto')
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
label_ax.set_xlabel(f'Time from shot {SHOT} (s)', labelpad=17)

# Remove some ticks
for ax in axs[:, 1:].flatten():
    ax.tick_params(left=False, which='both')
for ax in axs[:-1, :].flatten():
    ax.tick_params(bottom=False, which='both')

# Final figure tweaks to achieve proper borders
fig.tight_layout(pad=0.2, rect=(0, -0.06, 1, 1))
fig.subplots_adjust(hspace=0, wspace=0.05)

# Add colorbar
cax = fig.add_subplot(gs[0, :2])
cax_pos = cax.get_position()
cax.set_position(
    [
        cax_pos.xmin,
        cax_pos.ymax + (axs[1, 0].get_position().ymin - axs[3, 0].get_position().ymax),
        cax_pos.width,
        0.02,
    ]
)
extend_frac = 0.02
fig.colorbar(
    pcm, cax=cax, orientation='horizontal', extend='both', extendfrac=extend_frac
)
cax.text(
    1.05,
    0.5,
    f'Power (dB rel. {V_REF:g} [m/s]$^2$ Hz$^{{-1}}$)',
    ha='left',
    va='center',
    transform=cax.transAxes,
)
cax.xaxis.tick_top()
pos = cax.get_position()
triangle_width = extend_frac * pos.width
xmin = pos.xmin
width = pos.width
xmin -= triangle_width
width += 2 * triangle_width
cax.set_position([xmin, pos.ymin, width, pos.height])

# Print image axis aspect ratio (informs how to crop the image)
ratio = np.abs(np.diff(map_ax.get_ylim())[0] / np.diff(map_ax.get_xlim())[0])
print(f'Map height / width = {ratio:.2f}')

fig.show()

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

if False:
    fig.savefig(
        Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve()
        / 'spectrogram_comparison.png',
        dpi=400,
        bbox_inches=Bbox.from_bounds(0, 0, FIGSIZE[0], FIGSIZE[1] + 0.38),
    )
