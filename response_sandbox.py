import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import LogLocator
from obspy import UTCDateTime

from utils import NODAL_WORKING_DIR, get_stations, get_waveforms_shot

# Get station info and some data from Y5
inv = get_stations()
st = get_waveforms_shot('Y5')  # Using RAW data here, not processed

#%% Plot unique responses

# Get unique responses
vc = pd.value_counts([sta[0].response for sta in inv[0]])

# Plot
fig, axs = plt.subplots(
    nrows=2, ncols=len(vc), sharex=True, sharey='row', figsize=(15, 6)
)
for unique_response, count, ax_col in zip(vc.index, vc.values, axs.T):
    unique_response.plot(min_freq=0.1, axes=list(ax_col), plot_degrees=True)
    ref_val = unique_response.instrument_sensitivity.value
    atten = -3  # [dB]
    atten_val = (10 ** (atten / 20)) * ref_val  # https://en.wikipedia.org/wiki/Decibel
    ax_col[0].axhline(
        atten_val, zorder=-1, color='red', linestyle='--', label=f'${atten}$ dB'
    )
    ax_col[0].set_title(f'{count} nodes')
axs[0, 0].set_xlim(0.5, 100)
axs[0, 0].set_ylim(1e6, 2e9)
axs[1, 0].set_ylim(-180, 180)
axs[1, 1].set_xlabel('Frequency (Hz)', labelpad=10)
axs[0, 0].set_ylabel('Gain (counts m$^{-1}$ s)')
axs[1, 0].set_ylabel('Phase (degrees)')
axs[0, 0].yaxis.set_minor_locator(LogLocator(subs='auto'))
axs[1, 0].yaxis.set_major_locator(plt.MultipleLocator(60))
for ax in axs.flatten():
    ax.grid(linestyle=':')
axs[0, 1].legend(frameon=False, loc='lower right')
fig.tight_layout()
fig.show()

# fig.savefig(NODAL_WORKING_DIR / 'figures' / 'node_response.png', dpi=300, bbox_inches='tight')

#%% Response removal testing

# Pick a nice waveform to test on — this is a RAW seismogram!
tr_orig = st.select(station='4106')[0]

# Trim to GCA arrival
tr = tr_orig.copy().trim(
    UTCDateTime(2014, 8, 1, 10, 30, 36), UTCDateTime(2014, 8, 1, 10, 30, 44)
)

# Plot response removal
tr.remove_response(pre_filt=(0.5, 1, 50, 100), water_level=44, inventory=inv, plot=True)

# Plot bandpass-filtered trace
trf = tr.copy().filter('bandpass', freqmin=1, freqmax=50, zerophase=True)
trf.plot(fig=plt.figure(), method='full')
plt.show()
