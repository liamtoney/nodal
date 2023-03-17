import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import LogLocator

from utils import NODAL_WORKING_DIR, get_stations

# Get unique responses
vc = pd.value_counts([sta[0].response for sta in get_stations()[0]])

# Plot
fig, axs = plt.subplots(
    nrows=2, ncols=len(vc), sharex=True, sharey='row', figsize=(15, 6)
)
for unique_response, count, ax_col in zip(vc.index, vc.values, axs.T):
    unique_response.plot(min_freq=0.1, axes=list(ax_col), plot_degrees=True)
    ref_val = unique_response.instrument_sensitivity.value
    atten = -12  # [dB]
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
