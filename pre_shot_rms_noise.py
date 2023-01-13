import matplotlib.pyplot as plt
import numpy as np

from utils import get_shots, get_stations, get_waveforms_shot

WIN_DUR = 20  # [s] Seconds before shot time to include in RMS velocity calculation

df = get_shots()
inv = get_stations()

# Sort shots by time
df = df.sort_values(by='time')

rms_shot_dict = {}
for shot in df.index:
    print(shot)
    st = get_waveforms_shot(shot)

    # Merge, if needed (IMPORTANT since AO4 gaps are in the pre-shot RMS window!)
    if shot == 'AO4':
        st.merge()

    st.trim(df.loc[shot].time - WIN_DUR, df.loc[shot].time)

    # Convert to units of m/s
    # st.remove_response() is SLOW; I think just sensitivity removal is OK here?
    if shot == 'Y4':  # Shot Y4 data are from Brandon, so they don't match IRIS inv
        for tr in st:
            fudge_factor = 87921  # TODO: See _plot_node_shot_gather.py
            tr.data *= fudge_factor
            try:
                tr.remove_sensitivity(inventory=inv)
            except ValueError:
                print(f'{tr.id} not found in inventory. Removing.')
                st.remove(tr)
                continue
    else:
        st.remove_sensitivity(inventory=inv)

    # Compute RMS (within WIN_DUR) for each trace
    rms_vals = []
    for tr in st:
        rms_vals.append(np.sqrt(np.mean(tr.data**2)))

    rms_shot_dict[shot] = np.array(rms_vals)

#%% Make bar chart

no_detect_color = 'lightgray'

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.bar(
    x=range(len(rms_shot_dict)),
    # Take the median RMS value to get a single number for each shot, in μm/s
    height=[np.median(rms_vals) * 1e6 for rms_vals in rms_shot_dict.values()],
    tick_label=list(rms_shot_dict.keys()),
    color=['black' if detect else no_detect_color for detect in df.gcas_on_nodes],
    edgecolor='black',
    width=0.6,
)
ax.set_ylabel(f'Median RMS velocity (μm/s),\n{WIN_DUR} s window pre-shot')

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
