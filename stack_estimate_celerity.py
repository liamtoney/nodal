"""
For each shot, process the data and then use the shortest diffracted paths to find the
celerity which gives the largest stack value (via a simple 1D search).
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import NODAL_WORKING_DIR, get_shots, get_waveforms_shot

FREQMIN = 5  # [Hz]
FREQMAX = 50  # [Hz]
STA = 0.2  # [s]
LTA = 2  # [s]

PLOT_STACKS = False

df_shot = get_shots()

inc = 0.5
trial_celerities = np.arange(325, 350 + inc, inc)  # [m/s]

shot_stacks = {}
for shot in df_shot[df_shot.gcas_on_nodes].index:
    print(shot)

    # Get measurements and data
    df = pd.read_csv(NODAL_WORKING_DIR / 'shot_gather_measurements' / f'{shot}.csv')
    st = get_waveforms_shot(shot, processed=True)

    # Process
    st.detrend('demean')
    st.taper(0.05)
    st.filter('bandpass', freqmin=FREQMIN, freqmax=FREQMAX, zerophase=True)
    st.trigger('classicstalta', sta=STA, lta=LTA)
    st.merge(fill_value=np.nan)

    # Ensure data are all same length (UGLY)
    vc = pd.Series([tr.stats.npts for tr in st]).value_counts()
    most_common_npts = vc[vc == vc.max()].index.values[0]
    st = st.select(npts=most_common_npts)
    print(f'Removed {vc[vc.index != most_common_npts].sum()} Trace(s)')

    # Ensure that the order of the measurements and the Traces in the Stream match
    df = df[df.station.isin([int(tr.stats.station) for tr in st])]
    assert [tr.stats.station for tr in st] == df.station.astype(str).tolist()

    # Plot as a check
    if False:
        sorted_idx = df.diffracted_path_length.values.argsort()  # Increasing distance
        fig, ax = plt.subplots()
        pcm = ax.pcolormesh(
            st[0].times(reftime=df_shot.loc[shot].time),
            df.diffracted_path_length.values[sorted_idx],
            np.array([tr.data for tr in st])[sorted_idx, :],
        )
        fig.colorbar(pcm)
        fig.show()

    if PLOT_STACKS:
        fig_stack, ax_stack = plt.subplots()

    stack_maxes = []
    for trial_celerity in tqdm(trial_celerities):
        travel_time_remove = df.diffracted_path_length.values / trial_celerity
        st_shifted = st.copy()
        for tr, time in zip(st_shifted, travel_time_remove):
            starttime = tr.stats.starttime + time
            tr.trim(starttime, starttime + 60, pad=True, fill_value=np.nan)

        # Ensure data are all same length (UGLY)
        vc = pd.Series([tr.stats.npts for tr in st_shifted]).value_counts()
        most_common_npts = vc[vc == vc.max()].index.values[0]
        st_shifted = st_shifted.select(npts=most_common_npts)
        if len(vc) > 1:  # If we have multiple Trace lengths
            print(f'Removed {vc[vc.index != most_common_npts].sum()} shifted Trace(s)')

        stack = np.nansum([tr.data for tr in st_shifted], axis=0)
        stack /= st_shifted.count()  # KEY: Normalize by # of stations!

        stack_maxes.append(stack.max())

        if PLOT_STACKS:
            ax_stack.plot(st_shifted[0].times(), stack, label=f'{trial_celerity} m/s')

        # Plot as a check
        if False:
            sorted_idx = (
                df.diffracted_path_length.values.argsort()
            )  # Increasing distance
            fig, ((ax1, _), (ax2, cax)) = plt.subplots(
                nrows=2,
                ncols=2,
                sharex='col',
                gridspec_kw=dict(
                    height_ratios=(1, 6),
                    width_ratios=(50, 1),
                    wspace=0.1,
                    hspace=0,
                ),
            )
            _.remove()
            ax1.fill_between(st_shifted[0].times(), stack, lw=0, color='tab:gray')
            ax1.autoscale(tight=True)
            ax1.axis('off')
            pcm = ax2.pcolormesh(
                st_shifted[0].times(),
                df.diffracted_path_length.values[sorted_idx],
                np.array([tr.data for tr in st_shifted])[sorted_idx, :],
            )
            ax2.set_title(
                f' {trial_celerity} m/s',
                color='white',
                weight='bold',
                loc='left',
                fontsize=plt.rcParams['font.size'],
            )
            ax2.set_title(
                f'{stack.max():.2f} ',
                color='white',
                weight='bold',
                loc='right',
                fontsize=plt.rcParams['font.size'],
            )
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Shortest diffracted distance (m)')
            fig.colorbar(pcm, cax=cax, label='STA/LTA amplitude')
            fig.show()

    shot_stacks[shot] = np.array(stack_maxes)

    if PLOT_STACKS:
        ax_stack.legend(title=r'${\bf Celerity}$', frameon=False)
        ax_stack.set_xlabel('Time (s)')
        ax_stack.set_ylabel('Sum stack (normalized by # of stations)')
        fig_stack.show()

#%% Plot all stacks

# Sort by descending order of stack max
shot_stacks = dict(sorted(shot_stacks.items(), key=lambda x: x[1].max(), reverse=True))

fig, ax = plt.subplots(figsize=(3.7, 4.7))
min_stack = np.min([stack_function.min() for stack_function in shot_stacks.values()])
for shot, stack_function in shot_stacks.items():
    print(f'{shot.rjust(3)}: {trial_celerities[stack_function.argmax()]:g} m/s')
    ax.fill_between(
        trial_celerities, min_stack, stack_function, lw=0, alpha=0.1, clip_on=False
    )
    ax.plot(
        trial_celerities,
        stack_function,
        label=shot,
        clip_on=False,
        solid_capstyle='round',
    )
ax.set_xlabel('Trial celerity (m/s)')
ax.set_ylabel('Sum stack (normalized by # of stations)')
ax.legend(title=r'${\bf Shot}$', frameon=False)
ax.autoscale(tight=True)
for side in 'top', 'right':
    ax.spines[side].set_visible(False)
for side in 'bottom', 'left':
    ax.spines[side].set_position(('outward', 10))
fig.tight_layout()
yticks = ax.get_yticks()
ax.spines['left'].set_bounds(yticks[1], yticks[-2])
fig.show()

#%% Export as JSON

celerity_estimates = {}
for shot, stack_function in shot_stacks.items():
    celerity_estimates[shot] = trial_celerities[stack_function.argmax()]

with open(
    NODAL_WORKING_DIR / 'shot_gather_measurements' / 'celerity_estimates.json', 'w'
) as f:
    json.dump(celerity_estimates, f, indent='\t')
