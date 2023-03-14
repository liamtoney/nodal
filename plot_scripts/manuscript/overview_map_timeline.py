import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from pygmt.datasets import load_earth_relief

from utils import (
    INNER_RING_REGION,
    NODAL_WORKING_DIR,
    get_shots,
    get_stations,
    station_map,
)

# --------------------------------------------------------------------------------------
# (a) Overview map
# --------------------------------------------------------------------------------------

# Read in station info for plotting
net = get_stations()[0]

# Get DEM to use for obtaining real station elevations
dem = load_earth_relief(
    resolution='01s', registration='gridline', region=INNER_RING_REGION
)

# Plot
fig_gmt = station_map(
    [sta.longitude for sta in net],
    [sta.latitude for sta in net],
    [
        dem.sel(lon=sta.longitude, lat=sta.latitude, method='nearest').values
        for sta in net
    ],
    cbar_label='Node elevation (m)',
    cmap='dem1',
    plot_inset=True,
)

# --------------------------------------------------------------------------------------
# (b) Shot times and temperature time series from weather stations
# --------------------------------------------------------------------------------------

# Read in CSV files containing temp data (this code is format-specific!)
temp_df = pd.DataFrame()
for file in (NODAL_WORKING_DIR / 'data' / 'weather').glob('*.csv'):
    temp_df = pd.concat([temp_df, pd.read_csv(file, comment='#')])
temp_df.dropna(inplace=True)
temp_df.air_temp_set_1 = temp_df.air_temp_set_1.astype(float)

# Time ranges for two sections of plot
AX1_TIME_RANGE = UTCDateTime('2014-07-24'), UTCDateTime('2014-07-26')
AX2_TIME_RANGE = UTCDateTime('2014-08-01'), UTCDateTime('2014-08-02')

# Make plot
MPL_PLOT_WIDTH = 5.33  # [in]
fig, (ax1, ax2) = plt.subplots(
    ncols=2,
    sharey=True,
    gridspec_kw=dict(width_ratios=(np.diff(AX1_TIME_RANGE), np.diff(AX2_TIME_RANGE))),
    figsize=(MPL_PLOT_WIDTH, 3),
)

# Plot estimated speed of sound (relies on evenly sampled temps from above to get mean)
df_mean = temp_df.groupby('Date_Time').mean(numeric_only=True)
# Below calc from https://en.wikipedia.org/wiki/Speed_of_sound#Practical_formula_for_dry_air
c = 20.05 * np.sqrt(df_mean.air_temp_set_1 + 273.15)  # [m/s]
for ax, time_range in zip([ax1, ax2], [AX1_TIME_RANGE, AX2_TIME_RANGE]):
    mask = (df_mean.index >= time_range[0]) & (df_mean.index <= time_range[1])
    reg_slice = slice(None, 2)
    round_slice = slice(1, None)
    if ax == ax2:
        reg_slice = slice(None, 0)
        round_slice = slice(None, None)
    ax.plot(
        [UTCDateTime(t).matplotlib_date for t in df_mean[mask].index][reg_slice],
        c[mask][reg_slice],
        color='black',
    )
    ax.plot(
        [UTCDateTime(t).matplotlib_date for t in df_mean[mask].index][round_slice],
        c[mask][round_slice],
        color='black',
        clip_on=False,
        solid_capstyle='round',
    )
    ax.set_ylim(334, 348)
ax1.set_ylabel('Estimated dry air\nsound speed (m/s)')

# Plot shot times
df = get_shots()
df_sort = df.sort_values(by='time')
df_sort['yloc'] = np.array([357, 353, 349, 345] * 6)[:-1]  # Manual stagger
for _, row in df_sort.iterrows():
    for ax, time_range in zip([ax1, ax2], [AX1_TIME_RANGE, AX2_TIME_RANGE]):
        if (row.time >= time_range[0]) & (row.time <= time_range[1]):
            ax.plot(
                [row.time.matplotlib_date, row.time.matplotlib_date],
                [ax.get_ylim()[0], row.yloc],
                clip_on=False,
                linestyle='--',
                color='black',
                lw=0.5,
                zorder=-5,
            )  # Connecting lines
size_1000_lb = 100  # Marker size for the smaller, 1000-lb shots
kwargs = dict(edgecolor='black', lw=0.5, marker='s', clip_on=False)
scale = size_1000_lb / 1000  # [1/lb] Scale shot weights to marker sizes
for ax, time_range in zip([ax1, ax2], [AX1_TIME_RANGE, AX2_TIME_RANGE]):
    df_section = df_sort[
        (df_sort.time >= time_range[0]) & (df_sort.time <= time_range[1])
    ]
    ax.scatter(
        [t.matplotlib_date for t in df_section[~df_section.gcas_on_nodes].time],
        df_section[~df_section.gcas_on_nodes].yloc,
        s=df_section[~df_section.gcas_on_nodes].weight_lb * scale,
        color='white',
        label='GCAs not observed',
        **kwargs,
    )
    ax.scatter(
        [t.matplotlib_date for t in df_section[df_section.gcas_on_nodes].time],
        df_section[df_section.gcas_on_nodes].yloc,
        s=df_section[df_section.gcas_on_nodes].weight_lb * scale,
        color='black',
        label='GCAs observed',
        **kwargs,
    )
    for _, row in df_section.iterrows():
        ax.text(
            row.time.matplotlib_date,
            row.yloc,
            row.name,
            color='white' if row.gcas_on_nodes else 'black',
            va='center',
            ha='center',
            fontsize=5,
            clip_on=False,
        )

# Plot temp data
for ax, time_range in zip([ax1, ax2], [AX1_TIME_RANGE, AX2_TIME_RANGE]):
    ax_twin = ax.twinx()
    ax.set_zorder(1)
    ax.patch.set_alpha(0)
    for station in sorted(temp_df.Station_ID.unique()):
        station_df = temp_df[temp_df.Station_ID == station]
        station_df = station_df[
            (station_df.Date_Time >= time_range[0])
            & (station_df.Date_Time <= time_range[1])
        ]
        ax_twin.plot(
            [UTCDateTime(t).matplotlib_date for t in station_df.Date_Time],
            station_df.air_temp_set_1,
            label=station,
            solid_capstyle='round',
            clip_on=False,
        )
    ax_twin.spines['top'].set_visible(False)
    ax_twin.spines['left'].set_visible(False)
    ax_twin.set_ylim(5, 30)  # HARD-CODED based on data range
    if ax == ax2:
        ax_twin.set_ylabel('Temperature (°C)')
        ax_twin.yaxis.set_minor_locator(plt.MultipleLocator(5))
    else:
        ax_twin.spines['right'].set_visible(False)
        ax_twin.tick_params(right=False, labelright=False)
# leg_x = 0.17
# leg = ax_twin.legend(
#     ncol=2, loc='lower right', bbox_to_anchor=(leg_x, 1), frameon=False
# )

# Cleanup
for ax in ax1, ax2:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    locator = ax.xaxis.set_major_locator(mdates.HourLocator(range(0, 24, 12)))
    formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
    formatter.zero_formats[3] = '%-d\n%B'
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(mdates.HourLocator(range(0, 24, 3)))
ax2.spines['left'].set_visible(False)
ax2.tick_params(left=False)
ax1.set_xlim([t.matplotlib_date for t in AX1_TIME_RANGE])
ax2.set_xlim([t.matplotlib_date for t in AX2_TIME_RANGE])
ax2.set_xticklabels(
    [tl._text.replace('August', 'Aug.') for tl in ax2.get_xticklabels()]
)

# Plot little diagonal lines to show broken axis
dx = 0.05
dy = 0.7
for ax in ax1, ax2:
    if ax == ax1:
        x0 = ax.get_xlim()[1]
    else:
        x0 = ax.get_xlim()[0]
    y0 = ax.get_ylim()[0]
    ax.plot(
        (x0 - dx, x0 + dx),
        (y0 - dy, y0 + dy),
        color='black',
        clip_on=False,
        lw=plt.rcParams['axes.linewidth'],
    )

fig.tight_layout()
fig.subplots_adjust(wspace=0.2)

# --------------------------------------------------------------------------------------
# Now combine (b) into the GMT figure containing (a)
# --------------------------------------------------------------------------------------

# Note: EPS does not support transparency!
with tempfile.NamedTemporaryFile(suffix='.eps') as f:
    fig.savefig(f.name, bbox_inches='tight')
    fig_gmt.image(
        f.name,
        position=f'JBC+w{MPL_PLOT_WIDTH}i+o-0.11i/1.3i',  # "Tools -> Show Inspector" CLUTCH here!
    )
plt.close(fig)

# Plot (a) and (b) tags
tag_kwargs = dict(position='TL', no_clip=True, justify='TR', font='18p,Helvetica-Bold')
x_offset = -0.2  # [in]
fig_gmt.text(text='(a)', offset=f'{x_offset}i/0', **tag_kwargs)
fig_gmt.text(text='(b)', offset=f'{x_offset}i/-5i', **tag_kwargs)

fig_gmt.show(method='external')

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

# fig_gmt.savefig(Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve() / 'overview_map_timeline.png', dpi=600)
