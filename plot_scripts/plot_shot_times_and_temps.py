import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime

from utils import NODAL_WORKING_DIR, get_shots

# Read in CSV files containing temp data (this code is format-specific!)
temp_df = pd.DataFrame()
for file in (NODAL_WORKING_DIR / 'data' / 'weather').glob('*.csv'):
    temp_df = pd.concat([temp_df, pd.read_csv(file, comment='#')])
temp_df.dropna(inplace=True)
temp_df.air_temp_set_1 = temp_df.air_temp_set_1.astype(float)

# Make plot
fig, ax = plt.subplots(figsize=(14, 3.5))

# Plot estimated speed of sound (relies on evenly sampled temps from above to get mean)
df_mean = temp_df.groupby('Date_Time').mean(numeric_only=True)
# Below calc from https://en.wikipedia.org/wiki/Speed_of_sound#Practical_formula_for_dry_air
c = 20.05 * np.sqrt(df_mean.air_temp_set_1 + 273.15)  # [m/s]
ax.plot([UTCDateTime(t).matplotlib_date for t in df_mean.index], c, color='black')
ax.set_ylabel('Estimated dry air\nsound speed (m/s)')
ax.set_ylim(334, 348)

# Plot shot times
df = get_shots()
df_sort = df.sort_values(by='time')
df_sort['yloc'] = np.array([352, 350, 348, 346] * 6)[:-1]  # Manual stagger
for _, row in df_sort.iterrows():
    ax.plot(
        [row.time.matplotlib_date, row.time.matplotlib_date],
        [ax.get_ylim()[0], row.yloc],
        clip_on=False,
        linestyle='--',
        color='black',
        lw=0.5,
        zorder=-5,
    )  # Connecting lines
size_1000_lb = 130  # Marker size for the smaller, 1000-lb shots
kwargs = dict(edgecolor='black', lw=0.5, marker='s', clip_on=False)
scale = size_1000_lb / 1000  # [1/lb] Scale shot weights to marker sizes
ax.scatter(
    [t.matplotlib_date for t in df_sort[~df_sort.gcas_on_nodes].time],
    df_sort[~df_sort.gcas_on_nodes].yloc,
    s=df_sort[~df_sort.gcas_on_nodes].weight_lb * scale,
    color='white',
    label='GCAs not observed',
    **kwargs,
)
ax.scatter(
    [t.matplotlib_date for t in df_sort[df_sort.gcas_on_nodes].time],
    df_sort[df_sort.gcas_on_nodes].yloc,
    s=df_sort[df_sort.gcas_on_nodes].weight_lb * scale,
    color='black',
    label='GCAs observed',
    **kwargs,
)
for _, row in df_sort.iterrows():
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
alpha = 0.1
ax2 = ax.twinx()
ax.set_zorder(1)
ax.patch.set_alpha(0)
for station in sorted(temp_df.Station_ID.unique()):
    station_df = temp_df[temp_df.Station_ID == station]
    ax2.plot(
        [UTCDateTime(t).matplotlib_date for t in station_df.Date_Time],
        station_df.air_temp_set_1,
        label=station,
        alpha=alpha,
    )
leg_y = 0
leg = ax2.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, leg_y), frameon=False)
for text in leg.texts:
    text.set_alpha(alpha)
ax2.set_ylabel('Temperature (°C)', alpha=alpha)
for l in ax2.get_yticklines():
    l.set_alpha(alpha)
for t in ax2.get_yticklabels():
    t.set_alpha(alpha)
ax2.set_ylim(top=30)

# Cleanup
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_alpha(alpha)

loc = ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%-d\n%B'))
ax.xaxis.set_minor_locator(mdates.HourLocator(range(0, 24, 6)))
ax.set_xlim(
    UTCDateTime('2014-07-24').matplotlib_date, UTCDateTime('2014-08-02').matplotlib_date
)

# Hacky legend (ensuring proper order, and that sizes reflect the 1000-lbs shots)
h, l = ax.get_legend_handles_labels()
leg = ax.legend(
    h[::-1], l[::-1], loc='lower left', bbox_to_anchor=(0.59, leg_y), frameon=False
)
for handle in leg.legendHandles:
    handle.set_sizes([size_1000_lb])

fig.tight_layout()
fig.subplots_adjust(hspace=0)
fig.show()

# fig.savefig(NODAL_WORKING_DIR / 'figures' / 'shot_times_temps.png', dpi=300, bbox_inches='tight')
