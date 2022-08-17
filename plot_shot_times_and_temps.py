import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime

from utils import NODAL_WORKING_DIR, get_shots
from utils.utils import INNER_RING_REGION  # Since I haven't fully exposed this constant

# Read in CSV files containing temp data (this code is format-specific!)
temp_df = pd.DataFrame()
for file in (NODAL_WORKING_DIR / 'data' / 'weather').glob('*.csv'):
    temp_df = pd.concat([temp_df, pd.read_csv(file, comment='#')])
temp_df.dropna(inplace=True)
temp_df.air_temp_set_1 = temp_df.air_temp_set_1.astype(float)

# Make plot
fig, ax = plt.subplots(figsize=(14, 3.5))

# Plot estimated speed of sound (relies on evenly sampled temps from above to get mean)
df_mean = temp_df.groupby('Date_Time').mean()
# Below calc from https://en.wikipedia.org/wiki/Speed_of_sound#Practical_formula_for_dry_air
c = 20.05 * np.sqrt(df_mean.air_temp_set_1 + 273.15)  # [m/s]
ax.plot([UTCDateTime(t).matplotlib_date for t in df_mean.index], c, color='black', lw=2)
ax.set_ylabel('Estimated dry air\nsound speed (m/s)')
ax.set_ylim(334, 348)

# Plot shot times
df = get_shots()
df_sort = df.sort_values(by='time')
in_main_map = (
    (df_sort.lon > INNER_RING_REGION[0])
    & (df_sort.lon < INNER_RING_REGION[1])
    & (df_sort.lat > INNER_RING_REGION[2])
    & (df_sort.lat < INNER_RING_REGION[3])
)
df_sort['yloc'] = np.array([352, 350, 348, 346] * 6)[:-1]  # Manual stagger
for _, row in df_sort.iterrows():
    ax.plot(
        [row.time.matplotlib_date, row.time.matplotlib_date],
        [ax.get_ylim()[0], row.yloc],
        clip_on=False,
        linestyle='--',
        color='dimgray',
        lw=0.5,
        zorder=-5,
    )  # Connecting lines
ax.scatter(
    [t.matplotlib_date for t in df_sort[in_main_map].time],
    df_sort[in_main_map].yloc,
    s=160,
    facecolors='black',
    marker='s',
    clip_on=False,
)
ax.scatter(
    [t.matplotlib_date for t in df_sort[~in_main_map].time],
    df_sort[~in_main_map].yloc,
    edgecolors='black',
    facecolors='white',
    marker='s',
    clip_on=False,
)
for _, row in df_sort[in_main_map].iterrows():
    ax.text(
        row.time.matplotlib_date,
        row.yloc,
        row.name,
        color='white',
        va='center',
        ha='center',
        fontsize=8,
        clip_on=False,
    )

# Plot temp data
alpha = 0.4
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
ax2.legend(ncol=2, loc='lower center', frameon=False)
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

fig.tight_layout()
fig.subplots_adjust(hspace=0)
fig.show()

# fig.savefig(NODAL_WORKING_DIR / 'figures' / 'shot_times_temps.png', dpi=300, bbox_inches='tight')
