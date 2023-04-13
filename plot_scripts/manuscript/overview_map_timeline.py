import os
import subprocess
import tempfile
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmt
from matplotlib.colors import to_hex
from obspy import UTCDateTime
from pygmt.datasets import load_earth_relief

from utils import (
    FULL_REGION,
    INNER_RING_REGION,
    NODAL_WORKING_DIR,
    get_shots,
    get_stations,
)

FONT_SIZE = 10  # [pt]
plt.rcParams.update({'font.size': FONT_SIZE})

# --------------------------------------------------------------------------------------
# (a) Overview map
# --------------------------------------------------------------------------------------

# Set PyGMT defaults
pygmt.config(MAP_FRAME_TYPE='plain', FORMAT_GEO_MAP='D', FONT='10p')

# Get shot info
df = get_shots()

# Get DEM to use for obtaining real station elevations
dem = load_earth_relief(
    resolution='01s', registration='gridline', region=INNER_RING_REGION
)

# Get node lat/lon/elevation
net = get_stations()[0]
sta_lons = [sta.longitude for sta in net]
sta_lats = [sta.latitude for sta in net]
elevations = [
    dem.sel(lon=sta.longitude, lat=sta.latitude, method='nearest').values.tolist()
    for sta in net
]

# Plot
fig_gmt = pygmt.Figure()
shaded_relief = pygmt.grdgradient(
    '@earth_relief_01s_g', region=INNER_RING_REGION, azimuth=-45.0, normalize='t1+a0'
)
pygmt.makecpt(cmap='gray', series=[-2, shaded_relief.values.max()])  # -2 is nice(?)
fig_gmt.grdimage(
    shaded_relief,
    cmap=True,
    projection='M5.2i',
    region=INNER_RING_REGION,
    frame=False,
    transparency=30,
)
# Plot nodes
pygmt.makecpt(
    series=[np.min(elevations), np.max(elevations)],
    cmap=Path().home() / 'Documents' / 'CETperceptual_GMT' / 'CET-L10.cpt',
)
fig_gmt.plot(
    x=sta_lons, y=sta_lats, color=elevations, style='c0.05i', cmap=True, pen='black'
)
# Plot shots
size_1000_lb = 0.2  # [in] Marker size for the smaller, 1000-lb shots
shot_kw = dict(style='si', pen='0.25p')
scale = size_1000_lb / 1000  # [in/lb] Scale shot weights to marker sizes
fig_gmt.plot(
    x=df.lon[df.gcas_on_nodes],
    y=df.lat[df.gcas_on_nodes],
    size=df[df.gcas_on_nodes].weight_lb * scale,
    color='black',
    **shot_kw,
)
fig_gmt.plot(
    x=df.lon[~df.gcas_on_nodes],
    y=df.lat[~df.gcas_on_nodes],
    size=df[~df.gcas_on_nodes].weight_lb * scale,
    color='white',
    **shot_kw,
)
# Plot shot names
justify = 'CM'
fontsize = 5  # [pts]
fig_gmt.text(
    x=df.lon[df.gcas_on_nodes],
    y=df.lat[df.gcas_on_nodes],
    text=df[df.gcas_on_nodes].index,
    font=f'{fontsize}p,white',
    justify=justify,
)
fig_gmt.text(
    x=df.lon[~df.gcas_on_nodes],
    y=df.lat[~df.gcas_on_nodes],
    text=df[~df.gcas_on_nodes].index,
    font=f'{fontsize}p',
    justify=justify,
)
# Add frame on top
fig_gmt.basemap(map_scale='g-122.04/46.09+w5+f+l', frame=['WESN', 'a0.1f0.02'])
# Colorbar, shifted to the left
fig_gmt.colorbar(frame='+l"Node elevation (m)"', position='JBL+jML+o0/-0.5i+h')
# Inset map showing all shots
with fig_gmt.inset(position='JTR+w1.95i+o-0.71i/-1.3i', box='+gwhite+p1p'):
    # Plot patch corresponding to main map extent
    fig_gmt.plot(
        data=[
            [
                INNER_RING_REGION[0],
                INNER_RING_REGION[2],
                INNER_RING_REGION[1],
                INNER_RING_REGION[3],
            ]
        ],
        style='r+s',
        color='lightgray',
        region=FULL_REGION,
        projection='M?',
    )
    # Plot nodes as tiny black dots
    fig_gmt.plot(x=sta_lons, y=sta_lats, color='black', style='c0.01i')
    # Plot shots
    scale = 0.00007  # [in/lb] Scale shot weights to marker sizes
    fig_gmt.plot(
        x=df[df.gcas_on_nodes].lon,
        y=df[df.gcas_on_nodes].lat,
        size=df[df.gcas_on_nodes].weight_lb * scale,
        color='black',
        **shot_kw,
    )
    fig_gmt.plot(
        x=df[~df.gcas_on_nodes].lon,
        y=df[~df.gcas_on_nodes].lat,
        size=df[~df.gcas_on_nodes].weight_lb * scale,
        color='white',
        **shot_kw,
    )
    # Plot shot names (only those not appearing in main map!)
    inner_ring = (
        (df.lon > INNER_RING_REGION[0])
        & (df.lon < INNER_RING_REGION[1])
        & (df.lat > INNER_RING_REGION[2])
        & (df.lat < INNER_RING_REGION[3])
    )
    df_label = df[~inner_ring]
    # Place labels INSIDE the big shots
    df_big = df_label[df_label.weight_lb == 2000]
    justify = 'CM'
    fig_gmt.text(
        x=df_big[df_big.gcas_on_nodes].lon,
        y=df_big[df_big.gcas_on_nodes].lat,
        text=df_big[df_big.gcas_on_nodes].index,
        justify=justify,
        font='5p,white',
    )
    fig_gmt.text(
        x=df_big[~df_big.gcas_on_nodes].lon,
        y=df_big[~df_big.gcas_on_nodes].lat,
        text=df_big[~df_big.gcas_on_nodes].index,
        justify=justify,
        font='5p',
    )
    # Place labels OUTSIDE of the small shots - custom locations
    df_small = df_label[df_label.weight_lb == 1000]
    is_top = dict(AO2=False, AO3=True, AO4=True, X3=False, X6=True, Y3=True, Y6=False)
    assert list(is_top.keys()) == df_small.index.tolist()  # Trivial check
    for _, shot in df_small.iterrows():
        top = is_top[shot.name]
        if top:
            justify = 'CB'
            offset = '0/0.04i'
        else:
            justify = 'CT'
            offset = '0/-0.04i'
        fig_gmt.text(
            x=shot.lon,
            y=shot.lat,
            text=shot.name,
            justify=justify,
            offset=offset,
            font='5p',
        )
    fig_gmt.basemap(map_scale=f'g{np.mean(FULL_REGION[:2])}/45.75+w50')

# --------------------------------------------------------------------------------------
# Make fancy symbol for weather stations showing all colors
# --------------------------------------------------------------------------------------
shades = [to_hex(color) for color in plt.get_cmap('tab20b').colors[:4]]  # 4 stations!
met_pen = '0.25p'  # Overall symbol pen thickness
met_size = 0.15  # [inches] Overall symbol size (diameter of circumscribing circle)
width = (met_size / 2) * np.sqrt(3) / 2  # [inches] Side length of subtriangle
kwargs = dict(x=0, y=0, no_clip=True)  # Common params for all plotted symbol components
fig_symbol = pygmt.Figure()
fig_symbol.plot(  # Transparent background circle to set the proper exported symbol size
    region=[-1, 1, -1, 1],
    style=f'c{met_size}i',
    color='white',
    transparency=100,
    **kwargs,
)
fig_symbol.plot(  # RIGHT SUBTRIANGLE
    style=f'i{met_size / 2}i',
    color=shades[0],
    xshift=f'a{width / 2}i',
    yshift=f'a{(met_size / 2) / 4}i',
    **kwargs,
)
fig_symbol.plot(  # CENTER SUBTRIANGLE
    style=f't{met_size / 2}i',
    color=shades[1],
    **kwargs,
)
fig_symbol.plot(  # LEFT SUBTRIANGLE
    style=f'i{met_size / 2}i',
    color=shades[2],
    xshift=f'a{-width / 2}i',
    yshift=f'a{(met_size / 2) / 4}i',
    **kwargs,
)
fig_symbol.plot(  # BOTTOM SUBTRIANGLE
    style=f'i{met_size / 2}i',
    color=shades[3],
    yshift=f'a{-(met_size / 2) / 2}i',
    **kwargs,
)
fig_symbol.plot(style=f'i{met_size}i', pen=met_pen, **kwargs)  # Actual overall symbol!
# --------------------------------------------------------------------------------------

# Make legend
with NamedTemporaryFile(dir=Path.home() / '.gmt', suffix='.eps') as sym_f:
    prefix = os.path.splitext(sym_f.name)[0]  # Remove '.eps'
    fig_symbol.psconvert(prefix=prefix, fmt='e')
    symbolname = Path(sym_f.name).stem
    with NamedTemporaryFile(mode='w') as f:
        f.write(
            f'S - {shot_kw["style"][0]} {size_1000_lb}i black {shot_kw["pen"]} - Shot w/ GCAs\n'
        )
        f.write(
            f'S - {shot_kw["style"][0]} {size_1000_lb}i white {shot_kw["pen"]} - Shot w/o GCAs\n'
        )
        f.write(f'S - k{symbolname} {met_size}i - - - Weather station\n')
        f.flush()
        fig_gmt.legend(
            f.name, position='JBR+jML+o-0.6i/-0.65i+l1.5'
        )  # +l controls line spacing!

# --------------------------------------------------------------------------------------
# (b) Shot times and temperature time series from weather stations
# --------------------------------------------------------------------------------------

# Read in CSV files containing temp data (this code is format-specific!)
# https://explore.synopticdata.com/metadata/map/4619,-12228,10?vars=air_temp&bbox=-122.42,46.06,-121.98,46.36&status=ACTIVE
temp_df = pd.DataFrame()
met_station_coords = {}  # Each value is a list of [lat, lon, elevation in feet]
for file in (NODAL_WORKING_DIR / 'data' / 'weather').glob('*.csv'):
    temp_df_station = pd.read_csv(file, comment='#')
    met_station_coords[temp_df_station.Station_ID[1]] = np.loadtxt(
        file, skiprows=6, max_rows=2, comments=None, usecols=2
    ).tolist()
    met_station_coords[temp_df_station.Station_ID[1]] += [
        np.loadtxt(file, skiprows=8, max_rows=1, comments=None, usecols=3).tolist()
    ]
    temp_df = pd.concat([temp_df, temp_df_station])
temp_df.dropna(inplace=True)
temp_df.air_temp_set_1 = temp_df.air_temp_set_1.astype(float)

# Sort dictionary of met stations by increasing elevation
met_station_coords = dict(sorted(met_station_coords.items(), key=lambda x: x[1][2]))

# Time ranges for two sections of plot
AX1_TIME_RANGE = UTCDateTime('2014-07-24'), UTCDateTime('2014-07-26')
AX2_TIME_RANGE = UTCDateTime('2014-08-01'), UTCDateTime('2014-08-02')

# Make plot
MPL_PLOT_WIDTH = 6  # [in]
fig, (ax1, ax2) = plt.subplots(
    ncols=2,
    sharey=True,
    gridspec_kw=dict(width_ratios=(np.diff(AX1_TIME_RANGE), np.diff(AX2_TIME_RANGE))),
    figsize=(MPL_PLOT_WIDTH, 1.9),
)

# Compute static [dry air] sound speed, see first equation for c_air in
# https://en.wikipedia.org/wiki/Speed_of_sound#Speed_of_sound_in_ideal_gases_and_air
gamma = 1.4
R = 8.31446  # [J/(mol*K)]
M_air = 0.02896  # [kg/mol]
temp_df['c'] = np.sqrt(gamma * (R / M_air) * (temp_df.air_temp_set_1 + 273.15))  # [m/s]

# Plot estimated speed of sound
LINE_KWARGS = dict(lw=1, solid_capstyle='round')
for ax, time_range in zip([ax1, ax2], [AX1_TIME_RANGE, AX2_TIME_RANGE]):
    clip_slice = slice(None, 2)
    reg_slice = slice(1, None)
    if ax == ax2:
        clip_slice = slice(-2, None)
        reg_slice = slice(None, -1)
    for station, shade in zip(met_station_coords.keys(), shades):
        # Plot speed of sound on panel (b)
        station_df = temp_df[temp_df.Station_ID == station]
        station_df = station_df[
            (station_df.Date_Time >= time_range[0])
            & (station_df.Date_Time <= time_range[1])
        ]
        time = [UTCDateTime(t).matplotlib_date for t in station_df.Date_Time]
        ax.plot(
            time[reg_slice],
            station_df.c.iloc[reg_slice],
            color=shade,
            clip_on=False,
            **LINE_KWARGS,
        )
        ax.plot(
            time[clip_slice],
            station_df.c.iloc[clip_slice],
            color=shade,
            clip_on=True,
            **LINE_KWARGS,
        )
        # Plot station location on panel (a)
        fig_gmt.plot(
            x=met_station_coords[station][1],
            y=met_station_coords[station][0],
            color=shade,
            style=f'i{met_size}i',
            pen=met_pen,
        )
ax1.set_ylim(334, 348)
ax1.set_ylabel('Static sound\nspeed (m/s)')

# Plot shot times
df = get_shots()
df_sort = df.sort_values(by='time')
df_sort['yloc'] = np.array([354, 351, 348, 345] * 6)[:-1]  # Manual stagger
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
        **kwargs,
    )
    ax.scatter(
        [t.matplotlib_date for t in df_section[df_section.gcas_on_nodes].time],
        df_section[df_section.gcas_on_nodes].yloc,
        s=df_section[df_section.gcas_on_nodes].weight_lb * scale,
        color='black',
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

# Cleanup
for ax in ax1, ax2:
    ax.spines['top'].set_visible(False)
    locator = ax.xaxis.set_major_locator(mdates.HourLocator(range(0, 24, 12)))
    formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
    formatter.zero_formats[3] = '%-d\n%b.'
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(mdates.HourLocator(range(0, 24, 3)))
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.tick_params(which='both', left=False, right=True)
ax1.set_xlim([t.matplotlib_date for t in AX1_TIME_RANGE])
ax2.set_xlim([t.matplotlib_date for t in AX2_TIME_RANGE])
ax1.yaxis.set_minor_locator(plt.MultipleLocator(2))
ax1.yaxis.set_major_locator(plt.MultipleLocator(4))

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

fig.tight_layout(pad=0)
fig.subplots_adjust(wspace=0.15, top=0.71)

# --------------------------------------------------------------------------------------
# Now combine (b) into the GMT figure containing (a)
# --------------------------------------------------------------------------------------

# Note: EPS does not support transparency!
Y_OFF = 1.2  # [in] For (b), from bottom edge of (a) axes (positive down)
with tempfile.NamedTemporaryFile(suffix='.eps') as f:
    fig.savefig(f.name)
    fig_gmt.image(
        f.name,
        position=f'JBC+w{MPL_PLOT_WIDTH}i+o-0.26i/{Y_OFF}i',  # "Tools -> Show Inspector" CLUTCH here!
    )
plt.close(fig)

# Plot (a) and (b) tags
tag_kwargs = dict(position='TL', no_clip=True, justify='TR', font='12p,Helvetica-Bold')
x_offset = -0.2  # [in]
fig_gmt.text(text='(a)', offset=f'{x_offset}i/0', **tag_kwargs)
fig_gmt.text(text='(b)', offset=f'{x_offset}i/{-5.15 - Y_OFF}i', **tag_kwargs)

fig_gmt.show()

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

# fig_gmt.savefig(Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve() / 'overview_map_timeline.png', dpi=600, resize='+m2p')
