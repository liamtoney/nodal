import os
from pathlib import Path

import pandas as pd

from utils import NODAL_WORKING_DIR, get_shots

df = get_shots()

# Read in individual CSV files (one per shot)
shot_data = []
for shot in df.index:
    shot_data.append(
        pd.read_csv(NODAL_WORKING_DIR / 'shot_gather_measurements' / f'{shot}.csv')
    )
# RMS in μm/s and distance in km
df['pre_shot_rms'] = [data.pre_shot_rms.median() * 1e6 for data in shot_data]
df['median_dist'] = [data.dist_m.median() / 1000 for data in shot_data]

df = df.reset_index(level=0)  # Convert the index (which is shot name) to a column
df = df.sort_values(by='time')  # Sort from earliest to latest in time


# Convert pounds to kilograms and round
nearest_kg = 50  # [kg]
df.weight_lb = (nearest_kg * round((df.weight_lb / 2.20462) / nearest_kg)).astype(int)

# Format column names for table heading (also determines the order AND which columns!)
columns = dict(
    shot=('', 'Shot'),
    time=('UTC time', r'(MM-DD hh:mm:ss)'),
    weight_lb=('', 'Weight (kg)'),  # Note we *are* actually doing the unit conversion!
    pre_shot_rms=('Median RMS', r'velocity (\textmu m/s)'),
    median_dist=('Median shot--node', 'distance (km)'),
    gcas_on_nodes=('', 'GCAs?'),
)
columns.update(
    (k, tuple(rf'\textbf{{{vi}}}' for vi in v)) for k, v in columns.items()
)  # Make header bold

# Subset table
df = df[columns.keys()]


df.columns = pd.MultiIndex.from_tuples(columns.values())


# Color rows based on GCA presence/absence (conditional formatting, basically)
def color_rows_by_gca_presence(row):
    if not row[columns['gcas_on_nodes']]:
        return ['color: {lightgray}'] * len(row)
    else:
        return [None] * len(row)


# Style and output .tex file
df.rename(columns=columns).style.hide(axis='index').format(
    {
        columns['time']: lambda t: '{}'.format(t.strftime('%m-%d %H:%M:%S')),
        columns['gcas_on_nodes']: lambda x: rf'\texttt{{{x}}}',
        columns['pre_shot_rms']: lambda x: f'{x:.2f}',
        columns['median_dist']: lambda x: f'{x:.1f}',
    }
).apply(color_rows_by_gca_presence, axis='columns').to_latex(
    Path(os.environ['NODAL_FIGURE_DIR']).parent / 'shot_table.tex',
    hrules=True,
    column_format='c' * len(df.columns),
)
