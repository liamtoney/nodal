import os
from pathlib import Path

import pandas as pd

from utils import NODAL_WORKING_DIR, get_shots

# Get the original shot DataFrame
df = get_shots()

# Convert shot weights from pounds to kilograms
nearest_kg = 1  # [kg]
df['weight_kg'] = (nearest_kg * round((df.weight_lb / 2.2046) / nearest_kg)).astype(int)

# Read in individual CSV files (one per shot) and add summarized values to DataFrame
shot_data = []
for shot in df.index:
    shot_data.append(
        pd.read_csv(NODAL_WORKING_DIR / 'shot_gather_measurements' / f'{shot}.csv')
    )
# Medians: RMS in μm/s and distance in km
df['pre_shot_rms'] = [data.pre_shot_rms.median() * 1e6 for data in shot_data]
df['median_dist'] = [data.dist_m.median() / 1000 for data in shot_data]

df = df.sort_values(by='time')  # Sort from earliest to latest in time
df = df.reset_index(level=0)  # Convert the index (which is shot name) to a column

# Format column names for table heading (also determines the order AND which columns!)
columns = dict(
    shot='Shot',
    time='UTC time in 2014\n(MM-DD hh:mm:ss)',
    weight_kg='Weight\n(kg)',
    pre_shot_rms='Median RMS\nvelocity (\\textmu m/s)',
    median_dist='Median shot--node\ndistance (km)',
    gcas_on_nodes='Coupled\narrivals?',
)

# Re-format any newlines present in the column names so LaTeX can handle them properly
# (also make the header bold!) — note that `\thead{}` requires `\usepackage{makecell}`
for k in columns.keys():
    lines = columns[k].split('\n')
    lines_bold = [rf'\textbf{{{line}}}' for line in lines]
    columns[k] = r'\thead{{{}}}'.format(r' \\ '.join(lines_bold))

# Subset table
df = df[columns.keys()]


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
    column_format='c' * len(df.columns),  # Center every column
)
