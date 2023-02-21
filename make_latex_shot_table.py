import os
from pathlib import Path

from utils import get_shots

df = get_shots()

df = df.reset_index(level=0)  # Convert the index (which is shot name) to a column
df = df.sort_values(by='time')  # Sort from earliest to latest in time

# Format column names for table heading
columns = dict(
    shot='Shot',
    lat=r'Lat. (\textdegree)',
    lon=r'Lon. (\textdegree)',
    elev_m='Elev. (m)',
    weight_lb='Weight (lb)',
    time='UTC time',
    gcas_on_nodes='GCAs?',
)
columns.update((k, rf'\textbf{{{v}}}') for k, v in columns.items())  # Make header bold


# Color rows based on GCA presence/absence (conditional formatting, basically)
def color_rows_by_gca_presence(row):
    if not row[columns['gcas_on_nodes']]:
        return ['color: {lightgray}'] * len(row)
    else:
        return [None] * len(row)


# Style and output .tex file
df.rename(columns=columns).style.hide(axis='index').format(
    {
        columns['time']: lambda t: '{}'.format(t.strftime('%Y-%m-%d %H:%M:%S')),
        columns['gcas_on_nodes']: lambda x: rf'\texttt{{{x}}}',
    }
).format(lambda x: f'${x:.4f}$', subset=[columns['lat'], columns['lon']]).format(
    lambda x: f'${x:g}$', subset=[columns['elev_m'], columns['weight_lb']]
).apply(
    color_rows_by_gca_presence, axis='columns'
).to_latex(
    Path(os.environ['NODAL_FIGURE_DIR']).parent / 'shot_table.tex', hrules=True
)
