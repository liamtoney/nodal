import subprocess

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

tex_code_string = (
    df.rename(columns=columns)
    .style.hide(axis='index')
    .format({columns['time']: lambda t: '{}'.format(t.strftime('%Y-%m-%d %H:%M:%S'))})
    .format(lambda x: f'${x:.4f}$', subset=[columns['lat'], columns['lon']])
    .format(lambda x: f'${x:g}$', subset=[columns['elev_m'], columns['weight_lb']])
    .to_latex(hrules=True, column_format='l' * df.shape[1])
).rstrip()

# Form LaTeX code and add to clipboard for pasting into Overleaf
subprocess.run('pbcopy', text=True, input=tex_code_string)
print('LaTeX code copied to clipboard!')
