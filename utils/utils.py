import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read, read_inventory

# Read in and process shot metadata
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=UserWarning)  # Ignore "extension" warning
    df = pd.read_excel(
        Path(os.environ['NODAL_WORKING_DIR']) / 'metadata' / 'iMUSH_shot_metadata.xlsx'
    )
df.dropna(inplace=True)
df.rename(
    columns={'Shot': 'shot', 'Lat': 'lat', 'Lon': 'lon', 'Weight (lb)': 'weight_lb'},
    inplace=True,
)
df.set_index('shot', inplace=True)  # So we can easily look up shots by name
# fmt: off
def _form_time(row):
    """Convert iMUSH shot metadata spreadsheet times to UTCDateTime format."""
    frac_sec, whole_sec = np.modf(row.Sec)
    time = UTCDateTime(
        year=2014,  # Not included in spreadsheet
        julday=row.Julian,
        hour=int(row['UTC Hour']),
        minute=int(row.Min),
        second=int(whole_sec),
        microsecond=int(frac_sec * 1e6),  # Converting from seconds to microseconds
    )
    return time
# fmt: on
df['time'] = df.apply(_form_time, axis='columns')
df.drop(
    columns=[
        'Julian',
        'UTC Hour',
        'Min',
        'Sec',
        'Elev',
        'Est Mag (PNSN)',
        'Unnamed: 9',
    ],
    inplace=True,
)

# Read in station information
inv = read_inventory(str(Path(os.environ['NODAL_WORKING_DIR']) / 'metadata' / '1D.xml'))


def get_stations():
    """Return ObsPy Inventory containing 1D network information."""
    return inv


def get_waveforms_shot(shot):
    """Return ObsPy Stream containing waveforms for a given shot."""
    st = read(
        str(Path(os.environ['NODAL_WORKING_DIR']) / 'data' / 'mseed' / f'{shot}.mseed')
    )
    return st


def get_shots():
    """Return pandas DataFrame containing iMUSH shot metadata."""
    return df
