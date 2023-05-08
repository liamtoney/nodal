"""
Generate IRIS gmap URLs for the Texan configurations during each iMUSH shot
"""

import webbrowser

from utils import get_shots

df = get_shots().sort_values(by='time')

# URL params
CHANNEL = 'DPZ'  # Nodes and Texans
NETWORK = '1D,XD'  # "MSH Node Array" & "Collaborative Research: Illuminating the..."

# Form URL for each shot, print, and optionally open in browser
for shot, row in df.iterrows():
    url = f'https://ds.iris.edu/gmap/#net={NETWORK}&cha={CHANNEL}&starttime={row.time - 60}&endtime={row.time + 300}'
    # _ = webbrowser.open(url)
    print(f'{shot.rjust(3)}: {url}')
