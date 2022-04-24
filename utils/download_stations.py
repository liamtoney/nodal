#!/usr/bin/env python

"""Downloads StationXML file for iMUSH nodal stations."""

import os
from pathlib import Path

from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# Grab "MSH Node Array"
# http://ds.iris.edu/mda/1D/?starttime=2014-01-01T00:00:00&endtime=2014-12-31T23:59:59
print('Downloading coordinates and response information...')
inventory = Client('IRIS').get_stations(
    network='1D',
    starttime=UTCDateTime(2014, 1, 1),
    endtime=UTCDateTime(2014, 12, 31),
    level='response',
    format='xml',
)
print('Done')

# Save as StationXML file
inventory.write(
    str(Path(os.environ['NODAL_WORKING_DIR']) / 'data' / '1D.xml'), format='STATIONXML'
)
